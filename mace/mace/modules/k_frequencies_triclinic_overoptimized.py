from __future__ import annotations
import torch, math
from torch import nn
@torch.compile
def _reciprocal(cell: torch.Tensor) -> torch.Tensor:
    # recip row-vectors b_i = 2π * A^{-T}
    return 2.0 * math.pi * torch.inverse(cell).T  # (3,3)

# NOTE:
#   In addition to returning k-vectors k(n)=2π n A^{-T}, this module can return
#   the integer triplets n_sel directly (return_n=True). Downstream attention
#   uses the exact phase identity:
#       r · k(n) = 2π (f · n),   where f = r A^{-1}
#   so that the phase basis is cell-invariant and only weights depend on cell.
class EwaldPotentialTriclinic(nn.Module):
    """
    Triclinic reciprocal grid with Gaussian (Ewald-like) damping and
    automatic k-selection by cumulative mass.

    Parameters
    ----------
    dl : fallback real-space grid resolution (Å) to set a coarse k-sphere
         if auto_cut is False.
    sigma : Gaussian width (Å). If auto_sigma=True, this is ignored.
    auto_sigma : choose sigma from (r_cut, eps_real).
    eps_real : target real-space split tolerance (for sigma).
    auto_cut : choose k_max from (sigma, eps_k).
    eps_k    : tolerance for reciprocal-space tail (for k_max).
    eps_mass : cumulative-mass coverage; keep k until cumulative_w >= 1 - eps_mass.
    normalize_weights : if True, return w_k normalized to sum 1 (default).
    M_cap : Hard cap on the k-modes.
    """
    def __init__(self,
                 dl: float = 10.0,
                 sigma: float = 5.0,
                 auto_sigma: bool = True,
                 eps_real: float = 1e-3,
                 auto_cut: bool = True,
                 eps_k: float = 1e-4,
                 eps_mass: float = 1e-3,
                 normalize_weights: bool = True,
                 M_cap: int | None = None):
        super().__init__()
        self.dl = dl
        self._sigma_user = sigma
        self.auto_sigma = auto_sigma
        self.eps_real = eps_real
        self.auto_cut = auto_cut
        self.eps_k = eps_k
        self.eps_mass = eps_mass
        self.normalize_weights = normalize_weights
        self.M_cap  = M_cap
        self.two_pi = 2.0 * math.pi

    @torch.no_grad()
    def forward(self,
                r_cart: torch.Tensor,      # (N,3) unused, kept for API symmetry
                cell:   torch.Tensor,      # (3,3)
                r_cut:  float | None = None,  # SR cutoff to set sigma if auto
                return_n: bool = False  
               ):
        device, dtype = cell.device, cell.dtype
        recip = _reciprocal(cell)                 # (3,3)
        b_len = recip.norm(dim=1)                 # |b1|,|b2|,|b3|

        # -- decide sigma --
        if self.auto_sigma:
            assert r_cut is not None and r_cut > 0.0, \
                "auto_sigma=True requires r_cut (SR cutoff)."
            # alpha = sqrt(-ln eps_real) / r_cut
            alpha = math.sqrt(-math.log(self.eps_real)) / float(r_cut)
            sigma = 1.0 / (math.sqrt(2.0) * alpha)
        else:
            sigma = float(self._sigma_user)

        sigma_sq_half = (sigma * sigma) / 2.0

        # -- decide k_max --
        if self.auto_cut:
            # k_max = 2 alpha sqrt(-ln eps_k)
            alpha = 1.0 / (math.sqrt(2.0) * sigma)
            k_max = 2.0 * alpha * math.sqrt(-math.log(self.eps_k))
        else:
            # fallback to sphere implied by dl
            k_max = (self.two_pi / float(self.dl))

        # integer bounds along each reciprocal basis
        n_max = torch.ceil(torch.tensor(k_max, device=device, dtype=dtype) / b_len).to(torch.long)
        
        #print("sigma, k_max",sigma,k_max)
        #print("n_max", n_max)

        nx = torch.arange(0, n_max[0] + 1, device=device) # torch.arange(-n_max[0], n_max[0] + 1, device=device)
        ny = torch.arange(0, n_max[1] + 1, device=device) # torch.arange(-n_max[1], n_max[1] + 1, device=device)
        nz = torch.arange(0, n_max[2] + 1, device=device) # torch.arange(-n_max[2], n_max[2] + 1, device=device)
        nx_m, ny_m, nz_m = torch.meshgrid(nx, ny, nz, indexing="ij")

        # --- NEW: keep integer triplets as long ---
        n_vec_int = torch.stack((nx_m, ny_m, nz_m), dim=-1).reshape(-1, 3)  # (P,3), long
        # float copy for k computations
        n_vec = n_vec_int.to(dtype)

        # k = n @ recip, remove k=0
        kvec = n_vec @ recip                                # (P,3)
        k_sq = (kvec * kvec).sum(dim=1)                     # (P,)
        mask = k_sq > 0

        # --- NEW: apply mask consistently to integer and float versions ---
        n_vec_int = n_vec_int[mask]                         # (P',3) long
        kvec = kvec[mask]                                   # (P',3)
        k_sq = k_sq[mask]                                   # (P',)

        # raw importance scores: w_k = exp(-σ² k² / 2) / k²
        w = torch.exp(-sigma_sq_half * k_sq) / k_sq         # (P',)

        # --- handle degenerate cases early ---
        if w.numel() == 0:
            # No nonzero k-vectors survived masking; fallback
            n_sel = torch.zeros((1, 3), device=device, dtype=torch.long)
            w_sel = torch.ones((1,), device=device, dtype=dtype)
            if self.normalize_weights:
                w_sel = w_sel / w_sel.sum()
            return (n_sel, w_sel) if return_n else (torch.zeros((1,3), device=device, dtype=dtype), w_sel)

        total = w.sum()
        if not torch.isfinite(total) or total <= 0:
            # robust fallback: pick the best finite entry
            w_safe = torch.nan_to_num(w, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
            sel = torch.argmax(w_safe).view(1)
        else:
            cutoff_mass = (1.0 - self.eps_mass) * total

            if self.M_cap is not None:
                K = min(self.M_cap, w.numel())
                w_top, idx_top = torch.topk(w, k=K, largest=True, sorted=True)  # desc
                cum = torch.cumsum(w_top, dim=0)

                cut = torch.tensor(cutoff_mass, device=device, dtype=cum.dtype)
                m_keep = int(torch.searchsorted(cum, cut).item()) + 1
                m_keep = min(max(m_keep, 1), K)
                sel = idx_top[:m_keep]
            else:
                w_sorted, idx = torch.sort(w, descending=True)
                cum = torch.cumsum(w_sorted, dim=0)

                cut = torch.tensor(cutoff_mass, device=device, dtype=cum.dtype)
                m_keep = int(torch.searchsorted(cum, cut).item()) + 1
                m_keep = min(max(m_keep, 1), idx.numel())
                sel = idx[:m_keep]

        w_sel = w[sel]
        n_sel = n_vec_int[sel]

        if self.normalize_weights:
            w_sum = w_sel.sum()
            if w_sum > 0:
                w_sel = w_sel / w_sum

        if return_n:
            return n_sel, w_sel
        else:
            kvec_sel = kvec[sel]
            return kvec_sel, w_sel

