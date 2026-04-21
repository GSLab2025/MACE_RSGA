# -------------  mace/modules/rsga.py  -----------------
from __future__ import annotations

import math, torch
from torch import nn, einsum
from typing import Dict
from .k_frequencies_triclinic import  EwaldPotentialTriclinic
#from line_profiler import profile
#from mace.tools.scatter import scatter_sum
import  os
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")  # PyTorch 2.x

# ---------- helper: slice that contains ONLY the 0e channels --------------

def scalar_slice(irreps: Irreps) -> slice:
    """
    Returns a slice that grabs *all* 0e channels at the front of `irreps`,
    assuming they are stored first (default MACE ordering).
    Works with both new and old e3nn iterators.  
    """
    start = 0
    for mul_ir in irreps:
        # new API: _MulIr;  old API: Irrep
        ir   = mul_ir.ir if hasattr(mul_ir, "ir") else mul_ir
        mul  = mul_ir.mul if hasattr(mul_ir, "mul") else 1

        if ir.l != 0 or ir.p != 1:     # not a scalar-even channel
            break
        start += mul * ir.dim          # each copy contributes ir.dim dims
    return slice(0, start)             # [:start] are scalars


# -----------------------------------------------------------------------------
# Reciprocal-Space Linear Gated Attention (RS-LGA) with Fractional Fourier Phase Encoding
#
# Goal:
#   Implement a physically rigorous reciprocal-space attention mechanism that is:
#   (i) Periodicity-aware via Ewald summation logic.
#   (ii) Correct for Triclinic geometries (handling skewed lattice symmetries).
#   (iii) Invariant to cell definitions (supercells) via fractional coordinates.
#   (iv) Expressive via Gated Linear Attention (GLA) mechanisms.
#
# Key Idea (Fractional Fourier Encoding):
#   For a periodic cell with lattice matrix A (3x3), reciprocal vectors are:
#       k(n) = 2π * n * A^{-T},   where n ∈ Z^3 (integer triplets)
#   Atomic positions r are mapped to fractional coordinates f:
#       r = A f  =>  f = r A^{-1}
#
#   The Fourier phase satisfies the exact identity:
#       r · k(n) = (A f) · (2π n A^{-T}) = 2π (f · n)
#
#   Consequence:
#   The phase basis (cos/sin) depends solely on fractional positions f and
#   integer indices n, making it invariant to cell deformations. Geometry enters
#   only through the spectral weights w(k) derived from the Ewald kernel.
#
# Architecture (RS-LGA):
#   1. Input Gating ("Source Filter"):
#      - An element-wise sigmoid gate filters the Value vectors v before summation.
#      - Allows atoms to broadcast specific physical features (chemical identity)
#        while suppressing others based on local context.
#        v_gated = v * σ(W_in x)
#
#   2. Triclinic-Correct Phase Encoding:
#      - We generate a symmetric integer grid n ∈ [-N, N] to capture all
#        reflections in skewed triclinic lattices.
#      - Q and K are phase-encoded (rotated) by exp(i 2π f·n).
#
#   3. Linear Attention Aggregation:
#      - We compute the "global field" S_m for each frequency mode m:
#        S_m = Σ_t K_enc[m,t] ⊗ V_gated[t]      (Linear complexity)
#      - The unweighted update is retrieved via query projection:
#        β_m[i] = Q_enc[m,i]^T S_m
#
#   4. Weighted Accumulation:
#      - Modes are summed using Ewald spectral weights w_m:
#        update[i] = Σ_m w_m β_m[i]
#      - This is implemented as a highly optimized BLAS GEMV operation.
#
#   5. Output Gating ("Result Scaling"):
#      - A head-wise (scalar) tanh gate scales the accumulated field.
#      - Uses an identity-centered update rule for stability:
#        Output = Update * (1.0 + 0.2 * tanh(W_out x))
#
# Practical Notes:
#   - Fractional coordinates are wrapped to [0,1) to handle unwrapped MD trajectories.
#   - Processing is chunked over modes (Mc) to maintain constant memory footprint.
#   - Critical Float64 precision is enforced for grid generation and weight accumulation.
# -----------------------------------------------------------------------------
class ReciprocalSpaceGatedAttention(nn.Module): 
    def __init__(self, node_irreps, r_max: float, hidden: int = None, Mc: int = 128):
        super().__init__()
        self.scalar_sl = scalar_slice(node_irreps)
        S = self.scalar_sl.stop                      # #scalar channels

        if hidden is None:                               # ← NEW
            hidden = S                                   # ← NEW
        assert hidden % 2 == 0, "hidden must be even"    # ← keep existing check
        assert S % 2 == 0, "scalar channel count must be even (real+imag)."

        self.H = int(hidden) 
        self.Mc = int(Mc)

        # project scalar slice to hidden if needed (important if S != hidden)
        self.in_proj = nn.Identity() if S == hidden else nn.Linear(S, hidden, bias=False)

        self.qkv = nn.Linear(hidden, 3*hidden, bias=False)
        self.act    = nn.SiLU() 
        self.scale_q = 1 / math.sqrt(self.H)
        #self.norm   = nn.RMSNorm(hidden)

        # 1. Input Gate: Element-wise [Inspired by Gated Linear Attention (GLA) ]
        # We keep this element-wise (N, H) to allow "Source Filtering"
        # If this also decreases accuracy, we will downgrade it to (N, 1) later.
        self.val_gate = nn.Linear(hidden, hidden, bias=True)
        nn.init.zeros_(self.val_gate.weight)
        nn.init.constant_(self.val_gate.bias, 2.0) # Sigmoid(2) ≈ 0.88 (Open Valve)

        #Define the output and mixing  gates
        # 2. Output Gate: Head-wise (The Winner)
        # Reverted to (N, 1) as per P benchmarks
        self.head_gate = nn.Linear(hidden, 1, bias=True)   # headwise gate (N,1); Recommend option by https://arxiv.org/pdf/2505.06708
        # ZERO INITIALIZATION IS CRITICAL
        # This ensures tanh(0) = 0, so the multiplier starts at exactly 1.0
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)          # tanh(0)=0  -> scale starts at 1.0
        
        #self.elem_gate =  nn.Linear(hidden, hidden, bias=True)  # elementwise gate (N,H) #Not recommended option by https://arxiv.org/pdf/2505.06708
        # ZERO INITIALIZATION IS CRITICAL
        # This ensures tanh(0) = 0, so the multiplier starts at exactly 1.0
        #nn.init.zeros_(self.elem_gate.weight)
        #nn.init.zeros_(self.elem_gate.bias)

        #self.mix_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.get_default_dtype())) #global scalar
        self.mix_gate_node = nn.Linear(hidden, 1, bias=True)
        nn.init.zeros_(self.mix_gate_node.weight)
        nn.init.constant_(self.mix_gate_node.bias, -5.0)  # LR initially off


        self.kspace_freq = EwaldPotentialTriclinic(
            auto_sigma=True, 
            eps_real=1e-4, # TIGHT tolerance for smooth handover
            auto_cut=True, 
            eps_k=1e-3, # Minimum tolerance 
            eps_mass=1e-6, # TIGHT tolerance (crucial for accurate tail)
            normalize_weights=False, # <--- CRITICAL: Disable artificial normalization
            M_cap=1024 # Sufficient capacity
        )

        self.r_cut = r_max   # use your SR cutoff as r_c for auto-sigma

    @staticmethod
    def _rotate_from_phase(a: torch.Tensor, b: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        a,b: (N_g, H/2)
        phase: (N_g, M_blk)   (NOT transposed)
        returns: (M_blk, N_g, H)  with the same "cat" convention as previous code
        """
        # optional stability: keep phase small for trig, use if not float64
        #phase = torch.remainder(phase, 2 * math.pi)

        # a,b -> (1,N_g,H/2) broadcast against (M_blk,N_g,1)
        rot_a = a.unsqueeze(0) * cos - b.unsqueeze(0) * sin   # (M_blk,N_g,H/2)
        rot_b = a.unsqueeze(0) * sin + b.unsqueeze(0) * cos   # (M_blk,N_g,H/2)

        return torch.cat([rot_a, rot_b], dim=-1)              # (M_blk,N_g,H)

    
    def forward(self, data: Dict[str, torch.Tensor], node_feat:torch.Tensor) -> torch.Tensor:
        sl = self.scalar_sl
        scalars = node_feat[:, sl]                 # (N,S)
        scalars = self.in_proj(scalars)            # (N,H)

        pos = data['positions'].to(node_feat.dtype)                    # (N,3)
        # Explicitly cast cell to the same precision as the node features (Float64)
        cell = data['cell'].view(-1, 3, 3).to(node_feat.dtype)                            # (G,3,3)

        if data["batch"] is None:
            N = data['positions'].shape[0]
            batch = torch.zeros(N, dtype=torch.long, device=pos.device)
        else:
            batch = data["batch"]

        N = pos.shape[0]
        G = int(batch.max().item()) + 1
        H = self.H #Placeholder var
        
        # Q,K,V per node
        q, k, v = self.qkv(scalars).chunk(3, dim=-1)         # (N, H) each
        q, k    = self.act(q), self.act(k)       # (N,H)

        # ---------------------------------------------------------
        # GLA INSERTION (Input Valve)
        # ---------------------------------------------------------
        # Gate the Values *before* they enter the summation.
        # This allows the atom to "filter" what it broadcasts to the grid.
        g_in = torch.sigmoid(self.val_gate(scalars)) # (N, H)
        v = v * g_in
        
        # Pre-split real/imag pairs once
        a_q, b_q = q[..., 0::2], q[..., 1::2]           # (N,H/2)
        a_k, b_k = k[..., 0::2], k[..., 1::2]           # (N,H/2)

        update = torch.zeros((N, H), device=q.device, dtype=q.dtype)

        # Graphwise (no padding): each graph has its own M_g
        for g in range(G):
            idx = (batch == g)
            if idx.sum() == 0:
                continue

            pos_g = pos[idx]                            # (N_g,3)
            cell_g = cell[g]                            # (3,3)
            N_g_t = idx.sum().to(node_feat.dtype)  # scalar tensor on device

            # fractional coordinates f = r @ inv(cell)
            # solve cell_g.T * x = pos_g.T  => x.T = pos_g @ inv(cell_g)
            f_g = torch.linalg.solve(cell_g.T, pos_g.T).T  # (N_g,3), same dtype as pos_g
            # Optional but often beneficial in MD: wrap to [0,1)
            # This makes phase arguments well-behaved even if positions drift/unwrapped.
            f_g = f_g - torch.floor(f_g)

            v_g   = v[idx]                              # (N_g,H)
            # Stabilizer 1: This prevents the k=0 singularity from leaking into low-k modes.
            #v_g = v_g - v_g.mean(dim=0, keepdim=True)
            #print("position", pos_g)

            a_qg, b_qg = a_q[idx], b_q[idx]             # (N_g,H/2)
            a_kg, b_kg = a_k[idx], b_k[idx]             # (N_g,H/2)

            # k-grid for this triclinic cell (variable M_g)
            #k_vecs, w = self.kspace_freq(pos_g, cell[g], r_cut=self.r_cut)  # (M_g,3),(M_g,)
            #M_g = k_vecs.shape[0]
            n_vecs, w = self.kspace_freq(pos_g, cell_g, r_cut=self.r_cut, return_n=True)  # (M_g,3 long),(M_g,)
            M_g = n_vecs.shape[0]
            n_vecs_f = n_vecs.to(dtype=pos_g.dtype)   # (M_g,3) float copy for matmul

            upd_g = torch.zeros((idx.sum(), H), device=q.device, dtype=q.dtype)

            # chunk over k-modes to control memory
            for m0 in range(0, M_g, self.Mc):
                m1 = min(m0 + self.Mc, M_g)
                #k_blk = k_vecs[m0:m1]                   # (M_blk,3)
                n_blk = n_vecs_f[m0:m1].to(dtype=pos_g.dtype)          # (M_blk,3)
                w_blk = w[m0:m1].to(dtype=q.dtype) # Ensures Float64 matches Float64; (M_blk,)

                # phase for this block: (N_g, M_blk)
                #phase = pos_g @ k_blk.T
                phase = (2.0 * math.pi) * (f_g @ n_blk.T)              # (N_g, M_blk)
                phase_T = phase.transpose(0, 1)          # (M_blk, N_g)

                # cos/sin: (M_blk, N_g, 1) for broadcasting over channel pairs
                cos = phase_T.cos().unsqueeze(-1)
                sin = phase_T.sin().unsqueeze(-1)

                # rotate q,k for this block: (M_blk,N_g,H)
                q_rot = self._rotate_from_phase(a_qg, b_qg, cos, sin) * self.scale_q
                k_rot = self._rotate_from_phase(a_kg, b_kg, cos, sin)

                # S_blk[m,:,:] = sum_t k_rot[m,t,:] outer v_g[t,:]  -> (M_blk,H,H) with m=mode, t=node, a=in, b=out
                #S_blk = torch.einsum("m t a, t b -> m a b", k_rot, v_g) #Used for benchmark and works very well!
                #------- But matmul >> einsum on speed with NVIDIA GEMM, but so far einsum gives more accuracy wrt training  metrics!
                # -------------------------------------------------------------------------
                # OPTIMIZATION 1: Compute S_blk
                # Einsum: "m t a, t b -> m a b"
                # Matmul Equivalent: (M, H, N) @ (N, H) -> (M, H, H)
                # -------------------------------------------------------------------------
                # We use .contiguous() to ensure optimal memory strides for the BLAS kernel
                k_rot_T = k_rot.transpose(1, 2).contiguous() # (M_blk, H, N_g)
                S_blk = torch.matmul(k_rot_T, v_g)           # (M_blk, H, H) (Broadcasting v_g works automatically)
                # --- Stabilizer 2: Make the global sum extensive / size-consistent ---
                S_blk = S_blk / N_g_t

                
                # beta_blk[m,i,:] = q_rot[m,i,:]^T @ S_blk[m,:,:] -> (M_blk,N_g,H)
                beta_blk = torch.matmul(q_rot, S_blk)

                # -------------------------------------------------------------------------
                # OPTIMIZATION 2: Accumulate into upd_g
                # Einsum: "m, m n h -> n h" (Weighted sum over modes)
                # Optimization: Flatten (N, H) and treat this as a Vector-Matrix product.
                # This converts a slow reduction .sum() into a fast BLAS GEMV operation.
                # -------------------------------------------------------------------------
                N_curr = beta_blk.shape[1]

                # View beta as (M_blk, N_g * H)
                beta_flat = beta_blk.view(beta_blk.shape[0], -1) 

                # w_blk is (M_blk); view as (1, M_blk) for matmul
                # (1, M) @ (M, NH) -> (1, NH)
                update_flat = torch.matmul(w_blk.unsqueeze(0), beta_flat)

                # Reshape back to (N_g, H) and accumulate
                upd_g.add_(update_flat.view(N_curr, H))


            update[idx] = upd_g

        # ---------------------------------------------------------
        # OUTPUT GATING (The Winner)
        # ---------------------------------------------------------
        # Back to Head-wise (N, 1) based on P benchmarks.
        attn_gate = torch.tanh(self.head_gate(scalars)  )      # (N,1) head-wise gate beat element-wise for P benchmark. So we keep this!
        #attn_gate = torch.tanh(self.elem_gate(scalars))      # (N,H) ; element-wise leads to (N, H) * (N, H) element-wise multiplication for update!

        update = update * (1.0 + 0.2 * attn_gate )                        # Broadcast (N, H) * (N, 1)
        #print(attn_gate.mean(), attn_gate.min(), attn_gate.max())

        #gate_sr_lr  = torch.sigmoid(self.mix_gate) ; global scalar
        gate_sr_lr = torch.sigmoid(self.mix_gate_node(scalars))  # (N,1)
        #print("gate_sr_lr", gate_sr_lr)

        return update, gate_sr_lr
