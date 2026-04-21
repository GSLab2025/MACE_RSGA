# -------------  mace/modules/fourier_efa.py  -----------------
from __future__ import annotations

import math, torch
from torch import nn, einsum
from typing import Dict
#from .k_frequencies import EwaldPotential
from .k_frequencies_triclinic import  EwaldPotentialTriclinic
#from line_profiler import profile
from mace.tools.scatter import scatter_sum

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



class ReciprocalSpaceAttention(nn.Module): 
    def __init__(self, node_irreps, r_max: float,
                 hidden: int = 64):
        super().__init__()
        self.scalar_sl = scalar_slice(node_irreps)
        S = self.scalar_sl.stop                      # #scalar channels
        if hidden is None:                               # ← NEW
            hidden = S                                   # ← NEW
        assert hidden % 2 == 0, "hidden must be even"    # ← keep existing check

        assert S % 2 == 0, "scalar channel count must be even (real+imag)."

        self.H = int(hidden) 
        # self.embed_dim = hidden
        # self.output_dim = 3 * hidden
        # self.qkv = LinearReadoutBlock(o3.Irreps(str(self.embed_dim) + "x0e"), o3.Irreps(self.output_dim + "x0e"))
        self.qkv = nn.Linear(hidden, 3*hidden, bias=False)

        # self.rope_weights = nn.Parameter(torch.ones(58)) # for dimers

        #self.rope_weights = nn.Parameter(torch.ones(58), requires_grad=True) # for water
        self.rope_weights = None  # remove fixed-length parameter

        self.scale_q = 1 / math.sqrt(self.H)
        self.norm   = nn.LayerNorm(hidden)
        self.alpha  = nn.Parameter(torch.tensor(0.1))
        self.act    = nn.SiLU()   # SiLU
        #self.kspace_freq = EwaldPotential()
        #self.kspace_freq = EwaldPotentialTriclinic()
        self.kspace_freq = EwaldPotentialTriclinic(
            auto_sigma=True,   eps_real=1e-3,
            auto_cut=True,     eps_k=1e-4,
            eps_mass=1e-3,
            normalize_weights=True
        )
        self.r_cut = r_max   # use your SR cutoff as r_c for auto-sigma

    # rotary positional encoding ------------------------------------------------
    def _rope(self, h:torch.Tensor, pos:torch.Tensor,  cell:torch.Tensor) -> torch.Tensor:
        # h : (N,H) → (M,N,H)
        a, b = h[..., 0::2], h[..., 1::2]                    # (N,H/2)
        #u, _ = self.kspace_freq(pos, box)                    # (M,3)
        #k_vecs, _ = self.kspace_freq(pos, cell)      # k_vecs : (M,3)
        # ask triclinic module for (k, w)
        k_vecs, w_k = self.kspace_freq(pos, cell, r_cut=self.r_cut)  # (M,3),(M,)
        phase = pos @ k_vecs.T                       # (N,M)
        # if self.rope_weights is None:
            # init = torch.ones(u.shape[0], device=h.device, dtype=h.dtype)
            # self.rope_weights = torch.nn.Parameter(init, requires_grad=True)
        
        #phase = torch.matmul(pos, u.T)                        # (N,M)
        phase = phase[...,None]                               # (N,M,H/2)
        phase = phase.permute(1,0,2)                          # (M,N,H/2)

        cos, sin = phase.cos(), phase.sin()
        rot_a =  a.unsqueeze(0)*cos - b.unsqueeze(0)*sin
        rot_b =  a.unsqueeze(0)*sin + b.unsqueeze(0)*cos
        #w = torch.ones(k_vecs.shape[0], device=h.device, dtype=h.dtype) #Place holder dummy

        #return torch.cat([rot_a, rot_b], dim=-1), self.rope_weights             # (M,N,H)
        return torch.cat([rot_a, rot_b], dim=-1), w_k        # (M,N,H), (M,)

    # Graphwise rotary positional encoding iteratively ------------------------------------------------
    def _rope_graphwise(self,
                        h:    torch.Tensor,   # (N, H)
                        pos:  torch.Tensor,   # (N, D)
                        cell: torch.Tensor,
                        batch: torch.Tensor     
                    ) -> torch.Tensor:

        #out = []
        rot_blocks, w_blocks, M_sizes = [], [], []
        unique_graphs = torch.unique(batch)   # (G,)
        for g in unique_graphs:         
            idx = (batch == g)
            h_g   = h[idx]                   # (N_g, H)
            pos_g = pos[idx]                 # (N_g, 3)
            cell_g = cell[g]
            #out_g, weights = self._rope(h_g, pos_g, box_g)   # (M, N_g, H)
            #out_g, weights = self._rope(h_g, pos_g, box_g, cell_g)
            rot_g, w_g = self._rope(h_g, pos_g, cell_g)   # (M_g,N_g,H), (M_g,)
            #out.append(out_g)
            rot_blocks.append(rot_g)
            w_blocks.append(w_g)
            M_sizes.append(rot_g.shape[0])

        M_max = max(M_sizes)    # biggest grid in this batch
        # pad every block to (M_max, N_g, H)
        #padded = []
        rot_pad, w_pad = [], []
        #for rot_h in out_blocks:
        #    if rot_h.shape[0] < M_max:
        #        pad_M = M_max - rot_h.shape[0]
        #        pad   = torch.zeros(pad_M, *rot_h.shape[1:],
        #                        device=rot_h.device, dtype=rot_h.dtype)
        #        rot_h = torch.cat([rot_h, pad], dim=0)
        #    padded.append(rot_h)
        for rot_g, w_g in zip(rot_blocks, w_blocks):
            pad_M = M_max - rot_g.shape[0]
            if pad_M > 0:
                rot_g = torch.cat([rot_g,
                                   torch.zeros(pad_M, *rot_g.shape[1:],
                                               device=rot_g.device, dtype=rot_g.dtype)], dim=0)
                w_g   = torch.cat([w_g,
                                   torch.zeros(pad_M, device=w_g.device, dtype=w_g.dtype)], dim=0)
            rot_pad.append(rot_g);  w_pad.append(w_g)

        #out = torch.cat(out, dim=1)         # (M, N, H)
        #out = torch.cat(padded, dim=1)                  # (M_max, N, H)
        # print('kfreq weights', weights)

        #return out,None # we do not need the per-k weights later -->  return dummy
        rot_all = torch.cat(rot_pad, dim=1)        # (M_max, N, H)
        w_all   = torch.stack(w_pad, dim=1)        # (M_max, G)
        return rot_all, w_all
    
    # forward ------------------------------------------------------------------
    def forward(self, data: Dict[str, torch.Tensor], node_feat:torch.Tensor) -> torch.Tensor:
        sl = self.scalar_sl
        scalars = node_feat[:, sl]                 # (N,S)

        # --------------------------------  variables  ---------------------------
        pos = data['positions'].to(node_feat.dtype)                    # (N,3)

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch = data["batch"]

        # --------------------------------  inside forward()  -------------------------
        box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1) # old
        # new: keep full 3×3 cell matrix
        cell = data['cell'].view(-1, 3, 3)                            # (G,3,3)

        q, k, v = self.qkv(scalars).chunk(3, dim=-1)         # (N, H) each
        q, k    = self.act(q), self.act(k)       # ψ
        # Old 
        #(q, weights_q), (k, weights_k) = (self._rope_graphwise(x, pos,  cell, batch) for x in (q, k))    # (M, N, H)
        # ---- rotary encoding (padding handled inside) ----
        #q, _ = self._rope_graphwise(q, pos, cell, batch)   # (M_max, N, H)
        #k, _ = self._rope_graphwise(k, pos, cell, batch)   # (M_max, N, H)
        
        (q_rot, w_q), (k_rot, w_k) = (
        self._rope_graphwise(x, pos, cell, batch) for x in (q, k)
        )  # q_rot,k_rot: (M,N,H); w_q,w_k: (M,G)
                
        # Optional: combine the two weight sets. Using w = w_q (or average) works.
        # Project per-graph weights to per-node by batching index
        G = int(batch[-1]) + 1
        # pick one set:
        w = w_q    # (M,G)
        # expand to per-node:
        w_node = w[:, batch]             # (M,N)

        if not hasattr(self, "scale_q") or self.scale_q is None:
            self.scale_q = 1.0 / math.sqrt(q_rot.shape[-1])
        q = q_rot * self.scale_q

        # K ⊗ V per node
        kv_node = k_rot.unsqueeze(-1) * v.unsqueeze(-2)   # (M,N,H,H)

        # graph reduce -> broadcast as before
        kv_graph = scatter_sum(kv_node, batch, dim=1, dim_size=G)  # (M,G,H,H)
        kv_node  = kv_graph[:, batch]                              # (M,N,H,H)

        beta = (q.unsqueeze(-1) * kv_node).sum(-2)                 # (M,N,H)

        # ---- weighted sum across k, using w_node ----
        update = (w_node[..., None] * beta).sum(0)                 # (N,H)


        return update
        
# ----------------------------------------------------------------------------- 
