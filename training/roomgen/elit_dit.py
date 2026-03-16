"""ELIT-DiT: Elastic Latent Interface Transformer for room grid generation.

Architecture:
    Head blocks (spatial domain) → Read (spatial→latent) → Core blocks (latent) →
    Write (latent→spatial) → Tail blocks (spatial domain) → output velocity

Key features:
- 2D RoPE for resolution-independent spatial positions
- Grouped cross-attention Read/Write for variable grid sizes
- Multi-budget tail dropping for importance-ordered latents
- adaLN-Zero timestep conditioning throughout
- Text + dimension conditioning via cross-attention in core blocks
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D Rotary Position Embeddings
# ---------------------------------------------------------------------------

def build_2d_rope(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    """Build 2D RoPE frequencies for a (H, W) grid.

    Returns: (H*W, dim) complex rotation factors.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE"
    half = dim // 4

    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))

    rows = torch.arange(height, device=device).float()
    cols = torch.arange(width, device=device).float()

    # Row frequencies: (H, half)
    row_freqs = torch.outer(rows, freqs)
    # Col frequencies: (W, half)
    col_freqs = torch.outer(cols, freqs)

    # Expand to (H, W, half) each
    row_freqs = row_freqs.unsqueeze(1).expand(height, width, half)
    col_freqs = col_freqs.unsqueeze(0).expand(height, width, half)

    # Interleave: (H, W, dim//2) — [row_cos, row_sin, col_cos, col_sin]
    cos_r, sin_r = row_freqs.cos(), row_freqs.sin()
    cos_c, sin_c = col_freqs.cos(), col_freqs.sin()

    # Stack to (H*W, dim)
    rope = torch.cat([cos_r, sin_r, cos_c, sin_c], dim=-1).reshape(height * width, dim)
    return rope


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply 2D RoPE to query/key tensors.

    x: (B, N, H, D) or (B, N, D) where N=H*W
    rope: (N, D)
    """
    squeezed = x.dim() == 3
    if squeezed:
        x = x.unsqueeze(2)  # (B, N, 1, D)

    D = x.shape[-1]
    half = D // 2

    x1, x2 = x[..., :half], x[..., half:]
    cos_vals = rope[:, :half].unsqueeze(0).unsqueeze(2)  # (1, N, 1, half)
    sin_vals = rope[:, half:].unsqueeze(0).unsqueeze(2)

    out = torch.cat([
        x1 * cos_vals - x2 * sin_vals,
        x1 * sin_vals + x2 * cos_vals,
    ], dim=-1)

    if squeezed:
        out = out.squeeze(2)
    return out


# ---------------------------------------------------------------------------
# adaLN-Zero modulation
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """Adaptive Layer Norm with zero-initialized gate (DiT-style)."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Produces 6 modulation params: shift1, scale1, gate1, shift2, scale2, gate2
        self.proj = nn.Linear(d_cond, 6 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """Returns (normed_x_1, gate1, normed_x_2, gate2).

        cond: (B, d_cond) — timestep embedding
        """
        params = self.proj(cond)  # (B, 6*d)
        shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(6, dim=-1)

        # For attention input
        normed1 = self.norm(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        # For FFN input
        normed2 = self.norm(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)

        return normed1, gate1.unsqueeze(1), normed2, gate2.unsqueeze(1)


# ---------------------------------------------------------------------------
# Multi-Head Attention with optional RoPE
# ---------------------------------------------------------------------------

class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, qk_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.d_head)
            self.k_norm = nn.LayerNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, H, D_head)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope is not None:
            # rope is (N, D_head) — apply per-head
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)

        # (B, H, N, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Cross-Attention (for Read/Write and text conditioning)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, d_query: int, d_kv: int, n_heads: int, qk_norm: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_query // n_heads

        self.q_proj = nn.Linear(d_query, d_query)
        self.k_proj = nn.Linear(d_kv, d_query)
        self.v_proj = nn.Linear(d_kv, d_query)
        self.out_proj = nn.Linear(d_query, d_query)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.d_head)
            self.k_norm = nn.LayerNorm(self.d_head)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = query.shape
        Nkv = kv.shape[1]

        q = self.q_proj(query).reshape(B, Nq, self.n_heads, self.d_head)
        k = self.k_proj(kv).reshape(B, Nkv, self.n_heads, self.d_head)
        v = self.v_proj(kv).reshape(B, Nkv, self.n_heads, self.d_head)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


# ---------------------------------------------------------------------------
# Spatial DiT Block (for head/tail)
# ---------------------------------------------------------------------------

class SpatialBlock(nn.Module):
    """DiT block operating on spatial tokens with 2D RoPE."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_cond: int):
        super().__init__()
        self.adaln = AdaLNZero(d_model, d_cond)
        self.attn = MHA(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        normed1, gate1, normed2, gate2 = self.adaln(x, cond)
        x = x + gate1 * self.attn(normed1, rope=rope)
        x = x + gate2 * self.ffn(normed2)
        return x


# ---------------------------------------------------------------------------
# Core Latent Block (for core processing)
# ---------------------------------------------------------------------------

class CoreBlock(nn.Module):
    """Transformer block operating on latent tokens with text cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_cond: int, d_text: int):
        super().__init__()
        self.adaln = AdaLNZero(d_model, d_cond)
        self.self_attn = MHA(d_model, n_heads)
        self.cross_attn = CrossAttention(d_model, d_text, n_heads)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        normed1, gate1, normed2, gate2 = self.adaln(x, cond)
        x = x + gate1 * self.self_attn(normed1)
        # Text cross-attention
        x = x + self.cross_attn(self.cross_norm(x), text_emb)
        x = x + gate2 * self.ffn(normed2)
        return x


# ---------------------------------------------------------------------------
# Read / Write grouped cross-attention
# ---------------------------------------------------------------------------

class GroupedReadWrite(nn.Module):
    """ELIT grouped cross-attention for Read (spatial→latent) or Write (latent→spatial).

    Partitions spatial tokens into groups of group_area cells, each with J latent tokens.
    Cross-attention is within groups only.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_cond: int,
        j_per_group: int = 8,
        group_size: int = 4,  # group_size × group_size spatial tokens per group
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.j_per_group = j_per_group
        self.group_size = group_size

        # Learnable latent token embeddings (shared across groups)
        self.latent_tokens = nn.Parameter(torch.randn(j_per_group, d_model) * 0.02)
        self.latent_pos = nn.Parameter(torch.randn(j_per_group, d_model) * 0.02)

        self.cross_attn = CrossAttention(d_model, d_model, n_heads)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # adaLN for timestep modulation
        self.adaln_proj = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.adaln_proj.weight)
        nn.init.zeros_(self.adaln_proj.bias)

    def _assign_groups(self, H: int, W: int, device: torch.device):
        """Assign each spatial position to a group.

        Returns group_ids (H*W,) and G (number of groups).
        """
        gs = self.group_size
        gH = math.ceil(H / gs)
        gW = math.ceil(W / gs)
        G = gH * gW

        group_ids = torch.zeros(H * W, dtype=torch.long, device=device)
        for r in range(H):
            for c in range(W):
                gr = min(r // gs, gH - 1)
                gc = min(c // gs, gW - 1)
                group_ids[r * W + c] = gr * gW + gc

        return group_ids, G

    def read(
        self,
        spatial: torch.Tensor,
        cond: torch.Tensor,
        H: int,
        W: int,
        j_budget: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Read: spatial tokens → latent tokens.

        spatial: (B, N, d_model) where N = H * W
        Returns: latent (B, K, d_model), group_ids (N,), G
        """
        B, N, D = spatial.shape
        device = spatial.device

        group_ids, G = self._assign_groups(H, W, device)
        J = j_budget if j_budget is not None else self.j_per_group
        K = G * J

        # Initialize latent tokens
        latent = self.latent_tokens[:J].unsqueeze(0).expand(B, -1, -1)  # (B, J, D)
        pos = self.latent_pos[:J].unsqueeze(0)
        latent = latent + pos
        # Tile across groups
        latent = latent.unsqueeze(1).expand(B, G, J, D).reshape(B, K, D)

        # adaLN modulation
        mod = self.adaln_proj(cond)  # (B, 2*D)
        shift, scale = mod.chunk(2, dim=-1)
        latent = latent * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Per-group cross-attention: latent queries attend to spatial keys in same group
        # For efficiency with variable group sizes, process all at once with masking
        # Build attention mask: (K, N) where mask[k, n] = True if same group
        latent_group = torch.arange(G, device=device).repeat_interleave(J)  # (K,)
        attn_mask = (latent_group.unsqueeze(1) == group_ids.unsqueeze(0))  # (K, N)
        # Expand to (B*H, K, N) for SDPA
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention with mask
        q = self.norm_q(latent)
        kv = self.norm_kv(spatial)

        # Manual cross-attention with mask
        Nq, Nkv = K, N
        q_proj = self.cross_attn.q_proj(q).reshape(B, Nq, self.n_heads, D // self.n_heads)
        k_proj = self.cross_attn.k_proj(kv).reshape(B, Nkv, self.n_heads, D // self.n_heads)
        v_proj = self.cross_attn.v_proj(kv).reshape(B, Nkv, self.n_heads, D // self.n_heads)

        if self.cross_attn.qk_norm:
            q_proj = self.cross_attn.q_norm(q_proj)
            k_proj = self.cross_attn.k_norm(k_proj)

        q_proj = q_proj.transpose(1, 2)  # (B, H, K, Dh)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        # Convert bool mask to float mask for SDPA
        float_mask = torch.where(
            attn_mask.unsqueeze(1),  # (B, 1, K, N)
            torch.zeros(1, device=device),
            torch.full((1,), float("-inf"), device=device),
        )

        out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, attn_mask=float_mask)
        out = out.transpose(1, 2).reshape(B, K, D)
        latent = latent + self.cross_attn.out_proj(out)

        return latent, group_ids, G

    def write(
        self,
        latent: torch.Tensor,
        spatial: torch.Tensor,
        cond: torch.Tensor,
        group_ids: torch.Tensor,
        G: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Write: latent tokens → spatial tokens.

        latent: (B, K, d_model)
        spatial: (B, N, d_model)
        Returns: updated spatial (B, N, d_model)
        """
        B, N, D = spatial.shape
        K = latent.shape[1]
        J = K // G
        device = spatial.device

        # adaLN modulation
        mod = self.adaln_proj(cond)
        shift, scale = mod.chunk(2, dim=-1)
        spatial_mod = spatial * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Build mask: spatial queries attend to latent keys in same group
        latent_group = torch.arange(G, device=device).repeat_interleave(J)  # (K,)
        attn_mask = (group_ids.unsqueeze(1) == latent_group.unsqueeze(0))  # (N, K)
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)

        q = self.norm_q(spatial_mod)
        kv = self.norm_kv(latent)

        Nq, Nkv = N, K
        q_proj = self.cross_attn.q_proj(q).reshape(B, Nq, self.n_heads, D // self.n_heads)
        k_proj = self.cross_attn.k_proj(kv).reshape(B, Nkv, self.n_heads, D // self.n_heads)
        v_proj = self.cross_attn.v_proj(kv).reshape(B, Nkv, self.n_heads, D // self.n_heads)

        if self.cross_attn.qk_norm:
            q_proj = self.cross_attn.q_norm(q_proj)
            k_proj = self.cross_attn.k_norm(k_proj)

        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        float_mask = torch.where(
            attn_mask.unsqueeze(1),
            torch.zeros(1, device=device),
            torch.full((1,), float("-inf"), device=device),
        )

        out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, attn_mask=float_mask)
        out = out.transpose(1, 2).reshape(B, N, D)
        spatial = spatial + self.cross_attn.out_proj(out)

        return spatial


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP projection."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0, 1]. Returns (B, d_cond)."""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Cell Token Embedding
# ---------------------------------------------------------------------------

class CellTokenEmbedding(nn.Module):
    """Embed multi-channel cell data into d_model tokens."""

    def __init__(self, d_model: int, n_obs_types: int = 9):
        super().__init__()
        d_type = 32
        d_height = 16
        d_elev = 16

        self.obs_embed = nn.Embedding(n_obs_types, d_type)
        self.height_proj = nn.Linear(1, d_height)
        self.elev_proj = nn.Linear(1, d_elev)
        self.out_proj = nn.Linear(d_type + d_height + d_elev, d_model)

    def forward(self, obs_type: torch.Tensor, height: torch.Tensor, elevation: torch.Tensor):
        """
        obs_type: (B, N) long
        height: (B, N) float
        elevation: (B, N) float
        Returns: (B, N, d_model)
        """
        type_emb = self.obs_embed(obs_type)  # (B, N, d_type)
        h_emb = self.height_proj(height.unsqueeze(-1))  # (B, N, d_height)
        e_emb = self.elev_proj(elevation.unsqueeze(-1))  # (B, N, d_elev)
        return self.out_proj(torch.cat([type_emb, h_emb, e_emb], dim=-1))


# ---------------------------------------------------------------------------
# Output Head
# ---------------------------------------------------------------------------

class OutputHead(nn.Module):
    """Project spatial tokens back to multi-channel cell predictions."""

    def __init__(self, d_model: int, n_obs_types: int = 9):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # obstacle_type logits + height + elevation = 9 + 1 + 1 = 11
        self.proj = nn.Linear(d_model, n_obs_types + 2)
        self.n_obs_types = n_obs_types

    def forward(self, x: torch.Tensor):
        """x: (B, N, d_model). Returns dict of predictions."""
        x = self.norm(x)
        out = self.proj(x)  # (B, N, 11)
        return {
            "obs_logits": out[..., :self.n_obs_types],  # (B, N, 9)
            "height": out[..., self.n_obs_types],  # (B, N)
            "elevation": out[..., self.n_obs_types + 1],  # (B, N)
        }


# ---------------------------------------------------------------------------
# Dimension Predictor
# ---------------------------------------------------------------------------

class DimensionPredictor(nn.Module):
    """Predict room dimensions from text embedding + room type."""

    def __init__(self, d_text: int, n_room_types: int = 7):
        super().__init__()
        d_type = 32
        self.room_type_embed = nn.Embedding(n_room_types, d_type)
        self.mlp = nn.Sequential(
            nn.Linear(d_text + d_type, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 2),  # (width_raw, depth_raw)
        )

    def forward(self, text_emb: torch.Tensor, room_type: torch.Tensor):
        """
        text_emb: (B, d_text) — pooled text embedding
        room_type: (B,) long
        Returns: (B, 2) — (width, depth) in [8, 64]
        """
        rt_emb = self.room_type_embed(room_type)
        x = torch.cat([text_emb, rt_emb], dim=-1)
        raw = self.mlp(x)  # (B, 2)
        # Sigmoid → scale to [8, 64]
        dims = torch.sigmoid(raw) * 56.0 + 8.0
        return dims


# ---------------------------------------------------------------------------
# Full ELIT-DiT Model
# ---------------------------------------------------------------------------

class ELITDiT(nn.Module):
    """ELIT-DiT for room grid generation with rectified flow.

    Architecture:
        B_in spatial head blocks → Read → B_core latent blocks → Write →
        B_out spatial tail blocks → output
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        d_text: int = 384,
        n_head_blocks: int = 2,
        n_core_blocks: int = 8,
        n_tail_blocks: int = 2,
        j_per_group: int = 8,
        group_size: int = 4,
        n_obs_types: int = 9,
        n_room_types: int = 7,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.j_per_group = j_per_group
        self.group_size = group_size

        # Timestep embedding
        d_cond = d_model * 4
        self.time_embed = TimestepEmbedding(d_model, d_cond)

        # Dimension conditioning: project (w, d) to d_cond
        self.dim_proj = nn.Sequential(
            nn.Linear(2, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Cell token embedding
        self.cell_embed = CellTokenEmbedding(d_model, n_obs_types)

        # Text embedding projection (text → token sequence for cross-attn)
        self.text_proj = nn.Linear(d_text, d_model)
        # Expand single text embedding to a short sequence for cross-attn
        self.text_expand = nn.Linear(d_model, d_model * 4)

        # Head blocks (spatial domain)
        self.head_blocks = nn.ModuleList([
            SpatialBlock(d_model, n_heads, d_ff, d_cond)
            for _ in range(n_head_blocks)
        ])

        # Read/Write interface
        self.read_write = GroupedReadWrite(d_model, n_heads, d_cond, j_per_group, group_size)

        # Core blocks (latent domain)
        self.core_blocks = nn.ModuleList([
            CoreBlock(d_model, n_heads, d_ff, d_cond, d_model)
            for _ in range(n_core_blocks)
        ])

        # Tail blocks (spatial domain)
        self.tail_blocks = nn.ModuleList([
            SpatialBlock(d_model, n_heads, d_ff, d_cond)
            for _ in range(n_tail_blocks)
        ])

        # Output head
        self.output_head = OutputHead(d_model, n_obs_types)

        # Dimension predictor (separate, trained with L2 loss)
        self.dim_predictor = DimensionPredictor(d_text, n_room_types)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        obs_type: torch.Tensor,
        height: torch.Tensor,
        elevation: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        width: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        j_budget: Optional[int] = None,
    ) -> dict:
        """Forward pass for denoising.

        Args:
            obs_type: (B, D_max, W_max) long — noisy obstacle types (as soft indices)
            height: (B, D_max, W_max) float — noisy height values
            elevation: (B, D_max, W_max) float — noisy elevation values
            t: (B,) float — diffusion timestep in [0, 1]
            text_emb: (B, d_text) — text embedding
            width: (B,) long — room widths
            depth: (B,) long — room depths
            mask: (B, D_max, W_max) bool — valid cells
            j_budget: optional int — latent tokens per group (for multi-budget)

        Returns dict with:
            velocity: dict with obs_logits (B, N, 9), height (B, N), elevation (B, N)
        """
        B = obs_type.shape[0]
        device = obs_type.device
        D_max = obs_type.shape[1]
        W_max = obs_type.shape[2]

        # Flatten spatial dimensions
        N = D_max * W_max
        obs_flat = obs_type.reshape(B, N)
        h_flat = height.reshape(B, N)
        e_flat = elevation.reshape(B, N)

        # Embed cells
        x = self.cell_embed(obs_flat, h_flat, e_flat)  # (B, N, d_model)

        # Build condition embedding
        t_emb = self.time_embed(t)  # (B, d_cond)
        dim_cond = self.dim_proj(
            torch.stack([width.float(), depth.float()], dim=-1)
        )  # (B, d_cond)
        cond = t_emb + dim_cond

        # Build text tokens for cross-attention
        text_tok = self.text_proj(text_emb)  # (B, d_model)
        text_expanded = self.text_expand(text_tok).reshape(B, 4, self.d_model)  # (B, 4, d_model)

        # Build 2D RoPE for max grid dimensions (per-head dimension)
        d_head = self.d_model // self.n_heads
        rope = build_2d_rope(D_max, W_max, d_head, device)

        # === Head blocks (spatial) ===
        for block in self.head_blocks:
            x = block(x, cond, rope)

        # === Read: spatial → latent ===
        latent, group_ids, G = self.read_write.read(x, cond, D_max, W_max, j_budget)

        # === Core blocks (latent) ===
        for block in self.core_blocks:
            latent = block(latent, cond, text_expanded)

        # === Write: latent → spatial ===
        x = self.read_write.write(latent, x, cond, group_ids, G, D_max, W_max)

        # === Tail blocks (spatial) ===
        for block in self.tail_blocks:
            x = block(x, cond, rope)

        # === Output ===
        out = self.output_head(x)

        # Reshape back to grid
        out["obs_logits"] = out["obs_logits"].reshape(B, D_max, W_max, -1)
        out["height"] = out["height"].reshape(B, D_max, W_max)
        out["elevation"] = out["elevation"].reshape(B, D_max, W_max)

        # Apply mask — zero out predictions for padding cells
        if mask is not None:
            out["obs_logits"] = out["obs_logits"] * mask.unsqueeze(-1).float()
            out["height"] = out["height"] * mask.float()
            out["elevation"] = out["elevation"] * mask.float()

        return out

    def predict_dims(self, text_emb: torch.Tensor, room_type: torch.Tensor) -> torch.Tensor:
        """Predict room dimensions from text + room type. Returns (B, 2)."""
        return self.dim_predictor(text_emb, room_type)
