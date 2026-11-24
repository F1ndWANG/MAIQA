from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import timm
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MAIQAConfig:
    """Configuration for the MAIQA structured quality decomposition model."""

    backbone: str = "vit_base_patch16_224"
    pretrained: bool = True
    backbone_path: Optional[str] = None
    image_size: int = 224
    backbone_chunk_size: int = 192
    backbone_drop_path_rate: float = 0.0
    feature_layers: Tuple[int, int, int] = (3, 7, 11)

    # GA-Adapter
    adapter_dim: int = 96
    adapter_bottleneck: int = 24
    # Default values preserve the archived MAIQA modulation exactly.  The
    # centered/bounded options are reserved for explicitly configured
    # post-audit experiments and never activate implicitly.
    qie_adapter_transform: str = "sigmoid"  # sigmoid | centered_logits
    qie_adapter_temperature: float = 1.0
    adapter_alpha_mode: str = "free"  # free | bounded
    adapter_alpha_init: float = 0.0
    adapter_alpha_max: float = 0.10
    # ``feature_layers`` keeps the original three post-stage adapters.
    # ``every_block`` inserts an independent adapter after every ViT block up
    # to the deepest requested feature layer.  Blocks are assigned to the
    # fine/medium/coarse QIE prior by the three feature-layer boundaries.
    adapter_placement: str = "feature_layers"  # feature_layers | every_block

    # GR-QIE
    use_gr_qie: bool = True
    qie_low_res_size: int = 56
    # Controlled mechanism studies.  These switches separate using QIE before
    # representation formation from using it only for post-hoc pooling.
    qie_adapter_conditioning: bool = True
    qie_role_pooling: bool = True
    # Spatial-field sharing audit. ``shared`` preserves archived checkpoints;
    # ``shared_matched`` adds an unused, parameter-matching role estimator;
    # ``unshared`` uses that estimator for local-role pooling.
    qie_route_sharing: str = "shared"  # shared | shared_matched | unshared

    # Distortion Sidecar
    use_distortion_sidecar: bool = True
    sidecar_min_retain: float = 0.20
    sidecar_fusion: str = "controlled"  # controlled | gated | additive

    # Spatially aligned distortion evidence.  Unlike the legacy sidecar, which
    # globally pools its 14x14 feature map into one node, this path preserves a
    # distortion token for every ViT patch and injects it before role pooling.
    use_spatial_distortion_tokens: bool = False
    spatial_distortion_dim: int = 128
    spatial_injection_scale: float = 0.10

    # Role Graph
    node_dim: int = 128
    graph_heads: int = 4
    graph_layers: int = 1
    graph_mlp_ratio: float = 2.0
    role_interaction: str = "graph"  # graph | mlp | identity

    # Pooling & Head
    spatial_focus: float = 0.65
    role_pooling: str = "global_local"  # global_local | global_only | local_only
    dropout: float = 0.10
    use_role_sub_scores: bool = True

    # Image-level global/local set reasoning.  All local crops and the global
    # thumbnail share the same ViT; this is a set aggregator, not an MoE.
    use_global_local_set_pooling: bool = False
    set_pool_heads: int = 4
    set_pool_hidden: int = 256
    set_pool_entropy_floor: float = 0.55
    set_pool_use_uncertainty: bool = False
    # The learned set path is a bounded correction to the strong arithmetic
    # crop mean.  A zero-initialized residual makes the initial grouped model
    # exactly recover independent-crop mean pooling.
    set_residual_logit_scale: float = 1.0

    # ---- Multi-scale patch sampling (resolution-agnostic) ----
    # When None, multi-scale mode is DISABLED and the model uses the original
    # single-patch / grouped-patch forward path (full backward compatibility).
    # Set to e.g. (1.0, 0.5, 0.25) to enable multi-scale patch sampling.
    multiscale_scales: Optional[Tuple[float, ...]] = None
    # Number of patches per scale (must match length of multiscale_scales)
    patches_per_scale: Optional[Tuple[int, ...]] = None
    # Aggregation method: "mean" | "attention" | "variance_weighted"
    patch_aggregation: str = "attention"

    # Legacy (kept for config compatibility, not used)
    artifact_channels: int = 64
    use_fragments: bool = False

    # ---- Partial backbone fine-tuning ----
    # ``backbone_finetune_all`` is the literal end-to-end setting, including
    # patch embedding and positional/class tokens.  The layer list retains the
    # historical block-only fine-tuning behaviour.
    backbone_finetune_all: bool = False
    # When backbone_finetune_layers is set (e.g. [10, 11]), those ViT blocks
    # will be unfrozen and trained with a scaled learning rate.
    backbone_finetune_layers: Optional[Tuple[int, ...]] = None
    backbone_lr_scale: float = 0.1

    # ---- Standalone minimal multi-depth controls ----
    # These fields are ignored by the full MAIQA architecture.  The role-layer
    # tuple may contain one layer (single-depth control), two layers (pairwise
    # complementarity), three distinct layers, or a repeated layer
    # (capacity-matched redundancy test).
    minimal_role_layers: Optional[Tuple[int, ...]] = None
    minimal_fusion: str = "roles"  # roles | concat | adaptive_coarse
    minimal_role_interaction: str = "mlp"  # mlp | identity | gated_exchange
    set_weight_uniform_floor: float = 0.50

    # ---- Sufficient-depth conserved evidence model ----
    # A non-zero floor prevents the spatial reliability distribution from
    # starving most tokens.  Routing switches are exposed only for controlled
    # ablations; the proposed model learns both spatial and depth reliability.
    evidence_attention_floor: float = 0.05
    evidence_temperature: float = 1.0
    evidence_spatial_routing: str = "learned"  # learned | uniform
    evidence_depth_routing: str = "learned"  # learned | uniform

    # ---- Competitive conserved residual evidence model ----
    # CCRE uses parameter-free token-wise Gram--Schmidt residualization, a
    # single depth--spatial evidence simplex, and one shared bounded evidence
    # function.  These options are consumed only by ``ccre_iqa``.
    ccre_residual_eps: float = 1.0e-6
    ccre_detach_projection_basis: bool = True
    ccre_logit_scale_init: float = 4.0

    # ---- Conditional Orthogonal Depth-Innovation IQA ----
    # CODI learns image-conditioned projection strengths instead of imposing
    # unconditional Gram--Schmidt removal.  Bounds keep both the shared
    # representation and the depth-exclusive innovation identifiable.
    codi_projection_min: float = 0.10
    codi_projection_max: float = 0.90
    codi_router_hidden: int = 192

    # ---- Shared--Innovation Conserved Evidence model ----
    sice_role_floor: float = 0.02
    sice_spatial_floor: float = 0.02
    sice_router_temperature: float = 1.0
    sice_evidence_scale_min: float = 1.0
    sice_evidence_scale_max: float = 8.0

    # ---- Invertible Contrastive Conserved Evidence model ----
    # ICCE replaces lossy sequential projection with an orthonormal Helmert
    # transform over depth.  The transform yields one shared coordinate and
    # two exclusive depth increments without discarding information.
    icce_spatial_floor: float = 0.05
    icce_spatial_temperature: float = 1.0
    icce_spatial_routing: str = "learned"  # learned | uniform

    # ---- Head-preserving Interventional Conserved Evidence model ----
    # HICE retains the strongest minimal predictor.  A bounded zero-initialized
    # residual may use only the two depth-contrast coordinates; setting the
    # bound to zero is exactly the minimal predictor with parameter-free
    # interventional Shapley explanations.
    hice_residual_scale_max: float = 0.0
    hice_residual_detach_coordinates: bool = True

    # ---- Optional complementary semantic evidence tower ----
    # These options are consumed only by ``dual_evidence``.  The original
    # timm/HF backbone remains the local-distortion stream; this second tower
    # contributes independently decoded, exactly additive semantic evidence.
    semantic_hf_backbone: Optional[str] = None
    semantic_hf_backbone_path: Optional[str] = None
    semantic_feature_layers: Tuple[int, ...] = (7, 25)
    semantic_token_grid_side: int = 14
    semantic_finetune_all: bool = False
    semantic_finetune_layers: Optional[Tuple[int, ...]] = None
    evidence_source_routing: str = "global"  # global | uniform
    evidence_source_floor: float = 0.05

    # ---- HuggingFace vision backbones (optional, replaces the timm ViT) ----
    # One of {"siglip2_naflex", "qwen25_vl_vit"} or None (default timm path).
    # feature_layers are 0-based block indices into the HF encoder, and
    # token_grid_side is the token grid side at cfg.image_size
    # (SigLIP2: 224/16 = 14, Qwen2.5-VL: 224/14 = 16).
    hf_backbone: Optional[str] = None
    hf_backbone_path: Optional[str] = None
    token_grid_side: int = 14


def _load_backbone_checkpoint(model: nn.Module, path_value: str) -> None:
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained backbone not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for wrapper_key in ("model", "state_dict"):
            if wrapper_key in state and isinstance(state[wrapper_key], dict):
                state = state[wrapper_key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Backbone checkpoint must contain a state dict: {path}")
    state = {
        str(key).removeprefix("module.").removeprefix("model."): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    if len(incompatible.unexpected_keys) == len(state):
        raise ValueError(f"No tensors in {path} matched the configured backbone")


# ---------------------------------------------------------------------------
# GR-QIE: Granularity-Redistributed Quality Importance Estimator
# ---------------------------------------------------------------------------


class PatchAggregator(nn.Module):
    """Aggregate multi-scale patch embeddings into a single image-level representation.

    Supports three aggregation modes:
    - "mean": simple average over all patch embeddings
    - "attention": learnable attention-weighted aggregation (learns "where to look")
    - "variance_weighted": inverse-variance weighting from sub-score variance
    """

    def __init__(
        self,
        embed_dim: int = 768,
        node_dim: int = 128,
        num_scales: int = 3,
        aggregation: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.node_dim = node_dim
        self.aggregation = aggregation

        # Project ViT embedding to node_dim
        self.patch_project = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Scale-aware positional Embedding: each scale gets a learnable embedding
        # This lets the model distinguish patches from different scales
        self.scale_embeddings = nn.Parameter(
            torch.randn(1, num_scales, node_dim) * 0.02
        )

        if aggregation == "attention":
            # Cross-attention: query = [IMG] token, keys/values = patch embeddings
            self.img_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
            self.attn_norm = nn.LayerNorm(node_dim)
            self.attn = nn.MultiheadAttention(
                node_dim, num_heads=4, dropout=dropout, batch_first=True,
            )
            self.attn_out_norm = nn.LayerNorm(node_dim)
        elif aggregation == "variance_weighted":
            # Sub-score head for per-patch quality (used to compute variance weights)
            self.patch_score_head = nn.Sequential(
                nn.Linear(node_dim, node_dim // 4),
                nn.GELU(),
                nn.Linear(node_dim // 4, 1),
            )
            nn.init.normal_(self.patch_score_head[-1].weight, std=0.02)
            nn.init.zeros_(self.patch_score_head[-1].bias)
        # "mean" needs no extra parameters

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        scale_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Aggregate patch embeddings.

        Args:
            patch_embeddings: [B, N, embed_dim] - ViT embeddings for all patches
            scale_ids: [B, N] - integer scale index for each patch (0, 1, 2, ...)

        Returns:
            (aggregated, extras) where:
              aggregated: [B, node_dim*2] - image-level representation (mean + std)
              extras: dict with patch_scores, patch_weights etc.
        """
        B, N, D = patch_embeddings.shape
        projected = self.patch_project(patch_embeddings)  # [B, N, node_dim]

        # Add scale-aware positional embedding
        scale_emb = self.scale_embeddings[:, scale_ids.long(), :]  # [1, B, N, node_dim]
        scale_emb = scale_emb.squeeze(0)  # [B, N, node_dim]
        projected = projected + scale_emb

        extras: Dict[str, torch.Tensor] = {}

        if self.aggregation == "attention":
            # Attention-weighted aggregation
            q = self.img_query.expand(B, -1, -1)  # [B, 1, node_dim]
            kv = self.attn_norm(projected)  # [B, N, node_dim]
            attn_out, attn_weights = self.attn(
                q, kv, kv, need_weights=True,
            )  # attn_out: [B, 1, node_dim], attn_weights: [B, 1, N]
            pooled = self.attn_out_norm(attn_out).squeeze(1)  # [B, node_dim]
            extras["patch_weights"] = attn_weights.squeeze(1)  # [B, N]
            # Also compute patch-level std for richer representation
            std_pool = projected.std(dim=1, unbiased=False)  # [B, node_dim]
            aggregated = torch.cat([pooled, std_pool], dim=-1)  # [B, node_dim*2]

        elif self.aggregation == "variance_weighted":
            # Variance-aware weighting (similar to forward_grouped)
            patch_scores = torch.sigmoid(
                self.patch_score_head(projected).squeeze(-1)
            )  # [B, N]
            sub_var = projected.var(dim=-1)  # [B, N]
            confidence = 1.0 / (1.0 + 10.0 * sub_var)  # [B, N]
            weights = confidence / confidence.sum(dim=1, keepdim=True)  # [B, N]
            pooled = (projected * weights.unsqueeze(-1)).sum(dim=1)  # [B, node_dim]
            std_pool = projected.std(dim=1, unbiased=False)  # [B, node_dim]
            aggregated = torch.cat([pooled, std_pool], dim=-1)
            extras["patch_scores"] = patch_scores
            extras["patch_weights"] = weights

        else:  # "mean"
            mean_pool = projected.mean(dim=1)  # [B, node_dim]
            std_pool = projected.std(dim=1, unbiased=False)  # [B, node_dim]
            aggregated = torch.cat([mean_pool, std_pool], dim=-1)
            extras["patch_weights"] = torch.full(
                (B, N), 1.0 / N, device=projected.device
            )

        return aggregated, extras


class GranularityRedistributedQIE(nn.Module):
    """Predict granularity-specific quality importance from a low-resolution variant.

    Shared backbone extracts features; three lightweight depthwise-separable
    heads predict per-granularity saliency logits.  The caller chooses
    normalisation (sigmoid for GA-Adapter, softmax for DepthRoleEncoder).
    """

    def __init__(self, low_res_size: int = 56, target_tokens: int = 196):
        super().__init__()
        self.low_res_size = low_res_size
        side = math.isqrt(target_tokens)
        if side * side != target_tokens:
            raise ValueError(f"target_tokens must be a perfect square, got {target_tokens}")
        self.target_side = side

        # Shared backbone (operates on 56x56)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )

        # Three granularity-specific heads (depthwise-separable)
        def _make_head():
            return nn.Sequential(
                nn.Conv2d(96, 96, 3, padding=1, groups=96, bias=False),
                nn.Conv2d(96, 32, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(32, 1, 1),
            )

        self.fine_head = _make_head()
        self.medium_head = _make_head()
        self.coarse_head = _make_head()

    def forward_logits(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw logits for each granularity (no normalisation applied).

        Returns:
            logits_f, logits_m, logits_c: each [B, 196]
        """
        low_res = F.interpolate(
            images,
            size=(self.low_res_size, self.low_res_size),
            mode="bilinear",
            align_corners=False,
        )
        feat = self.backbone(low_res)  # [B, 96, 7, 7]

        def _head_to_logits(head):
            raw = head(feat)  # [B, 1, 7, 7]
            up = F.interpolate(
                raw, size=(self.target_side, self.target_side),
                mode="bilinear", align_corners=False,
            )
            return up.flatten(1)  # [B, 196]

        return _head_to_logits(self.fine_head), _head_to_logits(self.medium_head), _head_to_logits(self.coarse_head)


def adapter_field_from_logits(
    logits: torch.Tensor,
    transform: str = "sigmoid",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Map QIE logits to adapter conditioning without changing pooling.

    ``sigmoid`` is the archived MAIQA path. ``centered_logits`` removes the
    per-image spatially constant component and bounds the result in [-1, 1].
    The latter must be enabled explicitly by a resolved experiment config.
    """
    if transform == "sigmoid":
        return torch.sigmoid(logits)
    if transform == "centered_logits":
        if not temperature > 0:
            raise ValueError("qie_adapter_temperature must be positive")
        centered = logits - logits.mean(dim=-1, keepdim=True)
        return torch.tanh(centered / float(temperature))
    raise ValueError(
        "qie_adapter_transform must be sigmoid or centered_logits, "
        f"got {transform!r}"
    )


# ---------------------------------------------------------------------------
# GA-Adapter: Granularity-Aware Adapter
# ---------------------------------------------------------------------------


class GranularityAwareAdapter(nn.Module):
    """Importance-conditioned token adapter.

    Modulation: M = 1 + alpha * w_sigmoid + beta * tanh(MLP(Z))
    This provides per-token adaptive scaling driven by both external QIE
    saliency (alpha * w) and internal feature demand (beta * delta).
    """

    def __init__(
        self,
        input_dim: int = 768,
        adapter_dim: int = 96,
        bottleneck: int = 24,
        alpha_mode: str = "free",
        alpha_init: float = 0.0,
        alpha_max: float = 0.10,
    ):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.alpha_mode = str(alpha_mode)
        self.alpha_max = float(alpha_max)
        if self.alpha_mode not in {"free", "bounded"}:
            raise ValueError("alpha_mode must be free or bounded")
        if self.alpha_mode == "bounded":
            if not self.alpha_max > 0:
                raise ValueError("alpha_max must be positive in bounded mode")
            ratio = float(alpha_init) / self.alpha_max
            if not -1.0 < ratio < 1.0:
                raise ValueError("abs(alpha_init) must be smaller than alpha_max")
            alpha_parameter_init = math.atanh(ratio)
        else:
            alpha_parameter_init = float(alpha_init)

        # Channel compression
        self.down_norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, adapter_dim, bias=False)

        # Importance modulation
        self.alpha = nn.Parameter(torch.full((1,), alpha_parameter_init))
        self.beta = nn.Parameter(torch.zeros(1))
        self.token_mlp = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim // 3, bias=False),
            nn.GELU(),
            nn.Linear(adapter_dim // 3, 1, bias=False),
        )

        # Bottleneck
        self.bottleneck_down = nn.Linear(adapter_dim, bottleneck, bias=False)
        self.bottleneck_up = nn.Linear(bottleneck, adapter_dim, bias=False)

        # Channel decompression
        self.up_proj = nn.Linear(adapter_dim, input_dim, bias=False)

    def forward(
        self, x: torch.Tensor, w_sigmoid: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, 768] — ViT block output patch tokens
            w_sigmoid: [B, N] — sigmoid-normalised QIE importance

        Returns:
            [B, N, 768] — adapted tokens
        """
        z = self.down_proj(self.down_norm(x))  # [B, N, d]

        # Importance modulation
        delta = torch.tanh(self.token_mlp(z))  # [B, N, 1]
        alpha = (
            self.alpha_max * torch.tanh(self.alpha)
            if self.alpha_mode == "bounded"
            else self.alpha
        )
        modulation = 1.0 + alpha * w_sigmoid.unsqueeze(-1) + self.beta * delta
        z_mod = z * modulation

        # Bottleneck residual
        h = F.gelu(self.bottleneck_down(z_mod))
        z_adapted = z_mod + self.bottleneck_up(h)

        # Decompress + residual
        return x + self.up_proj(z_adapted)


class BasicResidualAdapter(nn.Module):
    """Single-bottleneck residual adapter used by the minimal model.

    The second argument is accepted for interface compatibility with
    ``_extract_and_adapt`` but intentionally ignored: this adapter contains no
    QIE-driven or token-demand modulation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        adapter_dim: int = 96,
    ):
        super().__init__()
        self.down_norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, adapter_dim, bias=False)
        self.up_proj = nn.Linear(adapter_dim, input_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        unused_spatial_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del unused_spatial_weights
        z = self.down_proj(self.down_norm(x))
        return x + self.up_proj(F.gelu(z))


# ---------------------------------------------------------------------------
# Distortion Sidecar
# ---------------------------------------------------------------------------


class DistortionSidecar(nn.Module):
    """Ultra-lightweight pixel-domain distortion awareness.

    Operates on edge-domain channels [luma, gradient, laplacian] with
    a 2-layer CNN to produce a single distortion_aware node.
    """

    def __init__(self, node_dim: int = 128):
        super().__init__()
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        sobel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
        ) / 8.0
        laplacian = torch.tensor(
            ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer(
            "sobel_y", sobel_x.t().view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "laplacian", laplacian.view(1, 1, 3, 3), persistent=False
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=4, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=4, padding=1, bias=False),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32, node_dim, bias=False),
            nn.LayerNorm(node_dim),
        )

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            normalized_images: [B, 3, 224, 224] — ImageNet-normalised

        Returns:
            [B, node_dim] — distortion_aware node
        """
        rgb = (
            normalized_images * self.imagenet_std.to(normalized_images.dtype)
            + self.imagenet_mean.to(normalized_images.dtype)
        ).clamp(0.0, 1.0)
        luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        grad_x = F.conv2d(luma, self.sobel_x.to(luma.dtype), padding=1)
        grad_y = F.conv2d(luma, self.sobel_y.to(luma.dtype), padding=1)
        gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
        lap = F.conv2d(luma, self.laplacian.to(luma.dtype), padding=1).abs()

        sidecar_input = torch.cat([luma, gradient, lap], dim=1)  # [B, 3, 224, 224]
        feat = self.cnn(sidecar_input)  # [B, 32, 14, 14]
        pooled = feat.mean(dim=[2, 3])  # [B, 32]
        return self.head(pooled)


class SpatialDistortionTokenizer(nn.Module):
    """Preserve pixel-domain distortion evidence on the ViT token grid.

    The fourth input channel is a local high-pass residual.  Together with
    luminance, Sobel magnitude and Laplacian response it provides an explicit
    spatial/frequency pyramid without introducing a second semantic expert.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        sobel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
        ) / 8.0
        laplacian = torch.tensor(
            ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().view(1, 1, 3, 3), persistent=False)
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, output_dim, 3, stride=4, padding=1, bias=False),
            nn.GroupNorm(8, output_dim),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        rgb = (
            normalized_images * self.imagenet_std.to(normalized_images.dtype)
            + self.imagenet_mean.to(normalized_images.dtype)
        ).clamp(0.0, 1.0)
        luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        grad_x = F.conv2d(luma, self.sobel_x.to(luma.dtype), padding=1)
        grad_y = F.conv2d(luma, self.sobel_y.to(luma.dtype), padding=1)
        gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
        lap = F.conv2d(luma, self.laplacian.to(luma.dtype), padding=1).abs()
        low = F.avg_pool2d(luma, kernel_size=5, stride=1, padding=2)
        high = (luma - low).abs()
        features = self.encoder(torch.cat((luma, gradient, lap, high), dim=1))
        if features.shape[-2:] != (14, 14):
            features = F.interpolate(
                features, size=(14, 14), mode="bilinear", align_corners=False
            )
        tokens = features.flatten(2).transpose(1, 2)
        return self.output_norm(tokens)


class SpatialDistortionInjector(nn.Module):
    """Inject aligned distortion tokens through a bounded residual gate."""

    def __init__(
        self,
        token_dim: int,
        distortion_dim: int,
        hidden_dim: int,
        initial_scale: float,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(token_dim)
        self.distortion_norm = nn.LayerNorm(distortion_dim)
        self.token_gate = nn.Linear(token_dim, hidden_dim, bias=False)
        self.distortion_gate = nn.Linear(distortion_dim, hidden_dim, bias=False)
        self.distortion_value = nn.Linear(distortion_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, token_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(
        self, tokens: torch.Tensor, distortion_tokens: torch.Tensor
    ) -> torch.Tensor:
        if tokens.shape[:2] != distortion_tokens.shape[:2]:
            raise ValueError(
                "ViT and distortion token grids must align, got "
                f"{tuple(tokens.shape)} and {tuple(distortion_tokens.shape)}"
            )
        token_hidden = self.token_gate(self.token_norm(tokens))
        distortion_hidden = self.distortion_norm(distortion_tokens)
        gate = torch.sigmoid(
            token_hidden + self.distortion_gate(distortion_hidden)
        )
        update = self.output(
            gate * F.gelu(self.distortion_value(distortion_hidden))
        )
        return tokens + torch.tanh(self.scale) * update


class GlobalLocalSetAggregator(nn.Module):
    """Aggregate local quality evidence conditioned on whole-image context."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        entropy_floor: float,
        use_uncertainty: bool,
    ):
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("set embedding_dim must be divisible by set_pool_heads")
        self.entropy_floor = float(entropy_floor)
        self.use_uncertainty = bool(use_uncertainty)
        self.local_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim)
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim)
        )
        self.position_proj = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, heads, dropout=dropout, batch_first=True
        )
        self.uncertainty_head = (
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.use_uncertainty else None
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid()
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        local_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, count, channels = local_embeddings.shape
        if positions is None:
            positions = local_embeddings.new_zeros(batch, count, 2)
        if positions.shape != (batch, count, 2):
            raise ValueError(
                f"patch_positions must be [B,N,2], got {tuple(positions.shape)}"
            )
        local = self.local_proj(local_embeddings) + self.position_proj(positions)
        query = self.global_proj(global_embedding).unsqueeze(1)
        attended, attention = self.cross_attention(
            query, local, local, need_weights=True, average_attn_weights=True
        )
        attention = attention.squeeze(1)
        log_variance = None
        if self.uncertainty_head is not None:
            log_variance = self.uncertainty_head(local).squeeze(-1).clamp(-4.0, 4.0)
            reliability = torch.exp(-F.softplus(log_variance))
            weights = attention * reliability
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            weights = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        local_pool = (local * weights.unsqueeze(-1)).sum(dim=1)
        global_value = query.squeeze(1)
        gate = self.fusion_gate(torch.cat((global_value, local_pool), dim=-1))
        fused = self.output_norm(
            global_value + attended.squeeze(1) + gate * local_pool
        )
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
        normalized_entropy = entropy / max(math.log(max(count, 2)), 1e-8)
        coverage_loss = F.relu(self.entropy_floor - normalized_entropy).mean()
        extras = {
            "patch_weights": weights,
            "patch_weight_entropy": entropy,
            "patch_coverage_loss": coverage_loss,
        }
        if log_variance is not None:
            extras["patch_uncertainty_logvar"] = log_variance
        return fused, extras


# ---------------------------------------------------------------------------
# Controlled Fusion
# ---------------------------------------------------------------------------


class ControlledFusion(nn.Module):
    """Gated residual fusion for sidecar node with minimum retain guarantee.

    Ensures the distortion sidecar always contributes at least eta * 100%
    of its original signal, preventing it from being drowned by the
    six semantic nodes in the role graph.
    """

    def __init__(self, node_dim: int = 128, eta: float = 0.20):
        super().__init__()
        self.eta = eta
        self.gate_proj = nn.Linear(node_dim * 2, 1, bias=True)

    def forward(
        self, semantic_nodes: torch.Tensor, distortion_node: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            semantic_nodes: [B, 6, node_dim]
            distortion_node: [B, node_dim]

        Returns:
            fused_node: [B, node_dim]
            gate_value: [B, 1]
        """
        S = semantic_nodes.mean(dim=1)  # [B, node_dim]
        gate_input = torch.cat([S, distortion_node], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, 1]

        blended = gate * distortion_node + (1.0 - gate) * S
        fused = self.eta * distortion_node + (1.0 - self.eta) * blended
        return fused, gate


class AdditiveFusion(nn.Module):
    """Parameter-light additive control without a learned semantic gate."""

    def __init__(self, node_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self, semantic_nodes: torch.Tensor, distortion_node: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        semantic_mean = semantic_nodes.mean(dim=1)
        return self.norm(semantic_mean + distortion_node), None


# ---------------------------------------------------------------------------
# DepthRoleEncoder (simplified)
# ---------------------------------------------------------------------------


class DepthRoleEncoder(nn.Module):
    """Turn one adapted ViT depth into global and local quality roles.

    Global: spatial mean — overall quality at this granularity.
    Local:  QIE-weighted sum — quality at quality-salient regions.
    """

    def __init__(self, input_dim: int, node_dim: int, spatial_focus: float):
        super().__init__()
        self.spatial_focus = float(spatial_focus)
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, node_dim, bias=False),
        )
        self.global_norm = nn.LayerNorm(node_dim)
        self.local_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        adapted_tokens: torch.Tensor,
        granularity_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            adapted_tokens: [B, N, 768] — output of GA-Adapter
            granularity_weights: [B, N] — softmax-normalised QIE weights

        Returns:
            global_node: [B, node_dim]
            local_node:  [B, node_dim]
        """
        projected = self.project(adapted_tokens)  # [B, N, node_dim]

        # Global: spatial mean
        global_node = self.global_norm(projected.mean(dim=1))

        # Local: QIE-weighted + anti-starvation
        count = granularity_weights.size(1)
        uniform = torch.full_like(granularity_weights, 1.0 / count)
        weights = self.spatial_focus * granularity_weights + (1.0 - self.spatial_focus) * uniform
        local_node = self.local_norm(
            (projected * weights.unsqueeze(-1)).sum(dim=1)
        )
        return global_node, local_node


class GlobalRoleEncoder(nn.Module):
    """Project one ViT depth and retain only its global quality role."""

    def __init__(self, input_dim: int, node_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, node_dim, bias=False),
        )
        self.global_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        adapted_tokens: torch.Tensor,
        unused_granularity_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del unused_granularity_weights
        global_node = self.global_norm(self.project(adapted_tokens).mean(dim=1))
        # The duplicated return value preserves MAIQA's encoder interface; the
        # minimal model's global-only topology consumes only the first value.
        return global_node, global_node


# ---------------------------------------------------------------------------
# RoleGraphBlock (simplified)
# ---------------------------------------------------------------------------


class RoleGraphBlock(nn.Module):
    """Single-layer multi-head self-attention over role nodes.

    No role_bias: role_embeddings already carry identity information
    into Q·K dot products.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        if dim % heads:
            raise ValueError("node_dim must be divisible by graph_heads")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.projection = nn.Linear(dim, dim, bias=False)
        hidden = int(round(dim * mlp_ratio))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, count, channels = nodes.shape
        qkv = self.qkv(self.norm1(nodes)).reshape(
            batch, count, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(logits, dim=-1)
        update = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, count, channels
        )
        nodes = nodes + self.projection(update)
        nodes = nodes + self.mlp(self.norm2(nodes))
        return nodes, attention


class RoleWiseMLPBlock(nn.Module):
    """Parameter-matched control that transforms roles without mixing them.

    With the default graph MLP ratio of two, this block has exactly the same
    number of trainable parameters as :class:`RoleGraphBlock`.  Its operations
    are applied independently to every node, so any difference from RoleGraph
    isolates cross-role communication rather than capacity.
    """

    def __init__(self, dim: int, graph_mlp_ratio: float):
        super().__init__()
        hidden = int(round(dim * (2.0 + graph_mlp_ratio)))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(
        self, nodes: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return nodes + self.mlp(self.norm(nodes)), None


# ---------------------------------------------------------------------------
# MAIQA main model
# ---------------------------------------------------------------------------


class MAIQA(nn.Module):
    """Structured Quality Decomposition IQA model.

    Seven role nodes: six from multi-granularity ViT depths (fine/medium/coarse
    x global/local) plus one distortion_aware from a lightweight pixel-domain
    sidecar.  A GR-QIE module provides per-granularity importance weights used
    both for importance-conditioned adaptation (GA-Adapter) and for local role
    node construction (DepthRoleEncoder).
    """

    # role_names is dynamic: includes "distortion_aware" only when sidecar is enabled
    _base_role_names = (
        "fine_global", "fine_local",
        "medium_global", "medium_local",
        "coarse_global", "coarse_local",
    )
    _full_role_names = _base_role_names + ("distortion_aware",)
    role_names = _full_role_names  # default (with sidecar)

    def __init__(self, cfg: MAIQAConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg or MAIQAConfig(**kwargs)
        cfg = self.cfg
        # Normalize list→tuple for hashability & consistency
        if not isinstance(cfg.feature_layers, tuple):
            cfg = _dc_replace(cfg, feature_layers=tuple(cfg.feature_layers))
        # Multi-scale configs: only normalize when enabled (not None)
        if cfg.multiscale_scales is not None:
            if not isinstance(cfg.multiscale_scales, tuple):
                cfg = _dc_replace(cfg, multiscale_scales=tuple(cfg.multiscale_scales))
            if not isinstance(cfg.patches_per_scale, tuple):
                cfg = _dc_replace(cfg, patches_per_scale=tuple(cfg.patches_per_scale))
        self.cfg = cfg
        if cfg.use_fragments:
            raise ValueError("MAIQA intentionally removes the fragment branch")
        if len(tuple(cfg.feature_layers)) != 3:
            raise ValueError("feature_layers must contain exactly three depths")
        if not 0.0 <= float(cfg.spatial_focus) <= 1.0:
            raise ValueError("spatial_focus must be in [0, 1]")
        if cfg.role_pooling not in {"global_local", "global_only", "local_only"}:
            raise ValueError(
                "role_pooling must be global_local, global_only, or local_only"
            )
        if cfg.role_interaction not in {"graph", "mlp", "identity"}:
            raise ValueError("role_interaction must be graph, mlp, or identity")
        if cfg.adapter_placement not in {"feature_layers", "every_block"}:
            raise ValueError(
                "adapter_placement must be feature_layers or every_block"
            )
        if cfg.qie_route_sharing not in {"shared", "shared_matched", "unshared"}:
            raise ValueError(
                "qie_route_sharing must be shared, shared_matched, or unshared"
            )
        if cfg.qie_route_sharing != "shared" and not cfg.use_gr_qie:
            raise ValueError("matched or unshared QIE routes require use_gr_qie=true")
        if cfg.sidecar_fusion not in {"controlled", "gated", "additive"}:
            raise ValueError(
                "sidecar_fusion must be controlled, gated, or additive"
            )
        if (cfg.qie_adapter_conditioning or cfg.qie_role_pooling) and not cfg.use_gr_qie:
            # A disabled estimator has no learned prior to route.  Normalise the
            # flags instead of silently constructing a misleading experiment.
            cfg = _dc_replace(
                cfg,
                qie_adapter_conditioning=False,
                qie_role_pooling=False,
            )
            self.cfg = cfg
        # Multi-scale validation (only when enabled)
        if cfg.multiscale_scales is not None:
            if cfg.patches_per_scale is None:
                raise ValueError("patches_per_scale is required when multiscale_scales is set")
            if len(cfg.multiscale_scales) != len(cfg.patches_per_scale):
                raise ValueError("multiscale_scales and patches_per_scale must have same length")
            if cfg.patch_aggregation not in ("mean", "attention", "variance_weighted"):
                raise ValueError(
                    f"patch_aggregation must be mean/attention/variance_weighted, "
                    f"got {cfg.patch_aggregation}"
                )
            self.total_patches = sum(cfg.patches_per_scale)
        else:
            self.total_patches = 0  # legacy mode: not applicable

        # ---- Backbone ----
        self.hf_backbone = None
        if cfg.hf_backbone is not None:
            from maiqa.hf_backbones import build_hf_vision_backbone

            self.vit = None
            self.hf_backbone = build_hf_vision_backbone(
                cfg.hf_backbone,
                cfg.hf_backbone_path,
                grid_side=int(cfg.token_grid_side),
            )
            self.hf_backbone.eval()
            for parameter in self.hf_backbone.parameters():
                parameter.requires_grad_(False)
            if cfg.backbone_finetune_all:
                for parameter in self.hf_backbone.parameters():
                    parameter.requires_grad_(True)
            elif cfg.backbone_finetune_layers is not None:
                hf_blocks = self.hf_backbone.blocks
                for layer_idx in cfg.backbone_finetune_layers:
                    if not 0 <= layer_idx < len(hf_blocks):
                        raise ValueError(
                            f"backbone_finetune_layers {layer_idx} out of range "
                            f"for {len(hf_blocks)} HF blocks"
                        )
                    for parameter in hf_blocks[layer_idx].parameters():
                        parameter.requires_grad_(True)
                final_norm = getattr(self.hf_backbone, "final_norm", None)
                if final_norm is not None:
                    for parameter in final_norm.parameters():
                        parameter.requires_grad_(True)
            layers = tuple(int(index) for index in cfg.feature_layers)
            if not all(0 <= i < len(self.hf_backbone.blocks) for i in layers):
                raise ValueError("feature_layers exceed the HF backbone depth")
            self.embed_dim = int(self.hf_backbone.num_features)
        else:
            self.vit = timm.create_model(
                cfg.backbone,
                pretrained=bool(cfg.pretrained and not cfg.backbone_path),
                num_classes=0,
                drop_path_rate=float(cfg.backbone_drop_path_rate),
            )
            if cfg.backbone_path:
                _load_backbone_checkpoint(self.vit, cfg.backbone_path)
            self._validate_backbone()
            self.vit.eval()
            for parameter in self.vit.parameters():
                parameter.requires_grad_(False)

            # ---- Partial backbone fine-tuning (P1 improvement) ----
            # Unfreeze selected ViT blocks for higher capacity on large datasets.
            if cfg.backbone_finetune_all:
                for parameter in self.vit.parameters():
                    parameter.requires_grad_(True)
            elif cfg.backbone_finetune_layers is not None:
                vit_blocks = self.vit.blocks  # timm ViT blocks list
                for layer_idx in cfg.backbone_finetune_layers:
                    if 0 <= layer_idx < len(vit_blocks):
                        for parameter in vit_blocks[layer_idx].parameters():
                            parameter.requires_grad_(True)
                # Also unfreeze the final norm layer so finetuned features are
                # properly normalised.
                for parameter in self.vit.norm.parameters():
                    parameter.requires_grad_(True)

            self.embed_dim = int(getattr(self.vit, "num_features", 768))

        if cfg.adapter_placement == "every_block" and self.hf_backbone is not None:
            raise ValueError(
                "adapter_placement=every_block currently requires a timm ViT; "
                "HuggingFace towers expose only selected hidden states"
            )
        if cfg.qie_adapter_transform not in {"sigmoid", "centered_logits"}:
            raise ValueError(
                "qie_adapter_transform must be sigmoid or centered_logits"
            )
        if not float(cfg.qie_adapter_temperature) > 0:
            raise ValueError("qie_adapter_temperature must be positive")

        # ---- Multi-scale Patch Aggregator (only when enabled) ----
        if cfg.multiscale_scales is not None:
            self.patch_aggregator = PatchAggregator(
                embed_dim=self.embed_dim,
                node_dim=cfg.node_dim,
                num_scales=len(cfg.multiscale_scales),
                aggregation=cfg.patch_aggregation,
                dropout=cfg.dropout,
            )
        else:
            self.patch_aggregator = None

        # ---- GR-QIE ----
        if cfg.use_gr_qie:
            self.gr_qie = GranularityRedistributedQIE(
                low_res_size=cfg.qie_low_res_size,
                target_tokens=int(cfg.token_grid_side) ** 2,
            )
            self.gr_qie_role = (
                GranularityRedistributedQIE(
                    low_res_size=cfg.qie_low_res_size,
                    target_tokens=int(cfg.token_grid_side) ** 2,
                )
                if cfg.qie_route_sharing in {"shared_matched", "unshared"}
                else None
            )
        else:
            self.gr_qie = None
            self.gr_qie_role = None

        # ---- GA-Adapter ----
        adapter_count = (
            int(cfg.feature_layers[-1]) + 1
            if cfg.adapter_placement == "every_block"
            else 3
        )
        self.ga_adapters = nn.ModuleList(
            GranularityAwareAdapter(
                input_dim=self.embed_dim,
                adapter_dim=cfg.adapter_dim,
                bottleneck=cfg.adapter_bottleneck,
                alpha_mode=cfg.adapter_alpha_mode,
                alpha_init=cfg.adapter_alpha_init,
                alpha_max=cfg.adapter_alpha_max,
            )
            for _ in range(adapter_count)
        )

        # ---- Depth Role Encoders ----
        self.depth_encoders = nn.ModuleList(
            DepthRoleEncoder(self.embed_dim, cfg.node_dim, cfg.spatial_focus)
            for _ in range(3)
        )

        # Spatial distortion evidence is deliberately injected before role
        # pooling so location-specific defects survive the semantic decoder.
        if cfg.use_spatial_distortion_tokens:
            self.spatial_distortion_tokenizer = SpatialDistortionTokenizer(
                cfg.spatial_distortion_dim
            )
            self.spatial_distortion_injectors = nn.ModuleList(
                SpatialDistortionInjector(
                    token_dim=self.embed_dim,
                    distortion_dim=cfg.spatial_distortion_dim,
                    hidden_dim=cfg.adapter_dim,
                    initial_scale=cfg.spatial_injection_scale,
                )
                for _ in range(3)
            )
        else:
            self.spatial_distortion_tokenizer = None
            self.spatial_distortion_injectors = nn.ModuleList()

        # ---- Distortion Sidecar ----
        if cfg.use_distortion_sidecar:
            self.distortion_sidecar = DistortionSidecar(cfg.node_dim)
            if cfg.sidecar_fusion == "additive":
                self.controlled_fusion = AdditiveFusion(cfg.node_dim)
            else:
                eta = cfg.sidecar_min_retain if cfg.sidecar_fusion == "controlled" else 0.0
                self.controlled_fusion = ControlledFusion(cfg.node_dim, eta=eta)
        else:
            self.distortion_sidecar = None
            self.controlled_fusion = None

        # Role identity follows the actual computation graph.  This makes
        # global-only/local-only studies true topology removals rather than
        # changes to a pooling coefficient that leave the node count intact.
        semantic_role_names = []
        for depth in ("fine", "medium", "coarse"):
            if cfg.role_pooling in {"global_local", "global_only"}:
                semantic_role_names.append(f"{depth}_global")
            if cfg.role_pooling in {"global_local", "local_only"}:
                semantic_role_names.append(f"{depth}_local")
        if cfg.use_distortion_sidecar:
            semantic_role_names.append("distortion_aware")
        self.role_names = tuple(semantic_role_names)

        # ---- Role Graph ----
        num_nodes = len(self.role_names)
        self.role_embeddings = nn.Parameter(
            torch.randn(1, num_nodes, cfg.node_dim) * 0.02
        )
        if cfg.role_interaction == "graph":
            interaction_factory = lambda: RoleGraphBlock(
                cfg.node_dim, cfg.graph_heads, cfg.graph_mlp_ratio
            )
        elif cfg.role_interaction == "mlp":
            interaction_factory = lambda: RoleWiseMLPBlock(
                cfg.node_dim, cfg.graph_mlp_ratio
            )
        else:
            interaction_factory = None
        self.graph = nn.ModuleList(
            interaction_factory() for _ in range(cfg.graph_layers)
        ) if interaction_factory is not None else nn.ModuleList()

        # ---- Score Head ----
        # Input is either from PatchAggregator (multiscale mode) or from
        # role-graph pooling (single-patch mode).  Both output [B, node_dim*2].
        pooled_dim = cfg.node_dim * 2  # mean + std
        self.output_norm = nn.LayerNorm(pooled_dim)
        self.score_head = nn.Sequential(
            nn.Linear(pooled_dim, cfg.node_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.node_dim, 1),
        )
        nn.init.normal_(self.score_head[-1].weight, std=0.02)
        nn.init.zeros_(self.score_head[-1].bias)

        if cfg.use_global_local_set_pooling:
            self.global_local_set_aggregator = GlobalLocalSetAggregator(
                embedding_dim=pooled_dim,
                hidden_dim=cfg.set_pool_hidden,
                heads=cfg.set_pool_heads,
                dropout=cfg.dropout,
                entropy_floor=cfg.set_pool_entropy_floor,
                use_uncertainty=cfg.set_pool_use_uncertainty,
            )
            self.set_residual_head = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, cfg.node_dim),
                nn.GELU(),
                nn.Linear(cfg.node_dim, 1),
            )
            nn.init.zeros_(self.set_residual_head[-1].weight)
            nn.init.zeros_(self.set_residual_head[-1].bias)
        else:
            self.global_local_set_aggregator = None
            self.set_residual_head = None

        # ---- Role Sub-score Head ----
        if cfg.use_role_sub_scores:
            self.role_score_head = nn.Sequential(
                nn.Linear(cfg.node_dim, cfg.node_dim // 4),
                nn.GELU(),
                nn.Linear(cfg.node_dim // 4, 1),
            )
            nn.init.normal_(self.role_score_head[-1].weight, std=0.02)
            nn.init.zeros_(self.role_score_head[-1].bias)
        else:
            self.role_score_head = None

        # ---- Coupling state (managed by training loop) ----
        self.register_buffer("_coupling_rho", torch.tensor(0.0), persistent=False)

    # ------------------------------------------------------------------
    # Backbone helpers
    # ------------------------------------------------------------------

    def _validate_backbone(self) -> None:
        required = ("patch_embed", "cls_token", "pos_embed", "blocks", "norm")
        missing = [name for name in required if not hasattr(self.vit, name)]
        if missing:
            raise TypeError(f"Unsupported ViT backbone; missing {missing}")
        layers = tuple(int(index) for index in self.cfg.feature_layers)
        if tuple(sorted(set(layers))) != layers:
            raise ValueError("feature_layers must be distinct and increasing")
        if not all(0 <= index < len(self.vit.blocks) for index in layers):
            raise ValueError("feature_layers exceed the configured ViT depth")
        if int(self.cfg.backbone_chunk_size) < 1:
            raise ValueError("backbone_chunk_size must be positive")
        backbone_size = getattr(self.vit.patch_embed, "img_size", None)
        if backbone_size is not None and tuple(map(int, backbone_size)) != (
            int(self.cfg.image_size), int(self.cfg.image_size)
        ):
            raise ValueError("model.image_size must match the ViT input size")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vit is not None:
            self.vit.train(mode) if self.cfg.backbone_finetune_all else self.vit.eval()
        if self.hf_backbone is not None:
            (
                self.hf_backbone.train(mode)
                if self.cfg.backbone_finetune_all
                else self.hf_backbone.eval()
            )
        return self

    def _resize(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] == (self.cfg.image_size, self.cfg.image_size):
            return images
        return F.interpolate(
            images,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bicubic",
            align_corners=False,
        )

    def _embed_tokens(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.patch_embed(images)
        cls = self.vit.cls_token.expand(tokens.size(0), -1, -1)
        if getattr(self.vit, "no_embed_class", False):
            tokens = tokens + self.vit.pos_embed
            tokens = torch.cat((cls, tokens), dim=1)
        else:
            tokens = torch.cat((cls, tokens), dim=1) + self.vit.pos_embed
        if hasattr(self.vit, "patch_drop"):
            tokens = self.vit.patch_drop(tokens)
        if hasattr(self.vit, "norm_pre"):
            tokens = self.vit.norm_pre(tokens)
        return tokens

    def _extract_and_adapt(
        self,
        images: torch.Tensor,
        w_adapters: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Run frozen ViT to extract multi-depth tokens, then adapt with GA-Adapter.

        Args:
            images: [B, 3, 224, 224]
            w_adapters: tuple of 3 tensors [B, 196] (sigmoid-normalised QIE)
                        These may be detached (warmup) or soft-coupled.

        Returns:
            (adapted_fine, adapted_medium, adapted_coarse), each [B, N, D]
        """
        if self.cfg.adapter_placement == "every_block":
            return self._extract_and_adapt_every_block(images, w_adapters)
        if self.hf_backbone is not None:
            return self._extract_and_adapt_hf(images, w_adapters)
        layers = tuple(int(index) for index in self.cfg.feature_layers)
        layer_to_slot = {layer: slot for slot, layer in enumerate(layers)}

        # When backbone_finetune_layers is set, those blocks need gradient
        # computation. We run frozen blocks under no_grad and finetuned
        # blocks with gradients enabled, preserving the computation graph
        # so that loss can backprop through finetuned blocks.
        finetune_set = (
            set(range(len(self.vit.blocks)))
            if self.cfg.backbone_finetune_all
            else (
                set(self.cfg.backbone_finetune_layers)
                if self.cfg.backbone_finetune_layers is not None
                else set()
            )
        )
        has_finetune = bool(finetune_set)
        # Sort blocks: first all frozen (no_grad), then finetuned (grad).
        # This matches the sequential order of ViT blocks.
        all_block_indices = list(range(layers[-1] + 1))

        raw_outputs: List[List[torch.Tensor]] = [[], [], []]

        for chunk in images.split(int(self.cfg.backbone_chunk_size), dim=0):
            tokens = self._embed_tokens(self._resize(chunk))

            # Run blocks sequentially, switching grad context at finetune boundaries
            in_finetune_zone = False
            for idx in all_block_indices:
                block = self.vit.blocks[idx]
                is_finetune = idx in finetune_set

                # When entering finetune zone, detach from frozen graph
                # and re-enable grad for the finetuned blocks.
                if is_finetune and not in_finetune_zone:
                    if not self.cfg.backbone_finetune_all:
                        tokens = tokens.detach().requires_grad_(True)
                    in_finetune_zone = True

                if is_finetune:
                    # Finetuned block: full gradient computation
                    tokens = block(tokens)
                else:
                    # Frozen block: no gradient needed
                    with torch.no_grad():
                        tokens = block(tokens)

                if idx in layer_to_slot:
                    slot_tokens = tokens[:, 1:]
                    if is_finetune:
                        # Keep graph connected for backprop through finetuned block
                        raw_outputs[layer_to_slot[idx]].append(slot_tokens)
                    else:
                        # Detach: no gradient flows through frozen blocks
                        raw_outputs[layer_to_slot[idx]].append(slot_tokens.detach())

        raw_tokens = tuple(
            torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
            for chunks in raw_outputs
        )

        # When backbone finetuning is active and the feature layer is
        # finetuned, keep the gradient graph; otherwise detach.
        adapted = []
        for slot, adapter in enumerate(self.ga_adapters):
            if has_finetune and layers[slot] in finetune_set:
                leaf = raw_tokens[slot]
            else:
                leaf = raw_tokens[slot].detach().requires_grad_(True)
            adapted.append(adapter(leaf, w_adapters[slot]))
        return tuple(adapted)

    def _extract_and_adapt_every_block(
        self,
        images: torch.Tensor,
        w_adapters: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Insert a granularity-conditioned adapter after every ViT block.

        The three feature layers define contiguous granularity stages.  With
        the default ``(3, 7, 11)``, blocks 0--3 use the fine prior, blocks
        4--7 use the medium prior, and blocks 8--11 use the coarse prior.  The
        adapters are independent across blocks, while the stage prior remains
        shared so the original three-granularity hypothesis is preserved.

        Frozen blocks after the first adapter must run with autograd enabled:
        their parameters remain frozen, but their Jacobians are required to
        train earlier adapters.  Only the prefix before the first adapter can
        safely use ``no_grad``.
        """
        if self.hf_backbone is not None:
            raise RuntimeError("every-block adapters require a timm ViT")

        layers = tuple(int(index) for index in self.cfg.feature_layers)
        layer_to_slot = {layer: slot for slot, layer in enumerate(layers)}
        finetune_set = (
            set(range(len(self.vit.blocks)))
            if self.cfg.backbone_finetune_all
            else (
                set(self.cfg.backbone_finetune_layers)
                if self.cfg.backbone_finetune_layers is not None
                else set()
            )
        )
        captured: List[List[torch.Tensor]] = [[], [], []]

        for chunk in images.split(int(self.cfg.backbone_chunk_size), dim=0):
            tokens = self._embed_tokens(self._resize(chunk))
            for block_index in range(layers[-1] + 1):
                block = self.vit.blocks[block_index]
                # At block zero no adapter precedes the block.  A frozen block
                # may therefore execute without an autograd graph.  All later
                # blocks receive adapter-dependent tokens and must preserve the
                # input gradient even when their parameters are frozen.
                needs_graph = (
                    block_index in finetune_set
                    or bool(tokens.requires_grad)
                )
                if needs_graph:
                    tokens = block(tokens)
                else:
                    with torch.no_grad():
                        tokens = block(tokens)

                stage = next(
                    slot for slot, boundary in enumerate(layers)
                    if block_index <= boundary
                )
                adapted_patches = self.ga_adapters[block_index](
                    tokens[:, 1:], w_adapters[stage]
                )
                tokens = torch.cat((tokens[:, :1], adapted_patches), dim=1)

                if block_index in layer_to_slot:
                    captured[layer_to_slot[block_index]].append(tokens[:, 1:])

        return tuple(
            torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
            for chunks in captured
        )

    def _extract_and_adapt_hf(
        self,
        images: torch.Tensor,
        w_adapters: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """HF-backbone counterpart of _extract_and_adapt.

        Runs the HF vision tower once per chunk, capturing block outputs via
        the wrapper.  When fine-tune layers are active the forward runs with
        gradients enabled; frozen blocks do not record autograd history
        because neither their parameters nor their inputs require grad.
        """
        layers = tuple(int(index) for index in self.cfg.feature_layers)
        finetune_set = (
            set(range(len(self.hf_backbone.blocks)))
            if self.cfg.backbone_finetune_all
            else (
                set(self.cfg.backbone_finetune_layers)
                if self.cfg.backbone_finetune_layers is not None
                else set()
            )
        )
        enable_grad = bool(finetune_set)

        raw_outputs: List[List[torch.Tensor]] = [[], [], []]
        for chunk in images.split(int(self.cfg.backbone_chunk_size), dim=0):
            feats = self.hf_backbone.extract(
                self._resize(chunk), layers, enable_grad=enable_grad
            )
            for slot, feat in enumerate(feats):
                raw_outputs[slot].append(feat)

        raw_tokens = tuple(
            torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
            for chunks in raw_outputs
        )

        adapted = []
        for slot, adapter in enumerate(self.ga_adapters):
            if enable_grad and (
                self.cfg.backbone_finetune_all or layers[slot] in finetune_set
            ):
                leaf = raw_tokens[slot]
            else:
                leaf = raw_tokens[slot].detach().requires_grad_(True)
            adapted.append(adapter(leaf, w_adapters[slot]))
        return tuple(adapted)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scale_ids(patches_per_scale: Tuple[int, ...], batch: int, device) -> torch.Tensor:
        """Create scale-id tensor for multi-scale patches.

        Returns: [B, N_total] int tensor where each patch is tagged with
        its scale index (0, 1, 2, ...).
        """
        ids = []
        for scale_idx, count in enumerate(patches_per_scale):
            ids.append(torch.full((batch, count), scale_idx, dtype=torch.long, device=device))
        return torch.cat(ids, dim=1)

    def _extract_patch_embeddings(self, patches: torch.Tensor) -> torch.Tensor:
        """Extract ViT embeddings for a batch of 224x224 patches.

        Args:
            patches: [B, 3, 224, 224]

        Returns:
            embeddings: [B, 768] - CLS-token embedding from the last feature layer
        """
        if self.hf_backbone is not None:
            raise NotImplementedError(
                "multi-scale patch mode is not supported with HF backbones; "
                "use the grouped-patch path (multiscale_scales: null)"
            )
        layers = tuple(int(index) for index in self.cfg.feature_layers)
        with torch.no_grad():
            embeddings = []
            for chunk in patches.split(int(self.cfg.backbone_chunk_size), dim=0):
                tokens = self._embed_tokens(self._resize(chunk))
                captured = None
                for index, block in enumerate(self.vit.blocks):
                    tokens = block(tokens)
                    if index == layers[-1]:
                        captured = tokens[:, 0]  # CLS token
                        break
                embeddings.append(captured)
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        images: torch.Tensor,
        scale_ids: Optional[torch.Tensor] = None,
        fragment_images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        global_images: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_evidence: bool = False,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del fragment_images

        # ---- Multi-scale patch mode: [B, N, C, H, W] with scale_ids ----
        if images.ndim == 5 and scale_ids is not None:
            return self.forward_multiscale(
                images, scale_ids,
                return_features=return_features,
                return_evidence=return_evidence,
                return_details=return_details,
            )

        # ---- 5D input WITHOUT scale_ids (legacy grouped-patch mode) ----
        if images.ndim == 5 and scale_ids is None:
            return self.forward_grouped(
                images,
                targets=targets,
                global_images=global_images,
                patch_positions=patch_positions,
                return_features=return_features,
                return_evidence=return_evidence,
                return_details=return_details,
            )

        # ---- Single-patch mode: [B, C, H, W] ----
        if images.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] or [B,N,C,H,W], got {tuple(images.shape)}")

        del targets, global_images, patch_positions

        resized = self._resize(images)

        # ---- GR-QIE ----
        qie_out: Dict[str, torch.Tensor] = {}
        if self.gr_qie is not None:
            logits_f, logits_m, logits_c = self.gr_qie.forward_logits(resized)
            role_logits = (
                self.gr_qie_role.forward_logits(resized)
                if self.cfg.qie_route_sharing == "unshared"
                else (logits_f, logits_m, logits_c)
            )

            # The archived path uses sigmoid.  A centered, bounded transform
            # is available only to explicitly configured post-audit runs.
            adapter_transform = self.cfg.qie_adapter_transform
            adapter_temperature = self.cfg.qie_adapter_temperature
            w_f_adapter = adapter_field_from_logits(
                logits_f, adapter_transform, adapter_temperature
            )
            w_m_adapter = adapter_field_from_logits(
                logits_m, adapter_transform, adapter_temperature
            )
            w_c_adapter = adapter_field_from_logits(
                logits_c, adapter_transform, adapter_temperature
            )

            # Softmax for DepthRoleEncoder (probability distribution, sum=1)
            w_f_role, w_m_role, w_c_role = (
                F.softmax(role_logit, dim=-1) for role_logit in role_logits
            )

            qie_out = {
                "qie_logits_f": logits_f,
                "qie_logits_m": logits_m,
                "qie_logits_c": logits_c,
                "qie_weights_f": w_f_adapter,
                "qie_weights_m": w_m_adapter,
                "qie_weights_c": w_c_adapter,
            }
        else:
            num_tokens = int(self.cfg.token_grid_side) ** 2
            w_f_adapter = w_m_adapter = w_c_adapter = torch.zeros(
                images.size(0), num_tokens, device=images.device
            )
            w_f_role = w_m_role = w_c_role = torch.full(
                (images.size(0), num_tokens), 1.0 / num_tokens, device=images.device
            )

        if not self.cfg.qie_adapter_conditioning:
            w_f_adapter = torch.zeros_like(w_f_adapter)
            w_m_adapter = torch.zeros_like(w_m_adapter)
            w_c_adapter = torch.zeros_like(w_c_adapter)
        if not self.cfg.qie_role_pooling:
            w_f_role = torch.full_like(w_f_role, 1.0 / w_f_role.size(1))
            w_m_role = torch.full_like(w_m_role, 1.0 / w_m_role.size(1))
            w_c_role = torch.full_like(w_c_role, 1.0 / w_c_role.size(1))

        # Soft-coupling: during warmup, w_adapters are detached.
        rho = self._coupling_rho.item()
        if rho < 1e-6:
            w_adapters = (w_f_adapter.detach(), w_m_adapter.detach(), w_c_adapter.detach())
        else:
            w_adapters = (
                (1 - rho) * w_f_adapter.detach() + rho * w_f_adapter,
                (1 - rho) * w_m_adapter.detach() + rho * w_m_adapter,
                (1 - rho) * w_c_adapter.detach() + rho * w_c_adapter,
            )

        # ---- ViT + GA-Adapter ----
        adapted_tokens = self._extract_and_adapt(resized, w_adapters)

        # ---- Spatially aligned distortion-token injection ----
        spatial_distortion_tokens = None
        if self.spatial_distortion_tokenizer is not None:
            spatial_distortion_tokens = self.spatial_distortion_tokenizer(resized)
            adapted_tokens = tuple(
                injector(tokens, spatial_distortion_tokens)
                for injector, tokens in zip(
                    self.spatial_distortion_injectors, adapted_tokens
                )
            )

        # ---- DepthRoleEncoder ----
        role_nodes = []
        weights_role = [w_f_role, w_m_role, w_c_role]
        for encoder, tokens, w in zip(self.depth_encoders, adapted_tokens, weights_role):
            global_node, local_node = encoder(tokens, w)
            if self.cfg.role_pooling in {"global_local", "global_only"}:
                role_nodes.append(global_node)
            if self.cfg.role_pooling in {"global_local", "local_only"}:
                role_nodes.append(local_node)

        # ---- Distortion Sidecar ----
        gate_value = None
        if self.distortion_sidecar is not None and self.controlled_fusion is not None:
            node_d = self.distortion_sidecar(resized)
            semantic_stack = torch.stack(role_nodes, dim=1)  # [B, 6, node_dim]
            node_d_fused, gate_value = self.controlled_fusion(semantic_stack, node_d)
            role_nodes.append(node_d_fused)

        # ---- Role Graph ----
        nodes = torch.stack(role_nodes, dim=1) + self.role_embeddings
        attentions = []
        for block in self.graph:
            nodes, attention = block(nodes)
            if attention is not None:
                attentions.append(attention)

        # ---- Score ----
        mean_pool = nodes.mean(dim=1)
        std_pool = nodes.std(dim=1, unbiased=False)
        pooled = torch.cat((mean_pool, std_pool), dim=-1)
        embedding = self.output_norm(pooled)
        score = torch.sigmoid(self.score_head(embedding)).squeeze(-1)

        output: Dict[str, torch.Tensor] = {"score": score}

        # ---- Role sub-scores ----
        if self.role_score_head is not None:
            role_sub = torch.sigmoid(
                self.role_score_head(nodes).squeeze(-1)
            )
            output["role_sub_scores"] = role_sub

        # ---- QIE outputs (always expose for training loss) ----
        output.update(qie_out)

        if return_features or return_evidence or return_details:
            output.update({
                "quality_embedding": embedding,
                "role_nodes": nodes,
                "role_weights": torch.full(
                    (images.size(0), len(self.role_names)),
                    1.0 / len(self.role_names),
                    device=images.device,
                    dtype=score.dtype,
                ),
            })
            if attentions:
                output["role_graph_attention"] = torch.stack(attentions, dim=1)
            if gate_value is not None:
                output["sidecar_gate"] = gate_value
            if spatial_distortion_tokens is not None:
                output["spatial_distortion_tokens"] = spatial_distortion_tokens

        return output

    def forward_multiscale(
        self,
        images: torch.Tensor,
        scale_ids: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_evidence: bool = False,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Multi-scale patch forward pass.

        Args:
            images: [B, N_total, 3, 224, 224] - patches from all scales
            scale_ids: [B, N_total] - integer scale index per patch.
                       If None, auto-generated from patches_per_scale config.
        """
        if images.ndim != 5:
            raise ValueError(f"Expected [B,N,C,H,W], got {tuple(images.shape)}")

        B, N, C, H, W = images.shape
        cfg = self.cfg

        # Auto-generate scale_ids if not provided
        if scale_ids is None:
            scale_ids = self._build_scale_ids(cfg.patches_per_scale, B, images.device)

        # Flatten to [B*N, 3, 224, 224] for ViT processing
        flat_patches = images.reshape(B * N, C, H, W)

        # Extract ViT CLS embeddings for all patches
        patch_embs = self._extract_patch_embeddings(flat_patches)  # [B*N, embed_dim]
        patch_embs = patch_embs.reshape(B, N, self.embed_dim)  # [B, N, embed_dim]

        # ---- GR-QIE on the global-scale patch (last patch = scale 0.25) ----
        qie_out: Dict[str, torch.Tensor] = {}
        # Find the global patch index (last scale, typically 0.25)
        global_scale_idx = len(cfg.multiscale_scales) - 1
        global_mask = (scale_ids == global_scale_idx)  # [B, N]
        # Use the first global patch for QIE
        global_idx = global_mask.int().argmax(dim=1)  # [B]
        global_patch = images[torch.arange(B), global_idx]  # [B, 3, 224, 224]

        if self.gr_qie is not None:
            logits_f, logits_m, logits_c = self.gr_qie.forward_logits(global_patch)
            role_logits = (
                self.gr_qie_role.forward_logits(global_patch)
                if self.cfg.qie_route_sharing == "unshared"
                else (logits_f, logits_m, logits_c)
            )
            adapter_transform = self.cfg.qie_adapter_transform
            adapter_temperature = self.cfg.qie_adapter_temperature
            w_f_adapter = adapter_field_from_logits(
                logits_f, adapter_transform, adapter_temperature
            )
            w_m_adapter = adapter_field_from_logits(
                logits_m, adapter_transform, adapter_temperature
            )
            w_c_adapter = adapter_field_from_logits(
                logits_c, adapter_transform, adapter_temperature
            )
            w_f_role, w_m_role, w_c_role = (
                F.softmax(role_logit, dim=-1) for role_logit in role_logits
            )
            qie_out = {
                "qie_logits_f": logits_f,
                "qie_logits_m": logits_m,
                "qie_logits_c": logits_c,
                "qie_weights_f": w_f_adapter,
                "qie_weights_m": w_m_adapter,
                "qie_weights_c": w_c_adapter,
            }
        else:
            num_tokens = int(self.cfg.token_grid_side) ** 2
            w_f_adapter = w_m_adapter = w_c_adapter = torch.zeros(
                B, num_tokens, device=images.device
            )
            w_f_role = w_m_role = w_c_role = torch.full(
                (B, num_tokens), 1.0 / num_tokens, device=images.device
            )

        if not self.cfg.qie_adapter_conditioning:
            w_f_adapter = torch.zeros_like(w_f_adapter)
            w_m_adapter = torch.zeros_like(w_m_adapter)
            w_c_adapter = torch.zeros_like(w_c_adapter)
        if not self.cfg.qie_role_pooling:
            w_f_role = torch.full_like(w_f_role, 1.0 / w_f_role.size(1))
            w_m_role = torch.full_like(w_m_role, 1.0 / w_m_role.size(1))
            w_c_role = torch.full_like(w_c_role, 1.0 / w_c_role.size(1))

        # ---- GA-Adapter on the global patch ----
        rho = self._coupling_rho.item()
        if rho < 1e-6:
            w_adapters = (w_f_adapter.detach(), w_m_adapter.detach(), w_c_adapter.detach())
        else:
            w_adapters = (
                (1 - rho) * w_f_adapter.detach() + rho * w_f_adapter,
                (1 - rho) * w_m_adapter.detach() + rho * w_m_adapter,
                (1 - rho) * w_c_adapter.detach() + rho * w_c_adapter,
            )

        adapted_tokens = self._extract_and_adapt(global_patch, w_adapters)

        # ---- DepthRoleEncoder ----
        role_nodes = []
        weights_role = [w_f_role, w_m_role, w_c_role]
        for encoder, tokens, w in zip(self.depth_encoders, adapted_tokens, weights_role):
            global_node, local_node = encoder(tokens, w)
            if self.cfg.role_pooling in {"global_local", "global_only"}:
                role_nodes.append(global_node)
            if self.cfg.role_pooling in {"global_local", "local_only"}:
                role_nodes.append(local_node)

        # ---- Distortion Sidecar ----
        gate_value = None
        if self.distortion_sidecar is not None and self.controlled_fusion is not None:
            node_d = self.distortion_sidecar(global_patch)
            semantic_stack = torch.stack(role_nodes, dim=1)  # [B, 6, node_dim]
            node_d_fused, gate_value = self.controlled_fusion(semantic_stack, node_d)
            role_nodes.append(node_d_fused)

        # ---- Role Graph ----
        nodes = torch.stack(role_nodes, dim=1) + self.role_embeddings
        attentions = []
        for block in self.graph:
            nodes, attention = block(nodes)
            if attention is not None:
                attentions.append(attention)

        # ---- Patch Aggregation (primary path) ----
        aggregated, agg_extras = self.patch_aggregator(patch_embs, scale_ids)
        embedding = self.output_norm(aggregated)
        score = torch.sigmoid(self.score_head(embedding)).squeeze(-1)

        output: Dict[str, torch.Tensor] = {"score": score}

        # Add patch-level outputs from aggregator
        output.update(agg_extras)

        # Compute patch-level variance metrics (when patch_scores available)
        if "patch_scores" in agg_extras:
            patch_scores = agg_extras["patch_scores"]
            variance = (patch_scores - score.unsqueeze(1)).square().mean(dim=1)
            output["patch_score_mean"] = patch_scores.mean(dim=1)
            output["patch_score_variance"] = variance
            output["crop_dispersion"] = variance.sqrt()
            if "patch_weights" in agg_extras:
                weights = agg_extras["patch_weights"]
                output["patch_weight_entropy"] = -(
                    weights * (weights + 1e-8).log()
                ).sum(dim=1)

        # ---- Role sub-scores ----
        if self.role_score_head is not None:
            role_sub = torch.sigmoid(
                self.role_score_head(nodes).squeeze(-1)
            )
            output["role_sub_scores"] = role_sub

        # ---- QIE outputs (always expose for training loss) ----
        output.update(qie_out)

        if return_features or return_evidence or return_details:
            output["quality_embedding"] = embedding
            output["role_nodes"] = nodes
            output["role_weights"] = torch.full(
                (B, len(self.role_names)),
                1.0 / len(self.role_names),
                device=images.device,
                dtype=score.dtype,
            )
            if attentions:
                output["role_graph_attention"] = torch.stack(attentions, dim=1)
            if gate_value is not None:
                output["sidecar_gate"] = gate_value

        return output

    def forward_grouped(
        self,
        images: torch.Tensor,
        fragment_images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        global_images: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_evidence: bool = False,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Legacy grouped-patch forward with variance-aware weighting.

        This is the ORIGINAL forward_grouped logic preserved for backward
        compatibility. When num_patches > 1 and multiscale is NOT enabled,
        this path is used. Each patch is independently scored, then patches
        are aggregated via variance-aware weighting.

        When multi-scale mode is enabled and scale_ids are provided,
        ``forward_multiscale`` is used instead.
        """
        del fragment_images
        if images.ndim != 5:
            raise ValueError(f"Expected [B,N,C,H,W], got {tuple(images.shape)}")

        batch, count, channels, height, width = images.shape
        flat = images.reshape(batch * count, channels, height, width)
        needs_set_features = self.global_local_set_aggregator is not None
        flat_output = self.forward(
            flat,
            return_features=return_features or needs_set_features,
            return_evidence=return_evidence or return_details,
        )
        patch_scores = flat_output["score"].reshape(batch, count)

        if needs_set_features:
            if global_images is None:
                # Backward-compatible fallback for callers without a dedicated
                # whole-image view. The patch mean is deterministic but new
                # grouped benchmark configs always provide global_images.
                global_images = images.mean(dim=1)
            if global_images.shape != (batch, channels, height, width):
                raise ValueError(
                    "global_images must be [B,C,H,W], got "
                    f"{tuple(global_images.shape)}"
                )
            global_output = self.forward(global_images, return_features=True)
            local_embeddings = flat_output["quality_embedding"].reshape(
                batch, count, -1
            )
            embedding, set_extras = self.global_local_set_aggregator(
                local_embeddings,
                global_output["quality_embedding"],
                patch_positions,
            )
            # Strong-path preservation: independent crop scores and arithmetic
            # mean are the baseline.  Set/global reasoning can only add a
            # bounded logit residual, initialized to exactly zero.
            base_score = patch_scores.mean(dim=1)
            residual = torch.tanh(
                self.set_residual_head(embedding).squeeze(-1)
            ) * float(self.cfg.set_residual_logit_scale)
            base_logit = torch.logit(base_score.clamp(1e-5, 1.0 - 1e-5))
            score = torch.sigmoid(base_logit + residual)
            weights = set_extras["patch_weights"]
            weighted_patch_score = (patch_scores * weights).sum(dim=1)
            variance = (
                weights * (patch_scores - weighted_patch_score.unsqueeze(1)).square()
            ).sum(dim=1)
            output: Dict[str, torch.Tensor] = {
                "score": score,
                "quality_embedding": embedding,
                "patch_scores": patch_scores,
                "set_base_score": base_score,
                "set_logit_residual": residual,
                "patch_score_mean": patch_scores.mean(dim=1),
                "patch_score_variance": variance,
                "crop_dispersion": variance.sqrt(),
                "crop_consistency_loss": F.smooth_l1_loss(
                    score, weighted_patch_score
                ),
                **set_extras,
            }
            if targets is not None and "patch_uncertainty_logvar" in set_extras:
                image_logvar = torch.logsumexp(
                    set_extras["patch_uncertainty_logvar"]
                    + weights.clamp_min(1e-8).log(),
                    dim=1,
                ).clamp(-6.0, 6.0)
                squared_error = (score.float() - targets.flatten().float()).square()
                output["uncertainty_loss"] = (
                    squared_error * torch.exp(-image_logvar.float())
                    + image_logvar.float()
                ).mean()

            for key in (
                "qie_logits_f", "qie_logits_m", "qie_logits_c",
                "qie_weights_f", "qie_weights_m", "qie_weights_c",
            ):
                if key in flat_output:
                    output[key] = flat_output[key].reshape(batch, count, -1)
            if "role_sub_scores" in flat_output:
                output["role_sub_scores"] = flat_output["role_sub_scores"].reshape(
                    batch, count, len(self.role_names)
                )
            if return_features or return_evidence or return_details:
                output["global_quality_embedding"] = global_output[
                    "quality_embedding"
                ]
                for key in ("role_nodes", "role_weights", "sidecar_gate"):
                    if key in flat_output:
                        output[key] = flat_output[key].reshape(
                            batch, count, *flat_output[key].shape[1:]
                        )
            return output

        # Variance-aware weighting via role sub-score variance
        if "role_sub_scores" in flat_output:
            sub_scores = flat_output["role_sub_scores"]  # [B*N, 7]
            sub_var = sub_scores.var(dim=-1).reshape(batch, count)
            confidence = 1.0 / (1.0 + 10.0 * sub_var)
            weights = confidence / confidence.sum(dim=1, keepdim=True)
        else:
            weights = torch.full_like(patch_scores, 1.0 / count)

        score = (patch_scores * weights).sum(dim=1)
        variance = (patch_scores - score.unsqueeze(1)).square().mean(dim=1)

        output: Dict[str, torch.Tensor] = {
            "score": score,
            "patch_scores": patch_scores,
            "patch_weights": weights,
            "patch_score_mean": patch_scores.mean(dim=1),
            "patch_score_variance": variance,
            "crop_dispersion": variance.sqrt(),
            "patch_weight_entropy": -(
                weights * (weights + 1e-8).log()
            ).sum(dim=1),
        }

        # ---- QIE & sub-score outputs (always expose for training loss) ----
        for key in ("qie_logits_f", "qie_logits_m", "qie_logits_c",
                    "qie_weights_f", "qie_weights_m", "qie_weights_c"):
            if key in flat_output:
                output[key] = flat_output[key].reshape(batch, count, -1)
        if "role_sub_scores" in flat_output:
            output["role_sub_scores"] = flat_output["role_sub_scores"].reshape(
                batch, count, len(self.role_names)
            )

        if return_features or return_evidence or return_details:
            for key in ("quality_embedding", "role_nodes", "role_weights", "sidecar_gate"):
                if key in flat_output:
                    output[key] = flat_output[key].reshape(
                        batch, count, *flat_output[key].shape[1:]
                    )
            if "role_graph_attention" in flat_output:
                value = flat_output["role_graph_attention"]
                output["role_graph_attention"] = value.reshape(
                    batch, count, *value.shape[1:]
                )
        return output


# ---------------------------------------------------------------------------
# Factory and parameter counter
# ---------------------------------------------------------------------------


class SpatialSetMAIQA(MAIQA):
    """MAIQA with spatial distortion tokens and global/local set reasoning."""

    def __init__(self, cfg: MAIQAConfig):
        if not cfg.use_spatial_distortion_tokens:
            raise ValueError(
                "spatial_set_maiqa requires use_spatial_distortion_tokens=true"
            )
        if not cfg.use_global_local_set_pooling:
            raise ValueError(
                "spatial_set_maiqa requires use_global_local_set_pooling=true"
            )
        super().__init__(cfg)


def build_maiqa_from_config(config: dict) -> MAIQA:
    architecture = str(config.get("architecture", "maiqa")).strip().lower()
    model_cfg = {
        key: value
        for key, value in config.items()
        if key not in {"version", "architecture", "max_trainable_parameters"}
    }
    cfg = MAIQAConfig(**model_cfg)
    if architecture in {"maiqa", "full"}:
        return MAIQA(cfg)
    if architecture in {
        "spatial_set_maiqa", "spatial-set-maiqa", "spatial_set",
        "global_local_spatial_evidence",
    }:
        return SpatialSetMAIQA(cfg)
    if architecture in {
        "minimal_multidepth", "minimal-multidepth", "minimal_multidepth_iqa"
    }:
        from maiqa.minimal_multidepth import MinimalMultiDepthIQA

        return MinimalMultiDepthIQA(cfg)
    if architecture in {
        "sufficient_evidence", "sufficient-evidence", "suffiqa",
        "conserved_evidence_iqa",
    }:
        from maiqa.sufficient_evidence import SufficientEvidenceIQA

        return SufficientEvidenceIQA(cfg)
    if architecture in {
        "dual_evidence", "dual-evidence", "dual_evidence_iqa",
        "conserved_dual_evidence",
    }:
        from maiqa.dual_evidence import DualEvidenceIQA

        return DualEvidenceIQA(cfg)
    if architecture in {
        "ccre_iqa", "ccre-iqa", "ccre",
        "competitive_conserved_residual_evidence",
    }:
        from maiqa.ccre_iqa import CCREIQA

        return CCREIQA(cfg)
    if architecture in {
        "codi_iqa", "codi-iqa", "codi",
        "conditional_orthogonal_depth_innovation",
    }:
        from maiqa.codi_iqa import CODIIQA

        return CODIIQA(cfg)
    if architecture in {
        "sice_iqa", "sice-iqa", "sice",
        "shared_innovation_conserved_evidence",
    }:
        from maiqa.sice_iqa import SICEIQA

        return SICEIQA(cfg)
    if architecture in {
        "icce_iqa", "icce-iqa", "icce",
        "invertible_contrastive_conserved_evidence",
    }:
        from maiqa.icce_iqa import ICCEIQA

        return ICCEIQA(cfg)
    if architecture in {
        "hice_iqa", "hice-iqa", "hice",
        "head_preserving_interventional_evidence",
    }:
        from maiqa.hice_iqa import HICEIQA

        return HICEIQA(cfg)
    if architecture in {
        "set_weighted_multidepth", "set-weighted-multidepth",
        "set_weighted_maiqa", "crop_set_maiqa",
    }:
        from maiqa.set_weighted_multidepth import SetWeightedMultiDepthIQA

        return SetWeightedMultiDepthIQA(cfg)
    if architecture in {
        "heterogeneity_shrinkage_multidepth", "heterogeneity-shrinkage-multidepth",
        "shrinkage_set_maiqa", "adaptive_crop_shrinkage",
    }:
        from maiqa.heterogeneity_shrinkage import HeterogeneityShrinkageIQA

        return HeterogeneityShrinkageIQA(cfg)
    if architecture in {
        "rank_calibrated_set_multidepth", "rank-calibrated-set-multidepth",
        "rank_calibrated_set_maiqa", "metric_decoupled_crop_set",
    }:
        from maiqa.rank_calibrated_set import RankCalibratedSetIQA

        return RankCalibratedSetIQA(cfg)
    raise ValueError(f"Unknown model architecture: {architecture!r}")


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    backbone_modules = []
    primary_backbone = (
        model.vit if getattr(model, "vit", None) is not None
        else getattr(model, "hf_backbone", None)
    )
    if primary_backbone is not None:
        backbone_modules.append(primary_backbone)
    semantic_backbone = getattr(model, "semantic_backbone", None)
    if semantic_backbone is not None:
        backbone_modules.append(semantic_backbone)
    if not backbone_modules:
        backbone = 0
        backbone_trainable = 0
    else:
        backbone = sum(
            parameter.numel()
            for module in backbone_modules
            for parameter in module.parameters()
        )
        backbone_trainable = sum(
            parameter.numel()
            for backbone_module in backbone_modules
            for parameter in backbone_module.parameters()
            if parameter.requires_grad
        )
    return {
        "total": total,
        "trainable": trainable,
        "frozen_backbone": backbone,
        "backbone_trainable": backbone_trainable,
        "trainable_millions": trainable / 1.0e6,
    }
