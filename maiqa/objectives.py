"""Lightweight IQA objectives for the MAIQA model.

Primary: CCCLoss calibrates absolute scores.
Auxiliary:
  - MarginRankingLoss preserves pairwise quality ordering.
  - QIERegularization keeps GR-QIE heads concentrated yet diverse.
  - SubScoreConsistencyLoss regularises role sub-score behaviour.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CCCLoss(nn.Module):
    """Align correlation, prediction mean, and prediction score range."""

    def __init__(self, eps: float = 1e-8, fallback_beta: float = 0.1):
        super().__init__()
        self.eps = float(eps)
        self.fallback_beta = float(fallback_beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten().float()
        target = target.flatten().float()
        if pred.numel() != target.numel():
            raise ValueError(
                "pred and target must contain the same number of scores"
            )
        if pred.numel() < 2 or torch.var(target, unbiased=False) < self.eps:
            return F.smooth_l1_loss(
                pred, target, beta=self.fallback_beta
            )

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        pred_var = pred_centered.square().mean()
        target_var = target_centered.square().mean()
        covariance = (pred_centered * target_centered).mean()
        mean_error = (pred.mean() - target.mean()).square()
        ccc = (2.0 * covariance) / (
            pred_var + target_var + mean_error + self.eps
        )
        return 1.0 - ccc


class PLCCLoss(nn.Module):
    """PLCC (Pearson Linear Correlation Coefficient) loss.

    Minimising 1 - PLCC drives predicted scores to be linearly correlated
    with ground-truth MOS.  Unlike CCC, PLCC does NOT penalise
    mean-shift or scale mismatch — it only cares about monotonic linear
    alignment.  This makes it a natural choice for IQA models whose
    output range may differ from the MOS scale (e.g. sigmoid outputs
    trained against [0, 100] MOS).

    When the batch has < 2 valid samples (or target variance ≈ 0),
    the loss falls back to Smooth-L1 so that gradients are still
    well-defined.
    """

    def __init__(self, eps: float = 1e-8, fallback_beta: float = 0.1):
        super().__init__()
        self.eps = float(eps)
        self.fallback_beta = float(fallback_beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten().float()
        target = target.flatten().float()
        if pred.numel() != target.numel():
            raise ValueError(
                "pred and target must contain the same number of scores"
            )
        if pred.numel() < 2 or torch.var(target, unbiased=False) < self.eps:
            return F.smooth_l1_loss(
                pred, target, beta=self.fallback_beta
            )

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        covariance = (pred_centered * target_centered).mean()
        pred_std = pred_centered.square().mean().sqrt()
        target_std = target_centered.square().mean().sqrt()
        plcc = covariance / (pred_std * target_std + self.eps)
        return 1.0 - plcc


class CCCPLCCLoss(nn.Module):
    """Mixed CCC + PLCC loss.

    Combines CCC (which penalises mean-shift and scale mismatch) with
    PLCC (which only optimises linear correlation).  The blend lets the
    model learn correct absolute scale (via CCC) while retaining the
    ranking-friendly gradient signal of PLCC.

    ``loss = ccc_alpha * (1 - CCC) + plcc_beta * (1 - PLCC)``

    Default weights: ``ccc_alpha=1.0, plcc_beta=0.5`` gives CCC dominant
    but PLCC supplements ranking gradients.

    When the batch has < 2 valid samples (or target variance ≈ 0),
    both terms fall back to Smooth-L1.
    """

    def __init__(
        self,
        ccc_alpha: float = 1.0,
        plcc_beta: float = 0.5,
        eps: float = 1e-8,
        fallback_beta: float = 0.1,
    ):
        super().__init__()
        self.ccc_alpha = float(ccc_alpha)
        self.plcc_beta = float(plcc_beta)
        self.eps = float(eps)
        self.fallback_beta = float(fallback_beta)
        self._ccc = CCCLoss(eps=eps, fallback_beta=fallback_beta)
        self._plcc = PLCCLoss(eps=eps, fallback_beta=fallback_beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ccc_val = self._ccc(pred, target)
        plcc_val = self._plcc(pred, target)
        return self.ccc_alpha * ccc_val + self.plcc_beta * plcc_val


class MarginRankingLoss(nn.Module):
    """Pairwise margin ranking loss for quality ordering.

    Unlike the distributional preference loss, this operates directly on
    continuous MOS scores and does not require ordinal bin logits.
    """

    def __init__(self, margin: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.margin = float(margin)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten().float()
        target = target.flatten().float()
        batch = pred.numel()
        if batch < 2:
            return pred.sum() * 0.0

        i, j = torch.triu_indices(batch, batch, offset=1, device=pred.device)
        diff = target[i] - target[j]  # positive when i > j
        label = (diff > self.eps).float() - (diff < -self.eps).float()
        mask = label.abs() > 0.5

        if mask.sum() < 1:
            return pred.sum() * 0.0

        pred_diff = pred[i[mask]] - pred[j[mask]]
        label_masked = label[mask]
        loss = torch.clamp(-label_masked * pred_diff + self.margin, min=0.0)
        return loss.mean()


class SoftSpearmanLoss(nn.Module):
    """Differentiable batchwise approximation of ``1 - SRCC``.

    Pairwise sigmoids form soft ranks for predictions. Targets use the same
    temperature but are detached, so ties remain smooth and deterministic.
    """

    def __init__(self, temperature: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if temperature <= 0:
            raise ValueError("SoftSpearman temperature must be positive")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def _soft_rank(self, values: torch.Tensor) -> torch.Tensor:
        differences = values.unsqueeze(0) - values.unsqueeze(1)
        return 1.0 + torch.sigmoid(
            differences / self.temperature
        ).sum(dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten().float()
        target = target.flatten().float()
        if pred.numel() != target.numel():
            raise ValueError("pred and target must have the same number of scores")
        if pred.numel() < 3 or target.var(unbiased=False) < self.eps:
            return pred.sum() * 0.0
        pred_rank = self._soft_rank(pred)
        with torch.no_grad():
            target_rank = self._soft_rank(target)
        pred_rank = pred_rank - pred_rank.mean()
        target_rank = target_rank - target_rank.mean()
        correlation = (pred_rank * target_rank).mean() / (
            pred_rank.square().mean().sqrt()
            * target_rank.square().mean().sqrt()
            + self.eps
        )
        return 1.0 - correlation


class QIERegularization(nn.Module):
    """Combined concentration + diversity regularisation for GR-QIE heads.

    L_qie = lambda_c * L_concentration + lambda_d * L_diversity

    L_concentration: penalises flat (uniform) importance maps.
        softplus(theta_c - avg_var) + tail * exp(-decay * avg_var)
        The softplus provides a smooth hinge, while the exponential tail
        guarantees nonzero gradient even when the variance constraint is
        satisfied — preventing QIE from drifting after convergence.

    L_diversity: penalises when all three granularity heads produce
    identical importance patterns.
        softplus(theta_d - mean_cos_dist) + tail * exp(-decay * mean_cos_dist)
        Same hinge + tail design for continuous gradient signal.
    """

    def __init__(
        self,
        theta_c: float = 0.04,
        theta_d: float = 0.3,
        lambda_c: float = 1.0,
        lambda_d: float = 1.0,
        tail_weight: float = 0.01,
        tail_decay: float = 10.0,
    ):
        super().__init__()
        self.theta_c = float(theta_c)
        self.theta_d = float(theta_d)
        self.lambda_c = float(lambda_c)
        self.lambda_d = float(lambda_d)
        self.tail_weight = float(tail_weight)
        self.tail_decay = float(tail_decay)

    def forward(
        self,
        logits_f: torch.Tensor,
        logits_m: torch.Tensor,
        logits_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_f, logits_m, logits_c: [B, 196] raw QIE logits

        Returns:
            scalar loss
        """
        w_f = torch.sigmoid(logits_f)
        w_m = torch.sigmoid(logits_m)
        w_c = torch.sigmoid(logits_c)

        # Concentration: penalise flat importance (hinge + exponential tail)
        var_f = w_f.var(dim=-1)
        var_m = w_m.var(dim=-1)
        var_c = w_c.var(dim=-1)
        avg_var = (var_f + var_m + var_c).mean() / 3.0
        l_concentration = (
            F.softplus(self.theta_c - avg_var)
            + self.tail_weight * torch.exp(-self.tail_decay * avg_var)
        )

        # Diversity: penalise identical patterns across granularities (hinge + tail)
        w_f_n = F.normalize(w_f.flatten(0), dim=-1)
        w_m_n = F.normalize(w_m.flatten(0), dim=-1)
        w_c_n = F.normalize(w_c.flatten(0), dim=-1)

        cos_fm = F.cosine_similarity(w_f_n, w_m_n, dim=-1)
        cos_fc = F.cosine_similarity(w_f_n, w_c_n, dim=-1)
        cos_mc = F.cosine_similarity(w_m_n, w_c_n, dim=-1)
        mean_cos_sim = (cos_fm + cos_fc + cos_mc) / 3.0
        mean_cos_dist = 1.0 - mean_cos_sim
        l_diversity = (
            F.softplus(self.theta_d - mean_cos_dist)
            + self.tail_weight * torch.exp(-self.tail_decay * mean_cos_dist)
        )

        return self.lambda_c * l_concentration + self.lambda_d * l_diversity


class SubScoreConsistencyLoss(nn.Module):
    """Regularise role sub-scores to be meaningful and not degenerate.

    Encourages sub-score variance (prevents collapse to constant) and
    penalises extreme values (keeps scores in a reasonable range).

    Uses softplus + exponential tail instead of hard clamp, ensuring
    nonzero gradient even when the variance constraint is satisfied.
    """

    def __init__(
        self,
        min_variance: float = 0.005,
        margin: float = 0.95,
        tail_weight: float = 0.005,
        tail_decay: float = 10.0,
    ):
        super().__init__()
        self.min_variance = float(min_variance)
        self.margin = float(margin)
        self.tail_weight = float(tail_weight)
        self.tail_decay = float(tail_decay)

    def forward(self, role_sub_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            role_sub_scores: [B, 7] sigmoid-activated sub-scores

        Returns:
            scalar loss
        """
        # Penalise near-constant sub-scores (hinge + exponential tail)
        var = role_sub_scores.var(dim=-1).mean()
        l_var = (
            F.softplus(self.min_variance - var)
            + self.tail_weight * torch.exp(-self.tail_decay * var)
        )

        # Penalise extreme saturation (all scores near 0 or 1)
        l_extreme = torch.clamp(
            role_sub_scores - self.margin, min=0.0
        ).mean() + torch.clamp(
            1.0 - self.margin - role_sub_scores, min=0.0
        ).mean()

        return l_var + l_extreme


def pairwise_preference_matrix(probabilities: torch.Tensor) -> torch.Tensor:
    """Return P(i > j) + 0.5 P(i = j) for ordinal distributions."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, ordinal_bins]")
    num_bins = probabilities.size(-1)
    indices = torch.arange(num_bins, device=probabilities.device)
    kernel = (
        (indices[:, None] > indices[None, :]).to(probabilities.dtype)
        + 0.5
        * (indices[:, None] == indices[None, :]).to(probabilities.dtype)
    )
    return torch.einsum(
        "ik,kl,jl->ij", probabilities, kernel, probabilities
    )


def distributional_preference_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Match distribution-derived preferences to continuous MOS ordering."""
    if temperature <= 0:
        raise ValueError("preference temperature must be positive")
    batch = logits.size(0)
    if batch < 2:
        return logits.sum() * 0.0
    targets = targets.flatten()
    if targets.numel() != batch:
        raise ValueError("targets must contain one score per distribution")

    probabilities = F.softmax(logits.float(), dim=-1)
    preferences = pairwise_preference_matrix(probabilities)
    i, j = torch.triu_indices(batch, batch, offset=1, device=logits.device)
    predicted = preferences[i, j]
    desired = torch.sigmoid(
        (targets[i].float() - targets[j].float()) / temperature
    )

    eps = torch.finfo(predicted.dtype).eps
    predicted = predicted.clamp(eps, 1.0 - eps)
    desired = desired.clamp(eps, 1.0 - eps)
    kl = (
        desired * (desired.log() - predicted.log())
        + (1.0 - desired)
        * ((1.0 - desired).log() - (1.0 - predicted).log())
    )
    return kl.mean().clamp_min(0.0)


__all__ = [
    "CCCLoss",
    "PLCCLoss",
    "CCCPLCCLoss",
    "MarginRankingLoss",
    "QIERegularization",
    "SubScoreConsistencyLoss",
    "distributional_preference_loss",
    "pairwise_preference_matrix",
]
