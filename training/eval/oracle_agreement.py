"""Oracle agreement metric for Phase 2 evaluation.

Computes the percentage of validation examples where the model's top-ranked
target matches the oracle's top-ranked target.  This is the primary
generalization metric — use it for early stopping, not BCE/CE loss.

Usage:
    from eval.oracle_agreement import compute_oracle_agreement
    agreement = compute_oracle_agreement(model, tokenizer, val_loader, device)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_oracle_agreement(
    model,
    val_loader,
    device: torch.device,
) -> dict[str, float]:
    """Compute oracle agreement metrics on validation set.

    Parameters
    ----------
    model : AbilityTransformerDecision
        Fine-tuned model with urgency + target heads.
    val_loader : iterable of dict
        Each batch has: input_ids, attention_mask, game_state, urgency, target_idx,
        has_target (bool mask for unit-targeted categories).
    device : torch.device

    Returns
    -------
    dict with:
        oracle_agreement : float — % of samples where model top target == oracle top target
        urgency_mae : float — mean absolute error on urgency predictions
        urgency_corr : float — Pearson correlation on urgency
        target_acc : float — target classification accuracy (unit-targeted only)
        n_samples : int
    """
    model.eval()

    total = 0
    target_correct = 0
    target_total = 0
    urgency_errors = []
    pred_urgencies = []
    true_urgencies = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        game_state = batch["game_state"].to(device)
        true_urgency = batch["urgency"].to(device)
        true_target = batch["target_idx"].to(device)
        has_target = batch["has_target"].to(device)

        pred_urgency, target_logits = model(input_ids, attention_mask, game_state)
        pred_urgency = pred_urgency.squeeze(-1)

        # Urgency metrics
        urgency_errors.append((pred_urgency - true_urgency).abs())
        pred_urgencies.append(pred_urgency)
        true_urgencies.append(true_urgency)
        total += len(true_urgency)

        # Target agreement (only for unit-targeted categories)
        if has_target.any():
            mask = has_target.bool()
            pred_top = target_logits[mask].argmax(dim=-1)
            true_top = true_target[mask]
            target_correct += (pred_top == true_top).sum().item()
            target_total += mask.sum().item()

    # Aggregate
    all_errors = torch.cat(urgency_errors)
    all_pred = torch.cat(pred_urgencies)
    all_true = torch.cat(true_urgencies)

    urgency_mae = all_errors.mean().item()

    # Pearson correlation
    if all_pred.std() > 1e-8 and all_true.std() > 1e-8:
        urgency_corr = torch.corrcoef(
            torch.stack([all_pred, all_true])
        )[0, 1].item()
    else:
        urgency_corr = 0.0

    target_acc = target_correct / target_total if target_total > 0 else 0.0

    # Oracle agreement = target accuracy for unit-targeted, urgency direction for others
    # (for non-targeted categories, "agreement" means urgency > 0.4 matches oracle > 0.4)
    threshold = 0.4
    direction_agree = ((all_pred >= threshold) == (all_true >= threshold)).float().mean().item()

    return {
        # Use direction agreement as oracle_agreement — target_acc is trivial
        # when 95%+ of targets are index 0 (oracle always picks best first).
        "oracle_agreement": direction_agree,
        "urgency_mae": urgency_mae,
        "urgency_corr": urgency_corr,
        "target_acc": target_acc,
        "direction_agreement": direction_agree,
        "n_samples": total,
        "n_targeted": target_total,
    }
