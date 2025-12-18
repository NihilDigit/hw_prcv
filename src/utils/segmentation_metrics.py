from __future__ import annotations

import torch


@torch.no_grad()
def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Compute confusion matrix for semantic segmentation.

    Args:
        preds: (N, H, W) int64 predicted class ids
        targets: (N, H, W) int64 ground-truth class ids
        num_classes: number of valid classes (0..num_classes-1)
        ignore_index: target value to ignore
    """
    if preds.shape != targets.shape:
        raise ValueError(f"preds/targets shape mismatch: {preds.shape} vs {targets.shape}")

    preds = preds.reshape(-1).to(torch.int64)
    targets = targets.reshape(-1).to(torch.int64)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=preds.device)

    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


@torch.no_grad()
def intersection_over_union(conf_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        conf_mat: (C, C) confusion matrix

    Returns:
        iou: (C,) per-class IoU
        miou: scalar mean IoU over classes that appear in GT or prediction
    """
    conf_mat = conf_mat.to(torch.float32)
    true_pos = torch.diag(conf_mat)
    false_pos = conf_mat.sum(dim=0) - true_pos
    false_neg = conf_mat.sum(dim=1) - true_pos
    denom = true_pos + false_pos + false_neg
    iou = torch.where(denom > 0, true_pos / denom, torch.zeros_like(denom))
    valid = denom > 0
    miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=conf_mat.device)
    return iou, miou

