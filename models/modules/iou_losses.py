"""
SIoU and EIoU Loss for Ultralytics YOLOv11
Monkey-patch approach: replace bbox_iou in BboxLoss.forward
"""

import math
import torch


def bbox_siou(box1, box2, xywh=True, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    sigma = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) / 2).pow(2) + \
            ((b2_y1 + b2_y2 - b1_y1 - b1_y2) / 2).pow(2)
    sin_alpha = torch.clamp(sigma / (c2 + eps), min=0, max=1)
    angle_cost = torch.cos(torch.arcsin(torch.sqrt(sin_alpha)) * 2 - math.pi / 2)
    gamma = 2 - angle_cost

    rho_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) / 2).pow(2) / (cw.pow(2) + eps)
    rho_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) / 2).pow(2) / (ch.pow(2) + eps)
    distance_cost = (1 - torch.exp(-gamma * rho_x)) + (1 - torch.exp(-gamma * rho_y))

    omiga_w = torch.abs(w1 - w2) / (w1.maximum(w2) + eps)
    omiga_h = torch.abs(h1 - h2) / (h1.maximum(h2) + eps)
    shape_cost = torch.pow(1 - torch.exp(-omiga_w), 4) + torch.pow(1 - torch.exp(-omiga_h), 4)

    return iou - 0.5 * (distance_cost + shape_cost)


def bbox_eiou(box1, box2, xywh=True, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) / 2).pow(2) + \
           ((b2_y1 + b2_y2 - b1_y1 - b1_y2) / 2).pow(2)

    rho_w = ((b2_x2 - b2_x1 - b1_x2 + b1_x1) / 2).pow(2)
    cw2 = cw.pow(2) + eps
    rho_h = ((b2_y2 - b2_y1 - b1_y2 + b1_y1) / 2).pow(2)
    ch2 = ch.pow(2) + eps

    return iou - (rho2 / c2 + rho_w / cw2 + rho_h / ch2)


def bbox_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2

    inner_b1_x1 = b1_cx - (b1_x2 - b1_x1) * ratio / 2
    inner_b1_y1 = b1_cy - (b1_y2 - b1_y1) * ratio / 2
    inner_b1_x2 = b1_cx + (b1_x2 - b1_x1) * ratio / 2
    inner_b1_y2 = b1_cy + (b1_y2 - b1_y1) * ratio / 2

    inner_b2_x1 = b2_cx - (b2_x2 - b2_x1) * ratio / 2
    inner_b2_y1 = b2_cy - (b2_y2 - b2_y1) * ratio / 2
    inner_b2_x2 = b2_cx + (b2_x2 - b2_x1) * ratio / 2
    inner_b2_y2 = b2_cy + (b2_y2 - b2_y1) * ratio / 2

    inner_inter = (inner_b1_x2.minimum(inner_b2_x2) - inner_b1_x1.maximum(inner_b2_x1)).clamp_(0) * \
                  (inner_b1_y2.minimum(inner_b2_y2) - inner_b1_y1.maximum(inner_b2_y1)).clamp_(0)
    inner_w1 = (inner_b1_x2 - inner_b1_x1).clamp(min=0)
    inner_h1 = (inner_b1_y2 - inner_b1_y1).clamp(min=0)
    inner_w2 = (inner_b2_x2 - inner_b2_x1).clamp(min=0)
    inner_h2 = (inner_b2_y2 - inner_b2_y1).clamp(min=0)
    inner_union = inner_w1 * inner_h1 + inner_w2 * inner_h2 - inner_inter + eps
    inner_iou = inner_inter / inner_union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) / 2).pow(2) + \
           ((b2_y1 + b2_y2 - b1_y1 - b1_y2) / 2).pow(2)

    d = rho2 / c2

    with torch.no_grad():
        focal_weight = torch.where(iou < 0.5, inner_iou.detach(), iou.detach())

    return iou - d * focal_weight


def apply_iou_loss(loss_type='siou'):
    valid = {'siou', 'eiou', 'ciou', 'giou', 'diou', 'inner_iou'}
    if loss_type not in valid:
        print(f"Warning: Unknown loss type '{loss_type}', using CIoU")
        return False

    if loss_type == 'ciou':
        print("Using default CIoU (no patch needed)")
        return True

    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.tal import bbox2dist

    if getattr(BboxLoss, '_iou_patched', None) == loss_type:
        return True

    iou_fn_map = {
        'siou': bbox_siou,
        'eiou': bbox_eiou,
        'inner_iou': bbox_inner_iou,
    }
    iou_fn = iou_fn_map[loss_type]

    original_forward = BboxLoss.forward

    def patched_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz=None, stride=None):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        if pred_fg.shape[0] > 0:
            iou = iou_fn(pred_fg, target_fg, xywh=False)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        else:
            loss_iou = torch.tensor(0.0, device=pred_bboxes.device)

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl

    BboxLoss.forward = patched_forward
    BboxLoss._iou_patched = loss_type
    print(f"IoU loss patched to {loss_type.upper()}")
    return True
