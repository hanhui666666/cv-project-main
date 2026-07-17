"""WIoU Loss
Wise Intersection over Union
参考: https://arxiv.org/abs/2301.10051
"""
import torch
import torch.nn as nn


def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    if xywh:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps

    iou = inter_area / union_area

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    return iou, inter_area, union_area, cw, ch, b1_area, b2_area


class WIoULoss(nn.Module):
    def __init__(self):
        super(WIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes):
        iou, inter_area, union_area, cw, ch, b1_area, b2_area = bbox_iou(pred_boxes, target_boxes)

        dist2 = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2
        c2 = cw ** 2 + ch ** 2 + 1e-7

        with torch.no_grad():
            beta = dist2 / c2
            alpha = beta / (beta + 1 - iou + 1e-7)

        wiou = iou - dist2 / c2
        wiou_loss = alpha * wiou

        return 1 - wiou_loss.mean()
