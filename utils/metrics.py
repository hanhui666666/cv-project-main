"""评估指标计算工具"""
import numpy as np
from typing import List, Dict, Tuple


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_map(predictions: List, ground_truths: List, 
                num_classes: int, iou_threshold: float = 0.5) -> Dict[str, float]:
    aps = {}
    
    for cls in range(num_classes):
        gt_boxes = [gt for gt in ground_truths if gt['class'] == cls]
        pred_boxes = [pred for pred in predictions if pred['class'] == cls]
        pred_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched_gt = set()
        
        for i, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / max(len(gt_boxes), 1)
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-10)
        
        aps[f'class_{cls}'] = compute_ap(recall, precision)
    
    aps['mAP50'] = np.mean(list(aps.values())) if aps else 0.0
    return aps


def compute_classification_metrics(confusion_matrix: np.ndarray) -> Dict[str, float]:
    num_classes = confusion_matrix.shape[0]
    metrics = {}
    
    total = confusion_matrix.sum()
    correct = np.trace(confusion_matrix)
    metrics['accuracy'] = correct / total if total > 0 else 0.0
    
    precisions = []
    recalls = []
    f1s = []
    
    for cls in range(num_classes):
        tp = confusion_matrix[cls, cls]
        fp = confusion_matrix[:, cls].sum() - tp
        fn = confusion_matrix[cls, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    metrics['precision'] = np.mean(precisions)
    metrics['recall'] = np.mean(recalls)
    metrics['f1'] = np.mean(f1s)
    
    return metrics
