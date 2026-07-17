"""可视化工具函数"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


def draw_bboxes(image: np.ndarray, bboxes: List[Dict], 
                class_names: Dict[int, str] = None, 
                colors: Dict[int, Tuple[int, int, int]] = None) -> np.ndarray:
    result = image.copy()
    
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox['bbox'])
        cls = bbox.get('class', 0)
        conf = bbox.get('confidence', None)
        
        color = colors.get(cls, (0, 255, 0)) if colors else (0, 255, 0)
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        label = class_names.get(cls, str(cls)) if class_names else str(cls)
        if conf is not None:
            label = f'{label} {conf:.2f}'
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result


def plot_training_curves(results_dict: Dict, save_path: str = None) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_map = {
        0: ('train/box_loss', 'Box Loss'),
        1: ('train/cls_loss', 'Class Loss'),
        2: ('train/dfl_loss', 'DFL Loss'),
        3: ('metrics/precision(B)', 'Precision'),
        4: ('metrics/recall(B)', 'Recall'),
        5: ('metrics/mAP50(B)', 'mAP50'),
    }
    
    for idx, (key, title) in metrics_map.items():
        if key in results_dict:
            axes[idx].plot(results_dict[key], label=title)
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Epoch')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Training curves saved: {save_path}')
    
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                          save_path: str = None, normalize: bool = True) -> None:
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Confusion matrix saved: {save_path}')
    
    plt.close()


def create_comparison_image(images: List[np.ndarray], titles: List[str] = None,
                            save_path: str = None) -> np.ndarray:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    
    for idx, (img, ax) in enumerate(zip(images, axes)):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Comparison image saved: {save_path}')
    
    plt.close()
