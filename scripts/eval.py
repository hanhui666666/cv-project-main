"""统一评估入口
Usage:
    python scripts/eval.py --config configs/baseline.yaml --weights checkpoints/best.pt
"""
import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    args = parser.parse_args()

    cfg = load_config(args.config)

    model = YOLO(args.weights)

    metrics = model.val(
        data=cfg['data'],
        imgsz=cfg.get('imgsz', 640),
        batch=cfg.get('batch', 16),
        split='val',
        verbose=True,
    )

    print('Evaluation complete!')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')
    return metrics


if __name__ == '__main__':
    main()
