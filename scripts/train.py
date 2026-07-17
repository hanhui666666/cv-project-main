"""统一训练入口
Usage:
    python scripts/train.py --config configs/baseline.yaml
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
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)

    model = YOLO(cfg['model'])

    results = model.train(
        data=cfg['data'],
        epochs=cfg.get('epochs', 100),
        imgsz=cfg.get('imgsz', 640),
        batch=cfg.get('batch', 16),
        patience=cfg.get('patience', 20),
        optimizer=cfg.get('optimizer', 'SGD'),
        lr0=cfg.get('lr0', 0.01),
        lrf=cfg.get('lrf', 0.01),
        momentum=cfg.get('momentum', 0.937),
        weight_decay=cfg.get('weight_decay', 0.0005),
        warmup_epochs=cfg.get('warmup_epochs', 3.0),
        warmup_momentum=cfg.get('warmup_momentum', 0.8),
        warmup_bias_lr=cfg.get('warmup_bias_lr', 0.1),
        box=cfg.get('box', 7.5),
        cls=cfg.get('cls', 0.5),
        dfl=cfg.get('dfl', 1.5),
        augment=cfg.get('augment', False),
        mosaic=cfg.get('mosaic', 0.0),
        mixup=cfg.get('mixup', 0.0),
        close_mosaic=cfg.get('close_mosaic', 10),
        project=cfg.get('project', 'runs/train'),
        name=cfg.get('name', 'exp'),
        exist_ok=cfg.get('exist_ok', True),
        verbose=True,
    )

    print('Training complete!')
    return results


if __name__ == '__main__':
    main()
