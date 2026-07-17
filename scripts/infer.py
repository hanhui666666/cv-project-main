"""推理/预测入口
Usage:
    python scripts/infer.py --config configs/baseline.yaml --weights checkpoints/best.pt --source data/test.jpg
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
    parser = argparse.ArgumentParser(description='Run inference with YOLOv8')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Source for inference (image/video/folder)')
    parser.add_argument('--save', action='store_true', default=True, help='Save inference results')
    args = parser.parse_args()

    cfg = load_config(args.config)

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        imgsz=cfg.get('imgsz', 640),
        save=args.save,
        conf=0.25,
        iou=0.45,
        verbose=True,
    )

    print('Inference complete!')
    return results


if __name__ == '__main__':
    main()
