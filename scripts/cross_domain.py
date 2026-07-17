"""跨域迁移脚本 - 第09组
Usage:
    python scripts/cross_domain.py --config configs/exp09_domain.yaml
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
    parser = argparse.ArgumentParser(description='Cross-domain adaptation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    da_cfg = cfg.get('domain_adaptation', {})

    if da_cfg.get('enabled', False):
        method = da_cfg.get('method', 'fine_tune')

        if method == 'fine_tune':
            print('Step 1: Training on source domain...')
            model = YOLO(cfg['model'])
            model.train(
                data=cfg.get('data_source', cfg['data']),
                epochs=da_cfg.get('source_epochs', 50),
                imgsz=cfg.get('imgsz', 640),
                batch=cfg.get('batch', 16),
                project=cfg.get('project', 'experiments/exp09_domain'),
                name='source_training',
                exist_ok=True,
            )

            print('Step 2: Fine-tuning on target domain...')
            source_weights = Path(cfg.get('project', 'experiments/exp09_domain')) / 'source_training' / 'weights' / 'best.pt'
            if source_weights.exists():
                model = YOLO(str(source_weights))

            if da_cfg.get('freeze_backbone', False):
                for param in model.model.parameters():
                    param.requires_grad = False

            model.train(
                data=cfg.get('data_target', cfg['data']),
                epochs=da_cfg.get('target_epochs', 50),
                imgsz=cfg.get('imgsz', 640),
                batch=cfg.get('batch', 16),
                project=cfg.get('project', 'experiments/exp09_domain'),
                name='target_finetune',
                exist_ok=True,
            )
    else:
        print('Domain adaptation not enabled, running standard training...')
        model = YOLO(cfg['model'])
        model.train(
            data=cfg['data'],
            epochs=cfg.get('epochs', 100),
            imgsz=cfg.get('imgsz', 640),
            batch=cfg.get('batch', 16),
            project=cfg.get('project', 'experiments/exp09_domain'),
            name='standard_training',
            exist_ok=True,
        )

    print('Cross-domain training complete!')


if __name__ == '__main__':
    main()
