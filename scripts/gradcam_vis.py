"""Grad-CAM可视化脚本 - 第10组
Usage:
    python scripts/gradcam_vis.py --config configs/exp10_gradcam.yaml --weights checkpoints/best.pt --source data/test.jpg
"""
import argparse
import yaml
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, ScoreCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print('Warning: pytorch-grad-cam not installed. Run: pip install pytorch-grad-cam')


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_target_layer(model):
    cfg = load_config(args.config) if 'args' in dir() else {}
    gradcam_cfg = cfg.get('gradcam', {})
    target_layer_str = gradcam_cfg.get('target_layer', 'model.model[-2]')
    
    try:
        target_layer = eval(target_layer_str, {'model': model})
        return [target_layer]
    except Exception as e:
        print(f'Warning: Could not get target layer {target_layer_str}: {e}')
        return [model.model[-2]]


def run_gradcam(model, image_path, output_dir, cfg):
    if not GRADCAM_AVAILABLE:
        print('pytorch-grad-cam is required for Grad-CAM visualization')
        return

    gradcam_cfg = cfg.get('gradcam', {})
    method = gradcam_cfg.get('method', 'GradCAMPlusPlus')
    colormap = gradcam_cfg.get('colormap', 'jet')
    alpha = gradcam_cfg.get('alpha', 0.5)

    target_layers = get_target_layer(model)

    cam_methods = {
        'GradCAM': GradCAM,
        'GradCAMPlusPlus': GradCAMPlusPlus,
        'HiResCAM': HiResCAM,
    }
    cam_class = cam_methods.get(method, GradCAMPlusPlus)

    rgb_image = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_image = cv2.resize(rgb_image, (640, 640))
    rgb_image_float = np.float32(rgb_image) / 255

    input_tensor = preprocess_image(rgb_image_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with cam_class(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_image_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=alpha)

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem

    if gradcam_cfg.get('save_overlay', True):
        overlay_path = os.path.join(output_dir, f'{base_name}_overlay.jpg')
        cv2.imwrite(overlay_path, visualization[:, :, ::-1])
        print(f'Saved overlay: {overlay_path}')

    if gradcam_cfg.get('save_heatmap', True):
        heatmap_path = os.path.join(output_dir, f'{base_name}_heatmap.jpg')
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap)
        print(f'Saved heatmap: {heatmap_path}')

    if gradcam_cfg.get('save_comparison', True):
        comp_path = os.path.join(output_dir, f'{base_name}_comparison.jpg')
        original = cv2.resize(cv2.imread(image_path), (640, 640))
        comparison = np.hstack([original, visualization[:, :, ::-1]])
        cv2.imwrite(comp_path, comparison)
        print(f'Saved comparison: {comp_path}')


def main():
    global args
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Source image/folder')
    args = parser.parse_args()

    cfg = load_config(args.config)

    model = YOLO(args.weights)

    gradcam_cfg = cfg.get('gradcam', {})
    output_dir = gradcam_cfg.get('output_dir', 'experiments/exp10_gradcam/heatmaps')

    source = args.source
    if os.path.isdir(source):
        image_files = [os.path.join(source, f) for f in os.listdir(source)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for img_file in image_files:
            print(f'Processing: {img_file}')
            run_gradcam(model, img_file, output_dir, cfg)
    else:
        run_gradcam(model, source, output_dir, cfg)

    print('Grad-CAM visualization complete!')


if __name__ == '__main__':
    main()
