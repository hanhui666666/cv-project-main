"""数据集处理工具函数"""
import os
import yaml
import shutil
from pathlib import Path
from typing import List, Tuple, Dict


def create_data_yaml(data_dir: str, train_dir: str, val_dir: str, 
                     names: Dict[int, str], output_path: str = None) -> str:
    data_config = {
        'path': data_dir,
        'train': train_dir,
        'val': val_dir,
        'names': names,
    }
    
    if output_path is None:
        output_path = os.path.join(data_dir, 'data.yaml')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True)
    
    print(f'Data YAML created: {output_path}')
    return output_path


def check_dataset_structure(data_dir: str) -> bool:
    required = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for sub in required:
        path = os.path.join(data_dir, sub)
        if not os.path.exists(path):
            print(f'Missing: {path}')
            return False
    print('Dataset structure OK')
    return True


def count_samples(data_dir: str) -> Dict[str, int]:
    counts = {}
    for split in ['train', 'val']:
        img_dir = os.path.join(data_dir, 'images', split)
        if os.path.exists(img_dir):
            counts[split] = len([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    print(f'Sample counts: {counts}')
    return counts


def split_dataset(src_dir: str, dst_dir: str, train_ratio: float = 0.8, 
                  seed: int = 42) -> None:
    import random
    random.seed(seed)
    
    img_dir = os.path.join(src_dir, 'images')
    label_dir = os.path.join(src_dir, 'labels')
    
    if not os.path.exists(img_dir):
        print(f'No images directory found in {src_dir}')
        return
    
    all_images = [f for f in os.listdir(img_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    for split, images in [('train', train_images), ('val', val_images)]:
        dst_img = os.path.join(dst_dir, 'images', split)
        dst_lbl = os.path.join(dst_dir, 'labels', split)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        
        for img in images:
            src_img_path = os.path.join(img_dir, img)
            dst_img_path = os.path.join(dst_img, img)
            shutil.copy2(src_img_path, dst_img_path)
            
            label_name = os.path.splitext(img)[0] + '.txt'
            src_lbl_path = os.path.join(label_dir, label_name)
            if os.path.exists(src_lbl_path):
                dst_lbl_path = os.path.join(dst_lbl, label_name)
                shutil.copy2(src_lbl_path, dst_lbl_path)
    
    print(f'Dataset split: train={len(train_images)}, val={len(val_images)}')
