#!/bin/bash
# ============================================================
# GitHub仓库一键初始化脚本 - 第10组工业缺陷检测项目
# 用法: bash init_repo.sh
# ============================================================

echo "=========================================="
echo "第10组 GitHub仓库初始化"
echo "=========================================="
echo ""

# 创建目录结构
echo "[1/4] 创建项目目录结构..."

mkdir -p .github/workflows
mkdir -p data/aluminum/images/{train,val,test}
mkdir -p data/aluminum/labels/{train,val,test}
mkdir -p data/fabric/images
mkdir -p data/fabric/labels
mkdir -p models/loss
mkdir -p configs
mkdir -p scripts
mkdir -p utils
mkdir -p experiments/exp{02..10}
mkdir -p docs/答辩PPT
mkdir -p docs/会议纪要
mkdir -p checkpoints

echo "  目录创建完成"

# 创建Python包标记文件
echo "[2/4] 创建Python包标记..."

touch models/__init__.py
touch models/loss/__init__.py
touch scripts/__init__.py
touch utils/__init__.py

echo "  __init__.py 创建完成"

# 创建占位文件（说明每个文件由谁负责）
echo "[3/4] 创建占位文件..."

# models目录
cat > models/cbam.py << 'EOF'
"""
CBAM注意力模块 - 02号方向
负责人: 成员B

参考: Woo et al. "CBAM: Convolutional Block Attention Module", ECCV 2018
"""
# TODO: 实现CBAM模块
EOF

cat > models/eca.py << 'EOF'
"""
ECA高效通道注意力模块 - 03号方向
负责人: 成员C

参考: Wang et al. "ECA-Net: Efficient Channel Attention", CVPR 2020
"""
# TODO: 实现ECA模块
EOF

cat > models/bifpn.py << 'EOF'
"""
BiFPN双向特征金字塔模块 - 05号方向
负责人: 成员E

参考: Tan et al. "EfficientDet", CVPR 2020
"""
# TODO: 实现BiFPN模块
EOF

cat > models/loss/focal_loss.py << 'EOF'
"""
Focal Loss损失函数 - 04号方向
负责人: 成员D

参考: Lin et al. "Focal Loss for Dense Object Detection", ICCV 2017
"""
# TODO: 实现Focal Loss
EOF

cat > models/loss/wiou.py << 'EOF'
"""
WIoU / NWD损失函数 - 04号方向
负责人: 成员D

参考: WIoUv3 / Normalized Wasserstein Distance
"""
# TODO: 实现WIoU/NWD
EOF

# configs目录
cat > configs/baseline.yaml << 'EOF'
# 基线训练配置 - 01号方向
# 负责人: 成员A

model: yolov8s.pt
data: data/aluminum/data.yaml
epochs: 100
imgsz: 640
batch: 16
optimizer: SGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
seed: 42
name: baseline_yolov8s
EOF

cat > configs/exp02_cbam.yaml << 'EOF'
# CBAM实验配置 - 02号方向
# 负责人: 成员B
# TODO: 基于baseline.yaml添加CBAM相关参数
EOF

cat > configs/exp03_eca.yaml << 'EOF'
# ECA实验配置 - 03号方向
# 负责人: 成员C
# TODO: 基于baseline.yaml添加ECA相关参数
EOF

cat > configs/exp04_focal.yaml << 'EOF'
# Focal Loss实验配置 - 04号方向
# 负责人: 成员D
# TODO: 基于baseline.yaml添加Focal Loss相关参数
EOF

cat > configs/exp05_bifpn.yaml << 'EOF'
# BiFPN实验配置 - 05号方向
# 负责人: 成员E
# TODO: 基于baseline.yaml添加BiFPN相关参数
EOF

cat > configs/exp06_augment.yaml << 'EOF'
# 数据增强实验配置 - 06号方向
# 负责人: 成员F
# TODO: 定义不同增强策略组合
EOF

cat > configs/exp07_rtdetr.yaml << 'EOF'
# RT-DETR对比实验配置 - 07号方向
# 负责人: 成员G
# TODO: RT-DETR训练配置
EOF

cat > configs/exp08_swa.yaml << 'EOF'
# SWA实验配置 - 08号方向
# 负责人: 成员H
# TODO: SWA相关参数
EOF

cat > configs/exp09_domain.yaml << 'EOF'
# 跨域迁移实验配置 - 09号方向
# 负责人: 成员I
# TODO: 迁移学习相关参数
EOF

cat > configs/exp10_gradcam.yaml << 'EOF'
# Grad-CAM可视化配置 - 10号方向
# 负责人: PM

target_layer: backbone.layer4  # TODO: 根据实际模型调整
output_dir: experiments/exp10_gradcam/heatmaps/
overlay_alpha: 0.5
EOF

# scripts目录
cat > scripts/train.py << 'EOF'
"""
统一训练入口脚本
负责人: 成员A（基线）/ 各方向负责人（改进实验）

用法:
    python scripts/train.py --config configs/baseline.yaml
"""
import argparse
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # TODO: 实现训练逻辑
    print(f"加载配置: {args.config}")
    print(f"配置内容: {config}")
    print("[INFO] 训练脚本待实现")


if __name__ == "__main__":
    main()
EOF

cat > scripts/eval.py << 'EOF'
"""
统一评估入口脚本
负责人: 各方向负责人

用法:
    python scripts/eval.py --weights runs/exp02_cbam/weights/best.pt --data data/aluminum/data.yaml
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="评估脚本")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data", type=str, required=True, help="数据集配置路径")
    args = parser.parse_args()

    # TODO: 实现评估逻辑
    print(f"评估模型: {args.weights}")
    print(f"数据集: {args.data}")
    print("[INFO] 评估脚本待实现")


if __name__ == "__main__":
    main()
EOF

cat > scripts/infer.py << 'EOF'
"""
推理/预测入口脚本
负责人: 各方向负责人

用法:
    python scripts/infer.py --weights runs/exp02_cbam/weights/best.pt --source data/aluminum/images/test/
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="推理脚本")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="输入图像路径")
    parser.add_argument("--save-txt", action="store_true", help="保存检测结果")
    parser.add_argument("--save-conf", action="store_true", help="保存置信度")
    args = parser.parse_args()

    # TODO: 实现推理逻辑
    print(f"推理模型: {args.weights}")
    print(f"输入路径: {args.source}")
    print("[INFO] 推理脚本待实现")


if __name__ == "__main__":
    main()
EOF

cat > scripts/gradcam_vis.py << 'EOF'
"""
Grad-CAM可视化脚本 - 10号方向
负责人: PM

用法:
    python scripts/gradcam_vis.py \
        --weights runs/exp02_cbam/weights/best.pt \
        --source data/aluminum/images/test/ \
        --output experiments/exp10_gradcam/heatmaps/

参考:
    - pytorch-grad-cam库: https://github.com/jacobgil/pytorch-grad-cam
    - Grad-CAM论文: Selvaraju et al., ICCV 2017
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM可视化")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", type=str, default="experiments/exp10_gradcam/heatmaps/", help="输出目录")
    parser.add_argument("--target-layer", type=str, default=None, help="目标层名称")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # TODO: 实现Grad-CAM可视化逻辑
    print(f"模型权重: {args.weights}")
    print(f"输入路径: {args.source}")
    print(f"输出目录: {args.output}")
    print("[INFO] Grad-CAM可视化脚本待实现")
    print("[提示] 安装: pip install pytorch-grad-cam")


if __name__ == "__main__":
    main()
EOF

cat > scripts/cross_domain.py << 'EOF'
"""
跨域迁移脚本 - 09号方向
负责人: 成员I

用法:
    python scripts/cross_domain.py --source-domain aluminum --target-domain fabric
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="跨域迁移")
    parser.add_argument("--source-domain", type=str, default="aluminum", help="源域")
    parser.add_argument("--target-domain", type=str, default="fabric", help="目标域")
    args = parser.parse_args()

    # TODO: 实现跨域迁移逻辑
    print(f"源域: {args.source_domain}")
    print(f"目标域: {args.target_domain}")
    print("[INFO] 跨域迁移脚本待实现")


if __name__ == "__main__":
    main()
EOF

# utils目录
cat > utils/dataset_utils.py << 'EOF'
"""数据集处理工具函数"""
# TODO: 实现数据集处理工具
EOF

cat > utils/metrics.py << 'EOF'
"""评估指标计算工具"""
# TODO: 实现评估指标计算
EOF

cat > utils/visualizer.py << 'EOF'
"""可视化工具函数"""
# TODO: 实现可视化工具
EOF

cat > utils/logger.py << 'EOF'
"""日志工具"""
# TODO: 实现日志工具
EOF

# experiments目录 - 创建.gitignore和README
cat > experiments/.gitignore << 'EOF'
# 实验结果中的大文件不上传
**/weights/
**/*.pt
**/*.pth
**/*.ckpt
EOF

for i in $(seq 2 10); do
    cat > experiments/exp$(printf "%02d" $i)/README.md << EOF
# 实验$(printf "%02d" $i)结果目录

存放本方向实验结果：
- results.csv: 实验结果表格
- plots/: 可视化图表
- heatmaps/: Grad-CAM热力图（仅限10号方向）

注意：模型权重文件(.pt)不上传GitHub
EOF
done

# data目录
cat > data/.gitignore << 'EOF'
# 数据集不上传GitHub
aluminum/images/
aluminum/labels/
fabric/images/
fabric/labels/
*.jpg
*.png
*.jpeg
*.txt
*.json
*.xml
EOF

cat > data/aluminum/data.yaml << 'EOF'
# 铝型材数据集配置
path: data/aluminum
train: images/train
val: images/val
test: images/test

nc: 10
names:
  0: budaodian      # 不导电
  1: cahua          # 擦花
  2: jiaoweiloudi   # 角位漏底
  3: jupi           # 桔皮
  4: loudi          # 漏底
  5: penliu         # 喷流
  6: qipao          # 漆泡
  7: qikeng         # 起坑
  8: zase           # 杂色
  9: zangdian       # 脏点
EOF

# checkpoints目录
cat > checkpoints/.gitignore << 'EOF'
# 模型权重不上传GitHub
*.pt
*.pth
*.ckpt
*.onnx
*.engine
EOF

echo "  占位文件创建完成"

# 创建.gitignore（如果不存在）
echo "[4/4] 检查.gitignore..."
if [ ! -f .gitignore ]; then
    echo "  .gitignore 不存在，请手动添加"
fi

echo ""
echo "=========================================="
echo "初始化完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "  1. 将本目录下的文件复制到你的GitHub仓库"
echo "  2. git add ."
echo "  3. git commit -m "chore: 初始化项目结构""
echo "  4. git push"
echo ""
echo "目录结构预览:"
tree -L 2 -I '__pycache__|*.pyc'
