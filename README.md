# 工业缺陷检测项目 - 第10组

> 基于YOLOv8的多模态智能巡检助手

## 项目简介

本项目是AI算法工程师培训班CV项目第10组的工业缺陷检测项目，基于YOLOv8目标检测框架，针对铝型材表面缺陷进行检测和分类，并探索多种改进方向以提升检测性能。

## 项目结构

```
defect-detection-10th/
├── data/              # 数据集（本地存放，不上传GitHub）
├── models/            # 模型定义（核心代码）
├── configs/          # 配置文件
├── scripts/          # 可运行脚本
├── utils/            # 工具函数
├── experiments/      # 实验结果
├── docs/             # 文档
├── checkpoints/      # 模型权重（本地存放）
└── README.md
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证环境

```bash
python verify_env.py
```

## 快速开始

### 训练模型

```bash
python scripts/train.py --config configs/baseline.yaml
```

### 评估模型

```bash
python scripts/eval.py --config configs/baseline.yaml --weights checkpoints/best.pt
```

### 推理预测

```bash
python scripts/infer.py --config configs/baseline.yaml --weights checkpoints/best.pt --source data/test.jpg
```

## 改进方向

| 编号 | 方向 | 负责人 | 配置文件 |
|-----|------|--------|---------|
| 01 | 基线模型 | 成员A | baseline.yaml |
| 02 | CBAM注意力模块 | 成员B | exp02_cbam.yaml |
| 03 | ECA注意力模块 | 成员C | exp03_eca.yaml |
| 04 | Focal Loss + WIoU | 成员D | exp04_focal.yaml |
| 05 | BiFPN改进 | 成员E | exp05_bifpn.yaml |
| 06 | 数据增强 | 成员F | exp06_augment.yaml |
| 07 | RT-DETR对比 | 成员G | exp07_rtdetr.yaml |
| 08 | SWA权重平均 | 成员H | exp08_swa.yaml |
| 09 | 跨域迁移 | 成员I | exp09_domain.yaml |
| 10 | Grad-CAM可视化 | PM | exp10_gradcam.yaml |

## 数据集

请将数据集放置在 `data/` 目录下：

```
data/
├── aluminum/        # 铝型材数据集
│   ├── images/
│   └── labels/
└── fabric/       # 布匹数据集（迁移用）
```

## 许可证

本项目仅用于学习和研究目的。

## 团队

第10组全体成员
