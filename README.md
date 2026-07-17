# 工业缺陷检测系统 - 第10组

> AI算法工程师培训班CV项目 | 答辩日期：2026年8月15日

## 项目简介

本项目基于YOLOv8目标检测框架，构建面向铝型材、布匹、瓷砖等工业产品表面的智能缺陷检测系统。通过多维度改进策略提升检测精度，并提供Grad-CAM可解释性分析。

## 项目目标

- **核心目标**：铝型材10类缺陷检测，mAP@0.5 >= 85%
- **扩展目标**：跨域迁移至布匹/瓷砖，mAP >= 75%
- **工程目标**：单帧推理 <= 50ms，支持CPU fallback
- **答辩目标**：完整技术报告 + 可运行Demo + 答辩PPT

## 团队成员与分工

| 方向 | 负责人 | 负责内容 | 状态 |
|------|--------|---------|------|
| 01-A/B | 成员A | YOLOv8基线训练（n/s/m/l/x对比） | 待开始 |
| 02 | 成员B | CBAM注意力机制改进 | 待开始 |
| 03 | 成员C | ECA轻量注意力改进 | 待开始 |
| 04 | 成员D | Focal Loss + WIoU损失函数改进 | 待开始 |
| 05 | 成员E | BiFPN Neck结构改进 | 待开始 |
| 06 | 成员F | 数据增强策略对比 | 待开始 |
| 07 | 成员G | RT-DETR Transformer架构对比 | 待开始 |
| 08 | 成员H | SWA随机权重平均 | 待开始 |
| 09 | 成员I | 跨域迁移（铝型材->布匹/瓷砖） | 待开始 |
| 10 | PM | Grad-CAM可解释性分析 | 待开始 |

## 环境配置

### 硬件要求
- GPU：>= 8GB 显存（推荐RTX 3060/4060及以上）
- 无GPU可用CPU运行（仅限推理和Grad-CAM可视化）
- 备选：魔搭平台（36小时免费GPU）

### 软件环境
- Python 3.10
- PyTorch 2.7.0
- CUDA 12.6
- torchvision 0.24.0

### 快速安装

```bash
# 1. 创建conda环境
conda create -n defect_det python=3.10 -y
conda activate defect_det

# 2. 安装PyTorch（清华镜像源）
pip install torch==2.7.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126     -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 安装YOLOv8及其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 验证环境
python scripts/verify_env.py
```

## 项目结构

```
defect-detection-10th/
├── data/               # 数据集（本地存放，不上传GitHub）
├── models/             # 模型定义（改进模块）
│   ├── cbam.py
│   ├── eca.py
│   ├── bifpn.py
│   └── loss/
├── configs/            # 实验配置文件
├── scripts/            # 可运行脚本
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── gradcam_vis.py      # Grad-CAM可视化
│   └── cross_domain.py     # 跨域迁移
├── utils/              # 工具函数
├── experiments/        # 实验结果
├── docs/               # 项目文档
├── checkpoints/        # 模型权重（本地存放）
├── README.md
├── requirements.txt
└── .gitignore
```

## 快速开始

### 1. 基线训练

```bash
python scripts/train.py --config configs/baseline.yaml
```

### 2. 改进实验（以CBAM为例）

```bash
# 训练CBAM改进模型
python scripts/train.py --config configs/exp02_cbam.yaml

# 评估
python scripts/eval.py --weights runs/exp02_cbam/weights/best.pt --data data/aluminum/data.yaml
```

### 3. Grad-CAM可视化

```bash
python scripts/gradcam_vis.py     --weights runs/exp02_cbam/weights/best.pt     --source data/aluminum/images/test/     --output experiments/exp10_gradcam/heatmaps/
```

### 4. 推理预测

```bash
python scripts/infer.py     --weights runs/exp02_cbam/weights/best.pt     --source data/aluminum/images/test/     --save-txt --save-conf
```

## 实验结果

| 实验 | 模型 | mAP@0.5 | mAP@0.5:0.95 | FPS | 参数量 |
|------|------|---------|--------------|-----|--------|
| 基线 | YOLOv8s | 83.0% | 55.2% | 85 | 11.2M |
| Exp02 | +CBAM | - | - | - | - |
| Exp03 | +ECA | - | - | - | - |
| Exp04 | +Focal Loss | - | - | - | - |
| Exp05 | +BiFPN | - | - | - | - |
| Exp06 | +增强策略 | - | - | - | - |
| Exp07 | RT-DETR | - | - | - | - |
| Exp08 | +SWA | - | - | - | - |
| Exp09 | 跨域迁移 | - | - | - | - |

> 注：实验完成后更新此表格

## 里程碑

| 日期 | 里程碑 | 状态 |
|------|--------|------|
| 7.19 | 选型会，角色认领 | 待完成 |
| 7.21 | 环境配置完成 | 待完成 |
| 7.26 | 第一次联调 | 待完成 |
| 8.2 | 第二次联调 | 待完成 |
| 8.9 | 整合截止 | 待完成 |
| 8.15 | 答辩 | 待完成 |

## 文档

- 需求文档
- 技术方案
- GitHub仓库协作指南

## 协作规范

1. **分支策略**：每人一个功能分支 exp<编号>-<方向名>，禁止直接推送main
2. **代码审核**：通过Pull Request合并，至少1人审核
3. **提交规范**：使用 feat/fix/docs/style 等类型前缀
4. **大文件**：数据集和模型权重不上传GitHub，使用.gitignore过滤

## License

本项目仅供学习交流使用。

---

> 最后更新: 2026-07-17 | 版本: v1.0
