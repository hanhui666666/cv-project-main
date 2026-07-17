# Grad-CAM可解释性分析对比报告

> 第10组 | 实验编号：Exp10  
> 生成日期：2026-07-17

---

## 一、实验目的

使用Grad-CAM（Gradient-weighted Class Activation Mapping）技术对训练好的缺陷检测模型进行可解释性分析，通过可视化热力图展示模型在做出预测时关注的图像区域，验证模型是否正确学习到缺陷特征。

---

## 二、实验配置

| 项目 | 配置 |
|-----|------|
| 模型权重 | `checkpoints/best.pt` |
| 目标层 | `model.model[-2].cv3.0.act` |
| Grad-CAM方法 | GradCAM++ |
| 热力图Colormap | jet |
| 透明度（alpha） | 0.5 |
| 图像尺寸 | 640×640 |

---

## 三、可视化结果

### 3.1 各类缺陷热力图展示

| 缺陷类型 | 原图 | 热力图 | 叠加图 | 分析说明 |
|---------|------|-------|--------|---------|
| 划痕 | ![](heatmaps/scratch_original.jpg) | ![](heatmaps/scratch_heatmap.jpg) | ![](heatmaps/scratch_overlay.jpg) | 模型正确关注到划痕区域 |
| 凹坑 | ![](heatmaps/pit_original.jpg) | ![](heatmaps/pit_heatmap.jpg) | ![](heatmaps/pit_overlay.jpg) | 模型关注区域与凹坑位置吻合 |
| 氧化 | ![](heatmaps/oxide_original.jpg) | ![](heatmaps/oxide_heatmap.jpg) | ![](heatmaps/oxide_overlay.jpg) | 氧化区域被有效识别 |
| 脏污 | ![](heatmaps/dirt_original.jpg) | ![](heatmaps/dirt_heatmap.jpg) | ![](heatmaps/dirt_overlay.jpg) | 模型正确定位脏污区域 |

### 3.2 典型案例分析

#### 案例1：正确检测
- **图像**：`case1_correct.jpg`
- **预测类别**：划痕（置信度 0.95）
- **热力图分析**：热力图高激活区域与划痕位置完全重合，说明模型正确学习到划痕的纹理特征。

#### 案例2：误检分析
- **图像**：`case2_fp.jpg`
- **预测类别**：凹坑（置信度 0.72）
- **实际类别**：正常
- **热力图分析**：热力图高激活区域位于图像边缘的反光区域，模型将金属反光误判为凹坑。建议增加反光样本或使用数据增强缓解。

#### 案例3：漏检分析
- **图像**：`case3_fn.jpg`
- **预测类别**：正常
- **实际类别**：氧化（轻微）
- **热力图分析**：轻微氧化区域激活度较低，模型未能有效响应。建议增加轻微缺陷样本或调整损失函数权重。

---

## 四、与基线模型对比

| 对比维度 | 基线模型（YOLOv8s） | 改进模型 | 说明 |
|---------|-------------------|---------|------|
| 热力图清晰度 | 一般 | 更清晰 | 改进后关注区域更集中 |
| 缺陷定位准确度 | 85% | 92% | 热力图中心与GT的IoU提升 |
| 背景干扰 | 较多背景激活 | 背景激活减少 | 改进后抗干扰能力更强 |
| 小缺陷响应 | 响应较弱 | 响应增强 | 小缺陷区域激活度提升 |

---

## 五、结论与建议

### 5.1 主要发现
1. 模型大部分情况下能够正确关注到缺陷区域，说明特征学习有效
2. 对于反光、阴影等干扰因素，模型仍存在误判风险
3. 轻微缺陷的检测灵敏度有待提升
4. 改进模型在可解释性层面优于基线模型

### 5.2 改进建议
| 问题 | 建议方案 | 优先级 |
|-----|---------|-------|
| 反光导致误检 | 增加反光样本，使用光泽度归一化 | P1 |
| 轻微缺陷漏检 | 调整Focal Loss的alpha/gamma参数 | P1 |
| 背景干扰 | 引入注意力模块（CBAM/ECA） | P2 |
| 小目标检测 | 使用BiFPN增强多尺度特征融合 | P2 |

### 5.3 后续工作
- [ ] 针对误检案例进行针对性数据增强
- [ ] 探索更多可解释性方法（Score-CAM、Layer-CAM）
- [ ] 对模型各层进行逐层热力图分析
- [ ] 构建自动化的可解释性评估流程

---

## 附录

### A. 热力图文件清单
```
experiments/exp10_gradcam/heatmaps/
├── scratch_overlay.jpg       # 划痕叠加图
├── scratch_heatmap.jpg       # 划痕热力图
├── scratch_comparison.jpg    # 划痕对比图
├── pit_overlay.jpg          # 凹坑叠加图
├── ...
└── case3_fn_comparison.jpg   # 漏检案例对比图
```

### B. 运行命令
```bash
# 生成单张图片热力图
python scripts/gradcam_vis.py \
    --config configs/exp10_gradcam.yaml \
    --weights checkpoints/best.pt \
    --source data/test/scratch.jpg

# 批量生成文件夹热力图
python scripts/gradcam_vis.py \
    --config configs/exp10_gradcam.yaml \
    --weights checkpoints/best.pt \
    --source data/test/
```
