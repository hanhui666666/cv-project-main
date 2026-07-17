# GitHub仓库协作指南

## 第10组 | AI算法工程师培训班CV项目

> **适用对象**: 第10组全体成员（含从未使用过GitHub的新手）  
> **文档版本**: v1.0  
> **编制日期**: 2026-07-17  

---

## 一、为什么用GitHub？（给新手的3句话）

1. **代码保险箱**：你的代码存在云端，电脑坏了也不会丢
2. **时光机**：改错了可以一键回退到之前的版本
3. **协作神器**：10个人同时写代码不会互相覆盖

---

## 二、准备工作（每人必做，5分钟搞定）

### Step 1：注册GitHub账号
- 打开 https://github.com
- 用邮箱注册（建议用常用邮箱，方便接收通知）
- 用户名建议：`你的英文名/拼音` + `数字`，如 `zhangsan2024`

### Step 2：安装Git
**Windows用户**：
```bash
# 下载安装包：https://git-scm.com/download/win
# 安装时全部点"下一步"即可，不用改任何选项
```

**Mac用户**：
```bash
# 打开终端，输入：
git --version
# 如果没安装，系统会提示你安装，点击安装即可
```

**验证安装成功**：
```bash
git --version
# 应该显示类似：git version 2.45.0
```

### Step 3：配置Git身份信息（只需做一次）
```bash
# 打开终端/命令行，输入：
git config --global user.name "你的GitHub用户名"
git config --global user.email "你注册GitHub用的邮箱"
```

### Step 4：生成SSH密钥（免密码上传代码）
```bash
# 在终端输入：
ssh-keygen -t ed25519 -C "你的邮箱"
# 连续按3次回车（使用默认路径，不设密码）

# 复制密钥内容：
cat ~/.ssh/id_ed25519.pub
# Windows用户用：type %USERPROFILE%\.ssh\id_ed25519.pub
```

**把密钥添加到GitHub**：
1. 登录GitHub → 右上角头像 → Settings
2. 左侧边栏 → SSH and GPG keys → New SSH key
3. Title填：`我的电脑`
4. Key粘贴刚才复制的密钥内容
5. 点击 Add SSH key

---

## 三、项目目录结构（统一规范，必须遵守）

```
defect-detection-10th/          ← 仓库根目录（仓库名）
│
├── 📁 .github/                  ← GitHub配置（不用管）
│   └── workflows/               ← 自动运行脚本（高级功能）
│
├── 📁 data/                     ← 数据集（不上传到GitHub！）
│   ├── .gitignore               ← 告诉GitHub忽略这个文件夹
│   ├── aluminum/                ← 铝型材数据集（每人本地放）
│   │   ├── images/
│   │   └── labels/
│   └── fabric/                  ← 布匹数据集（迁移用）
│
├── 📁 models/                   ← 模型定义（核心代码）
│   ├── __init__.py              ← Python包标记
│   ├── cbam.py                  ← 02号：CBAM注意力模块
│   ├── eca.py                   ← 03号：ECA注意力模块
│   ├── bifpn.py                 ← 05号：BiFPN改进
│   └── loss/
│       ├── __init__.py
│       ├── focal_loss.py        ← 04号：Focal Loss
│       └── wiou.py              ← 04号：WIoU损失
│
├── 📁 configs/                  ← 配置文件（超参数）
│   ├── baseline.yaml            ← 基线配置
│   ├── exp02_cbam.yaml          ← 02号实验配置
│   ├── exp03_eca.yaml           ← 03号实验配置
│   ├── exp04_focal.yaml         ← 04号实验配置
│   ├── exp05_bifpn.yaml         ← 05号实验配置
│   ├── exp06_augment.yaml       ← 06号：数据增强配置
│   ├── exp07_rtdetr.yaml        ← 07号：RT-DETR配置
│   ├── exp08_swa.yaml           ← 08号：SWA配置
│   ├── exp09_domain.yaml        ← 09号：跨域迁移配置
│   └── exp10_gradcam.yaml       ← 10号：Grad-CAM配置
│
├── 📁 scripts/                  ← 可运行脚本（入口文件）
│   ├── __init__.py
│   ├── train.py                 ← 统一训练入口
│   ├── eval.py                  ← 统一评估入口
│   ├── infer.py                 ← 推理/预测入口
│   ├── gradcam_vis.py           ← 10号：Grad-CAM可视化
│   ├── cross_domain.py          ← 09号：跨域迁移
│   └── verify_env.py            ← 环境验证脚本
│
├── 📁 utils/                    ← 工具函数（公共代码）
│   ├── __init__.py
│   ├── dataset_utils.py         ← 数据集处理工具
│   ├── metrics.py               ← 评估指标计算
│   ├── visualizer.py            ← 可视化工具
│   └── logger.py                ← 日志工具
│
├── 📁 experiments/              ← 实验结果（选择性上传）
│   ├── .gitignore
│   ├── exp02_cbam/              ← 02号实验结果
│   │   ├── results.csv
│   │   └── plots/
│   ├── exp03_eca/
│   ├── ...
│   └── exp10_gradcam/           ← 10号：Grad-CAM结果
│       ├── heatmaps/            ← 热力图输出
│       └── comparison_report.md  ← 对比报告
│
├── 📁 docs/                     ← 文档
│   ├── 需求文档.md
│   ├── 技术方案.md
│   ├── 答辩PPT/
│   └── 会议纪要/
│
├── 📁 checkpoints/              ← 模型权重（不上传！）
│   └── .gitignore
│
├── 📄 README.md                 ← 项目说明（最重要！）
├── 📄 requirements.txt          ← Python依赖包清单
├── 📄 .gitignore                ← 忽略规则
├── 📄 LICENSE                   ← 开源协议（可选）
└── 📄 setup.py                  ← 安装脚本（可选）
```

---

## 四、每人负责的文件（按方向分配）

| 方向编号 | 负责人 | 负责的文件 | 说明 |
|---------|--------|-----------|------|
| 01-A/B | 成员A | `scripts/train.py`（基线部分）、`configs/baseline.yaml` | 基线模型训练 |
| 02 | 成员B | `models/cbam.py`、`configs/exp02_cbam.yaml` | CBAM模块 |
| 03 | 成员C | `models/eca.py`、`configs/exp03_eca.yaml` | ECA模块 |
| 04 | 成员D | `models/loss/focal_loss.py`、`models/loss/wiou.py`、`configs/exp04_focal.yaml` | 损失函数 |
| 05 | 成员E | `models/bifpn.py`、`configs/exp05_bifpn.yaml` | BiFPN模块 |
| 06 | 成员F | `configs/exp06_augment.yaml` + 数据增强实验记录 | 数据增强 |
| 07 | 成员G | `configs/exp07_rtdetr.yaml` | RT-DETR对比 |
| 08 | 成员H | `configs/exp08_swa.yaml` + SWA实现 | 随机权重平均 |
| 09 | 成员I | `scripts/cross_domain.py`、`configs/exp09_domain.yaml` | 跨域迁移 |
| 10 | **PM（你）** | `scripts/gradcam_vis.py`、`configs/exp10_gradcam.yaml`、`experiments/exp10_gradcam/` | **Grad-CAM可视化** |
| 公共 | 所有人 | `utils/`下的工具函数、`docs/`下的文档 | 协作维护 |

---

## 五、Git操作流程（新手版，一步一步跟着做）

### 场景1：第一次把代码上传到GitHub

```bash
# 1. 打开终端，进入你的项目文件夹
cd defect-detection-10th

# 2. 初始化Git仓库（只需做一次）
git init

# 3. 添加所有文件到暂存区
git add .

# 4. 写提交说明（描述你这次做了什么）
git commit -m "feat: 初始项目结构 + 基线训练脚本"

# 5. 连接远程仓库（PM创建后把链接发给大家）
git remote add origin git@github.com:你的用户名/仓库名.git

# 6. 推送到GitHub
# 第一次推送：
git push -u origin main
# 之后每次推送：
git push
```

### 场景2：每天更新代码（日常操作）

```bash
# 1. 先拉取别人的最新代码（防止冲突）
git pull

# 2. 修改你的代码...

# 3. 查看你改了哪些文件
git status

# 4. 添加你修改的文件
git add 文件名
# 或者添加所有修改：
git add .

# 5. 写提交说明
git commit -m "feat: 添加了CBAM注意力模块"

# 6. 推送到GitHub
git push
```

### 场景3：创建自己的分支（推荐！防止互相影响）

```bash
# 1. 基于主分支创建新分支
git checkout -b exp02-cbam
# 分支名建议：exp<编号>-<方向名>，如 exp02-cbam

# 2. 在这个分支上写代码、提交

# 3. 推送分支到GitHub
git push -u origin exp02-cbam

# 4. 在GitHub网页上发起 Pull Request（合并请求）
# 让其他同学/PM审核你的代码

# 5. 审核通过后，合并到主分支
```

---

## 六、代码提交规范（必须遵守！）

### 提交信息格式
```
<type>: <简短描述>

<详细描述（可选）>
```

### 类型说明

| 类型 | 含义 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: 添加CBAM注意力模块` |
| `fix` | 修复bug | `fix: 修复训练时显存溢出` |
| `docs` | 文档更新 | `docs: 更新README使用说明` |
| `style` | 代码格式 | `style: 统一代码缩进为4空格` |
| `refactor` | 重构代码 | `refactor: 优化数据加载逻辑` |
| `test` | 测试相关 | `test: 添加Grad-CAM单元测试` |
| `chore` | 杂项 | `chore: 更新requirements.txt` |

### 好的提交信息示例
```bash
git commit -m "feat: 添加ECA注意力模块到YOLOv8 Backbone"
git commit -m "fix: 修复Focal Loss中alpha参数设置错误"
git commit -m "docs: 添加数据增强策略对比实验记录"
git commit -m "feat(exp10): 实现Grad-CAM热力图可视化功能"
```

### 不好的提交信息（禁止！）
```bash
git commit -m "update"           # 太模糊
git commit -m "123"               # 无意义
git commit -m "修复了一些问题"     # 不具体
```

---

## 七、Pull Request流程（代码审核机制）

### 为什么用Pull Request？
- 防止直接把有问题的代码推送到主分支
- 让其他同学/PM能看到你改了什么
- 方便讨论和提出修改建议

### 操作步骤

1. **推送你的分支到GitHub**
```bash
git push -u origin exp02-cbam
```

2. **在GitHub网页操作**
   - 打开仓库页面
   - 点击 "Compare & pull request"
   - 填写标题：`[Exp02] 添加CBAM注意力模块`
   - 填写描述：
     ```
     ## 变更内容
     - 在YOLOv8 Backbone的C2f模块后插入CBAM模块
     - 添加通道注意力和空间注意力子模块

     ## 测试情况
     - 在铝型材数据集上训练10个epoch，mAP提升1.2%

     ## 需要审核的点
     - CBAM插入位置是否合理
     - 超参数设置是否合适
     ```
   - 选择审核人（如PM或其他同学）
   - 点击 "Create pull request"

3. **审核人操作**
   - 收到通知后，点击 "Files changed" 查看代码变更
   - 可以逐行评论："这里为什么要用256？"
   - 审核通过后，点击 "Approve"
   - 最后点击 "Merge pull request" 合并到主分支

---

## 八、常见问题和解决方案

### Q1：git push 提示 "Permission denied"
**原因**：SSH密钥没配置好
**解决**：
```bash
# 检查SSH连接
ssh -T git@github.com
# 如果失败，重新配置SSH密钥（见Step 4）
```

### Q2：git pull 提示 "CONFLICT"（冲突）
**原因**：你和别人修改了同一个文件的同一行
**解决**：
```bash
# 1. 打开冲突文件，找到 <<<<<<< HEAD 标记
# 2. 手动选择保留哪部分代码
# 3. 删除 <<<<<<<、=======、>>>>>>> 标记
# 4. 重新提交
git add .
git commit -m "fix: 解决合并冲突"
git push
```

### Q3：不小心把大文件（数据集/模型权重）上传到GitHub
**原因**：GitHub限制单文件100MB，仓库建议不超过1GB
**解决**：
```bash
# 1. 把大文件路径加入 .gitignore
echo "data/" >> .gitignore
echo "checkpoints/" >> .gitignore

# 2. 从Git历史中删除大文件
git rm --cached -r data/
git rm --cached -r checkpoints/

# 3. 重新提交
git add .
git commit -m "fix: 移除大文件，添加.gitignore"
git push
```

### Q4：想回退到之前的版本
```bash
# 查看历史提交
git log --oneline

# 回退到某个版本（保留修改）
git reset --soft HEAD~1

# 回退到某个版本（丢弃修改，慎用！）
git reset --hard abc1234  # abc1234是commit id
```

### Q5：提交信息写错了
```bash
# 修改最后一次提交信息
git commit --amend -m "正确的提交信息"

# 如果已经push了，需要强制推送
git push --force-with-lease
```

---

## 九、PM（你）的特殊职责

### 作为PM，你不需要审核代码细节，但需要：

1. **创建仓库并设置规则**
   - Settings → Branches → Add rule
   - 勾选 "Require a pull request before merging"
   - 勾选 "Require approvals"（至少1人审核）
   - 这样确保没人能直接推送到主分支

2. **管理分支策略**
   - `main`：主分支，只有审核通过的代码才能合并
   - `exp02-cbam`、`exp10-gradcam`：每人一个功能分支
   - 禁止直接在main分支上开发

3. **定期检查进度**
   - 每周看一次GitHub的Insights → Contributors，了解每人提交情况
   - 在Pull Request里提问（不需要懂代码，可以问"这个功能完成了吗？""测试结果如何？"）

4. **作为10号方向负责人**
   - 你的Grad-CAM代码在 `scripts/gradcam_vis.py`
   - 结果输出到 `experiments/exp10_gradcam/`
   - 同样需要发起Pull Request，找一位同学帮你审核

---

## 十、快速参考卡片（保存到手机/桌面）

```
🚀 日常三步走：
1. git pull          ← 拉取最新代码
2. 修改代码...
3. git add . && git commit -m "feat: xxx" && git push

🌿 创建新分支：
git checkout -b exp<编号>-<方向名>
git push -u origin exp<编号>-<方向名>
# 然后在GitHub发起Pull Request

↩️ 回退操作：
git checkout -- 文件名     ← 撤销文件修改
git reset --soft HEAD~1    ← 撤销最后一次提交

❓ 查看状态：
git status                 ← 查看修改了哪些文件
git log --oneline          ← 查看提交历史
```

---

> **变更记录**
> 
> | 版本 | 日期 | 变更内容 | 变更人 |
> |-----|------|---------|-------|
> | v1.0 | 2026-07-17 | 初始版本 | PM |
