# LoR-IA-3DLUT 快速启动

## 🚀 5 分钟快速测试

### 1. 检查数据集

```bash
python check_dataset.py --root /path/to/your/dataset
```

**期望输出**:
```
📁 TRAIN
   ✅ Matched pairs: 500
📁 VAL
   ✅ Matched pairs: 50   # ⚠️ 必须 > 1！
```

---

### 2. 本地测试（MacBook M4 - CPU only）

```bash
# 小规模测试（1000 步）
python train.py \
    --cfg config/default.yaml \
    --data.root /path/to/dataset \
    --work_dir runs/test_local
```

⚠️ **M4 Mac 限制**:
- 无 CUDA GPU，训练会很慢
- 仅用于**验证代码可运行**
- 真实训练请用 Google Colab

---

### 3. Google Colab 训练（推荐）

#### Step 1: 上传代码到 GitHub
```bash
cd /Users/yable/Projects/lor_ia3dlut_starter
git init
git add .
git commit -m "LoR-IA-3DLUT initial implementation"
git remote add origin https://github.com/YOUR_USERNAME/lor_ia3dlut.git
git push -u origin main
```

#### Step 2: 打开 Colab

1. 访问 https://colab.research.google.com
2. 创建新笔记本
3. 复制 `notebooks/colab_train.md` 中的内容到 Colab cells

#### Step 3: 运行

**Cell 1 - Setup**:
```python
!nvidia-smi  # 验证 GPU
!git clone https://github.com/YOUR_USERNAME/lor_ia3dlut.git
%cd lor_ia3dlut
!pip install -r requirements.txt
```

**Cell 2 - Mount Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 3 - 上传/下载数据集**:
```python
# 方案 A: 从本地上传（小数据集）
from google.colab import files
uploaded = files.upload()
!unzip dataset.zip -d /content/dataset

# 或方案 B: 从 Drive 复制
!cp -r /content/drive/MyDrive/your_dataset /content/dataset
```

**Cell 4 - 训练**:
```python
!python train.py \
    --cfg config/default.yaml \
    --data.root /content/dataset \
    --work_dir /content/drive/MyDrive/lor_checkpoints/exp1
```

---

## 🔍 训练监控

### 健康的训练日志应该长这样：

```
📊 Dataset loaded: 4500 train, 500 val samples  # ✅ 验证集 > 1

[    10/30000] loss=12.5 psnr=18.5 de2000=15.2 amax=0.28 dnorm=0.003 lr=1.0e-4
[    20/30000] loss=11.8 psnr=19.2 de2000=14.5 amax=0.32 dnorm=0.005 lr=1.0e-4
  -> val PSNR: 18.2  # ✅ 第一次验证

[   100/30000] loss=9.5 psnr=21.3 de2000=11.2 amax=0.41 dnorm=0.008 lr=9.9e-5
  -> val PSNR: 20.5  # ✅ 在增长！

[   500/30000] loss=7.2 psnr=23.8 de2000=8.5 amax=0.45 dnorm=0.012 lr=9.8e-5
  -> val PSNR: 22.9  # ✅ 持续提升
```

### 关键指标含义

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| `amax` | 0.2 - 0.6 | Alpha 最大值：多个 LUT 在融合 |
| `dnorm` | 0.001 - 0.02 | 残差幅度：微调作用 |
| `val PSNR` | 逐步增长 | **最关键**：不能恒定不变 |
| `de2000` | 下降 | 色差在减小 |

---

## ⚠️ 常见问题

### Q1: 验证 PSNR 不变？

**原因**: 验证集只有 1 张图或数据有问题

**解决**:
```bash
python check_dataset.py --root /path/to/dataset
# 检查 "VAL Matched pairs" 是否 > 1
```

### Q2: OOM (Out of Memory) 错误？

**解决**: 减小 batch size 或 patch size

```yaml
# config/default.yaml
train:
  batch: 4  # 从 8 降到 4
  patch: 384  # 从 512 降到 384
```

### Q3: 训练太慢？

**本地 Mac**: 正常，CPU 训练确实慢，用 Colab

**Colab**: 
```python
import torch
print(torch.cuda.is_available())  # 应该是 True
print(torch.cuda.get_device_name(0))  # 应该显示 GPU 型号
```

### Q4: `dnorm` 仍然很大（>0.1）？

**解决**: 降低 `residual_scale`

```yaml
# config/default.yaml
model:
  residual_scale: 0.5  # 或 0.1
```

---

## 📊 预期训练时间

| 配置 | 时间 (30k iter) | PSNR (FiveK) |
|------|-----------------|--------------|
| Colab T4 (free) | ~12-15 小时 | 23-25 |
| Colab A100 (Pro) | ~3-4 小时 | 24-26 |
| Local M4 Mac | ~7 天 ❌ | 不推荐 |

---

## 📈 实验优先级

### 第 1 周：验证修复

- [x] 修复代码问题
- [ ] 在小数据集（100-500 样本）上训练 5k iter
- [ ] 确认 val PSNR 正常增长

### 第 2 周：基线实验

- [ ] FiveK Expert C 完整训练（30k iter）
- [ ] 记录最佳 PSNR / ΔE2000 / LPIPS

### 第 3 周：消融研究

- [ ] K=4/8/12 对比
- [ ] R=0/4/8/16 对比（R=0 等价于纯 IA-3DLUT）
- [ ] residual_scale=0/0.5/1/2 对比

### 第 4 周：论文撰写

- [ ] 方法图（基础 LUT + 残差融合）
- [ ] 实验表格（vs IA-3DLUT, SepLUT, AdaInt）
- [ ] 主观对比图
- [ ] 写 Introduction + Method + Experiments

---

## 🎯 里程碑目标

| 日期 | 目标 | 验收标准 |
|------|------|---------|
| Day 1-2 | 代码修复验证 | val PSNR 正常增长 |
| Day 7 | FiveK 基线 | PSNR > 23 |
| Day 14 | 消融实验 | 4 组对比数据 |
| Day 20 | 初稿完成 | 4 页 draft + 图表 |

---

## 💡 提示

1. **先在小数据集上验证** - 不要一上来就跑 30k iter
2. **监控 val PSNR** - 这是判断训练是否正常的金标准
3. **保存到 Drive** - Colab 会话会断开，checkpoint 必须在 Drive
4. **定期导出 .cube** - 方便在 Photoshop/Premiere 中测试效果

---

**准备好了吗？开始训练吧！ 🚀**

```bash
# 第一步：检查数据
python check_dataset.py --root /path/to/dataset

# 第二步：开始训练（Colab）
# 复制 notebooks/colab_train.md 到 Colab
```


