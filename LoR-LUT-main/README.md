
# LoR-IA-3DLUT (Starter)

一个可直接在 Google Colab 上训练的最小可用实现，包含：
- 基于 IA-3DLUT 的多基 LUT 融合 + **低秩（CP）残差**（LoR-IA-3DLUT）
- 可导三线性插值（默认）；四面体插值留有接口位点
- 损失：L1 + ΔE2000（Lab） + TV(LUT) + L2(ΔL)（可选 LPIPS）
- Paired-Folder 数据加载（任意成对数据集，按文件名对齐即可）
- 导出任意输入图像对应的 `.cube` LUT（无空间门控版本）

> **目录结构**
```
lor_ia3dlut_starter/
  core/
    core_lut.py
  data/
    paired_folder.py
  losses/
    delta_e.py
    lpips_wrapper.py
  utils/
    color.py
    metrics.py
    tv.py
  export/
    export_image_lut.py
    export_cube.py
  config/
    default.yaml
  train.py
  evaluate.py
  requirements.txt
  README.md
  notebooks/
    colab_quickstart.ipynb
```

---

## 0. 在 Colab 启动（建议步骤）

1. 打开此仓并把整个文件夹上传到你的 Google Drive（例如：`/MyDrive/lor_ia3dlut_starter`）。
2. 在 Colab 新建 Notebook，或直接用 `notebooks/colab_quickstart.ipynb`（上传到 Drive 后在 Colab 打开）。
3. 在 Colab 里执行以下（如果用我们提供的 notebook，会自动执行）：
```python
!nvidia-smi  # 查看 GPU
%cd /content
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/lor_ia3dlut_starter
!pip install -r requirements.txt
```

---

## 1. 准备成对数据（Paired-Folder）

将数据组织为：
```
/dataset_root/
  train/
    input/   # 输入图像（如相机原图/较差画质）
    gt/      # 对应目标图像（如专家修图/DSLR 参考）
  val/
    input/
    gt/
```

**要求**：`input/` 与 `gt/` 文件名一一对应（如 `0001.jpg` 在两边都存在）。格式支持 `.jpg/.png/.tif`。  
如用 MIT-Adobe FiveK 的 sRGB 成对版本或 DPED，都可整理为此结构。

---

## 2. 启动训练（示例）

```bash
# 以默认参数在 /dataset_root 上训练
python train.py   --data.root /content/drive/MyDrive/datasets/FiveK_paired   --work_dir runs/fivek_lor   --cfg config/default.yaml
```

**可选参数覆盖**：
- `--train.patch 512`：训练随机裁剪边长（默认 512）
- `--train.batch 16`：batch 大小
- `--model.G 33`、`--model.K 8`、`--model.R 8`
- `--loss.lpips 0.05`：LPIPS 权重（需要安装 `lpips`）
- `--optim.lr 1e-3`、`--train.iters 120000` 等

训练日志保存在 `work_dir`，`best.ckpt` 会按验证 PSNR/ΔE 选择。

---

## 3. 评估与导出 .cube

**评估（计算 PSNR/ΔE2000/LPIPS 可选）**
```bash
python evaluate.py   --data.root /content/drive/MyDrive/datasets/FiveK_paired/val   --ckpt runs/fivek_lor/best.ckpt   --out_dir runs/fivek_lor/val_vis
```

**导出某张图像对应的最终 LUT（.cube）**
```bash
python export/export_image_lut.py   --ckpt runs/fivek_lor/best.ckpt   --image /content/sample.jpg   --out_cube /content/sample_Lstar.cube
```

> 导出的 `.cube` 可直接用于调色工具/ISP，插值假设为三线性（tetra 可在部署端切换）。

---

## 4. 小贴士（与论文一致）

- 模型以下采样至 256² 的图像预测参数，对整图做一次 LUT 查表，因此能 4K 实时推理。
- 建议**训练在 sRGB 0-1** 的通道上，并使用 ΔE2000 作为色差感知损失；若掌握 RAW→线性 sRGB 的正确流程，可在此基础上切换到线性空间。
- 四面体插值（tetrahedral）在小网格下更鲁棒，但实现更复杂；本实现默认三线性，保持数值稳定与易读性。

---

## 5. 许可证与致谢

此 Starter 用于学术/研究快速复现，欢迎在论文中引用你的实现。
