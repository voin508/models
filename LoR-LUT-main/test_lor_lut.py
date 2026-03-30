import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab

from core.core_lut import LoRIA3DLUT
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel


# ===================== METRICS =====================

def psnr(pred, target):
    return 10 * torch.log10(1.0 / F.mse_loss(pred, target))


def compute_ssim(pred, target):
    pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
    target_np = target.squeeze(0).permute(1,2,0).cpu().numpy()

    return ssim(pred_np, target_np, channel_axis=2, data_range=1.0)


def compute_ciede2000(pred, target):
    pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
    target_np = target.squeeze(0).permute(1,2,0).cpu().numpy()

    lab1 = rgb2lab(pred_np)
    lab2 = rgb2lab(target_np)

    # упрощённая версия (через skimage)
    from skimage.color import deltaE_ciede2000
    delta = deltaE_ciede2000(lab1, lab2)

    return float(delta.mean())

def compute_gmsd(pred, target):

    pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
    target_np = target.squeeze(0).permute(1,2,0).cpu().numpy()

    # перевод в grayscale
    pred_gray = np.mean(pred_np, axis=2)
    target_gray = np.mean(target_np, axis=2)

    # градиенты (через sobel)
    grad_pred = sobel(pred_gray)
    grad_target = sobel(target_gray)

    # similarity map
    c = 0.0026  # стабилизатор
    gms = (2 * grad_pred * grad_target + c) / (grad_pred**2 + grad_target**2 + c)

    # стандартное отклонение (главная идея GMSD)
    return float(np.std(gms))

def normalize_percentile(img, low_q=0.01, high_q=0.99):
    low = torch.quantile(img, low_q)
    high = torch.quantile(img, high_q)
    img = (img - low) / (high - low + 1e-6)
    return img.clamp(0, 1)
# ===================== ERROR MAP =====================

def compute_error_map(pred, target):

    from skimage.color import rgb2lab, deltaE_ciede2000

    pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
    target_np = target.squeeze(0).permute(1,2,0).cpu().numpy()

    # --- RGB → LAB ---
    pred_lab = rgb2lab(pred_np)
    target_lab = rgb2lab(target_np)

    # --- perceptual difference (CIEDE2000) ---
    error = deltaE_ciede2000(pred_lab, target_lab)

    # --- нормализация ---
    error = error / (error.max() + 1e-8)

    # --- инверсия (белое = хорошо) ---
    error = 1.0 - error

    # --- немного усилим контраст (очень полезно) ---
    error = error ** 0.5

    error_img = (error * 255).astype(np.uint8)

    return Image.fromarray(error_img)


# ===================== QUAD BUILDER =====================

def create_quad(input_img, output_img, gt_img, error_img,
                psnr_value, ssim_value, ciede_value, gmsd_value,
                save_path):

    w, h = gt_img.size

    input_img = input_img.resize((w, h))
    output_img = output_img.resize((w, h))
    error_img = error_img.resize((w, h)).convert("RGB")

    header_h = 140

    total_width = w * 2
    total_height = h * 2 + header_h

    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_big = ImageFont.truetype("arial.ttf", 48)
        font_small = ImageFont.truetype("arial.ttf", 28)
    except:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # ---- title ----
    title = (
        f"PSNR: {psnr_value:.2f} | "
        f"SSIM: {ssim_value:.3f} | "
        f"CIEDE2000: {ciede_value:.2f} | "
        f"GMSD: {gmsd_value:.4f}"
    )

    bbox = draw.textbbox((0,0), title, font=font_big)
    text_w = bbox[2] - bbox[0]

    draw.text(((total_width-text_w)//2, 20), title, fill="black", font=font_big)

    # ---- labels ----
    labels = ["Input", "Model Output", "Error Map", "Ground Truth"]

    positions = [
        (0, header_h),
        (w, header_h),
        (0, header_h + h),
        (w, header_h + h)
    ]

    for label, (x,y) in zip(labels, positions):
        draw.text((x + 20, y - 30), label, fill="black", font=font_small)

    # ---- paste ----
    canvas.paste(input_img, positions[0])
    canvas.paste(output_img, positions[1])
    canvas.paste(error_img, positions[2])
    canvas.paste(gt_img, positions[3])

    canvas.save(save_path)


# ===================== MAIN =====================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "runs/test/best.ckpt"
    test_dir = "test1"

    input_dir = os.path.join(test_dir, "input")
    gt_dir = os.path.join(test_dir, "output")

    save_dir = os.path.join(test_dir, "model")
    quad_dir = os.path.join(test_dir, "quad")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(quad_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt["cfg"]

    model = LoRIA3DLUT(
        G=cfg["model"]["G"],
        K=cfg["model"]["K"],
        R=cfg["model"]["R"]
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    transform = transforms.ToTensor()
    n = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_ciede = 0
    sum_gmsd = 0

    files = sorted(os.listdir(input_dir))

    with torch.no_grad():

        for fname in files:
            
            input_path = os.path.join(input_dir, fname)
            gt_path = os.path.join(gt_dir, fname)

            if not os.path.exists(gt_path):
                continue
            n+=1

            input_img = Image.open(input_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")

            input_tensor = transform(input_img)

            #нормализация 
            input_tensor = normalize_percentile(input_tensor)

            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            gt_tensor = transform(gt_img).unsqueeze(0).to(device)

            lowres = F.interpolate(
                input_tensor,
                size=(256,256),
                mode="bilinear",
                align_corners=False
            )

            output, aux = model(lowres, input_tensor)
            output = torch.clamp(output, 0, 1)

            save_image(output, os.path.join(save_dir, fname))

            output_pil = transforms.ToPILImage()(output.squeeze(0).cpu())

            psnr_value = psnr(output, gt_tensor).item()
            sum_psnr += psnr_value

            ssim_value = compute_ssim(output, gt_tensor)
            sum_ssim += ssim_value

            ciede_value = compute_ciede2000(output, gt_tensor)
            sum_ciede += ciede_value

            gmsd_value = compute_gmsd(output, gt_tensor)
            sum_gmsd += gmsd_value

            error_img = compute_error_map(output, gt_tensor)

            create_quad(
            input_img,
            output_pil,
            gt_img,
            error_img,
            psnr_value,
            ssim_value,
            ciede_value,
            gmsd_value,
            os.path.join(quad_dir, fname)
        )

            print(
            f"{fname} | "
            f"PSNR: {psnr_value:.2f} | "
            f"SSIM: {ssim_value:.3f} | "
            f"CIEDE2000: {ciede_value:.2f} | "
            f"GMSD: {gmsd_value:.4f}"
            )
        print(f"| Mean PSNR: {sum_psnr/n:.2f}")
        print(f"| Mean SSIM: {sum_ssim/n:.3f}")
        print(f"| Mean CIEDE2000: {sum_ciede/n:.2f}")
        print(f"| Mean GMSD: {sum_gmsd/n:.4f}")


if __name__ == "__main__":
    main()