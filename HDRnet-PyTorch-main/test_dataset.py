import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from models import HDRnetModel
from utils import load_test_ckpt

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.filters import sobel


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

    delta = deltaE_ciede2000(lab1, lab2)
    return float(delta.mean())


def compute_gmsd(pred, target):

    pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
    target_np = target.squeeze(0).permute(1,2,0).cpu().numpy()

    pred_gray = np.mean(pred_np, axis=2)
    target_gray = np.mean(target_np, axis=2)

    grad_pred = sobel(pred_gray)
    grad_target = sobel(target_gray)

    c = 0.0026
    gms = (2 * grad_pred * grad_target + c) / (grad_pred**2 + grad_target**2 + c)

    return float(np.std(gms))


def compute_error_map(pred, target):

    pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
    target_np = target.squeeze(0).permute(1,2,0).cpu().numpy()

    pred_lab = rgb2lab(pred_np)
    target_lab = rgb2lab(target_np)

    error = deltaE_ciede2000(pred_lab, target_lab)

    error = error / (error.max() + 1e-8)
    error = 1.0 - error
    error = error ** 0.5

    error_img = (error * 255).astype(np.uint8)

    return Image.fromarray(error_img)


def create_quad(input_img, output_img, gt_img, error_img,
                psnr_v, ssim_v, ciede_v, gmsd_v,
                save_path):

    w, h = gt_img.size

    input_img = input_img.resize((w, h))
    output_img = output_img.resize((w, h))
    error_img = error_img.resize((w, h)).convert("RGB")

    header_h = 140

    canvas = Image.new("RGB", (w*2, h*2 + header_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_big = ImageFont.truetype("arial.ttf", 48)
        font_small = ImageFont.truetype("arial.ttf", 28)
    except:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    title = (
        f"PSNR: {psnr_v:.2f} | "
        f"SSIM: {ssim_v:.3f} | "
        f"CIEDE2000: {ciede_v:.2f} | "
        f"GMSD: {gmsd_v:.4f}"
    )

    draw.text((50, 20), title, fill="black", font=font_big)

    positions = [
        (0, header_h),
        (w, header_h),
        (0, header_h + h),
        (w, header_h + h)
    ]

    labels = ["Input", "Output", "Error", "GT"]

    for (x,y), label in zip(positions, labels):
        draw.text((x+20, y-30), label, fill="black", font=font_small)

    canvas.paste(input_img, positions[0])
    canvas.paste(output_img, positions[1])
    canvas.paste(error_img, positions[2])
    canvas.paste(gt_img, positions[3])

    canvas.save(save_path)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "ckpts/best_model_new_combinedloss.pt"
    test_dir = "test1"

    input_dir = os.path.join(test_dir, "input")
    gt_dir = os.path.join(test_dir, "output")

    save_dir = os.path.join(test_dir, "model")
    quad_dir = os.path.join(test_dir, "quad")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(quad_dir, exist_ok=True)

    state_dict, params = load_test_ckpt(ckpt_path)

    model = HDRnetModel(params).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.ToTensor()

    files = sorted(os.listdir(input_dir))

    sum_psnr = 0
    sum_ssim = 0
    sum_ciede = 0
    sum_gmsd = 0
    n = 0

    with torch.no_grad():
        for fname in files:

            input_path = os.path.join(input_dir, fname)
            gt_path = os.path.join(gt_dir, fname)

            if not os.path.exists(gt_path):
                continue

            input_img = Image.open(input_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")

            input_tensor = transform(input_img).unsqueeze(0).to(device)
            gt_tensor = transform(gt_img).unsqueeze(0).to(device)

            lowres = F.interpolate(
                input_tensor,
                size=(params["input_res"], params["input_res"]),
                mode="bilinear",
                align_corners=False
            )

            output = model(lowres, input_tensor)
            output = torch.clamp(output, 0, 1)

            save_image(output, os.path.join(save_dir, fname))

            output_pil = transforms.ToPILImage()(output.squeeze(0).cpu())

            psnr_v = psnr(output, gt_tensor).item()
            ssim_v = compute_ssim(output, gt_tensor)
            ciede_v = compute_ciede2000(output, gt_tensor)
            gmsd_v = compute_gmsd(output, gt_tensor)

            sum_psnr += psnr_v
            sum_ssim += ssim_v
            sum_ciede += ciede_v
            sum_gmsd += gmsd_v
            n += 1

            error_img = compute_error_map(output, gt_tensor)

            create_quad(
                input_img,
                output_pil,
                gt_img,
                error_img,
                psnr_v,
                ssim_v,
                ciede_v,
                gmsd_v,
                os.path.join(quad_dir, fname)
            )

            print(
                f"{fname} | "
                f"PSNR: {psnr_v:.2f} | "
                f"SSIM: {ssim_v:.3f} | "
                f"CIEDE2000: {ciede_v:.2f} | "
                f"GMSD: {gmsd_v:.4f}"
            )

    print("\n==============================")
    print(f"Mean PSNR: {sum_psnr/n:.2f}")
    print(f"Mean SSIM: {sum_ssim/n:.3f}")
    print(f"Mean CIEDE2000: {sum_ciede/n:.2f}")
    print(f"Mean GMSD: {sum_gmsd/n:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()