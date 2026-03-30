import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from skimage.color import deltaE_ciede2000
from skimage.filters import sobel


CHECKPOINT_DIR = "./checkpoints/DRN_pix2pix3"
EPOCH = "80"  

DATASET_PATH = "test1"

INPUT_DIR = os.path.join(DATASET_PATH, "input")
GT_DIR = os.path.join(DATASET_PATH, "output")

SAVE_DIR = os.path.join(DATASET_PATH, "model")
QUAD_DIR = os.path.join(DATASET_PATH, "quad")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(QUAD_DIR, exist_ok=True)


def psnr(pred, target):
    return 10 * torch.log10(1.0 / F.mse_loss(pred, target))

def compute_ssim(pred, target):
    pred_np = pred.permute(1,2,0).cpu().numpy()
    target_np = target.permute(1,2,0).cpu().numpy()

    return ssim(pred_np, target_np, channel_axis=2, data_range=1.0)


def compute_ciede2000(pred, target):
    pred_np = pred.permute(1,2,0).cpu().numpy()
    target_np = target.permute(1,2,0).cpu().numpy()

    lab1 = rgb2lab(pred_np)
    lab2 = rgb2lab(target_np)

    delta = deltaE_ciede2000(lab1, lab2)
    return float(delta.mean())


def compute_gmsd(pred, target):
    pred_np = pred.permute(1,2,0).cpu().numpy()
    target_np = target.permute(1,2,0).cpu().numpy()

    pred_gray = np.mean(pred_np, axis=2)
    target_gray = np.mean(target_np, axis=2)

    grad_pred = sobel(pred_gray)
    grad_target = sobel(target_gray)

    c = 0.0026
    gms = (2 * grad_pred * grad_target + c) / (grad_pred**2 + grad_target**2 + c)

    return float(np.std(gms))

def compute_error_map(pred, target):
    pred_np = pred.permute(1,2,0).cpu().numpy()
    target_np = target.permute(1,2,0).cpu().numpy()

    diff = pred_np - target_np
    error = np.sqrt(np.sum(diff**2, axis=2))

    error = error / (error.max() + 1e-8)
    error = 1.0 - error

    error_img = (error * 255).astype(np.uint8)

    return Image.fromarray(error_img)


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

    title = (
        f"PSNR: {psnr_value:.2f} | "
        f"SSIM: {ssim_value:.3f} | "
        f"CIEDE2000: {ciede_value:.2f} | "
        f"GMSD: {gmsd_value:.4f}"
    )

    bbox = draw.textbbox((0,0), title, font=font_big)
    text_w = bbox[2] - bbox[0]

    draw.text(((total_width-text_w)//2, 20), title, fill="black", font=font_big)

    labels = ["Input", "Model Output", "Error Map", "Ground Truth"]

    positions = [
        (0, header_h),
        (w, header_h),
        (0, header_h + h),
        (w, header_h + h)
    ]

    for label, (x,y) in zip(labels, positions):
        draw.text((x + 20, y - 30), label, fill="black", font=font_small)

    canvas.paste(input_img, positions[0])
    canvas.paste(output_img, positions[1])
    canvas.paste(error_img, positions[2])
    canvas.paste(gt_img, positions[3])

    canvas.save(save_path)


def tensor_to_pil(t):

    t = (t + 1) / 2
    t = torch.clamp(t, 0, 1)

    return transforms.ToPILImage()(t.cpu())


def main():

    opt = TestOptions().parse()
    import torch

    if torch.cuda.is_available():
        opt.gpu_ids = [0]
        print("Using GPU")
    else:
        opt.gpu_ids = []
        print("Using CPU")
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    opt.name = "DRN_pix2pix3"
    opt.checkpoints_dir = "./checkpoints"
    opt.epoch = EPOCH

    opt.model = "pix2pix"
    opt.dataset_mode = "aligned"

    opt.netG = "rdnccut"
    opt.netD = "fe"

    opt.norm = "batch"

    opt.dataroot = DATASET_PATH

    opt.resize_or_crop = "none"

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_ciede = 0
    total_gmsd = 0
    count = 0

    with torch.no_grad():

        for data in dataset:

            model.set_input(data)
            model.test()

            visuals = model.get_current_visuals()

            real_A = visuals["real_A"][0]
            fake_B = visuals["fake_B"][0]
            real_B = visuals["real_B"][0]

            pred = (fake_B + 1) / 2
            target = (real_B + 1) / 2

            psnr_val = psnr(pred, target).item()
            ssim_val = compute_ssim(pred, target)
            ciede_val = compute_ciede2000(pred, target)
            gmsd_val = compute_gmsd(pred, target)

            total_psnr += psnr_val
            total_ssim += ssim_val
            total_ciede += ciede_val
            total_gmsd += gmsd_val
            count += 1
            error_img = compute_error_map(pred, target)

            input_pil = tensor_to_pil(real_A)
            output_pil = tensor_to_pil(fake_B)
            gt_pil = tensor_to_pil(real_B)

            fname = os.path.basename(data["A_paths"][0])

            output_pil.save(os.path.join(SAVE_DIR, fname))

            quad_path = os.path.join(QUAD_DIR, fname)

            create_quad(
                input_pil,
                output_pil,
                gt_pil,
                error_img,
                psnr_val,
                ssim_val,
                ciede_val,
                gmsd_val,
                quad_path
            )

            print(
                f"{fname} | "
                f"PSNR: {psnr_val:.2f} | "
                f"SSIM: {ssim_val:.3f} | "
                f"CIEDE2000: {ciede_val:.2f} | "
                f"GMSD: {gmsd_val:.4f}"
            )


    print(f"Mean PSNR: {total_psnr / count:.2f}")
    print(f"Mean SSIM: {total_ssim / count:.3f}")
    print(f"Mean CIEDE2000: {total_ciede / count:.2f}")
    print(f"Mean GMSD: {total_gmsd / count:.4f}")

if __name__ == "__main__":
    main()