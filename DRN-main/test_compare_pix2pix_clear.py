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
EPOCH = "80"  # или "latest"

DATASET_PATH = "data_full"

INPUT_DIR = os.path.join(DATASET_PATH, "input")
GT_DIR = os.path.join(DATASET_PATH, "output")

SAVE_DIR = os.path.join(DATASET_PATH, "model")
QUAD_DIR = os.path.join(DATASET_PATH, "quad")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(QUAD_DIR, exist_ok=True)


def tensor_to_pil(t):

    t = (t + 1) / 2
    t = torch.clamp(t, 0, 1)

    return transforms.ToPILImage()(t.cpu())


def main():

    opt = TestOptions().parse()
    import torch

    
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

    with torch.no_grad():

        for data in dataset:

            model.set_input(data)
            model.test()

            visuals = model.get_current_visuals()

            fake_B = visuals["fake_B"][0]

            output_pil = tensor_to_pil(fake_B)

            fname = os.path.basename(data["A_paths"][0])

            output_pil.save(os.path.join(SAVE_DIR, fname))


            print(
                f"{fname} | "                
            )

if __name__ == "__main__":
    main()