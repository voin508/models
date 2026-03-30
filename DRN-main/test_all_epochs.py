import os
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model


CHECKPOINT_DIR = "./checkpoints/DRN_pix2pix7"
DATASET_PATH = "test1"

MODEL_NAME = "DRN_pix2pix7"


def psnr(pred, target):
    return 10 * torch.log10(1.0 / F.mse_loss(pred, target))


def find_epochs():

    epochs = []

    for f in os.listdir(CHECKPOINT_DIR):

        match = re.match(r"(\d+)_net_G.pth", f)

        if match:
            epochs.append(int(match.group(1)))

    epochs = sorted(epochs)

    return epochs


def evaluate_epoch(epoch):

    opt = TestOptions().parse()

    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    opt.name = MODEL_NAME
    opt.checkpoints_dir = "./checkpoints"
    opt.epoch = str(epoch)

    opt.model = "pix2pix"
    opt.dataset_mode = "aligned"
    opt.netG = "rdnccut"
    opt.netD = "fe"
    opt.norm = "batch"
    opt.dataroot = DATASET_PATH

    opt.resize_or_crop = "none"
    opt.loadSize = 512
    opt.fineSize = 512

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    total_psnr = 0
    count = 0

    with torch.no_grad():

        for data in dataset:

            model.set_input(data)
            model.test()

            visuals = model.get_current_visuals()

            fake_B = visuals["fake_B"]
            real_B = visuals["real_B"]

            pred = (fake_B + 1) / 2
            target = (real_B + 1) / 2

            value = psnr(pred, target).item()

            total_psnr += value
            count += 1

    return total_psnr / count


def main():

    epochs = find_epochs()

    print("Found epochs:", epochs)

    psnr_values = []

    for epoch in epochs:

        print("\nTesting epoch:", epoch)

        value = evaluate_epoch(epoch)

        psnr_values.append(value)

        print("Average PSNR:", value)


    best_index = psnr_values.index(max(psnr_values))
    print("BEST EPOCH:", epochs[best_index])
    print("BEST PSNR:", psnr_values[best_index])

    plt.figure()

    plt.plot(epochs, psnr_values, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Epoch")

    plt.grid(True)

    plt.savefig("psnr_vs_epoch.png")

    plt.show()


if __name__ == "__main__":
    main()