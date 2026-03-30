import numpy as np
import os
import time
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from datasets import HDRDataset
from models import HDRnetModel
from utils import psnr, print_params, load_train_ckpt, save_model_stats, plot_per_check, AvgMeter

import torch.nn.functional as F


class ProductionLoss(nn.Module):
    def __init__(self,
                 w_l1=0.8,
                 w_ssim=0.2,
                 w_grad=0.3,
                 w_color=0.3):
        super().__init__()

        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_color = w_color

        self.l1 = nn.L1Loss()

    def ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)

        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) *
                    (sigma_x + sigma_y + C2))

        return ssim_map.mean()

    def gradient_loss(self, x, y):
        dx_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        dx_y = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])

        dy_x = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        dy_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

        return F.l1_loss(dx_x, dx_y) + F.l1_loss(dy_x, dy_y)

    def color_loss(self, x, y):
        mean_x = x.mean(dim=[2, 3])
        mean_y = y.mean(dim=[2, 3])
        return F.l1_loss(mean_x, mean_y)

    def forward(self, output, target):

        l1 = self.l1(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        grad = self.gradient_loss(output, target)
        color = self.color_loss(output, target)

        total = (
            self.w_l1 * l1 +
            self.w_ssim * ssim_loss +
            self.w_grad * grad +
            self.w_color * color
        )

        return total


def train(params):

    device = torch.device("cuda" if params['cuda'] else "cpu")

    # --- Загружаем уже готовые папки ---
    train_dataset = HDRDataset(params['train_data_dir'], params, is_train=True)
    val_dataset = HDRDataset(params['val_data_dir'], params, is_train=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=params['batch_size'],
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False)

    model = HDRnetModel(params).to(device)

    optimizer = Adam(model.parameters(),
                     params['learning_rate'],
                     weight_decay=1e-8)

    criterion = ProductionLoss()

    best_psnr = 0.0

    for epoch in range(params['epochs']):

        model.train()
        train_psnr_meter = AvgMeter()

        for low, full, target in train_loader:

            low = low.to(device) / 255.0
            full = full.to(device) / 255.0
            target = target.to(device) / 255.0

            output = model(low, full)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_psnr_meter.update(psnr(output, target).item())

        print(f"Epoch {epoch+1} | Train PSNR: {train_psnr_meter.avg:.2f}")

        val_psnr = evaluate(model, val_loader, device)
        print(f"Validation PSNR: {val_psnr:.2f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_model_stats(model, params, "best_model.pt", {})
            print("🔥 Saved BEST model")


def evaluate(model, loader, device):

    model.eval()
    meter = AvgMeter()

    with torch.no_grad():
        for low, full, target in loader:

            low = low.to(device) / 255.0
            full = full.to(device) / 255.0
            target = target.to(device) / 255.0

            output = model(low, full)
            meter.update(psnr(output, target).item())

    return meter.avg


def parse_args():
    parser = ArgumentParser(description='HDRnet training')

    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--ckpt_interval', default=600, type=int)
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str)
    parser.add_argument('--stats_dir', default='./stats', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--summary_interval', default=10, type=int)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--val_data_dir', type=str, required=True)
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--eval_out', default='./outputs', type=str)
    parser.add_argument('--hdr', action='store_true')

    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--input_res', default=256, type=int)
    parser.add_argument('--output_res', default=(1024, 1024),
                        type=int, nargs=2)

    return parser.parse_args()


if __name__ == '__main__':

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    params = vars(parse_args())
    print_params(params)

    train(params)