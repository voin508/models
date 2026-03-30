
import torch
import torch.nn.functional as F
from losses.delta_e import delta_e_2000_srgb

def psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction='none').mean(dim=[1,2,3])
    psnr = -10.0 * torch.log10(mse + eps)
    return psnr

def deltaE2000_mean(x, y):
    de = delta_e_2000_srgb(x, y)  # [B,H,W]
    # 兼容 3D/4D
    if de.dim() == 4:
        return de.mean(dim=[1,2,3])
    else:  # [B,H,W]
        return de.mean(dim=[1,2])