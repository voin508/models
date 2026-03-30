
import torch

# sRGB <-> linear helpers
def srgb_to_linear(x):
    # x in [0,1]
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x):
    a = 0.055
    return torch.where(x <= 0.0031308, x*12.92, (1+a)*torch.pow(x, 1/2.4) - a)

# sRGB (D65) -> XYZ
def srgb_to_xyz(srgb):
    # srgb: [...,3] in [0,1]; assume gamma-correct (non-linear)
    lin = srgb_to_linear(srgb)
    # Matrix for D65
    M = lin.new_tensor([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
    # (...,3) @ (3,3)^T
    xyz = torch.matmul(lin, M.t())
    return xyz

# XYZ -> Lab (D65), using CIE standard
def xyz_to_lab(xyz):
    # white point D65
    Xn = 0.95047
    Yn = 1.00000
    Zn = 1.08883
    x = xyz[...,0] / Xn
    y = xyz[...,1] / Yn
    z = xyz[...,2] / Zn

    eps = 216/24389
    kappa = 24389/27

    def f(t):
        return torch.where(t > eps, t.pow(1/3), (kappa * t + 16)/116)

    fx = f(x); fy = f(y); fz = f(z)
    L = 116*fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)

def srgb_to_lab(img):
    """ img: [B,3,H,W] in [0,1] """
    B, C, H, W = img.shape
    x = img.permute(0,2,3,1).contiguous()  # [B,H,W,3]
    xyz = srgb_to_xyz(x)
    lab = xyz_to_lab(xyz)
    return lab.permute(0,3,1,2).contiguous()
