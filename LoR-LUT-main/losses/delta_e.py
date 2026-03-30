
import torch
from utils.color import srgb_to_lab

def ciede2000(lab1, lab2, eps=1e-5):
    # lab1, lab2: [B,3,H,W]
    L1, a1, b1 = lab1[:,0], lab1[:,1], lab1[:,2]
    L2, a2, b2 = lab2[:,0], lab2[:,1], lab2[:,2]

    C1 = torch.sqrt(torch.clamp(a1*a1 + b1*b1, min=0.0) + eps)
    C2 = torch.sqrt(torch.clamp(a2*a2 + b2*b2, min=0.0) + eps)
    Cm = 0.5*(C1 + C2)

    G = 0.5*(1 - torch.sqrt((Cm**7)/(Cm**7 + (25**7) + eps)))
    a1p = (1+G)*a1; a2p = (1+G)*a2
    C1p = torch.sqrt(torch.clamp(a1p*a1p + b1*b1, min=0.0) + eps)
    C2p = torch.sqrt(torch.clamp(a2p*a2p + b2*b2, min=0.0) + eps)

    def atan2b(b, a):
        ang = torch.atan2(b, a)
        ang = ang % (2*torch.pi)
        return ang

    h1p = atan2b(b1, a1p)
    h2p = atan2b(b2, a2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = dhp - 2*torch.pi*torch.round(dhp/(2*torch.pi))
    dHp = 2*torch.sqrt(C1p*C2p + eps) * torch.sin(dhp/2)

    Lpm = 0.5*(L1 + L2)
    Cpm = 0.5*(C1p + C2p)

    # average hue
    hsum = h1p + h2p
    hdiff = torch.abs(h1p - h2p)
    hpm = torch.where((C1p*C2p).sqrt() < 1e-6, hsum,
                      torch.where(hdiff <= torch.pi, 0.5*hsum,
                                  hsum + torch.where(hsum < 2*torch.pi, torch.pi, -3*torch.pi)))
    hpm = (hpm + 2*torch.pi) % (2*torch.pi)

    T = (1
         - 0.17*torch.cos(hpm - torch.deg2rad(torch.tensor(30.0, device=hpm.device)))
         + 0.24*torch.cos(2*hpm)
         + 0.32*torch.cos(3*hpm + torch.deg2rad(torch.tensor(6.0, device=hpm.device)))
         - 0.20*torch.cos(4*hpm - torch.deg2rad(torch.tensor(63.0, device=hpm.device))))

    dtheta = torch.deg2rad(torch.tensor(30.0, device=hpm.device)) * torch.exp(
        - ((torch.rad2deg(hpm) - 275.0)/25.0)**2
    )
    Rc = 2*torch.sqrt((Cpm**7)/(Cpm**7 + (25**7) + eps))
    Sl = 1 + (0.015*((Lpm - 50)**2)) / torch.sqrt(20 + (Lpm - 50)**2 + eps)
    Sc = 1 + 0.045*Cpm
    Sh = 1 + 0.015*Cpm*T
    Rt = -torch.sin(2*dtheta) * Rc

    x = (dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2 + Rt*(dCp/Sc)*(dHp/Sh)
    x = torch.clamp(x, min=0.0)
    dE = torch.sqrt(x + eps)
    return dE

def delta_e_2000_srgb(pred, target):
    # pred/target: [B,3,H,W] in [0,1] (sRGB)
    lab1 = srgb_to_lab(pred.clamp(0,1))
    lab2 = srgb_to_lab(target.clamp(0,1))
    return ciede2000(lab1, lab2)
