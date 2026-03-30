
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 3D LUT application: Trilinear -----------------

class TrilinearLUTFunction(nn.Module):
    """
    Apply a 3D LUT with trilinear interpolation.
    img: [B,3,H,W] in [0,1]
    lut: [B, G, G, G, 3]
    """
    def __init__(self, grid_size: int = 33):
        super().__init__()
        self.G = grid_size

    def forward(self, img, lut):
        B, _, H, W = img.shape
        G = self.G
        # [0, G-1 - eps]
        xyz = img.permute(0,2,3,1).contiguous()
        xyz = torch.clamp(xyz * (G - 1), 0, G - 1 - 1e-6)

        x0 = torch.floor(xyz[...,0]).long()
        y0 = torch.floor(xyz[...,1]).long()
        z0 = torch.floor(xyz[...,2]).long()
        x1 = torch.clamp(x0 + 1, max=G-1)
        y1 = torch.clamp(y0 + 1, max=G-1)
        z1 = torch.clamp(z0 + 1, max=G-1)

        xd = (xyz[...,0] - x0.float()).unsqueeze(-1)
        yd = (xyz[...,1] - y0.float()).unsqueeze(-1)
        zd = (xyz[...,2] - z0.float()).unsqueeze(-1)

        def gather(ix, iy, iz):
            # lut: [B, G,G,G,3] -> [B,H,W,3]
            idx = (ix*G + iy)*G + iz
            lut_flat = lut.view(B, G*G*G, 3)
            out = torch.gather(lut_flat, 1, idx.view(B,-1).unsqueeze(-1).expand(-1,-1,3))
            return out.view(B, H, W, 3)

        c000 = gather(x0,y0,z0); c100 = gather(x1,y0,z0)
        c010 = gather(x0,y1,z0); c110 = gather(x1,y1,z0)
        c001 = gather(x0,y0,z1); c101 = gather(x1,y0,z1)
        c011 = gather(x0,y1,z1); c111 = gather(x1,y1,z1)

        c00 = c000*(1-xd) + c100*xd
        c01 = c001*(1-xd) + c101*xd
        c10 = c010*(1-xd) + c110*xd
        c11 = c011*(1-xd) + c111*xd

        c0  = c00*(1-yd) + c10*yd
        c1  = c01*(1-yd) + c11*yd

        out = c0*(1-zd) + c1*zd
        return out.permute(0,3,1,2).contiguous()

# --------------- Low-rank CP residual -> dense LUT ---------------

def cp_residual_to_lut(u, v, w, c):
    """
    u,v,w: [B,R,G] (softmax along G)
    c:     [B,R,3] (tanh-bounded recommended)
    return: [B,G,G,G,3]
    """
    B, R, G = u.shape
    U = u.unsqueeze(3).unsqueeze(4)   # [B,R,G,1,1]
    V = v.unsqueeze(2).unsqueeze(4)   # [B,R,1,G,1]
    W = w.unsqueeze(2).unsqueeze(3)   # [B,R,1,1,G]
    core = (U * V * W)                # [B,R,G,G,G]
    C = c.view(B, R, 1,1,1, 3)        # [B,R,1,1,1,3]
    delta = (core.unsqueeze(-1) * C).sum(1) # [B,G,G,G,3]
    return delta

# -------------------- Predictors (very small) --------------------

class WeightPredictor(nn.Module):
    def __init__(self, K=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, K)
        # Initialize bias to 1.0 (following original IA-3DLUT)
        nn.init.constant_(self.fc.bias, 1.0)
        
    def forward(self, x):
        h = self.backbone(x)
        w = self.fc(h.view(h.size(0), -1))
        # No softmax - use free weights with L2 regularization (like original IA-3DLUT)
        return w

class ResidualPredictor(nn.Module):
    def __init__(self, G=33, R=8):
        super().__init__()
        self.R, self.G = R, G
        self.enc = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(True),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_u = nn.Linear(32, R*G)
        self.fc_v = nn.Linear(32, R*G)
        self.fc_w = nn.Linear(32, R*G)
        self.fc_c = nn.Linear(32, R*3)
    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        u = self.fc_u(h).view(-1, self.R, self.G)
        v = self.fc_v(h).view(-1, self.R, self.G)
        w = self.fc_w(h).view(-1, self.R, self.G)
        c = self.fc_c(h).view(-1, self.R, 3)
        
        # Use sigmoid instead of softmax for larger values
        # u = torch.sigmoid(u)
        # v = torch.sigmoid(v)
        # w = torch.sigmoid(w)
        # c: no tanh, allow larger range (will be scaled by delta*100)
        # c = c * 0.1  # Scale to reasonable range
        return u, v, w, c

# -------------------- Identity LUT initializer -------------------

def create_identity_lut(G: int):
    rg = torch.linspace(0, 1, G)
    R, Gg, B = torch.meshgrid(rg, rg, rg, indexing='ij')
    lut = torch.stack([R, Gg, B], dim=-1) # [G,G,G,3]
    return lut

# ----------------------------- Model -----------------------------

class LoRIA3DLUT(nn.Module):
    def __init__(self, G=33, K=8, R=8):
        super().__init__()
        self.G, self.K, self.R = G, K, R
        base = create_identity_lut(G).unsqueeze(0).repeat(K, 1,1,1,1) # [K,G,G,G,3]
        
        # Add noise to break symmetry and help training
        if K > 1:
            noise = torch.randn_like(base) * 0.01  # 1% random perturbation
            base = base + noise
            base = base.clamp(0, 1)
        
        self.bases = nn.Parameter(base) # learnable
        self.weight_pred = WeightPredictor(K=K)
        self.resid_pred  = ResidualPredictor(G=G, R=R)
        self.apply_lut   = TrilinearLUTFunction(grid_size=G)
        
        # Initialize residual predictor with larger weights for faster learning
        nn.init.normal_(self.resid_pred.fc_c.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.resid_pred.fc_c.bias)

    def fuse_bases(self, alpha):
        """
        alpha: [B,K] - raw, unnormalized weights
        return fused bases (no residual): [B,G,G,G,3]
        """
        B = alpha.size(0)
        bases = self.bases.unsqueeze(0)         # [1,K,G,G,G,3]
        # Unconstrained weights, regulated by alpha_l2 loss (IA-3DLUT style)
        alpha_ = alpha.view(B, self.K, 1,1,1,1) # [B,K,1,1,1,1]
        L = (alpha_ * bases).sum(dim=1)         # [B,G,G,G,3]
        return L

    def forward(self, img_lr, img_full):
        """
        img_lr:  [B,3,h,w] (e.g., 256^2) for predicting parameters
        img_full:[B,3,H,W] full-res for lookup
        """
        alpha = self.weight_pred(img_lr)                    # [B,K]
        B = alpha.size(0)

        # If R <= 0, disable residual branch and use bases only
        if self.R <= 0:
            L = self.fuse_bases(alpha)                      # [B,G,G,G,3]
            out = self.apply_lut(img_full, L)               # [B,3,H,W]
            delta = torch.zeros(
                (B, self.G, self.G, self.G, 3),
                device=img_lr.device,
                dtype=self.bases.dtype,
            )
            aux = {
                "alpha": alpha,
                "delta": delta,
                "L_final": L,
                "delta_norm": torch.tensor(0.0, device=img_lr.device, dtype=self.bases.dtype),
            }
            return out, aux

        # Residual branch enabled
        u,v,w,c = self.resid_pred(img_lr)                  # low-rank
        delta = cp_residual_to_lut(u,v,w,c)                # [B,G,G,G,3]
        # Amplify residual to make it effective, regulated by dl2 loss
        delta = delta * 1.0
        
        L = self.fuse_bases(alpha) + delta                 # [B,G,G,G,3], unconstrained
        out = self.apply_lut(img_full, L)                  # [B,3,H,W]
        
        aux = {
            "alpha": alpha,
            "delta": delta, # for dl2 loss
            "L_final": L,   # for tv loss
            "delta_norm": delta.abs().mean(),
        }
        return out, aux
