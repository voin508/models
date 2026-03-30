
import os, argparse, yaml, time, math, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from core.core_lut import LoRIA3DLUT, cp_residual_to_lut
from data.paired_folder import PairedFolderDataset
from losses.delta_e import delta_e_2000_srgb
from losses.lpips_wrapper import LPIPSWrapper
from utils.metrics import psnr, deltaE2000_mean
from utils.tv import tv_3d, l2_residual, monotonicity_3d

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_dataloaders(cfg):
    root = cfg["data"]["root"]
    in_dir = cfg["data"]["in_dir"]
    gt_dir = cfg["data"]["gt_dir"]
    exts = tuple(cfg["data"]["ext"])

    ds_tr = PairedFolderDataset(root, "train", in_dir, gt_dir, exts, patch=cfg["train"]["patch"], augment=True)
    ds_va = PairedFolderDataset(root, "val",   in_dir, gt_dir, exts, patch=0, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch"], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return dl_tr, dl_va

def save_ckpt(path, model, optim, cfg, it, best_metric):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optim.state_dict(),
        "cfg": cfg,
        "iter": it,
        "best": best_metric
    }, path)

def train_one(cfg, work_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])

    dl_tr, dl_va = make_dataloaders(cfg)

    G, K, R = cfg["model"]["G"], cfg["model"]["K"], cfg["model"]["R"]
    model = LoRIA3DLUT(G=G, K=K, R=R).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    # scheduler (cosine)
    if cfg["sched"]["cosine"]:
        T = cfg["train"]["iters"]
        min_lr = cfg["sched"]["min_lr"]
        def lr_lambda(step):
            if step >= T: return min_lr / cfg["optim"]["lr"]
            cos = 0.5*(1 + math.cos(math.pi*step/T))
            return (min_lr/cfg["optim"]["lr"]) + (1 - min_lr/cfg["optim"]["lr"]) * cos
        sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        sch = None
    #old
    # scaler = torch.cuda.amp.GradScaler(enabled=(cfg["train"]["amp"] and torch.cuda.is_available()))
    #new
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg["train"]["amp"] and torch.cuda.is_available()))
    lpips_loss = LPIPSWrapper().to(device)
    os.makedirs(work_dir, exist_ok=True)

    iters = cfg["train"]["iters"]
    best_psnr = -1e9
    t0 = time.time()

    it = 0
    while it < iters:
        for batch in dl_tr:
            it += 1
            model.train()
            img_lr = batch["img_lr"].to(device, non_blocking=True)
            img_in = batch["img_in"].to(device, non_blocking=True)
            img_gt = batch["img_gt"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                pred, aux = model(img_lr, img_in)

                # losses (L1, TV, L2 in autocast)
                # Following IA-3DLUT baseline: NO clamp on main L1 loss to allow strong transformations
                loss = 0.0
                if cfg["loss"]["l1"] > 0:
                    # CRITICAL: No clamp on pred or gt, following IA-3DLUT baseline
                    loss_l1 = torch.nn.functional.l1_loss(pred, img_gt)
                    loss = loss + cfg["loss"]["l1"] * loss_l1
                
                # Alpha L2 regularization (following original IA-3DLUT)
                if cfg["loss"].get("alpha_l2", 0) > 0:
                    alpha = aux["alpha"]
                    weights_norm = (alpha ** 2).mean()
                    loss = loss + cfg["loss"]["alpha_l2"] * weights_norm
                
                # Regularization losses (TV, Monotonicity and Delta L2)
                # We use the LUT and delta passed from the model via aux dictionary
                if cfg["loss"]["tv"] > 0:
                    final_lut = aux['L_final']
                    # TV loss on raw LUT (no clamp), following IA-3DLUT baseline
                    loss = loss + cfg["loss"]["tv"] * tv_3d(final_lut)

                # Monotonicity regularization: penalize negative slopes along LUT axes
                if cfg["loss"].get("mono", 0) > 0:
                    final_lut = aux['L_final']
                    loss = loss + cfg["loss"]["mono"] * monotonicity_3d(final_lut)
                
                if cfg["loss"]["dl2"] > 0:
                    delta = aux['delta']
                    loss = loss + cfg["loss"]["dl2"] * (delta**2).mean()

            # Perceptual losses (e.g., LPIPS) and colorimetric losses (DE2000)
            # These NEED clamp for numerical stability in perceptual/metric calculations
            with torch.amp.autocast('cuda', enabled=False):
                pred_safe = pred.float().clamp(0, 1)  # Only for perceptual/metrics
                gt_safe = img_gt.float().clamp(0, 1)   # Only for perceptual/metrics
                
                de_weight = cfg["loss"]["de2000"]

                if it < cfg["loss"].get("de2000_start_iter", 0):
                    de_weight = 0

                if de_weight > 0:
                    de_map = delta_e_2000_srgb(pred_safe, gt_safe)
                    loss_de = de_map.mean()
                    loss = loss + de_weight * loss_de
                
                if cfg["loss"]["lpips"] > 0:
                    loss = loss + cfg["loss"]["lpips"] * lpips_loss(pred_safe, gt_safe)

            # Check for NaN BEFORE backward and parameter update
            if not torch.isfinite(loss):
                print(f"  WARNING: NaN/Inf at step {it}, skipping")
                opt.zero_grad(set_to_none=True)
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # tighter clipping for stability
            scaler.step(opt)
            scaler.update()
            
            
                
            if sch is not None:
                sch.step()

            if it % cfg["train"]["log_every"] == 0:
                with torch.no_grad():
                    p = psnr(pred.clamp(0,1), img_gt.clamp(0,1)).mean().item()
                    d = deltaE2000_mean(pred.clamp(0,1), img_gt.clamp(0,1)).mean().item()
                    # Monitor alpha sharpness and residual strength
                    a = aux["alpha"]
                    amax = a.max(dim=1).values.mean().item()  # alpha sharpness
                    dnorm = aux["delta_norm"].item()          # residual strength
                lr_cur = opt.param_groups[0]["lr"]
                dt = time.time() - t0
                print(f"[{it:6d}/{iters}] loss={loss.item():.4f} psnr={p:.2f} de2000={d:.3f} "
                      f"amax={amax:.3f} dnorm={dnorm:.4f} lr={lr_cur:.2e} time={dt/60:.1f}m")
                save_image(pred.clamp(0,1), os.path.join(work_dir, "last_pred.png"))
                save_image(img_gt, os.path.join(work_dir, "last_gt.png"))

            if it % cfg["train"]["val_every"] == 0 or it == iters:
                val_psnr = validate(model, dl_va, device)
                print(f"  -> val PSNR: {val_psnr:.2f}")
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    save_ckpt(os.path.join(work_dir, "best.ckpt"), model, opt, cfg, it, best_psnr)

            if it >= iters:
                break

    save_ckpt(os.path.join(work_dir, "last.ckpt"), model, opt, cfg, iters, best_psnr)

@torch.no_grad()
def validate(model, dl, device):
    model.eval()
    import torch
    from utils.metrics import psnr
    ps = []
    for batch in dl:
        img_lr = batch["img_lr"].to(device, non_blocking=True)
        img_in = batch["img_in"].to(device, non_blocking=True)
        img_gt = batch["img_gt"].to(device, non_blocking=True)

        pred, _ = model(img_lr, img_in)
        p = psnr(pred.clamp(0,1), img_gt.clamp(0,1))
        ps.append(p)
    return torch.cat(ps, dim=0).mean().item()

def save_ckpt(path, model, opt, cfg, it, best):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "cfg": cfg,
        "iter": it,
        "best": best
    }, path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/default.yaml")
    parser.add_argument("--data.root", dest="data_root", type=str, default="")
    parser.add_argument("--work_dir", type=str, default="runs/exp1")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg["data"]["root"] = args.data_root

    os.makedirs(args.work_dir, exist_ok=True)
    train_one(cfg, args.work_dir)
