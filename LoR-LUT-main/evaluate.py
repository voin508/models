
import os, argparse, torch, json
from torch.utils.data import DataLoader
from core.core_lut import LoRIA3DLUT
from data.paired_folder import PairedFolderDataset
from utils.metrics import psnr, deltaE2000_mean
from torchvision.utils import save_image
import numpy as np

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data.root", dest="data_root", type=str, required=True,
                    help="Path to val set root (expects val/input and val/gt under this root if you pass the top-level dataset root). If you pass '/.../val', it will use that directly.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    G = cfg["model"]["G"]; K = cfg["model"]["K"]; R = cfg["model"]["R"]

    # If user points at dataset_root, use 'val' split; if they point at val folder, adapt.
    root = args.data_root
    if os.path.isdir(os.path.join(root, "input")) and os.path.isdir(os.path.join(root, "gt")):
        # user passed '/.../val'
        ds = PairedFolderDataset(root=os.path.dirname(root), split=os.path.basename(root),
                                 in_dir="input", gt_dir="gt", exts=tuple(cfg["data"]["ext"]),
                                 patch=0, augment=False)
    else:
        ds = PairedFolderDataset(root=root, split="val",
                                 in_dir=cfg["data"]["in_dir"], gt_dir=cfg["data"]["gt_dir"],
                                 exts=tuple(cfg["data"]["ext"]),
                                 patch=0, augment=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = LoRIA3DLUT(G=G, K=K, R=R).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    P, D = [], []
    for i, batch in enumerate(dl):
        img_lr = batch["img_lr"].to(device, non_blocking=True)
        img_in = batch["img_in"].to(device, non_blocking=True)
        img_gt = batch["img_gt"].to(device, non_blocking=True)
        name = batch["name"][0]

        pred, _ = model(img_lr, img_in)
        P.append(psnr(pred.clamp(0,1), img_gt.clamp(0,1)).item())
        D.append(deltaE2000_mean(pred.clamp(0,1), img_gt.clamp(0,1)).item())

        save_image(pred.clamp(0,1), os.path.join(args.out_dir, name))

    report = {
        "N": len(P),
        "PSNR_mean": float(np.mean(P)),
        "PSNR_std": float(np.std(P)),
        "DeltaE2000_mean": float(np.mean(D)),
        "DeltaE2000_std": float(np.std(D)),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(report)

if __name__ == "__main__":
    main()
