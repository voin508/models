
import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def _list_files(d, exts):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(d, f"*{e}"))
    files = sorted(files)
    return files

class PairedFolderDataset(Dataset):
    """
    Expect structure:
    root/
      train/ or val/
        input/
        gt/
    Filenames must match one-to-one between input/ and gt/.
    """
    def __init__(self, root, split="train", in_dir="input", gt_dir="gt", exts=(".jpg",".jpeg",".png",".tif",".tiff"),
                 patch=512, augment=True):
        self.root = root
        self.split = split
        self.in_dir = os.path.join(root, split, in_dir)
        self.gt_dir = os.path.join(root, split, gt_dir)
        self.exts = exts
        self.patch = patch
        self.augment = augment

        ins = _list_files(self.in_dir, exts)
        gts = _list_files(self.gt_dir, exts)
        name2path_in = {os.path.basename(p): p for p in ins}
        name2path_gt = {os.path.basename(p): p for p in gts}
        names = sorted(list(set(name2path_in.keys()) & set(name2path_gt.keys())))
        if len(names) == 0:
            raise RuntimeError(f"No paired files found under {self.in_dir} and {self.gt_dir}.")
        self.pairs = [(name2path_in[n], name2path_gt[n]) for n in names]

    def __len__(self):
        return len(self.pairs)

    def _read_image(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        pin, pgt = self.pairs[idx]
        xin = self._read_image(pin)
        xgt = self._read_image(pgt)

        # to tensor [0,1]
        tin = TF.to_tensor(xin)
        tgt = TF.to_tensor(xgt)

        # 强制让 tin 和 tgt 具有相同空间尺寸（取两者公共的中心区域）
        _, Hin, Win = tin.shape
        _, Hgt, Wgt = tgt.shape
        if Hin != Hgt or Win != Wgt:
            Hc = min(Hin, Hgt)
            Wc = min(Win, Wgt)
            # 分别按各自中心裁到公共尺寸
            top_in  = (Hin - Hc) // 2; left_in  = (Win - Wc) // 2
            top_gt  = (Hgt - Hc) // 2; left_gt  = (Wgt - Wc) // 2
            tin = tin[:, top_in:top_in+Hc, left_in:left_in+Wc]
            tgt = tgt[:, top_gt:top_gt+Hc, left_gt:left_gt+Wc]

        # random crop (train) or center crop (val) if patch > 0
        if self.patch and self.patch > 0:
            _, H, W = tin.shape
            p = self.patch
            if self.split == "train":
                if H >= p and W >= p:
                    top = torch.randint(0, H - p + 1, (1,)).item()
                    left = torch.randint(0, W - p + 1, (1,)).item()
                    tin = tin[:, top:top+p, left:left+p]
                    tgt = tgt[:, top:top+p, left:left+p]
            else:
                if H >= p and W >= p:
                    top = (H - p) // 2
                    left = (W - p) // 2
                    tin = tin[:, top:top+p, left:left+p]
                    tgt = tgt[:, top:top+p, left:left+p]

        # simple augmentation
        if self.split == "train" and self.augment:
            if torch.rand(1).item() < 0.5:
                tin = TF.hflip(tin); tgt = TF.hflip(tgt)
            if torch.rand(1).item() < 0.5:
                tin = TF.vflip(tin); tgt = TF.vflip(tgt)

        
        # def normalize_percentile(img):
        #     low = torch.quantile(img, 0.005)
        #     high = torch.quantile(img, 0.995)
        #     return ((img - low) / (high - low + 1e-6)).clamp(0,1)

        # tin = normalize_percentile(tin)
        
        # if self.split == "train" and self.augment:

        #     if torch.rand(1) < 0.5:
        #         # exposure
        #         gain = torch.exp(torch.randn(1) * 0.3)
        #         tin = tin * gain

        #     if torch.rand(1) < 0.5:
        #         # gamma (контраст)
        #         gamma = torch.exp(torch.randn(1) * 0.3)
        #         tin = tin.clamp(1e-4, 1) ** gamma

        #     if torch.rand(1) < 0.5:
        #         # white balance
        #         wb = torch.randn(3) * 0.1 + 1.0
        #         tin = tin * wb.view(3,1,1)

        #     tin = tin.clamp(0,1)

        # predictor input (downsample to 256 for robustness & speed)
        h = 256
        img_lr = TF.resize(tin, [h, h], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        return {
            "img_lr": img_lr,
            "img_in": tin,
            "img_gt": tgt,
            "name": os.path.basename(pin),
        }
