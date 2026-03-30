import numpy as np
import os
import rawpy
import torch
from skimage import io
from torchvision import transforms
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset
from utils import get_files
from PIL import Image


class HDRDataset(Dataset):

    def __init__(self, data_dir, params, is_train=True):

        self.data_path = data_dir
        self.input_res = params['input_res']
        self.output_res = params['output_res']
        self.params = params
        self.is_train = is_train

        all_files = get_files(os.path.join(self.data_path, 'input'))
        self.file_list = sorted([f.split('/')[-1] for f in all_files])

        print(f"{'Train' if is_train else 'Validation'} samples: {len(self.file_list)}")

        if self.is_train:
            self.augment = transforms.RandomCrop(self.output_res)

    # ------------------------------------

    def load_img(self, fname):
        inp = io.imread(os.path.join(self.data_path, 'input', fname))
        out = io.imread(os.path.join(self.data_path, 'output', fname))

        inp = torch.from_numpy(inp.transpose((2, 0, 1))).float()
        out = torch.from_numpy(out.transpose((2, 0, 1))).float()

        return inp, out

    def load_img_hdr(self, fname):
        inp = rawpy.imread(os.path.join(self.data_path, 'input', fname))
        inp = inp.postprocess(use_camera_wb=True,
                              half_size=False,
                              no_auto_bright=True,
                              output_bps=16)
        inp = np.asarray(inp, dtype=np.float32)

        out = io.imread(os.path.join(
            self.data_path,
            'output',
            fname.split('.')[0] + '.jpg'))

        inp = torch.from_numpy(inp.transpose((2, 0, 1))).float()
        out = torch.from_numpy(out.transpose((2, 0, 1))).float()

        return inp, out

    # ------------------------------------

    def __getitem__(self, idx):

        fname = self.file_list[idx]

        if self.params['hdr']:
            inp, target = self.load_img_hdr(fname)
        else:
            inp, target = self.load_img(fname)

        assert inp.shape == target.shape

        if self.is_train:
            # совместный crop
            inout = torch.cat([inp, target], dim=0)
            inout = self.augment(inout)

            full = inout[:3]
            target = inout[3:]
        else:
            full = inp

        low = resize(full, (self.input_res, self.input_res), Image.NEAREST)

        return low, full, target

    # ------------------------------------

    def __len__(self):
        return len(self.file_list)