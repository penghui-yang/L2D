import json
import os.path as osp

import numpy as np
from PIL import Image, ImageFile
from randaugment import RandAugment
from torch.utils.data import Dataset
from torchvision import transforms

from tools.cut_out_pil import CutoutPIL

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CocoDataset(Dataset):

    def __init__(self, img_prefix, ann_file, class_name, img_size=224, train_mode=False):
        super(CocoDataset, self)

        self.CLASSES = class_name
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.img_list = json.load(open(self.ann_file, "r"))
        self.length = len(self.img_list)

        self.gt_labels = np.zeros((self.length, len(self.CLASSES)), dtype=np.int64)
        for i in range(self.length):
            for j in self.img_list[i]["labels"]:
                self.gt_labels[i][j] = 1

        if train_mode:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                CutoutPIL(cutout_factor=0.5),
                RandAugment(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = osp.join(self.img_prefix, self.img_list[idx]["file_name"])
        pil_img = Image.open(img_path).convert("RGB")
        img = self.transform(pil_img)
        return img, self.gt_labels[idx]
