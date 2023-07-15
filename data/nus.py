import os
import os.path as osp

import numpy as np
from PIL import Image
from randaugment import RandAugment
from torch.utils.data import Dataset
from torchvision import transforms

from tools.cut_out_pil import CutoutPIL


class NUSWideDataset(Dataset):

    def __init__(self, img_prefix, ann_prefix, img_size=224, train_mode=False, ls=0.0):
        self.img_list = []

        self.img_prefix = img_prefix
        train_or_test = "Train" if train_mode else "Test"
        self.img_list_file = f"{ann_prefix}/ImageList/{train_or_test}Imagelist.txt"
        with open(self.img_list_file, "r") as file:
            for line in file:
                line = line.strip("\n").replace("\\", "/")
                self.img_list.append({"file_name": line})
        
        self.CLASSES = []
        with open(os.path.join(ann_prefix, "Concepts81.txt"), "r") as file:
            for line in file:
                line = line.strip("\n")
                self.CLASSES.append(line)
        
        self.gt_labels = []
        for class_i in self.CLASSES:
            labels_i = []
            label_file = f"TrainTestLabels/Labels_{class_i}_{train_or_test}.txt"
            with open(os.path.join(ann_prefix, label_file), "r") as file:
                for line in file:
                    label = int(line.strip("\n"))
                    labels_i.append(label)
            self.gt_labels.append(np.expand_dims(np.array(labels_i), axis=1))
        self.gt_labels = np.concatenate(self.gt_labels, axis=1)

        assert len(self.gt_labels) == len(self.img_list)
        self.length = len(self.img_list)

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
