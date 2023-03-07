import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from PIL import Image
from randaugment import RandAugment
from torch.utils.data import Dataset
from torchvision import transforms

from data.coco import CutoutPIL


class VocDataset(Dataset):

    def __init__(self, img_prefix, ann_file, class_name, img_size=224, train_mode=False):
        super(VocDataset, self)

        self.CLASSES = class_name
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.img_premix = img_prefix
        self.ann_file = ann_file
        self.img_ids = mmcv.list_from_file(self.ann_file)
        self.img_infos = self.load_annotations()
        self.length = len(self.img_ids)
        self.gt_labels = np.zeros((self.length, len(self.CLASSES)), dtype=np.int64)
        for i in range(self.length):
            self.gt_labels[i] = self.get_ann_info(i)

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
        img_path = osp.join(self.img_premix, "JPEGImages", "{}.jpg".format(self.img_ids[idx]))
        pil_img = Image.open(img_path).convert("RGB")
        img = self.transform(pil_img)
        return img, self.gt_labels[idx]

    def load_annotations(self):
        img_infos = []
        for img_id in self.img_ids:
            filename = "JPEGImages/{}.jpg".format(img_id)
            xml_path = osp.join(self.img_premix, "Annotations", "{}.xml".format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            img_infos.append(dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]["id"]
        xml_path = osp.join(self.img_premix, "Annotations", "{}.xml".format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_labels = np.zeros((len(self.CLASSES)), dtype=np.int64)
        for obj in root.findall("object"):
            name = obj.find("name").text
            gt_labels[self.cat2label[name]] = 1
        return gt_labels
