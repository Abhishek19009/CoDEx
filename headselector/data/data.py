import json
import random
import time
from datetime import date
from os.path import join, exists
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from .transforms import (
    random_fliph,
    random_flipv,
    random_resize_crop,
    random_rotate,
)


class DEDataset(Dataset):
    def __init__(self,
                 path,
                 split,
                 feat_types,
                 label_types,
                 num_channels,
                 num_classes,
                 ):
        
        super(Dataset).__init__()
        self.path = path
        self.feat_folders = {}
        self.label_folders = {}
        self.gt_folder = []
        self.feat_types = feat_types
        self.label_types = label_types
        
        for feat_type in self.feat_types:
            feat_files = os.listdir(os.path.join(path, split, feat_type))
            npy_files = [f for f in feat_files if f.endswith('.npy')]
            npy_files_sorted = sorted(npy_files, key=lambda x: int(os.path.splitext(x)[0]))
            npy_files_sorted = [os.path.join(path, split, feat_type, f) for f in npy_files_sorted]
            self.feat_folders[feat_type] = npy_files_sorted

        for label_type in self.label_types:
            label_files = os.listdir(os.path.join(path, 'labels', split, label_type))
            label_files = [f for f in label_files if f.endswith('.npy')]
            label_files_sorted = sorted(label_files, key=lambda x: int(x.split('.')[0]))
            label_files_sorted = [os.path.join(path, 'labels', split, label_type, f) for f in label_files_sorted]
            self.label_folders[label_type] = label_files_sorted


        gt_files = os.listdir(os.path.join(path, split, 'gt'))
        npy_files = [gt for gt in gt_files if gt.endswith('.npy')]
        npy_files_sorted = sorted(npy_files, key=lambda x: int(os.path.splitext(x)[0]))
        npy_files_sorted = [os.path.join(path, split, 'gt', gt) for gt in npy_files_sorted]
        self.gt_folder = npy_files_sorted

        self.split = split
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.monthly_dates = get_monthly_dates_dict()
        self.collate_fn = collate_fn
        self.num_patches = len(self.feat_folders[self.feat_types[0]])


    def __len__(self):
        return self.num_patches
    

    def transform(self, data, gt=None, patch_loc_i=None, patch_loc_j=None):
        if self.split == "train":
            # data, gt = random_crop(data, gt, self.img_size, self.true_size)
            data, gt = random_fliph(data, gt)
            data, gt = random_flipv(data, gt)
            data, gt = random_rotate(data, gt)
            data, gt = random_resize_crop(data, gt)

        return data, gt


    def __getitem__(self, i):
        # Load patch and label
        output = {}
        for feat_type in self.feat_types:
            feat_file = self.feat_folders[feat_type][i]
            output[feat_type] = torch.from_numpy(np.load(feat_file)).type(torch.float32)

        for label_type in self.label_types:
            label_file = self.label_folders[label_type][i]
            file_idx = int(os.path.basename(label_file).split('.')[0])
            patch_idx = file_idx // 64
            labels = np.load(label_file)
            output[label_type] = torch.from_numpy(labels).type(torch.float32)

        gt_file = self.gt_folder[i]
        output['gt'] = torch.from_numpy(np.load(gt_file)).type(torch.uint8)

        data, gt = self.transform(output['input'], output['gt'])

        output['input'] = data
        output['gt'] = gt

        months = list(range(24))
        days = [self.monthly_dates[month] for month in months]
        positions = torch.tensor(days, dtype=torch.long)

        output['positions'] = positions
        output['patch_idx'] = torch.tensor([patch_idx])
        output['file_idx'] = torch.tensor([file_idx])

        return output
    

def collate_fn(batch):
    keys = list(batch[0].keys())
    output = {}

    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

def get_monthly_dates_dict():
    s_date = date(2018, 1, 1)
    e_date = date(2019, 12, 31)
    dates_monthly = [
        f"{year}-{month}-01"
        for year, month in zip(
            [2018 for _ in range(12)] + [2019 for _ in range(12)],
            [f"0{m}" for m in range(1, 10)]
            + ["10", "11", "12"]
            + [f"0{m}" for m in range(1, 10)]
            + ["10", "11", "12"],
        )
    ]
    dates_daily = pd.date_range(s_date, e_date, freq="d").strftime("%Y-%m-%d").tolist()
    monthly_dates = []
    i, j = 0, 0
    while i < 730 and j < 24:
        if dates_monthly[j] == dates_daily[i]:
            monthly_dates.append(i)
            j += 1
        i += 1
    return monthly_dates
