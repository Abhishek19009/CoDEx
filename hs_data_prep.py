import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from os.path import join
from shutil import copyfile, rmtree
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
import hydra
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_fabric.utilities.rank_zero import _get_rank
from models.module import SitsScdModel, SitsScdModel_Multihead
from data.data import DynamicEarthNet, Muds
from metrics.scd_metrics_multihead import SCDMetric

from pathlib import Path
import pandas as pd
import json
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import io
import base64
from PIL import Image

# Register new resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval)

# Set color map for classes
class_colors = {
    "impervi": "#606060",
    "agricult": "#cdcd00",  # Greenish Yellow
    "forest": "#00cd00",  # Green
    "wetlands": "#00009a",  # Dark Blue
    "soil": "#9a4b00",  # Reddish Brown
    "water": "#1852ff",  # Blue
}

cmap = mcolors.ListedColormap(list(class_colors.values()))
bounds = list(range(len(class_colors))) + [len(class_colors)]
norm = mcolors.BoundaryNorm(bounds, cmap.N)


def project_init(cfg):
    print(f"Working directory set to {os.getcwd()}")
    directory = cfg.checkpoints.dirpath
    os.makedirs(directory, exist_ok=True)
    copyfile(".hydra/config.yaml", join(directory, "config.yaml"))

def store_feats_labels(feats, metric, patch_idx, store_dir, split):
    # Precompute the CPU and detach operations
    feats_np = {ft_name: feats[ft_name].squeeze(1 if ft_name == 'feats_55x16x128x128' else 0).cpu().detach().numpy() 
                for ft_name in feats.keys()}
    labels_np = {mt_name: metric[mt_name].cpu().detach().numpy()
                for mt_name in metric.keys()}


    # Ensure the directories exist
    feats_base_dir = os.path.join(store_dir, split)
    labels_dir = os.path.join(store_dir, 'labels', split)
    
    os.makedirs(feats_base_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for ft_name, feat in feats_np.items():
        save_ft_path = os.path.join(feats_base_dir, ft_name, f'{patch_idx}.npy')
        os.makedirs(os.path.dirname(save_ft_path), exist_ok=True)
        np.save(save_ft_path, feat)

    for mt_name, mt in labels_np.items():
        save_label_path = os.path.join(labels_dir, mt_name, f'{patch_idx}.label')
        os.makedirs(os.path.dirname(save_label_path), exist_ok=True)
        np.save(save_label_path, mt)


@hydra.main(config_path="configs", config_name="mm_config", version_base=None)
def main(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    for dataset_type in ["val_dataset_domain_shift", "train_dataset_infer"]:
        dataset_config = dict_config["datamodule"][dataset_type]
        if cfg.dataset.name == "DynamicEarthNet":
            dataset = DynamicEarthNet(
                path=dataset_config["path"],
                split=dataset_config["split"],
                domain_shift=dataset_config.get("domain_shift", False),
            )

        elif cfg.dataset.name == "Muds":
            dataset = Muds(
                path=dataset_config["path"],
                split=dataset_config["split"],
                domain_shift=dataset_config.get("domain_shift", False),
            )

        # Needs to be batch_size=1 for data preparation.
        batch_size = 1

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
        metric = SCDMetric(
            cfg.dataset.num_classes, cfg.dataset.class_names, 
            cfg.dataset.num_areas, cfg.dataset.ignore_index
        )

        directory = cfg.checkpoints.dirpath
        checkpoint_path = Path(directory) / f"{cfg.checkpoints.filename}_out.ckpt"
        print(f"Loading from checkpoint ... {checkpoint_path}")
        model = SitsScdModel_Multihead.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
        model.eval()


        img_patches = []
        height = 1024
        patch_size = 128
        num_patches_per_sits = (height // patch_size) ** 2

        with open(os.path.join(dataset_config["path"], "split.json")) as f:
            splits = json.load(f)

        areas = splits[dataset_config["split"]]
        num_classes = cfg.dataset.num_classes
        class_names = cfg.dataset.class_names

        miou_dict = {area: np.zeros((num_classes, num_classes)) for area in areas}

        head_freq = {domain:np.zeros(55,) for domain in range(0, 10)}
        best_area_idx = []

        store_dir = f'{cfg.root_dir}/datasets/{cfg.dataset.name}_feats'

        if os.path.exists(store_dir):
            break

        for idx_batch, batch in enumerate(tqdm(dataloader)):
            batch["data"] = batch["data"].cuda().type(torch.float32)
            batch["gt"] = batch["gt"].cuda()
            batch["positions"] = batch["positions"].cuda()
            idx_sits = batch["idx"]
            positions = batch["positions"]
            gt = batch["gt"]
            data_unnorm = batch["data_unnorm"]
            patch_ij = batch["patch_ij"]
            _, pred = model.forward_pass(batch)

            metric.update(pred['pred'], gt)
            output = metric.compute()

            miou_heads = []
            acc_heads = []

            for idx, area in enumerate(output):
                miou = output[area]['miou']
                conf_mat = output[area]['conf_matrix']
                acc = output[area]['acc']
                miou_heads.append(miou)
                acc_heads.append(acc)
                del output[area]['conf_matrix']

            acc_heads = torch.tensor(acc_heads)
            miou_heads = torch.tensor(miou_heads) 

            store_feat_dict = {}
            store_feat_dict['input'] = batch['data']
            store_feat_dict['gt'] = batch['gt']

            store_feats_labels(
                store_feat_dict,
                {'acc': acc_heads, 'miou': miou_heads},
                idx_batch,
                store_dir,
                dataset_config['split']
            )


if __name__ == "__main__":
    main()
