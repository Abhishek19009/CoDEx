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
from models.module import SitsScdModel_Multihead
from data.data import DynamicEarthNet, Muds
from metrics.scd_metrics_multihead import SCDMetric as SCDMetric_mh
from metrics.scd_metrics import SCDMetric as SCDMetric_sh
from pathlib import Path
import pandas as pd
import json
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

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

def project_init(cfg):
    print(f"Working directory set to {os.getcwd()}")
    directory = cfg.checkpoints.dirpath
    os.makedirs(directory, exist_ok=True)
    copyfile(".hydra/config.yaml", join(directory, "config.yaml"))

@hydra.main(config_path="configs", config_name="mm_config", version_base=None)
def main(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    dataset_type = "val_dataset_domain_shift"
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
            domain_shift=dataset_config.get("domain_shift", False)
        )

    batch_size = 1

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    metric_mh = SCDMetric_mh(
        cfg.dataset.num_classes, cfg.dataset.class_names, 
        cfg.dataset.num_areas, cfg.dataset.ignore_index
    )

    metric_sh = SCDMetric_sh(
        cfg.dataset.num_classes, cfg.dataset.class_names,
        cfg.dataset.ignore_index
    )

    directory = cfg.checkpoints.dirpath
    checkpoint_path = Path(directory) / f"{cfg.checkpoints.filename}_in.ckpt"
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

    head_freq = {domain:np.zeros(55,) for domain in range(0, 55)}
    best_area_idx = []

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

        metric_mh.update(pred['pred'], gt)
        output = metric_mh.compute()

        miou_best = -1
        acc_best = -1
        conf_mat_best = None
        area_best = None
        miou_heads = []
        acc_heads = []

        for area in output:
            miou = output[area]['miou']
            acc = output[area]['acc']
            conf_mat = output[area]['conf_matrix']
            miou_heads.append(miou)

            if miou > miou_best:
                miou_best = miou
                area_best = area

            del output[area]['conf_matrix']

        area_best_idx = int(area_best.split('_')[-1])
        best_area_idx.append(area_best_idx)
        head_freq[idx_sits[0]][int(area_best.split('_')[-1])] += 1

        metric_sh.update(pred['pred'][area_best_idx], gt)

    output = metric_sh.compute()
    miou = output['miou']
    acc = output['acc']

    print(output)

        
if __name__ == "__main__":
    main()
