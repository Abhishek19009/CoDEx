import os
import json 
import torch
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
from data.data import DynamicEarthNet
from metrics.scd_metrics_multihead import SCDMetric as SCDMetric_mh
from pathlib import Path
import pandas as pd
import json

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

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    dataset_type = "val_dataset_domain_shift"
    dataset_config = dict_config["datamodule"][dataset_type]
    dataset = DynamicEarthNet(
        path=dataset_config["path"],
        split=dataset_config["split"],
        domain_shift=dataset_config.get("domain_shift", False),
    )

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

    metric = SCDMetric_mh(
        cfg.dataset.num_classes, cfg.dataset.class_names, 
        cfg.dataset.num_areas, cfg.dataset.ignore_index
    )

    directory = cfg.checkpoints.dirpath
    checkpoint_path = Path(directory) / f"{cfg.checkpoints.filename}_in.ckpt"

    print(f"Loading from checkpoint ... {checkpoint_path}")
    model = SitsScdModel_Multihead.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
    model.eval()

    device = 'cuda:1'
    model.to(device)

    img_patches = []
    height = 1024
    patch_size = 128
    num_patches_per_sits = (height // patch_size) ** 2

    with open(os.path.join(dataset_config["path"], "split.json")) as f:
        splits = json.load(f)
    areas = splits[dataset_config["split"]]
    areas.sort()
    miou_dict = {}

    best_heads = {}
    img_size = int(1024/128) ** 2

    for idx_batch, batch in enumerate(tqdm(dataloader)):
        batch["data"] = batch["data"].to(device).type(torch.float32)
        batch["gt"] = batch["gt"].to(device)
        batch["positions"] = batch["positions"].to(device)
        idx_sits = batch["idx"]
        positions = batch["positions"]
        gt = batch["gt"]
        data_unnorm = batch["data_unnorm"]
        patch_ij = batch["patch_ij"]
        _, pred = model.forward_pass(batch)

        if (idx_batch + 1) % (img_size // batch_size) == 0:
            metric.update(pred['pred'][:, :, :, :, :], gt[:, :, :, :])
            output = metric.compute()

            miou_heads = []
            acc_heads = []

            miou_best = -1
            acc_best = -1
            area_best = None

            for area in output:
                miou = output[area]['miou']
                acc = output[area]['acc']

                acc_heads.append(acc)
                miou_heads.append(miou)

                conf_mat = output[area]['conf_matrix']
                if acc > acc_best:
                    acc_best = acc
                    conf_mat_best = conf_mat.copy()
                    area_best = area

                del output[area]['conf_matrix']

            acc_heads = torch.sort(torch.tensor(acc_heads), descending=True)
            miou_heads = torch.sort(torch.tensor(miou_heads), descending=True)

            np.save(f"/home/abhishek/Hackathon-Imagine/HACKATHON-IMAGINE-2024/conf_matrices/{dataset_config['split']}/{areas[idx_sits[0]]}.npy", conf_mat_best)

            file_path = join(cfg.root_dir, f"webpages/multihead_{areas[idx_sits[0]]}.json")
            with open(file_path, "w") as json_file:
                json.dump(output, json_file, indent=4)
            
            print(f"Results successfully saved to {file_path}")

        else:
            metric.update(pred["pred"], gt)


def compute_miou(confusion_matrix):
    den_iou = (
        np.sum(confusion_matrix, axis=1)
        + np.sum(confusion_matrix, axis=0)
        - np.diag(confusion_matrix)
    )
    per_class_iou = np.divide(
        np.diag(confusion_matrix),
        den_iou,
        out=np.zeros_like(den_iou),
        where=den_iou != 0,
    )
    return np.nanmean(per_class_iou) * 100, per_class_iou * 100


def compute_miou_all():
    conf_dir = "/home/abhishek/Hackathon-Imagine/HACKATHON-IMAGINE-2024/conf_matrices/train"
    areas = os.listdir(conf_dir)
    num_classes = 6
    class_names = ["impervi", "agricult", "forest", "wetlands", "soil", "water"]

    conf_mat_all = np.zeros((num_classes, num_classes))
    
    for area in areas:
        conf_mat_all += np.load(os.path.join(conf_dir, area))

    miou, per_class_iou = compute_miou(conf_mat_all)

    output = {"miou": miou}

    for class_id, class_name in enumerate(class_names):
        output[class_name] = per_class_iou[class_id]

    return output

        
if __name__ == "__main__":
    # main()

    output = compute_miou_all()
    print(output)

