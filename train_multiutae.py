import os
from os.path import isfile, join
from shutil import copyfile

import hydra 
from hydra.utils import instantiate
from lightning_fabric.utilities.rank_zero import _get_rank
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor

from pathlib import Path
import wandb

from models.module import (
    SitsScdModel
)

OmegaConf.register_new_resolver("eval", eval)

def wandb_init(cfg):
    directory = cfg.checkpoints.dirpath
    if isfile(join(directory, "wandb_id.txt")):
        with open(join(directory, "wandb_id.txt"), "r") as f:
            wandb_id = f.readline()
    else:
        rank = _get_rank()
        wandb_id = wandb.util.generate_id()
        print(f"Generated wandb id: {wandb_id}")
        if rank == 0 or rank is None:
            with open(join(directory, "wandb_id.txt"), "w") as f:
                f.write(str(wandb_id))

    return wandb_id


def load_model(cfg, dict_config, wandb_id, callbacks):
    directory = Path(cfg.checkpoints.dirpath)
    print("Pretrained and directory", cfg.pretrained_ckpt_filename, directory)
    if (directory / Path("last.ckpt")).exists():
        ckpt_path = directory / Path("last.ckpt")
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        model = SitsScdModel.load_from_checkpoint(ckpt_path, cfg=cfg.model)
        print(f"Loading from checkpoint ... {ckpt_path}")
    else:
        ckpt_path = None
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        log_dict = {
            "model": dict_config["model"],
            "dataset": dict_config["dataset"]
        }
        logger._wandb_init.update({"config": log_dict})
        model = SitsScdModel(cfg.model)
        print(f"No checkpoint loaded, starting training from scratch.")


    trainer, strategy = cfg.trainer, cfg.trainer.strategy
    trainer = instantiate(
        trainer,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks
    )

    return trainer, model, ckpt_path

def project_init(cfg):
    print("Working directory set to {}".format(os.getcwd()))
    directory = cfg.checkpoints.dirpath
    os.makedirs(directory, exist_ok=True)
    if isfile(".hydra/config.yaml"):
        copyfile(".hydra/config.yaml", join(directory, "config.yaml"))


def callback_init(cfg):
    monitor = cfg.checkpoints["monitor"]
    filename = cfg.checkpoints["filename"]
    cfg.checkpoints["monitor"] = monitor + "_out"
    cfg.checkpoints["filename"] = filename + "_out"
    checkpoint_callback_out = instantiate(cfg.checkpoints)
    cfg.checkpoints["monitor"] = monitor + "_in"
    cfg.checkpoints["filename"] = filename + "_in"
    checkpoint_callback_in = instantiate(cfg.checkpoints)
    progress_bar = instantiate(cfg.progress_bar)
    lr_monitor = LearningRateMonitor()
    callbacks = [
        checkpoint_callback_out,
        checkpoint_callback_in,
        progress_bar,
        lr_monitor,
    ]

    return callbacks


def init_datamodule(cfg):
    datamodule = instantiate(cfg.datamodule)
    return datamodule

def hydra_boilerplate(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    callbacks = callback_init(cfg)
    datamodule = init_datamodule(cfg)
    project_init(cfg)
    wandb_id = wandb_init(cfg)
    trainer, model, ckpt_path = load_model(cfg, dict_config, wandb_id, callbacks)
    return trainer, model, datamodule, ckpt_path

@hydra.main(config_path="configs", config_name="mm_config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    trainer, model, datamodule, ckpt_path = hydra_boilerplate(cfg)
    model.datamodule = datamodule
    if cfg.mode == 'train':
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif cfg.mode == 'val':
        trainer.validate(model, datamodule=datamodule)
    elif cfg.mode == 'test':
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()



