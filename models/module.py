import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
import torch.nn.functional as F
import numpy as np
from .helper_functions import SCDMetric
import wandb


class HeadSelectorModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        self.loss = instantiate(cfg.loss.instance)
        self.val_metrics = {
            "out": instantiate(cfg.val_metrics),
            "in": instantiate(cfg.val_metrics),
        }

        self.test_metrics = {
            "out": instantiate(cfg.test_metrics),
            "in": instantiate(cfg.test_metrics)
        }
        self.domain_dict = {0: "out", 1: "in"}
        
        self.mh_metric = SCDMetric(
            cfg.mhmetrics.num_classes, cfg.mhmetrics.class_names, 
            cfg.mhmetrics.num_areas, cfg.mhmetrics.ignore_index
        )

    def get_hs_target(self, pred, batch):
        logits_heads = pred['logits_heads']
        gt = batch['gt']

        logits_preds = torch.argmax(logits_heads, dim=3)
        self.mh_metric.update(logits_preds, gt)

        acc_perf, miou_perf = self.mh_metric.compute()
        self.mh_metric.reset()

        batch['acc_perf'] = acc_perf / 100
        batch['miou_perf'] = miou_perf / 100

        return pred, batch

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)

        # Get hs targets
        pred, batch = self.get_hs_target(pred, batch)

        # Check for NaN values in acc_perf or miou_perf
        if torch.isnan(batch['acc_perf']).any() or torch.isnan(batch['miou_perf']).any():
            self.log(
                "train/skipped_nan_step",
                1,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
            # Skip this training step
            batch['acc_perf'] = torch.zeros(size=(batch['gt'].shape[0], pred['logits_heads'].shape[0])).to(batch['gt'].device)  
            batch['miou_perf'] = torch.zeros(size=(batch['gt'].shape[0], pred['logits_heads'].shape[0])).to(batch['gt'].device) 

      
        loss = self.loss(pred, batch, average=True)
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True
            )
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)

        # Get hs targets
        pred, batch = self.get_hs_target(pred, batch)

        # Combined pmoh preds
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics[self.domain_dict[dataloader_idx]].update(
            pred["pred"], batch["gt"]
        )
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        for dataloader_idx in ["out", "in"]:
            metrics = self.val_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                print(f"val/{metric_name}_{dataloader_idx}")
                self.log(
                    f"val/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True
                )


    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)

        # Get hs targets
        pred, batch = self.get_hs_target(pred, batch)

        # Combined pmoh preds
        pred['pred'] = torch.argmax(pred["logits"], dim=2)

        self.test_metrics[self.domain_dict[dataloader_idx]].update(
            pred['pred'], batch['gt']
        )
        

    @torch.no_grad()
    def forward_pass(self, batch):
        pred = self.model(batch)

        # Get hs targets
        pred, batch = self.get_hs_target(pred, batch)

        # Combined pmoh preds
        pred['pred'] = torch.argmax(pred['logits'], dim=2)

        return batch, pred
    
    def on_test_epoch_end(self):
        for dataloader_idx in ["out", "in"]:
            metrics = self.test_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"test/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True
                )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ] 
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )

        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.model.parameters())
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)


class SitsScdModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        self.loss = instantiate(cfg.loss.instance)
        self.ignore_index = self.loss.ignore_index
        self.val_metrics = {
            "out": instantiate(cfg.val_metrics),
            "in": instantiate(cfg.val_metrics),
        }
        self.test_metrics = {
            "out": instantiate(cfg.test_metrics),
            "in": instantiate(cfg.test_metrics),
        }

        self.domain_dict = {0: "out", 1: "in"}
        if "features" in cfg:
            self.return_features = cfg.features
        else:
            self.return_features = None

        if "coral_penalty" in cfg and cfg.coral_penalty:
            self.coral_penalty = cfg.coral_penalty
        else:
            self.coral_penalty = False

    def training_step(self, batch, batch_idx):
        if self.coral_penalty:
            pred_train = self.model(
                batch['train'], return_features=self.return_features
            )
            pred_test = self.model(
                batch['test'], return_features=self.return_features, features_only=True
            )
            loss = self.loss(pred_train, batch["train"], pred_test, average=True)

        else:
            pred = self.model(batch)
            loss = self.loss(pred, batch, average=True)
            
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True
            )

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics[self.domain_dict[dataloader_idx]].update(
            pred["pred"], batch["gt"]
        )
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        for dataloader_idx in ["out", "in"]:
            metrics = self.val_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"val/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        self.val_metrics[self.domain_dict[dataloader_idx]].update(
            pred["pred"], batch["gt"]
        )

    @torch.no_grad()
    def forward_pass(self, batch):
        pred = self.model(batch)
        pred['pred'] = torch.argmax(pred["logits"], dim=2)
        return batch, pred
    
    def on_test_epoch_end(self):
        for dataloader_idx in ["out", "in"]:
            metrics = self.test_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"test/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.model.parameters())
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)


class SitsScdModel_d3g(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        self.loss = instantiate(cfg.loss.instance)
        self.ignore_index = self.loss.ignore_index
        self.val_metrics = {
            "out": instantiate(cfg.val_metrics),
            "in": instantiate(cfg.val_metrics),
        }
        self.test_metrics = {
            "out": instantiate(cfg.test_metrics),
            "in": instantiate(cfg.test_metrics),
        }

        self.domain_dict = {0: "out", 1: "in"}
        if "features" in cfg:
            self.return_features = cfg.features
        else:
            self.return_features = None

        for name, param in self.model.named_parameters():
            if name == "all_coord_weights":
                print(f"{name} - Requires Grad: {param.requires_grad}")  # Should be True
                print(f"{name} - Grad before training: {param.grad}")    # Should not be None after backward()



    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss(pred, batch, average=True)

        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True
            )

        # Log all_coord_weights as a heatmap in W&B
        if batch_idx % 100 == 0:  # Log every 100 steps (adjust as needed)
            self.log_coord_weights(batch_idx)

        # for name, param in self.model.named_parameters():
        #     if name == "all_coord_weights":
        #         print(f"{name} - Requires Grad: {param.requires_grad}")
        #         print(f"{name} - Grad after backward: {param.grad}")  # Should NOT be None

        return loss
    
    def log_coord_weights(self, step):
        weights = self.model.all_coord_weights.detach().cpu().numpy()  # Convert to NumPy
        if self.model.all_coord_weights.grad is not None:
            print(self.model.all_coord_weights.grad.abs().mean().item())
        print(weights.min(), weights.max())
        self._plot_weights(weights)
        

    def _plot_weights(self, weights):
        """ Creates a Matplotlib figure for the heatmap. """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(6, 5))
        plt.imshow(weights, cmap="magma", aspect="auto")  # Try "cividis", "plasma", or "inferno" too
        plt.colorbar(label="Weight Value")
        plt.title("Affinity Matrix")
        plt.xlabel("Domain Index")
        plt.ylabel("Domain Index")
        
        plt.savefig('./affinity_denet_learned.png', format='png')
        
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["out_logits_s"], dim=2)
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics[self.domain_dict[dataloader_idx]].update(
            pred["pred"], batch["gt"]
        )
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

        if batch_idx % 100 == 0:  # Log every 100 steps (adjust as needed)
            self.log_coord_weights(batch_idx)

        

    def on_validation_epoch_end(self):
        for dataloader_idx in ["out", "in"]:
            metrics = self.val_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"val/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["out_logits_s"], dim=2)
        self.val_metrics[self.domain_dict[dataloader_idx]].update(
            pred["pred"], batch["gt"]
        )

    @torch.no_grad()
    def forward_pass(self, batch):
        pred = self.model(batch)
        pred['pred'] = torch.argmax(pred["out_logits_s"], dim=2)
        return batch, pred
    
    def on_test_epoch_end(self):
        for dataloader_idx in ["out", "in"]:
            metrics = self.test_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"test/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                },

                {  
                    "params": [p for n, p in self.model.named_parameters() if "all_coord_weights" in n],
                    "weight_decay": 0.0,  # Ensure NO weight decay
                    "lr": 1e-2  # Boost learning rate for faster updates
                }
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.model.parameters())
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)



class SitsScdModel_Multihead(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        self.loss = instantiate(cfg.loss.instance)
        self.ignore_index = self.loss.ignore_index
        self.val_metrics = {
            "out": instantiate(cfg.val_metrics),
            "in": instantiate(cfg.val_metrics),
        }
        self.test_metrics = {
            "out": instantiate(cfg.test_metrics),
            "in": instantiate(cfg.test_metrics),
        }
        self.domain_dict = {0: "out", 1: "in"}

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss(pred, batch, average=True)
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics[self.domain_dict[dataloader_idx]].update(
            pred["pred"], batch["gt"]
        )
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        for dataloader_idx in ["out", "in"]:
            metrics = self.val_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"val/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        self.test_metrics[self.domain_dict[dataloader_idx]].update(
            pred["pred"], batch["gt"]
        )

    @torch.no_grad()
    def forward_pass(self, batch):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=3)
        return batch, pred

    def on_test_epoch_end(self):
        for dataloader_idx in ["out", "in"]:
            metrics = self.test_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"test/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.model.parameters())
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)



def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result