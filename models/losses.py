"""
Code from
https://github.com/clcarwin/focal_loss_pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metrics.scd_metrics_multihead import SCDMetric
import math

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, cons_lambda=0.5, alpha=None, size_average=True, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "logits": torch.Tensor BxTxKxHxW
            y: dict that contains "gt": torch.Tensor BxTxHxW
        Returns:
            torch.Tensor: Focal loss between x and y: torch.Tensor([B])
        """
        prediction = x["logits"]
        target = y["gt"]
        
        if prediction.dim() > 2:
            prediction = prediction.contiguous().view(
                prediction.size(0) * prediction.size(1), prediction.size(2), -1
            )  # B,T,K,H,W => B*T,K,H*W
            prediction = prediction.transpose(1, 2)  # N,K,H*W => N,H*W,K
            prediction = prediction.contiguous().view(
                -1, prediction.size(2)
            )  # N,H*W,K => N*H*W,K
        target = target.contiguous().flatten()

        if self.ignore_index is not None:
            prediction = prediction[target != self.ignore_index]
            target = target[target != self.ignore_index]

        target = target[:, None]
        logpt = F.log_softmax(prediction, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != prediction.data.type():
                self.alpha = self.alpha.type_as(prediction.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class FocalLosswME(nn.Module):
    def __init__(self, gamma=0, cons_lambda=0.5, alpha=None, hs_lamdba=0.9, size_average=True, ignore_index=None):
        super(FocalLosswME, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.hs_lambda = hs_lamdba
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average
        self.hs_criterion = nn.L1Loss()


    def forward(self, x, y):
        """
        Args:
            x: dict that contains "logits": torch.Tensor BxTxKxHxW
            y: dict that contains "gt": torch.Tensor BxTxHxW
        Returns:
            torch.Tensor: Focal loss between x and y: torch.Tensor([B])
        """
        # hs logits
        hs_logits = x["hs_logits"]
        acc_perf = y['acc_perf']
        miou_perf = y['miou_perf']

        hs_loss = self.hs_criterion(hs_logits, acc_perf)

        # pmoh logits
        prediction = x["logits"]
        target = y["gt"]

        if prediction.dim() > 2:
            prediction = prediction.contiguous().view(
                prediction.size(0) * prediction.size(1), prediction.size(2), -1
            )  # B,T,K,H,W => B*T,K,H*W
            prediction = prediction.transpose(1, 2)  # N,K,H*W => N,H*W,K
            prediction = prediction.contiguous().view(
                -1, prediction.size(2)
            )  # N,H*W,K => N*H*W,K
        target = target.contiguous().flatten()

        if self.ignore_index is not None:
            prediction = prediction[target != self.ignore_index]
            target = target[target != self.ignore_index]

        target = target[:, None]
        logpt = F.log_softmax(prediction, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != prediction.data.type():
                self.alpha = self.alpha.type_as(prediction.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        loss = hs_loss + self.hs_lambda * loss
        
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        

class FocalLossd3g(nn.Module):
    def __init__(self, gamma=0, cons_lambda=0.5, alpha=None, size_average=True, ignore_index=None):
        super(FocalLossd3g, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average
        self.cons_lambda = cons_lambda

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "logits": torch.Tensor BxTxKxHxW
            y: dict that contains "gt": torch.Tensor BxTxHxW
        Returns:
            torch.Tensor: Focal loss between x and y: torch.Tensor([B])
        """
        out_logits_s = x["out_logits_s"]
        out_logits_c = x["out_logits_c"]
        losses = []

        for prediction in [out_logits_s, out_logits_c]:
            target = y["gt"]
            if prediction.dim() > 2:
                prediction = prediction.contiguous().view(
                    prediction.size(0) * prediction.size(1), prediction.size(2), -1
                )  # B,T,K,H,W => B*T,K,H*W
                prediction = prediction.transpose(1, 2)  # N,K,H*W => N,H*W,K
                prediction = prediction.contiguous().view(
                    -1, prediction.size(2)
                )  # N,H*W,K => N*H*W,K
            target = target.contiguous().flatten()

            if self.ignore_index is not None:
                prediction = prediction[target != self.ignore_index]
                target = target[target != self.ignore_index]

            target = target[:, None]
            logpt = F.log_softmax(prediction, dim=-1)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            if self.alpha is not None:
                if self.alpha.type() != prediction.data.type():
                    self.alpha = self.alpha.type_as(prediction.data)
                at = self.alpha.gather(0, target.data.view(-1))
                logpt = logpt * Variable(at)

            loss = -1 * (1 - pt) ** self.gamma * logpt
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()

            losses.append(loss)

        # lamdba parameter set to 1.0
        loss = losses[0] + self.cons_lambda * losses[1]

        return loss


LOSSES = {
    "focal": FocalLoss,
    "focald3g": FocalLossd3g,
    "focalwme": FocalLosswME
}

AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}

class Losses(nn.Module):
    """The Losses meta-object that can take a mix of losses."""

    def __init__(self, mix={}, cons_lambda=1.0, ignore_index=None):
        """Initializes the Losses object.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        super(Losses, self).__init__()
        assert len(mix)
        self.ignore_index = ignore_index
        self.cons_lambda = cons_lambda
        self.init_losses(mix)

    def init_losses(self, mix):
        """Initializes the losses.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        self.loss = {}
        for m, v in mix.items():
            m = m.lower()
            try:
                self.loss[m] = (LOSSES[m](ignore_index=self.ignore_index, cons_lambda=self.cons_lambda), v)
            except KeyError:
                raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")
            

    # from https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/examples/algorithms/deepCORAL.py#L55
    def coral_penalty(self, x, y):
        if x.dim() > 2:
            # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
            # we flatten to Tensors of size (*, feature dimensionality)
            # x = x.view(-1, x.size(-1))
            # y = y.view(-1, y.size(-1))

            # reshape BxTxCxHxW to (BxTxHxW)xC using einops
            x = x.permute(0, 1, 3, 4, 2)
            x = x.reshape(-1, x.size(-1))
            y = y.permute(0, 1, 3, 4, 2)
            y = y.reshape(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y

        div_x = math.sqrt((len(x) - 1))
        div_y = math.sqrt((len(y) - 1))

        cent_x = cent_x / div_x
        cent_y = cent_y / div_y

        cova_x = cent_x.t() @ cent_x
        cova_y = cent_y.t() @ cent_y

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff
    

    def forward(self, *args, x_test=None, average=True, alpha=1):
        """Computes the losses.
        Args:
            args: passed to all losses
            average (bool): whether to average the losses or not
        Returns:
            dict: dictionary with losses
        """
        losses = {n: AVERAGE[average](f(*args)) for n, (f, _) in self.loss.items()}
        losses["loss"] = sum([losses[n] * w for n, (_, w) in self.loss.items()])

        if x_test is not None:
            losses["coral_penalty"] = self.coral_penalty(
                x["features"], x_test["features"]
            )
            losses["loss"] += alpha * losses["coral_penalty"]

        return losses
