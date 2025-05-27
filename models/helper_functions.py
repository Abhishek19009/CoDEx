import torch
from torchmetrics import Metric
import numpy as np


class SCDMetric(Metric):
    """
    Computes the mean intersection-over-union (miou) and accuracy (acc) for each batch and area.
    """

    def __init__(self, num_classes, class_names, num_areas, ignore_index=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.num_areas = num_areas

        # Add states to store metrics
        self.add_state("acc_tensor", default=torch.zeros(0, 0), dist_reduce_fx="mean")
        self.add_state("miou_tensor", default=torch.zeros(0, 0), dist_reduce_fx="mean")
        self.conf_matrices = [np.zeros((num_classes, num_classes)) for _ in range(num_areas)]


    def update(self, pred, gt):
        """
        Update confusion matrices for each area based on predictions and ground truth.
        :param pred: Tensor of shape (num_areas, batch_size, T, H, W)
        :param gt: Tensor of shape (batch_size, T, H, W)
        """
        batch_size = gt.shape[0]
        num_areas = pred.shape[0]

        # Initialize accuracy and IoU tensors if not already done
        if self.acc_tensor.numel() == 0:
            self.acc_tensor = torch.zeros((batch_size, num_areas), device=pred.device)
        if self.miou_tensor.numel() == 0:
            self.miou_tensor = torch.zeros((batch_size, num_areas), device=pred.device)


        gt_copy = gt.clone().unsqueeze(dim=1)  # (batch_size, 1, T, H, W)
        pred_copy = pred.clone().permute(1, 0, 2, 3, 4).unsqueeze(dim=2)  # (batch_size, num_areas, T, H, W)

        for batch_idx in range(batch_size):
            for area_idx in range(num_areas):
                # Flatten time and spatial dimensions for confusion matrix calculation
                gt_flat = gt_copy[batch_idx].permute(1, 0, 2, 3).reshape(-1).long()  # (T * H * W,)
                pred_flat = pred_copy[batch_idx, area_idx].permute(1, 0, 2, 3).reshape(-1).long()  # (T * H * W,)

                # Ignore undefined data using the mask
                mask = (gt_flat != self.ignore_index)
                gt_flat = gt_flat[mask]
                pred_flat = pred_flat[mask]

                # Compute confusion matrix for this batch and area
                conf_mat = np.bincount(
                    self.num_classes * gt_flat.cpu().numpy() + pred_flat.cpu().numpy(),
                    minlength=self.num_classes ** 2,
                ).reshape(self.num_classes, self.num_classes)

                # Update the confusion matrix for this area
                self.conf_matrices[area_idx] += conf_mat

                # Compute metrics for this batch and area
                acc = (np.diag(conf_mat).sum() / (conf_mat.sum() + 1e-7)) * 100
                miou, _ = compute_miou(conf_mat)


                # Store in tensors
                self.acc_tensor[batch_idx, area_idx] = acc
                self.miou_tensor[batch_idx, area_idx] = miou

    def compute(self):
        """
        Return batch-wise accuracy and mIoU tensors.
        :return: (acc_tensor, miou_tensor) with shape (batch_size, num_areas)
        """
        return self.acc_tensor, self.miou_tensor

    def reset(self):
        """
        Reset all internal states for a new evaluation.
        """
        self.conf_matrices = [np.zeros((self.num_classes, self.num_classes)) for _ in range(self.num_areas)]
        self.acc_tensor = torch.zeros(0, 0)
        self.miou_tensor = torch.zeros(0, 0)


def compute_miou(confusion_matrix):
    """
    Helper function to compute mIoU from a confusion matrix.
    :param confusion_matrix: Confusion matrix as a numpy array.
    :return: mIoU and per-class IoU.
    """
    den_iou = (
        np.sum(confusion_matrix, axis=1)
        + np.sum(confusion_matrix, axis=0)
        - np.diag(confusion_matrix)
    )

    valid_class = np.sum(confusion_matrix, axis=1) != 0

    per_class_iou = np.divide(
        np.diag(confusion_matrix),
        den_iou,
        out=np.zeros_like(den_iou, dtype=np.float64),
        where=den_iou != 0,
    )
    miou = np.nanmean(per_class_iou[valid_class]) * 100
    return miou, per_class_iou * 100



# import torch
# from torchmetrics import Metric
# import numpy as np

# class SCDMetric(Metric):
#     """
#     Computes the mean intersection-over-union (miou), the binary change score (bc), the semantic change score (sc)
#     and the semantic change segmentation score (scs). Additionally provides the accuracy (acc) and mean accuracy (macc)
#     for semantic segmentation.

#     Args:
#         num_classes (int): the number of semantic classes.
#         ignore_index (int): ground truth index to ignore in the metrics.
#     """

#     def __init__(self, num_classes, class_names, num_areas, ignore_index=None, dist_sync_on_step=False):
#         super().__init__()
#         self.num_classes = num_classes
#         self.class_names = class_names
#         self.ignore_index = ignore_index
#         self.num_areas = num_areas

#         self.conf_matrices = {f"Area_{i}": np.zeros((num_classes, num_classes)) for i in range(num_areas)}
#         # self.conf_matrix_change = np.zeros((2, 2))
#         # self.conf_matrix_sc = np.zeros((num_classes, num_classes))

#     def update_and_compute(self, pred, gt):
#         """
#         Update the confusion matrices
#         :param pred: 55 x B x T x H x W
#         :param gt: B x T x H x W

#         :return: None
#         """
#         gt_copy = gt.clone().unsqueeze(dim=1)
#         pred_copy = pred.clone().permute(1, 0, 2, 3, 4).unsqueeze(dim=2)

#         acc_output = torch.zeros(shape=(gt.shape[0], pred.shape[0]))
#         miou_output = torch.zeros(shape=(gt.shape[0], pred.shape[0]))

#         for batch_idx in range(gt_copy.shape[0]):
#             acc_output_area = []
#             miou_output = []
#             pred_temp = pred_copy[batch_idx]
#             gt_temp = gt_copy[batch_idx]
#             for area_idx in range(pred_copy.shape[1]):
#                 gt = gt_temp.permute(1, 0, 2, 3).reshape(gt_temp.shape[1], -1).long()  # T x N
#                 pred = pred_temp[area_idx].long().permute(1, 0, 2, 3).reshape(pred_temp.shape[2], -1)  # T x N

#                 # gt_change = (gt[1:] != gt[:-1]).int()  # (T-1) x N
#                 # pred_change = (pred[1:] != pred[:-1]).int()  # (T-1) x N, 1 if change, 0 otherwise
#                 udm_mask = (gt == self.ignore_index).int()  # undefined data mask
#                 udm_mask = 1 - udm_mask

#                 gt = gt.cpu().numpy()
#                 pred = pred.cpu().numpy()
#                 udm_mask = udm_mask.flatten().cpu().numpy()

#                 mask = (gt.flatten() >= 0) & (gt.flatten() < self.num_classes) & (udm_mask == 1)
#                 # mask_change = (gt_change >= 0) & (gt_change < 2) & (mixed_udm_mask == 1)
#                 # mask_semantic_change = (gt[1:].flatten() >= 0) & (gt[1:].flatten() < self.num_classes) & (mixed_udm_mask == 1) & (gt_change == 1)

#                 conf_mat += np.bincount(
#                     self.num_classes * gt.flatten()[mask].astype(int) + pred.flatten()[mask],
#                     minlength=self.num_classes**2,
#                 ).reshape(self.num_classes, self.num_classes)

#                 self.conf_matrices[f"Area_{area_idx}"] = conf_mat
            
#                 miou, per_class_iou = compute_miou(self.conf_matrices[f"Area_{area_idx}"])
#                 macc = (
#                 np.nanmean( 
#                     np.divide(
#                         np.diag(conf_mat),
#                         np.sum(conf_mat, axis=1),
#                         out=np.zeros_like(np.diag(conf_mat)),
#                         where=np.sum(conf_mat, axis=1) != 0,
#                     )
#                     )
#                     * 100
#                 )

#                 area_output = {
#                     "acc": (np.diag(conf_mat).sum() / (conf_mat.sum() + 1e-7)) * 100,
#                     "macc": macc,
#                     "miou": miou
#                 }

#                 acc_output[batch_idx][area_idx] = area_output['acc']
#                 miou_output[batch_idx][area_idx] = miou_output['miou']

#                 # for class_id, class_name in enumerate(self.class_names):
#                 #     area_output[class_name] = per_class_iou[class_id]

#         return acc_output, miou_output
    

# def compute_miou(confusion_matrix):
#     den_iou = (
#         np.sum(confusion_matrix, axis=1)
#         + np.sum(confusion_matrix, axis=0)
#         - np.diag(confusion_matrix)
#     )

#     valid_class = np.sum(confusion_matrix, axis=1) != 0

#     per_class_iou = np.divide(
#         np.diag(confusion_matrix),
#         den_iou,
#         out=np.zeros_like(den_iou),
#         where=den_iou != 0,
#     )
#     miou = np.nanmean(per_class_iou[valid_class]) * 100 
#     return miou, per_class_iou * 100