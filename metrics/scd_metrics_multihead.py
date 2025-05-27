import torch
from torchmetrics import Metric
import numpy as np

class SCDMetric(Metric):
    """
    Computes the mean intersection-over-union (miou), the binary change score (bc), the semantic change score (sc)
    and the semantic change segmentation score (scs). Additionally provides the accuracy (acc) and mean accuracy (macc)
    for semantic segmentation.

    Args:
        num_classes (int): the number of semantic classes.
        ignore_index (int): ground truth index to ignore in the metrics.
    """

    def __init__(self, num_classes, class_names, num_areas, ignore_index=None, dist_sync_on_step=False):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.num_areas = num_areas

        self.conf_matrices = {f"Area_{i}": np.zeros((num_classes, num_classes)) for i in range(num_areas)}
        # self.conf_matrix_change = np.zeros((2, 2))
        # self.conf_matrix_sc = np.zeros((num_classes, num_classes))

    def update(self, pred, gt):
        """
        Update the confusion matrices
        :param pred: 55 x B x T x H x W
        :param gt: B x T x H x W
        :param area_idx: index of the area
        :return: None
        """
        gt_copy = gt.clone()
        pred_copy = pred.clone()
        for area_idx in range(pred_copy.shape[0]):
            gt = gt_copy.permute(1, 0, 2, 3).reshape(gt_copy.shape[1], -1).long()  # T x N
            pred = pred_copy[area_idx].long().permute(1, 0, 2, 3).reshape(pred_copy.shape[2], -1)  # T x N

            # gt_change = (gt[1:] != gt[:-1]).int()  # (T-1) x N
            # pred_change = (pred[1:] != pred[:-1]).int()  # (T-1) x N, 1 if change, 0 otherwise
            udm_mask = (gt == self.ignore_index).int()  # undefined data mask
            udm_mask = 1 - udm_mask

            gt = gt.cpu().numpy()
            pred = pred.cpu().numpy()
            udm_mask = udm_mask.flatten().cpu().numpy()

            mask = (gt.flatten() >= 0) & (gt.flatten() < self.num_classes) & (udm_mask == 1)
            # mask_change = (gt_change >= 0) & (gt_change < 2) & (mixed_udm_mask == 1)
            # mask_semantic_change = (gt[1:].flatten() >= 0) & (gt[1:].flatten() < self.num_classes) & (mixed_udm_mask == 1) & (gt_change == 1)

            self.conf_matrices[f"Area_{area_idx}"] += np.bincount(
                self.num_classes * gt.flatten()[mask].astype(int) + pred.flatten()[mask],
                minlength=self.num_classes**2,
            ).reshape(self.num_classes, self.num_classes)


    def compute(self):
        output = {}
        for area, conf_mat in self.conf_matrices.items():
            miou, per_class_iou = compute_miou(conf_mat)
            macc = (
                np.nanmean( 
                    np.divide(
                        np.diag(conf_mat),
                        np.sum(conf_mat, axis=1),
                        out=np.zeros_like(np.diag(conf_mat)),
                        where=np.sum(conf_mat, axis=1) != 0,
                    )
                )
                * 100
            )
            area_output = {
                "acc": (np.diag(conf_mat).sum() / (conf_mat.sum() + 1e-7)) * 100,
                "macc": macc,
                "miou": miou,
                "conf_matrix": conf_mat
            }

            for class_id, class_name in enumerate(self.class_names):
                area_output[class_name] = per_class_iou[class_id]
            output[area] = area_output
            self.conf_matrices[area] = np.zeros((self.num_classes, self.num_classes))
        return output
    

def compute_miou(confusion_matrix):
    den_iou = (
        np.sum(confusion_matrix, axis=1)
        + np.sum(confusion_matrix, axis=0)
        - np.diag(confusion_matrix)
    )

    valid_class = np.sum(confusion_matrix, axis=1) != 0

    per_class_iou = np.divide(
        np.diag(confusion_matrix),
        den_iou,
        out=np.zeros_like(den_iou),
        where=den_iou != 0,
    )
    miou = np.nanmean(per_class_iou[valid_class]) * 100 
    return miou, per_class_iou * 100