from random import randint
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from typing import Union
from torchvision.transforms import ColorJitter


def random_flipv(data, label):
    flip = randint(0, 1)
    if flip:
        if label is not None:
            return transforms.functional.vflip(data), transforms.functional.vflip(label)
        else:
            return transforms.functional.vflip(data), None
    else:
        return data, label


def random_fliph(data, label):
    flip = randint(0, 1)
    if flip:
        if label is not None:
            return transforms.functional.hflip(data), transforms.functional.hflip(label)
        else:
            return transforms.functional.hflip(data), None
    else:
        return data, label


def random_rotate(data, label):
    rotate = randint(0, 3)
    if label is not None:
        return torch.rot90(data, rotate, (2, 3)), torch.rot90(label, rotate, (1, 2))
    else:
        return torch.rot90(data, rotate, (2, 3)), None


def random_crop(data, label, img_size, true_size):
    i = randint(0, true_size - img_size - 1)
    j = randint(
        0, true_size - (1 + 2 * int(i >= true_size - 3 * img_size)) * img_size - 1
    )
    data = transforms.functional.crop(data, i, j, img_size, img_size)
    if label is not None:
        label = transforms.functional.crop(label, i, j, img_size, img_size)
        return data, label
    else:
        return data


def random_resize_crop(data, label):
    height, width = data.shape[-2:]
    top, left = randint(0, height // 2 - 1), randint(0, width // 2 - 1)
    new_height, new_width = randint(height // 2, height - top), randint(
        width // 2, width - left
    )
    data = transforms.functional.resized_crop(
        data,
        top=top,
        left=left,
        height=new_height,
        width=new_width,
        size=[height, width],
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    if label is not None:
        label = transforms.functional.resized_crop(
            label,
            top=top,
            left=left,
            height=new_height,
            width=new_width,
            size=[height, width],
            interpolation=transforms.InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )
        return data, label
    else:
        return data, None


class ClassMix:
    def __init__(
        self,
        alpha: float = 0.5,
        do_normalize: bool = False,
        random_n_samples: bool = True,
    ) -> None:
        """
        Implementation of the ClassMix augmentation technique from https://arxiv.org/abs/2007.07936
        args:
            alpha: float, is defined by |c| / |C| (see line 5 of Algorithm 1 in the paper)
        """
        self.alpha = alpha
        self.do_normalize = do_normalize
        self.random_n_samples = random_n_samples

    def generate_mask(self, labels: torch.Tensor) -> Union[torch.Tensor, None]:
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        if self.random_n_samples:
            n_samples = randint(0, int(self.alpha * n_classes))
        else:
            n_samples = int(self.alpha * n_classes)
        if n_samples == 0:
            return None
        sampled_classes = torch.randperm(n_classes, device=labels.device)[:n_samples]
        mask = torch.zeros_like(labels, device=labels.device)
        mask[torch.isin(labels, sampled_classes)] = 1
        if len(torch.unique(mask)) == 1:
            return None
        return mask

    def __call__(self, data: list, data_unnorm: list, labels: list) -> tuple:
        assert len(data) == len(labels) == 2, "Data and labels must have length 2"
        mask = self.generate_mask(labels[0])
        if mask is None:
            return data[0], data_unnorm[0], labels[0], mask
        else:
            labels = mask * labels[0] + (1 - mask) * labels[1]
            mask = mask.unsqueeze(2).expand_as(data[0])
            data = mask * data[0] + (1 - mask) * data[1]
            data_unnorm = mask * data_unnorm[0] + (1 - mask) * data_unnorm[1]
            if self.do_normalize:
                data = F.normalize(data, dim=2)
            return data, data_unnorm, labels, mask


class ColorJitterCustom(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, infra=False):
        self.jitter_transform = ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.infra = infra

    def __call__(self, sample):
        assert sample.shape[1] == 4
        sample = sample / 255

        sample = sample.type(torch.float32)
        rg_img = self.jitter_transform(sample[:, :3])
        if self.infra:
            infra_img = self.jitter_transform(sample[:, 3:4])
        else:
            infra_img = sample[:, 3:4]
        sample_jittered = torch.concat([rg_img, infra_img], axis=1) * 255
        return sample_jittered.type(torch.float16)
