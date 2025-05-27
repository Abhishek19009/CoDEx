import time

import torch
import torch.utils
import torch.utils.data
import pytorch_lightning as L
import torch
# from styleaug import StyleAugmentor
from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader
from .transforms import ClassMix
import matplotlib.pyplot as plt
import numpy as np


class ImageDataModule_Muds(L.LightningDataModule):
    def __init__(
        self,
        all_dataset,
        train_dataset,
        train_dataset_infer,
        val_dataset_domain_shift,
        val_dataset_no_domain_shift,
        test_dataset_domain_shift,
        test_dataset_no_domain_shift,
        global_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        self._builders = {
            "all": all_dataset,
            "train": train_dataset,
            "train_dataset_infer": train_dataset_infer,
            "val_domain_shift": val_dataset_domain_shift,
            "val_no_domain_shift": val_dataset_no_domain_shift,
            "test_domain_shift": test_dataset_domain_shift,
            "test_no_domain_shift": test_dataset_no_domain_shift,
        }
        self.num_workers = num_workers
        self.batch_size = global_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")

    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        else:
            return self._builders["train"]().num_classes

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"]()
            self.val_dataset_domain_shift = self._builders["val_domain_shift"]()
            self.val_dataset_no_domain_shift = self._builders["val_no_domain_shift"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Out-of-domain val dataset size: {len(self.val_dataset_domain_shift)}")
            print(f"In-domain val dataset size: {len(self.val_dataset_no_domain_shift)}")
        else:
            self.val_dataset_domain_shift = self._builders["val_domain_shift"]()
            self.val_dataset_no_domain_shift = self._builders["val_no_domain_shift"]()
            self.test_dataset_domain_shift = self._builders["test_domain_shift"]()
            self.test_dataset_no_domain_shift = self._builders["test_no_domain_shift"]()
            print(f"Out-of-domain test dataset size: {len(self.test_dataset_domain_shift)}")
            print(f"In-domain test dataset size: {len(self.test_dataset_no_domain_shift)}")
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return [DataLoader(
            self.val_dataset_domain_shift,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_domain_shift.collate_fn,
        ), DataLoader(
            self.val_dataset_no_domain_shift,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_no_domain_shift.collate_fn,
        )
        ]

    def test_dataloader(self):
        return [DataLoader(
            self.test_dataset_domain_shift,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset_domain_shift.collate_fn,
        ), DataLoader(
            self.test_dataset_no_domain_shift,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset_no_domain_shift.collate_fn,
        )
        ]



class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        all_dataset,
        train_dataset,
        train_dataset_infer,
        val_dataset_domain_shift,
        val_dataset_no_domain_shift,
        test_dataset_domain_shift,
        test_dataset_no_domain_shift,
        global_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
        transforms={},
    ):
        super().__init__()
        self._builders = {
            "all": all_dataset,
            "train": train_dataset,
            "train_dataset_infer": train_dataset_infer,
            "val_domain_shift": val_dataset_domain_shift,
            "val_no_domain_shift": val_dataset_no_domain_shift,
            "test_domain_shift": test_dataset_domain_shift,
            "test_no_domain_shift": test_dataset_no_domain_shift,
        }
        self.num_workers = num_workers
        self.batch_size = global_batch_size // (num_nodes * num_devices)

        
        self.transforms_gpu = transforms["gpu"]
        self.transforms_cpu = transforms["cpu"]
        print(f"Each GPU will receive {self.batch_size} images")
        if not torch.cuda.is_available():
            print("GPU not available, style-augmentation is disable (only support gpu)")
            self.transforms_gpu["style-augmentation"]["enable"] = False

        if self.transforms_gpu["style-augmentation"]["enable"]:
            self.style_augmentor = StyleAugmentor()
            # TODO: should load style statistic associated with our dataset

        if self.transforms_gpu["i2i"]["enable"]:
            self.gen_type = self.transforms_gpu["i2i"]["gen_type"]
            self.gen = load_generator(self.gen_type)
            print("Loaded generator", self.gen_type)

            
    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        else:
            return self._builders["train"]().num_classes

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"](transforms=self.transforms_cpu)
            self.val_dataset_shift = self._builders["val_domain_shift"]()
            self.val_dataset_no_shift = self._builders["val_no_domain_shift"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset with domain shift size: {len(self.val_dataset_shift)}")
            print(
                f"Val dataset without domain shift size: {len(self.val_dataset_no_shift)}"
            )

            if self.transforms_gpu["contrastive"]["trainvaltest"]:
                self.all_dataset = self._builders["all"](transforms=self.transforms_cpu)
        else:
            self.val_dataset_shift = self._builders["val_domain_shift"]()
            self.val_dataset_no_shift = self._builders["val_no_domain_shift"]()
            self.test_dataset_shift = self._builders["test_domain_shift"]()
            self.test_dataset_no_shif = self._builders["test_no_domain_shift"]()
            print(
                f"Test dataset with domain shift size: {len(self.test_dataset_shift)}"
            )
            print(
                f"Test dataset without domain shiftsize: {len(self.test_dataset_no_shif)}"
            )
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        if self.transforms_gpu["classmix"]["enable"]:
            loaders = {
                "A": DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True,
                    num_workers=self.num_workers,
                    collate_fn=self.train_dataset.collate_fn,
                ),
                "B": DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True,
                    num_workers=self.num_workers,
                    collate_fn=self.train_dataset.collate_fn,
                ),
            }
            return CombinedLoader(loaders, self.transforms_gpu["classmix"]["mode"])
        elif self.transforms_gpu["contrastive"]["trainvaltest"]:
            return [
                DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True,
                    num_workers=self.num_workers,
                    collate_fn=self.train_dataset.collate_fn,
                ),
                DataLoader(
                    self.all_dataset,
                    batch_size=self.transforms_gpu["contrastive"][
                        "contrastive_batch_size"
                    ],
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True,
                    num_workers=self.num_workers,
                    collate_fn=self.train_dataset.collate_fn,
                ),
            ]
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=False,
                drop_last=True,
                num_workers=self.num_workers,
                collate_fn=self.train_dataset.collate_fn,
            )

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_dataset_shift,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                collate_fn=self.val_dataset_shift.collate_fn,
            ),
            DataLoader(
                self.val_dataset_no_shift,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                collate_fn=self.val_dataset_no_shift.collate_fn,
            ),
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                self.test_dataset_shift,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                collate_fn=self.test_dataset_shift.collate_fn,
            ),
            DataLoader(
                self.test_dataset_no_shif,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                collate_fn=self.test_dataset_no_shif.collate_fn,
            ),
        ]

    def on_after_batch_transfer(
        self, batch: torch.Tensor, dataloader_idx: int
    ) -> torch.Tensor:
        if self.trainer.training:
            # apply augmentation
            if self.transforms_gpu["classmix"]["enable"]:
                class_mix_transform = ClassMix(
                    alpha=self.transforms_gpu["classmix"]["alpha"],
                    do_normalize=self.transforms_gpu["classmix"]["do_normalize"],
                    random_n_samples=self.transforms_gpu["classmix"][
                        "random_n_samples"
                    ],
                )
                data, data_unnorm, label, _ = class_mix_transform(
                    [batch["A"]["data"], batch["B"]["data"]],
                    [batch["A"]["data_unnorm"], batch["B"]["data_unnorm"]],
                    [batch["A"]["gt"], batch["B"]["gt"]],
                )
                assert self.transforms_gpu["classmix"]["init_length"] == data.shape[1]
                positions = torch.sort(
                    torch.randperm(data.shape[1])[
                        : self.transforms_gpu["classmix"]["train_length"]
                    ]
                )[0]
                batch.update(
                    {
                        "idx": batch["A"]["idx"],
                        "data": data[:, positions],
                        "data_unnorm": data_unnorm[:, positions],
                        "gt": label[:, positions],
                        "positions": batch["A"]["positions"][:, positions],
                    }
                )
                del batch["A"]
                del batch["B"]

            if self.transforms_gpu["style-augmentation"]["enable"]:
                shape = batch["data"].shape
                # get style embedding and repeate it along the temporal dimention
                embedding = self.style_augmentor.sample_embedding(shape[0])
                embedding = embedding.unsqueeze(1).expand(shape[0], shape[1], 100)

                # fuse temporal dimension in the batch
                data = (
                    batch["data"]
                    .reshape(-1, shape[2], shape[3], shape[4])
                    .to(dtype=torch.float32)
                )
                data_rgb = data[:, :3, :, :]
                data_ir = data[:, 3:, :, :].expand(-1, 3, -1, -1)
                embedding = embedding.reshape(-1, 100)

                data_rgb = self.style_augmentor(
                    data_rgb,
                    embedding=embedding,
                    alpha=self.transforms_gpu["style-augmentation"]["alpha"],
                )
                data_ir = self.style_augmentor(
                    data_ir,
                    embedding=embedding,
                    alpha=self.transforms_gpu["style-augmentation"]["alpha"],
                ).mean(dim=1, keepdim=True)
                data = torch.cat([data_rgb, data_ir], dim=1)
                batch["data"] = data.reshape(shape).to(dtype=torch.float16)


            if self.transforms_gpu["i2i"]["enable"]:
                # Input is batch['data'], dim: (bs, seq_len, num_channels, H, W)
                data = batch["data"]
                self.gen.to(data.device)
                bs, seq_len, num_channels, H, W = data.size()
                data_reshaped = data.view(bs * seq_len, num_channels, H, W).to(dtype=torch.float32)

                # labels_reshaped = labels.view(bs * seq_len, H, W)
                # data_reshaped = minmax_channel(data_reshaped)
                # visualize_img(data_reshaped)
                # visualize_labels(labels_reshaped)

                # time.sleep(5)
                if self.gen_type == "cycada":
                    gen_output_reshaped = self.gen(data_reshaped[:, :3, :, :])
                    gen_output_reshaped = torch.cat((gen_output_reshaped, data_reshaped[:, [3], :, :]), dim=1)
                gen_output = gen_output_reshaped.view(bs, seq_len, num_channels, H, W)

                batch["data"] = gen_output.to(dtype=torch.float16)
                

        return super().on_after_batch_transfer(batch, dataloader_idx)

def visualize_img(gen_output):
    numpy_output = gen_output.detach().cpu().numpy()
    numpy_output = (numpy_output + 1) / 2
    numpy_output = numpy_output[0, :3, :, :].T
    plt.imsave("/home/abhishek/Hackathon-Imagine/HACKATHON-IMAGINE-2024/gen_output.png", numpy_output)

def visualize_labels(labels):
    labels = labels[0].detach().cpu().numpy()
    plt.imsave("/home/abhishek/Hackathon-Imagine/HACKATHON-IMAGINE-2024/label_output.png", labels)

def minmax_channel(image):
    # print(image.shape)
    min_max_img = torch.zeros_like(image)
    for i in range(image.shape[1]):
        channel = image[:, i, :, :]
        channel_min = channel.min()
        channel_max = channel.max()
        min_max_img[:, i, :, :] = (channel - channel_min) / (channel_max - channel_min)
    image = min_max_img

    return image

class ImageDataModule_train_test(L.LightningDataModule):
    def __init__(
        self,
        all_dataset,
        train_dataset,
        val_dataset_domain_shift,
        val_dataset_no_domain_shift,
        test_dataset_domain_shift,
        test_dataset_no_domain_shift,
        global_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
        transforms={},
    ):
        super().__init__()
        self._builders = {
            "all_dataset": all_dataset,
            "train": train_dataset,
            "val_domain_shift": val_dataset_domain_shift,
            "val_no_domain_shift": val_dataset_no_domain_shift,
            "test_domain_shift": test_dataset_domain_shift,
            "test_no_domain_shift": test_dataset_no_domain_shift,
        }
        self.num_workers = num_workers
        self.batch_size = global_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")
        self.transforms_gpu = transforms["gpu"]
        self.transforms_cpu = transforms["cpu"]
        print(f"Each GPU will receive {self.batch_size} images")
        if not torch.cuda.is_available():
            print("GPU not available, style-augmentation is disable (only support gpu)")
            self.transforms_gpu["style-augmentation"]["enable"] = False

        if self.transforms_gpu["style-augmentation"]["enable"]:
            self.style_augmentor = StyleAugmentor()

    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        else:
            return self._builders["train"]().num_classes

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"](transforms=self.transforms_cpu)
            self.val_dataset_shift = self._builders["val_domain_shift"]()
            self.val_dataset_no_shift = self._builders["val_no_domain_shift"]()
            self.test_dataset_shift = self._builders["test_domain_shift"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset with domain shift size: {len(self.val_dataset_shift)}")
            print(
                f"Val dataset without domain shift size: {len(self.val_dataset_no_shift)}"
            )
        else:
            self.test_dataset_shift = self._builders["test_domain_shift"]()
            self.test_dataset_no_shif = self._builders["test_no_domain_shift"]()
            print(
                f"Test dataset with domain shift size: {len(self.test_dataset_shift)}"
            )
            print(
                f"Test dataset without domain shiftsize: {len(self.test_dataset_no_shif)}"
            )
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_dataset_shift,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                collate_fn=self.val_dataset_shift.collate_fn,
            ),
            DataLoader(
                self.val_dataset_no_shift,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                collate_fn=self.val_dataset_no_shift.collate_fn,
            ),
        ]

    def train_dataloader(self):
        train = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )

        test = DataLoader(
            self.test_dataset_shift,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset_shift.collate_fn,
        )

        return CombinedLoader(
            {
                "train": train,
                "test": test,
            },
            "max_size_cycle",
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.test_dataset_shift,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                collate_fn=self.test_dataset_shift.collate_fn,
            ),
            DataLoader(
                self.test_dataset_no_shif,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                collate_fn=self.test_dataset_no_shif.collate_fn,
            ),
        ]
