import torch
import random
from math import ceil


class LocationSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()

        self.batch_size = batch_size

        self.sits_ids = dataset.sits_ids
        self.split = dataset.split
        self.is_domain_shift = dataset.domain_shift

        print(
            f"Instantiating LocationSampler for split {self.split} with domain shif {self.is_domain_shift}"
        )

        if self.split == "train":
            self.num_patches_per_sits = (dataset.true_size // dataset.img_size) ** 2 - 4
        elif self.is_domain_shift:
            self.num_patches_per_sits = (dataset.true_size // dataset.img_size) ** 2
        else:
            self.num_patches_per_sits = 2

    def __iter__(self):
        all_candidates = {
            sits_id: list(range(self.num_patches_per_sits))
            for sits_id in range(len(self.sits_ids))
        }

        batches = []
        while all_candidates:
            sits_id = random.choice(list(all_candidates.keys()))
            n_remaining = len(all_candidates[sits_id])
            patch_idxs = random.sample(
                range(n_remaining), min(self.batch_size, n_remaining)
            )

            patch_ids = [
                sits_id * self.num_patches_per_sits + all_candidates[sits_id].pop(idx)
                for idx in sorted(patch_idxs, reverse=True)
            ]

            if len(all_candidates[sits_id]) == 0:
                del all_candidates[sits_id]

            batches.append(patch_ids)

        yield from batches

    def __len__(self):
        return sum(
            ceil(self.num_patches_per_sits / self.batch_size) for _ in self.sits_ids
        )
