import torch
import torch.nn as nn

from .multiutae import MultiUTAE
import torchvision.transforms as tfm


class ContraSegNet(MultiUTAE):
    def __init__(
        self,
        input_dim,
        num_classes,
        in_features,
        where_contrastive="encoder",
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
        T=730,
        offset=0,
        use_memory_bank=False,
        num_vectors_in_memory_bank=None,
        vector_dim_in_memory_bank=None,
        random_select_date=0,
    ):
        """Constructor.

        Parameters
        ----------
        model : str
            Which part to use for contrastive learning.
            Choices : ["encoder", "all"].
            Default : "encoder"

        use_memory_bank : bool
            Whether to add memory bank or not.
            Default : False

        num_vectors_in_memory_bank : int
            Number of feature vectors for the memory bank.
            Default : None

        vector_dim_in_memory_bank : int
            Dimension of each vector in the memory bank.
            Default : None
            Note: Make it same as the number of output channels.
        """
        super().__init__(
            input_dim,
            num_classes,
            in_features,
            str_conv_k,
            str_conv_s,
            str_conv_p,
            agg_mode,
            encoder_norm,
            n_head,
            d_k,
            pad_value,
            padding_mode,
            T,
            offset,
        )

        ## Setting the new attributes ##
        self.where_contrastive = where_contrastive
        self.use_memory_bank = use_memory_bank
        self.vector_dim_in_memory_bank = vector_dim_in_memory_bank
        self.random_select_date = random_select_date

        if self.use_memory_bank:
            assert (num_vectors_in_memory_bank != None) and (
                vector_dim_in_memory_bank != None
            ), f"num_vectors_in_memory_bank or vector_dim_in_memory_bank is None!"

            ## Setting the memory bank (This is something similar to the prototypes) ##
            self.memory_bank = nn.Embedding(
                num_embeddings=num_vectors_in_memory_bank,
                embedding_dim=vector_dim_in_memory_bank,
            )

        ## Color Jitter ##
        self.colorjit = tfm.ColorJitter(
            brightness=0.9, contrast=0.8, saturation=0.9, hue=0.05
        )

    @torch.no_grad()
    def transform_batch(self, batch):
        # data has shape B, T, C, H, W
        # apply color jitter to batch1 and batch2

        batch_1 = {
            "data": self.colorjit(batch["data"].flatten(0, 2).unsqueeze(1)).view(
                batch["data"].shape
            )
        }
        batch_2 = {
            "data": self.colorjit(batch["data"].flatten(0, 2).unsqueeze(1)).view(
                batch["data"].shape
            )
        }

        # if self.random_select_date != 0:
        #     choice = torch.randint(
        #         0,
        #         batch["data"].shape[1],
        #         (self.random_select_date,),
        #         device=batch["data"].device,
        #     )
        #     batch_1["data"] = batch_1["data"][:, choice, :, :, :]
        #     batch_2["data"] = batch_2["data"][:, choice, :, :, :]

        return batch_1, batch_2

    def forward(self, batch):
        if type(batch) is not list:
            return super().forward(batch)

        out = super().forward(batch[0])

        batch_1, batch_2 = self.transform_batch(batch[1])

        ## If we are doing CL in the middle ##
        if self.where_contrastive == "encoder":
            out_1, _ = self.encode(batch=batch_1)
            out_2, _ = self.encode(batch=batch_2)
        else:
            raise ValueError(
                f"Invalid value for where_contrastive : {self.where_contrastive}"
            )

        ## For memory bank ##
        if self.use_memory_bank:
            out_1_store, out_2_store = out_1, out_2
            b, t, c, h, w = out_1.shape

            assert (
                c == self.vector_dim_in_memory_bank
            ), f"Embedding dim must be same as {c}"

            ## Flattening ##
            flattened_out_1, flattened_out_2 = (
                out_1.permute(0, 1, 3, 4, 2).view(b, t, -1, c),
                out_2.permute(0, 1, 3, 4, 2).view(b, t, -1, c),
            )

            ## Distance calculation ##
            dist_1 = torch.cdist(
                flattened_out_1,
                self.memory_bank.weight[None, None, :].repeat(b, t, 1, 1),
            )
            dist_2 = torch.cdist(
                flattened_out_2,
                self.memory_bank.weight[None, None, :].repeat(b, t, 1, 1),
            )

            ## Get the minimum index embedding ##
            min_index_1 = torch.argmin(dist_1, dim=-1).view(-1)
            min_index_2 = torch.argmin(dist_2, dim=-1).view(-1)

            ## Getting the quantized codes ##
            code_1 = (
                self.memory_bank.weight[min_index_1]
                .view(b, t, h, w, c)
                .permute(0, 1, 4, 2, 3)
            )
            code_2 = (
                self.memory_bank.weight[min_index_2]
                .view(b, t, h, w, c)
                .permute(0, 1, 4, 2, 3)
            )
            out["contra"] = [out_1_store, code_2, out_2_store, code_1]

        ## In case of no memory bank ##
        else:
            out["contra"] = [out_1, out_2]

        return out

    def encode(self, batch):
        x = batch["data"]
        out = self.in_conv.smart_forward(x)
        feature_maps = [out]

        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        return out, feature_maps
