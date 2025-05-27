"""
Code adapted from https://github.com/zhu-xlab/DOFA
Copyright (c) 2024 Zhitong Xiong
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
from timm.models.vision_transformer import Block

from .multiltae import MultiLTAE


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        dynamic_weight = weight.view(
            self.embed_dim, inplanes, self.kernel_size, self.kernel_size
        )  # 3xoutdx16x16
        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat,
            weights,
            bias=bias,
            stride=self.kernel_size,
            padding=1,
            dilation=1,
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class DOFA(nn.Module):
    """
    DOFA model with pretrained weights
    """

    def __init__(
        self,
        num_classes,
        wavelengths=[],
        path="",
        pretrained=True,
        unfreeze_last_x_blocks=False,
        lora_rank=0,
    ):
        super().__init__()
        self.model = vit_base_patch16()
        self.wavelengths = wavelengths
        self.head = SimpleSegmentationHead(
            embed_dim=768,
            downsample_factor=16,
            remove_cls_token=False,
            features_format="NHWC",
            features_sizes=14,
            num_classes=num_classes,
        )
        if pretrained:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint, strict=False)

        for param in self.model.parameters():
            param.requires_grad = False
        if unfreeze_last_x_blocks:
            print(f"unfreezing last {unfreeze_last_x_blocks} blocks")
            for param in self.model.blocks[-unfreeze_last_x_blocks:].parameters():
                param.requires_grad = True
        if lora_rank > 0:
            print(f"Applying LORA with rank {lora_rank}")
            assert (
                not unfreeze_last_x_blocks
            ), "LORA and unfreeze_last_x_blocks are exclusive"
            apply_lora(self.model, lora_rank)

    def forward_late(self, x, wavelenghts):
        # x: [batch, n_bands, h, w]
        # wavelengths: [batch, n_bands]
        return self.model.forward_features(x, wavelenghts)

    def forward(self, batch):
        x = batch["data"]
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = transforms.functional.resize(x, 224)
        feature_maps = self.model.forward_features(x, self.wavelengths)
        out = self.head(feature_maps)
        out = transforms.functional.resize(out, 128)
        out = out.reshape(b, t, self.head.num_classes, h, w)
        return {"logits": out}


class DOFA_with_attention(DOFA):
    def __init__(
        self,
        num_classes,
        wavelengths=[],
        path="",
        pretrained=True,
        unfreeze_last_x_blocks=False,
        lora_rank=0,
        n_head=16,
        d_k=4,
        T=1000,
        offset=10,
        pad_value=0,
    ):
        super().__init__(
            num_classes=num_classes,
            wavelengths=wavelengths,
            path=path,
            pretrained=pretrained,
            unfreeze_last_x_blocks=unfreeze_last_x_blocks,
            lora_rank=lora_rank,
        )
        self.temporal_encoder = MultiLTAE(
            in_channels=768,
            n_head=n_head,
            return_att=True,
            d_k=d_k,
            T=T,
            offset=offset,
        )
        self.pad_value = pad_value

    def forward(self, batch):
        x = batch["data"]
        batch_positions = batch["positions"]
        pad_mask = (
            (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = transforms.functional.resize(x, 224)
        feature_maps = self.model.forward_features(x, self.wavelengths)
        feature_maps = feature_maps.reshape(b, t, *feature_maps.shape[-3:])
        feature_maps = feature_maps.permute(0, 1, 4, 2, 3)

        out_att, _ = self.temporal_encoder(
            feature_maps, batch_positions=batch_positions, pad_mask=pad_mask
        )
        out_att = out_att.reshape(b * t, *feature_maps.shape[-3:])
        out_att = out_att.permute(0, 2, 3, 1)

        out = self.head(out_att)
        out = transforms.functional.resize(out, 128)
        out = out.reshape(b, t, self.head.num_classes, h, w)
        return {"logits": out}


def vit_base_patch16(**kwargs):
    model = OFAViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


class OFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.hw_size = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist
        x, _ = self.patch_embed(x, self.waves)
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        # remove cls_token
        x = x[:, 1:, :]
        # reshape
        outcome = x.reshape(x.shape[0], self.hw_size, self.hw_size, -1)
        return outcome

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        return x


class SimpleSegmentationHead(nn.Module):
    """Simple segmentation head."""

    def __init__(
        self,
        embed_dim,
        downsample_factor,
        remove_cls_token,
        features_format,
        features_sizes,
        num_classes,
        decoder_stride=2,
        **kwargs,
    ):
        """Simple segmentation head.

        Args:
            embed_dim (int): Embedding dimension of the backbone model.
            downsample_factor (int): The downsample factor of the backbone model.
            remove_cls_token (bool): Whether to remove the cls token from the output features.
            features_format (str): The format of the output features.
            features_sizes (int): The size of the output feature map.
            num_classes (int): Number of classes.
            decoder_stride (int): The stride of the decoder.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor
        self.remove_cls_token = remove_cls_token
        self.features_format = features_format
        self.feature_size = features_sizes
        self.num_classes = num_classes
        self.decoder_stride = decoder_stride

        self.layered_output = isinstance(self.embed_dim, (list, tuple))
        if self.layered_output:
            self.embed_dim = self.embed_dim[-1]
            self.downsample_factor = self.downsample_factor[-1]
            self.feature_size = self.feature_size[-1]

        depth = math.log(self.downsample_factor, decoder_stride)
        assert (
            depth.is_integer()
        ), f"decoder stride({decoder_stride}) must be a power of the downsample factor({self.downsample_factor})"
        depth = int(depth)
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.embed_dim // 2 ** (d),
                        self.embed_dim // 2 ** (d + 1),
                        decoder_stride,
                        stride=decoder_stride,
                    ),
                    nn.BatchNorm2d(self.embed_dim // 2 ** (d + 1)),
                    nn.GELU(),
                    nn.Conv2d(
                        self.embed_dim // 2 ** (d + 1),
                        self.embed_dim // 2 ** (d + 1),
                        3,
                        padding="same",
                    ),
                    nn.BatchNorm2d(self.embed_dim // 2 ** (d + 1)),
                    nn.GELU(),
                )
                for d in range(depth - 1)
            ]
            + [
                nn.ConvTranspose2d(
                    self.embed_dim // 2 ** (depth - 1),
                    num_classes,
                    decoder_stride,
                    stride=decoder_stride,
                )
            ]
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): The input to the segmentation head.

        Returns:
            torch.Tensor: The output of the segmentation head.
        """
        if self.layered_output:
            x = x[-1]
        if self.remove_cls_token:
            x = x[:, 1:, :]
        if self.features_format == "NLC":
            # Convert from NLC to NCHW
            x = x.reshape(x.shape[0], self.feature_size, self.feature_size, x.shape[-1])
            x = x.permute(0, 3, 1, 2)
        if self.features_format == "NHWC":
            # Convert from NHWC to NCHW
            x = x.permute(0, 3, 1, 2)
        return self.layers(x)


class _LORA_attention(nn.Module):
    def __init__(self, old_attention, r) -> None:
        super().__init__()
        self.att = old_attention
        self.dim = self.att.num_heads * self.att.head_dim
        self.in_layer_q = nn.Linear(self.dim, r)
        self.out_layer_q = nn.Linear(r, self.dim)
        self.in_layer_k = nn.Linear(self.dim, r)
        self.out_layer_k = nn.Linear(r, self.dim)
        self.in_layer_v = nn.Linear(self.dim, r)
        self.out_layer_v = nn.Linear(r, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.att.qkv(x)
        lora_qkv = torch.cat(
            [
                self.out_layer_q(self.in_layer_q(x)),
                self.out_layer_k(self.in_layer_k(x)),
                self.out_layer_v(self.in_layer_v(x)),
            ],
            dim=-1,
        )

        qkv = qkv.reshape(B, N, 3, self.att.num_heads, self.att.head_dim).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv.unbind(0)
        q, k = self.att.q_norm(q), self.att.k_norm(k)

        if self.att.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.att.attn_drop.p if self.att.training else 0.0,
            )
        else:
            q = q * self.att.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.att.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.att.proj(x)
        x = self.att.proj_drop(x)
        return x

    def freeze_not_lora(self):
        for param in self.att.parameters():
            param.requires_grad = False


def apply_lora(model, r):
    for i, block in enumerate(model.blocks):
        if hasattr(block, "attn"):
            model.blocks[i].attn = _LORA_attention(block.attn, r)
            model.blocks[i].attn.freeze_not_lora()
