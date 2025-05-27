import torch.nn as nn
import torch

import torch.nn.functional as F
from .multiltae import MultiLTAE
from .ltae import LTAE2d
from .blocks import ConvBlock, DownConvBlock, UpConvBlock 
import torchvision.models as models
from .multiutae_multihead_affinity import MultiUTAE
import numpy as np
from torch.autograd import Function

class HeadSelector(nn.Module):
    def __init__(self,
                input_dim,
                num_domains,
                num_classes,
                n_clusters,
                feat_type,
                kernel_size,
                num_convs,
                out_first,
                in_features,
                seq_length,
                checkpoint_path,
                return_all_heads=True,
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                agg_mode="att_group",
                encoder_norm="group",
                n_head=16,
                d_k=4,
                pad_value=0,
                padding_mode='reflect',
                T=730,
                offset=0,
                buffer_size=50):
        
        super().__init__()
        self.pretrained = MultiUTAE(input_dim,
                                    num_domains,
                                    num_classes,
                                    in_features,
                                    return_all_heads,
                                    seq_length,
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
                                    buffer_size)
        
        state_dict = torch.load(checkpoint_path)

        state_dict = state_dict['state_dict']
        filtered_state_dict = {}
        for k,v in state_dict.items():
            key = k.replace('model.', '')
            filtered_state_dict[key] = v


        self.pretrained.load_state_dict(filtered_state_dict)
        print("Pretrained MultiHead MultiUTAE loaded successfully...")
        # self.pretrained.eval()

        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.num_convs = num_convs

        # Change to 'batch'
        self.norm_type = 'batch'

        self.conv_block_input = nn.ModuleList([
            ConvBlock(
                nkernels=[input_dim, out_first],
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )]
            + [
            ConvBlock(
                nkernels=[out_first, out_first],
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )
            for _ in range(num_convs - 1)
        ])


        self.conv_block_e_64x16x16 = nn.ModuleList([
            ConvBlock(
                nkernels=[64, out_first],
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )]
            + [
            ConvBlock(
                nkernels=[out_first, out_first],
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )
            for _ in range(num_convs - 1)    
        ])

        self.conv_block_e_32x32x32 = nn.ModuleList([
            ConvBlock(
                nkernels=[32, out_first],
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )]
            + [
            ConvBlock(
                nkernels=[out_first, out_first],
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )
            for _ in range(num_convs - 1)   
        ])


        self.conv_block_e_32x64x64 = nn.ModuleList([
            ConvBlock(
                nkernels=[32, out_first],
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )]
            + [
            ConvBlock(
                nkernels=[out_first, out_first],  # Subsequent blocks: 32 -> 32
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )
            for _ in range(num_convs - 1)
        ])


        self.conv_block_e_32x128x128 = nn.ModuleList([
            ConvBlock(
                nkernels=[32, out_first],
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )]
            + [
            ConvBlock(
                nkernels=[out_first, out_first],  # Subsequent blocks: 32 -> 32
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )
            for _ in range(num_convs - 1)
        ])

        self.conv_block_d_32x32x32 = nn.ModuleList([
            ConvBlock(
                    nkernels=[32, out_first],
                    padding_mode=padding_mode,
                    norm=self.norm_type,
                    k=kernel_size,
                    pad_value=pad_value,
                    store_act=False
            )]
            + [
            ConvBlock(
                nkernels=[out_first, out_first],  # Subsequent blocks: 32 -> 32
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )
            for _ in range(num_convs - 1)
        ])

        self.conv_block_d_16x64x64 = nn.ModuleList([
            ConvBlock(
                    nkernels=[16, out_first],
                    padding_mode=padding_mode,
                    norm=self.norm_type,
                    k=kernel_size,
                    pad_value=pad_value,
                    store_act=False
            )]
            + [
            ConvBlock(
                nkernels=[out_first, out_first],  # Subsequent blocks: 32 -> 32
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )
            for _ in range(num_convs - 1)
        ])

        self.conv_block_d_16x128x128 = nn.ModuleList([
            ConvBlock(
                    nkernels=[16, out_first],
                    padding_mode=padding_mode,
                    norm=self.norm_type,
                    k=kernel_size,
                    pad_value=pad_value,
                    store_act=False
            )]
            + [
            ConvBlock(
                nkernels=[out_first, out_first],  # Subsequent blocks: 32 -> 32
                padding_mode=padding_mode,
                norm=self.norm_type,
                k=kernel_size,
                pad_value=pad_value,
                store_act=False
            )
            for _ in range(num_convs - 1)
        ])

        self.feat_type = feat_type

        if not n_clusters:
            out_last = num_domains
        else:
            out_last = n_clusters

        if self.feat_type == 'encoder':
            self.linear2 = nn.Linear(
                    in_features=5*out_first,
                    out_features=out_last
            )
        
        elif self.feat_type == 'decoder':
            self.linear2 = nn.Linear(
                    in_features=3*out_first,
                    out_features=out_last
            )

        elif self.feat_type == 'encoder+decoder':
            self.linear2 = nn.Linear(
                    in_features=8*out_first,
                    out_features=out_last
            )

        elif self.feat_type == 'bottleneck':
            self.linear2 = nn.Linear(
                    in_features=out_first,
                    out_features=out_last
            )

    def forward(self, batch):
        self.pretrained.training = False
        with torch.no_grad():
            out = self.pretrained(batch)
            logits_heads = out['logits']


        domain_feats = out['domain_feats']

        if self.feat_type == 'bottleneck':
            feats_64x16x16 = domain_feats['feats_e_64x16x16']
            bs, T, C, H, W = feats_64x16x16.shape
            feats_64x16x16 = feats_64x16x16.reshape(bs * T, *feats_64x16x16.shape[2:])

            out_dict = {}

            for conv_block in self.conv_block_e_64x16x16:
                feats_64x16x16 = conv_block(feats_64x16x16)

            out_64x16x16 = feats_64x16x16.view(bs, T, *feats_64x16x16.shape[1:])

            out_64x16x16 = out_64x16x16[:, :, :, :, :].mean(dim=1)
            out_64x16x16 = self.global_pool(out_64x16x16).view(bs, -1)

            out_feats = out_64x16x16

        if self.feat_type in ['encoder', 'encoder+decoder']:
            if 'data' in batch.keys():
                feats_input = batch['data']
            else:
                feats_input = batch['input']

            feats_64x16x16 = domain_feats['feats_e_64x16x16']
            feats_32x64x64 = domain_feats['feats_e_32x64x64']
            feats_32x32x32 = domain_feats['feats_e_32x32x32']
            feats_32x128x128 = domain_feats['feats_e_32x128x128']

            bs, T, C, H, W = feats_64x16x16.shape

            feats_input = feats_input.reshape(bs * T, *feats_input.shape[2:])
            feats_64x16x16 = feats_64x16x16.reshape(bs * T, *feats_64x16x16.shape[2:])
            feats_32x64x64 = feats_32x64x64.reshape(bs * T, *feats_32x64x64.shape[2:])
            feats_32x32x32 = feats_32x32x32.reshape(bs * T, *feats_32x32x32.shape[2:])
            feats_32x128x128 = feats_32x128x128.reshape(bs * T, *feats_32x128x128.shape[2:])

            out_dict = {}

            for conv_block in self.conv_block_input:
                feats_input = conv_block(feats_input)

            for conv_block in self.conv_block_e_64x16x16:
                feats_64x16x16 = conv_block(feats_64x16x16)

            for conv_block in self.conv_block_e_32x64x64:
                feats_32x64x64 = conv_block(feats_32x64x64)

            for conv_block in self.conv_block_e_32x32x32:
                feats_32x32x32 = conv_block(feats_32x32x32)

            for conv_block in self.conv_block_e_32x128x128:
                feats_32x128x128 = conv_block(feats_32x128x128)

            out_input = feats_input.view(bs, T, *feats_input.shape[1:])
            out_64x16x16 = feats_64x16x16.view(bs, T, *feats_64x16x16.shape[1:])
            out_32x64x64 = feats_32x64x64.view(bs, T, *feats_32x64x64.shape[1:])
            out_32x32x32 = feats_32x32x32.view(bs, T, *feats_32x32x32.shape[1:])
            out_32x128x128 = feats_32x128x128.view(bs, T, *feats_32x128x128.shape[1:])

            # Temporal Pooling
            out_input = out_input[:, :, :, :, :].mean(dim=1)
            out_64x16x16 = out_64x16x16[:, :, :, :, :].mean(dim=1)
            out_32x64x64 = out_32x64x64[:, :, :, :, :].mean(dim=1)
            out_32x32x32 = out_32x32x32[:, :, :, :, :].mean(dim=1)
            out_32x128x128 = out_32x128x128[:, :, :, :, :].mean(dim=1)

            # Spatial pooling
            out_input = self.global_pool(out_input).view(bs, -1)
            out_64x16x16 = self.global_pool(out_64x16x16).view(bs, -1)
            out_32x32x32 = self.global_pool(out_32x32x32).view(bs, -1)
            out_32x64x64 = self.global_pool(out_32x64x64).view(bs, -1)
            out_32x128x128 = self.global_pool(out_32x128x128).view(bs, -1)

            # out_feats = out_64x16x16
            enc_feats = torch.cat([out_input, out_32x64x64, out_32x32x32, out_64x16x16, out_32x128x128], dim=-1)
            out_feats = enc_feats

        if self.feat_type in ['decoder', 'encoder+decoder']:
            feats_32x32x32 = domain_feats['feats_d_32x32x32']
            feats_16x64x64 = domain_feats['feats_d_16x64x64']
            feats_16x128x128 = domain_feats['feats_d_16x128x128']

            bs, T, C, H, W = feats_16x64x64.shape

            out_dict = {}

            feats_16x64x64 = feats_16x64x64.reshape(bs * T, *feats_16x64x64.shape[2:])
            feats_32x32x32 = feats_32x32x32.reshape(bs * T, *feats_32x32x32.shape[2:])
            feats_16x128x128 = feats_16x128x128.reshape(bs * T, *feats_16x128x128.shape[2:])

            for conv_block in self.conv_block_d_32x32x32:
                feats_32x32x32 = conv_block(feats_32x32x32)

            for conv_block in self.conv_block_d_16x64x64:
                feats_16x64x64 = conv_block(feats_16x64x64)

            for conv_block in self.conv_block_d_16x128x128:
                feats_16x128x128 = conv_block(feats_16x128x128)

            out_16x64x64 = feats_16x64x64.view(bs, T, *feats_16x64x64.shape[1:])
            out_32x32x32 = feats_32x32x32.view(bs, T, *feats_32x32x32.shape[1:])
            out_16x128x128 = feats_16x128x128.view(bs, T, *feats_16x128x128.shape[1:])

            out_32x32x32 = out_32x32x32[:, :, :, :, :].mean(dim=1)
            out_16x64x64 = out_16x64x64[:, :, :, :, :].mean(dim=1)
            out_16x128x128 = out_16x128x128[:, :, :, :].mean(dim=1)

            out_32x32x32 = self.global_pool(out_32x32x32).view(bs, -1)
            out_16x64x64 = self.global_pool(out_16x64x64).view(bs, -1)
            out_16x128x128 = self.global_pool(out_16x128x128).view(bs, -1)
            
            dec_feats = torch.cat([out_32x32x32, out_16x64x64, out_16x128x128], dim=-1)
            out_feats = dec_feats

        if self.feat_type == 'encoder+decoder':
            out_feats = torch.cat([enc_feats, dec_feats], dim=-1)

        out_feats = self.linear2(out_feats)

        p_cls = torch.softmax(out_feats / 1.0, dim=-1)
        p_seg = logits_heads

        p_moh = torch.sum(p_cls.view(p_cls.shape[1], p_cls.shape[0], 1, 1, 1, 1) * p_seg, dim=0)

        out_dict['logits'] = p_moh
        out_dict['hs_logits'] = out_feats
        out_dict['logits_heads'] = logits_heads

        return out_dict