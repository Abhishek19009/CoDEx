import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .multiltae import MultiLTAE
from .blocks import ConvBlock, DownConvBlock, UpConvBlock
from .ltae import LTAE2d


def compute_mmd(x, y):
    """ Maximum Mean Discrepancy (MMD) """
    x_flat = x.view(x.size(0), -1)  # Flatten to (seq_len, channels * h * w)
    y_flat = y.view(y.size(0), -1)  # Flatten to (seq_len, channels * h * w)

    xx = torch.mm(x_flat, x_flat.t())
    yy = torch.mm(y_flat, y_flat.t())
    xy = torch.mm(x_flat, y_flat.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- (rx.t() + rx - 2 * xx))
    L = torch.exp(- (ry.t() + ry - 2 * yy))
    P = torch.exp(- (rx.t() + ry - 2 * xy))

    beta = (1. / (x_flat.size(0) * x_flat.size(0)))
    gamma = (2. / (x_flat.size(0) * y_flat.size(0)))

    return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)

def compute_jsd(p, q, dim=1, epsilon=1e-8):
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.

    Args:
    p (torch.Tensor): First probability distribution.
    q (torch.Tensor): Second probability distribution.
    dim (int): The dimension along which to compute the JSD.
    epsilon (float): Small value to avoid log(0).

    Returns:
    torch.Tensor: The Jensen-Shannon Divergence.
    """
    p = p[:q.size(0)]

    p = p.view(p.size(0), -1)
    q = q.view(q.size(0), -1)

    # Normalize the distributions
    p = p / (p.sum(dim=dim, keepdim=True) + epsilon)
    q = q / (q.sum(dim=dim, keepdim=True) + epsilon)
    
    # Calculate the mean distribution
    m = 0.5 * (p + q)

    # Avoid log(0) by small epsilon value
    log_m = torch.log(m + epsilon)
    
    # Compute the KL divergence for p and q with respect to m
    kl_pm = F.kl_div(log_m, p+epsilon, reduction='none').sum(dim=dim)
    kl_qm = F.kl_div(log_m, q+epsilon, reduction='none').sum(dim=dim)
    
    # Return the Jensen-Shannon Divergence
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


def normalize_features(train_feats_all, eval_feats_all):
    # Normalize the features
    train_feats_all_norm = train_feats_all / np.linalg.norm(train_feats_all, axis=1, keepdims=True)
    eval_feats_all_norm = eval_feats_all / np.linalg.norm(eval_feats_all, axis=1, keepdims=True)

    return train_feats_all_norm, eval_feats_all_norm

def compute_similarity(train_feats_norm, eval_feats_norm, metric):
    if metric == 'l2':
        distances = np.linalg.norm(eval_feats_norm[:, np.newaxis, :] - train_feats_norm[np.newaxis, :, :], axis=2)
        argmin_distance = np.argmin(distances, axis=1)
        return argmin_distance
    elif metric == 'cosine':
        similarities = np.dot(eval_feats_norm, train_feats_norm.T)
        argmax_similarity = np.argmax(similarities, axis=1)
        return argmax_similarity
    else:
        raise ValueError("Unsupported metric: choose 'l2' or 'cosine'")

def select_head(x, domains, split, feature_type, metric='cosine', patch_wise=True):
    base_path = "/home/abhishek/Hackathon-Imagine/HACKATHON-IMAGINE-2024/train_feats"
    train_feats_all = np.load(f"{base_path}/train_{feature_type}_feats_numpy.npy")
    if patch_wise:
        eval_feats_all = x.mean(dim=(1,3,4)).detach().cpu().numpy()
    else:    
        eval_feats_all = np.load(f"{base_path}/{split}_{feature_type}_feats_numpy.npy")

    train_feats_norm, eval_feats_norm = normalize_features(train_feats_all, eval_feats_all)

    assert train_feats_norm.shape[1] == eval_feats_norm.shape[1], f"{feature_type} feature dimensions do not match."

    indices = compute_similarity(train_feats_norm, eval_feats_norm, metric)
    
    heads_idx = indices if patch_wise else indices[domains]

    return heads_idx


class DomainClassifier(nn.Module):
    def __init__(self, in_channels, num_domains=55):
        super(DomainClassifier, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Pool H and W dimensions to 1x1
        self.fc1 = nn.Linear(in_channels, in_channels)  # Fully connected layer to output num_domains
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.fc2 = nn.Linear(in_channels, num_domains)
    def forward(self, x):
        # # x: (bs, T, C, H, W)
        bs, T, C, H, W = x.shape
        x = self.global_pool(x)
        x = x.mean(dim=1).view(bs, C)

        # Fully connected layer: (bs, C) -> (bs, C//2)
        out = F.relu(self.fc1(x))
        out = self.batch_norm(out)
        # Transform to num_domains: (bs, C//2) -> (bs, 55)
        out = self.fc2(out)

        out = F.softmax(out, dim=-1)
        
        return out



class DomainClassifier_v2(nn.Module):
    def __init__(self, in_channels, num_domains=55):
        super(DomainClassifier_v2, self).__init__()
        self.out_ltae = LTAE2d(
                in_channels=64
            )
        
        # self.conv_block = 
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_domains)

    def forward(self, x, batch_positions):
        # bs, T, C, H, W = x.shape
        x = self.out_ltae(x, batch_positions)

        bs, C_out, H, W = x.shape

        x = x.reshape(bs*C_out, H*W)
        x = self.global_pool(x)
        x = x.reshape(bs, C_out)
        x = F.relu(self.fc1(self.batch_norm(x)))
        x = self.fc2(x)

        return x




class MultiUTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        num_domains,
        num_classes,
        in_features,
        seq_length,
        return_all_heads,
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
        buffer_size=50
    ):
        """
        Multi-temporal U-TAE architecture for multi-stamp spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            num_classes_list (list): List of number of classes for each domain.
            in_features (int): Feature size at the innermost stage.
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_k (int): Key-Query space dimension
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
        """
        super().__init__()
        self.encoder_widths = [
            in_features // 2,
            in_features // 2,
            in_features // 2,
            in_features,
        ]
        self.decoder_widths = [
            in_features // 4,
            in_features // 4,
            in_features // 2,
            in_features,
        ]
        self.n_stages = len(self.encoder_widths)
        self.enc_dim = (
            self.decoder_widths[0]
            if self.decoder_widths is not None
            else self.encoder_widths[0]
        )
        self.stack_dim = (
            sum(self.decoder_widths)
            if self.decoder_widths is not None
            else sum(self.encoder_widths)
        )
        self.pad_value = pad_value

        if self.decoder_widths is not None:
            assert len(self.encoder_widths) == len(self.decoder_widths)
            assert self.encoder_widths[-1] == self.decoder_widths[-1]
        else:
            self.decoder_widths = self.encoder_widths

        in_conv_kernels = [input_dim] + [self.encoder_widths[0], self.encoder_widths[0]]
        self.in_conv = ConvBlock(
            nkernels=in_conv_kernels,
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=self.encoder_widths[i],
                d_out=self.encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=self.decoder_widths[i],
                d_out=self.decoder_widths[i - 1],
                d_skip=self.encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="group",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = MultiLTAE(
            in_channels=self.encoder_widths[-1],
            n_head=n_head,
            return_att=True,
            d_k=d_k,
            T=T,
            offset=offset,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # Create multiple output heads for each domain
        self.out_convs = nn.ModuleList(
            [
                ConvBlock(
                    nkernels=[self.decoder_widths[0]] + [in_features // 4, num_classes],
                    padding_mode=padding_mode,
                    norm="None",
                    store_act=False
                )
                for _ in range(num_domains)
            ]
        )
        self.num_domains = num_domains
        self.feature_type = 'aggregate'
        self.return_all_heads = return_all_heads
        self.seq_length = seq_length
              
    def forward(self, batch):
        if "data" in batch.keys():
            x = batch["data"]
        elif "input" in batch.keys():
            x = batch["input"]

        batch_positions = batch["positions"]

        if "idx" in batch.keys():
            domains = batch["idx"]
            split = batch['split'][0]
            
        # out_acts = []
        domain_feats = {}

        batch_size, seq_len, c, h, w = x.size()
        if batch_positions is None:
            batch_positions = torch.tensor(
                range(1, x.shape[1] + 1), dtype=torch.long, device=x.device
            )[None].expand(x.shape[0], -1, -1, -1, -1)

        pad_mask = (
            (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        out = self.in_conv.smart_forward(x)
        out_sh = out.shape
        domain_feats[f'feats_e_{out_sh[2]}x{out_sh[3]}x{out_sh[4]}'] = out
        feature_maps = [out]

        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            out_sh = out.shape
            domain_feats[f'feats_e_{out_sh[2]}x{out_sh[3]}x{out_sh[4]}'] = out
            feature_maps.append(out)

        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )
    

        # SPATIAL DECODER
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            out_sh = out.shape
            domain_feats[f'feats_d_{out_sh[2]}x{out_sh[3]}x{out_sh[4]}'] = out


        if self.training:
            out_logits_s = []

            for sits_idx, domain in enumerate(domains):
                domain_out = self.out_convs[domain](out[sits_idx].view(1 * seq_len, -1, h, w)).view(
                    1, seq_len, -1, h, w
                )
                
                out_logits_s.append(domain_out)
                

            out_logits_s = torch.cat(out_logits_s)
            out_logits_c = torch.zeros_like(out_logits_s)

            for sits_idx, domain_i in enumerate(domains):
                # angle difference between coord and all heads in sequential order. 
                geo_weight = batch['geo_weight'][sits_idx]
                
                for domain_j in range(self.num_domains):
                    if domain_j != domain_i:
                        domain_out = self.out_convs[domain_j](out[sits_idx].view(1 * seq_len, -1, h, w)).view(
                            seq_len, -1, h, w
                        )
                        domain_out = domain_out * geo_weight[domain_j]
                        out_logits_c[sits_idx] = out_logits_c[sits_idx] + domain_out

                mask = torch.ones_like(geo_weight, dtype=torch.bool)
                mask[domain_i] = False

                out_logits_c[sits_idx] = out_logits_c[sits_idx] / (geo_weight[mask].sum() + 1e-6)

            return {"out_logits_s": out_logits_s, "out_logits_c": out_logits_c}
        

        elif not self.training:
            out_logits_s = [None for _ in range(len(domains))]
            head_logits = [[] for _ in range(len(domains))]
            for sits_idx, domain_i in enumerate(domains):
                # Simply use geo_weight, computes angle difference b/w coord in test with all coords of heads (sequential).
                geo_weight = batch["geo_weight"][sits_idx]
                for domain_j in range(self.num_domains):
                    domain_out = self.out_convs[domain_j](out[sits_idx].view(1 * seq_len, -1, h, w)).view(
                        seq_len, -1, h, w
                    ) 

                    head_logits[sits_idx].append(domain_out)
                    
                    domain_out *= geo_weight[domain_j]

                    if out_logits_s[sits_idx] is None:
                        out_logits_s[sits_idx] = domain_out
                    else:
                        out_logits_s[sits_idx] += domain_out

                
                head_logits[sits_idx] = torch.stack(head_logits[sits_idx])
                out_logits_s[sits_idx] = out_logits_s[sits_idx] / (geo_weight.sum() + 1e-6)
            
            out_logits_s = torch.stack(out_logits_s)
            head_logits = torch.cat(head_logits)
            if len(domains) == 1:
                head_logits = head_logits.unsqueeze(dim=0)

            head_logits = head_logits.permute(1, 0, *range(2, head_logits.ndim))
            return {"logits": head_logits, "out_logits_s": out_logits_s, "out_logits_c": torch.zeros_like(out_logits_s)}
        
        
class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, _, h, w = attn_mask.shape
                attn = attn_mask.contiguous().view(n_heads * b * t, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, None, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = torch.einsum("nbtuhw, nbukhw -> nbtukhw", attn, out)
                out = out.sum(dim=3)  # sum on temporal dim -> hxBxTx(C/h)xHxW
                out = torch.cat([group for group in out], dim=2)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, _, h, w = attn_mask.shape
                attn = attn_mask.contiguous().view(n_heads * b * t, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, t, *x.shape[-2:])  # hxBxTxTxHxW
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTx(C/h)xHxW
                out = torch.einsum("nbtuhw, nbukhw -> nbtukhw", attn, out)
                out = out.sum(dim=3)  # sum on temporal dim -> hxBxTx(C/h)xHxW
                out = torch.cat([group for group in out], dim=2)  # -> BxTxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)

def create_mlp(in_ch, out_ch, n_hidden_units, n_layers):
    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_hidden_units), nn.ReLU(True)]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU(True)]
        seq += [nn.Linear(n_hidden_units, out_ch)]
    else:
        seq = [nn.Linear(in_ch, out_ch)]
    return nn.Sequential(*seq)
