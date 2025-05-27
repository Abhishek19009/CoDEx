import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from headselector.data.data import DEDataset
from models.networks.head_selector import HeadSelector
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from headselector.losses.focalloss import FocalLoss
from metrics.scd_metrics import SCDMetric

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for HeadSelector")

    # Data and model parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU Device')
    parser.add_argument('--reset_optim', type=bool, default=False, help='Reset the optimizer after a interval')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--lr_scheduler', type=str, default=None, help='Which scheduler to choose? Default: None')

    # Model-specific hyperparameters
    parser.add_argument('--num_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--num_domains', type=int, default=40, help='Number of domains')
    parser.add_argument('--feat_type', type=str, default='encoder', help='Which blocks to select?')
    parser.add_argument('--label_type', type=str, default='miou', help="Which label type")
    parser.add_argument('--hs_lambda', type=float, default=0.9, help='Head selection lambda for focal loss')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for Conv Block') 
    parser.add_argument('--num_convs', type=int, default=1, help='Number of conv blocks for head selector')
    parser.add_argument('--out_first', type=int, default=16, help='Projection dimension in head selector')
    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension of the model')
    parser.add_argument('--in_features', type=int, default=64, help='Number of input features for the model')
    parser.add_argument('--seq_length', type=int, default=6, help='Sequence length for temporal data')
    parser.add_argument('--str_conv_k', type=int, default=4, help='Convolution kernel size')
    parser.add_argument('--str_conv_s', type=int, default=2, help='Convolution stride size')
    parser.add_argument('--str_conv_p', type=int, default=1, help='Convolution padding size')
    parser.add_argument('--agg_mode', type=str, default="att_group", help='Aggregation mode')
    parser.add_argument('--encoder_norm', type=str, default="group", help='Encoder normalization type')
    parser.add_argument('--n_head', type=int, default=16, help='Number of heads in multi-head attention')
    parser.add_argument('--d_k', type=int, default=4, help='Dimensionality of the key in attention mechanism')
    parser.add_argument('--pad_value', type=int, default=0, help='Padding value')
    parser.add_argument('--padding_mode', type=str, default="reflect", help='Padding mode')
    parser.add_argument('--T', type=int, default=1000, help='T value for positional encoding')
    parser.add_argument('--offset', type=int, default=10, help='Offset value for positional encoding')
    parser.add_argument('--buffer_size', type=int, default=20, help='Buffer size for the model')
    parser.add_argument('--pretrained_checkpoint_path', type=str, default="../checkpoints/Muds_MultiHead_MultiUTAE/best_miou_ckpt_in.ckpt", help='Path to pretrained checkpoint')
    parser.add_argument('--temp', type=float, default=2.0, help='Temperature for target distribution')
    parser.add_argument('--criterion', type=str, default='me')
    parser.add_argument('--ignore_index', type=int, default=2)

    parser.add_argument('--data_dir', type=str, default="../datasets/Muds_feats", help='Directory containing dataset')
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints", help='Path to save checkpoints')
    parser.add_argument('--head', type=int, default=0, help='Head index for multi-head attention')
    parser.add_argument('--load_save', type=bool, default=False, help='Whether to load from a saved checkpoint')

    return parser.parse_args()

args = parse_args()

train_dataset = DEDataset(
    path=args.data_dir,
    split='train',
    feat_types=['input'],  # or customize based on needs
    label_types=[args.label_type],
    num_channels=args.input_dim,
    num_classes=args.num_classes,
)

val_dataset = DEDataset(
    path=args.data_dir,
    split='val',
    feat_types=['input'],  # or customize based on needs
    label_types=[args.label_type],
    num_channels=args.input_dim,
    num_classes=args.num_classes,
    )

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=False,
    drop_last=True,
    num_workers=args.num_workers,
    collate_fn=train_dataset.collate_fn)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=args.num_workers,
    collate_fn=val_dataset.collate_fn)

eval_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=args.num_workers,
    collate_fn=val_dataset.collate_fn)

model = HeadSelector(
    input_dim=args.input_dim,
    num_domains=args.num_domains,
    num_classes=args.num_classes,
    feat_type=args.feat_type,
    kernel_size=args.kernel_size,
    num_convs=args.num_convs,
    out_first=args.out_first,
    n_clusters=None,
    in_features=args.in_features,
    return_all_heads=True,
    seq_length=args.seq_length,
    checkpoint_path=args.pretrained_checkpoint_path,
    str_conv_k=args.str_conv_k,
    str_conv_s=args.str_conv_s,
    str_conv_p=args.str_conv_p,
    agg_mode=args.agg_mode,
    encoder_norm=args.encoder_norm,
    n_head=args.n_head,
    d_k=args.d_k,
    pad_value=args.pad_value,
    padding_mode=args.padding_mode,
    T=args.T,
    offset=args.offset,
    buffer_size=args.buffer_size
)

device = args.device if torch.cuda.is_available() else "cpu"
model.to(device)

model_params = sum(p.numel() for p in model.parameters())
param_memory = model_params * 4 / (1024 ** 2)
# print("Model size in GB:", param_memory)

print(model)

if args.criterion == 'mse':
    criterion = nn.MSELoss()
elif args.criterion == 'me':
    criterion = nn.L1Loss()
elif args.criterion == 'crossentropy':
    criterion = nn.CrossEntropyLoss()

focal_criterion = FocalLoss(ignore_index=args.ignore_index)

if args.load_save:
    ckpt_path = os.path.join(args.checkpoint_path, f'best_{args.criterion}_{args.label_type}_{args.feat_type}_{args.kernel_size}_{args.out_first}_classifier.ckpt')
    print("Loading checkpoint...", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))

def soft_target(target):
    target = target / args.temp
    target = F.softmax(torch.clamp(target, min=1e-7), dim=-1)
    return target

wandb.init(project="arch_pmoh_muds", entity="imaginelab", name=f"{args.criterion}_{args.label_type}_{args.num_convs}_{args.feat_type}_{args.kernel_size}_{args.out_first}_{args.sgd_momentum}", config=vars(args))
def train():
    best_acc = 0
    best_miou = -1
    best_pixel_acc = -1
    least_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), betas=[0.9, 0.999], weight_decay=0.01, lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        running_top5_acc = 0.0
        running_total_samples = 0.0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) 
                        for key, value in batch.items()}
            
            outputs = model(batch)
            logits = outputs["hs_logits"]
            logits_heads = outputs['logits_heads']

            gt = batch['gt'].type(torch.int64)

            if args.criterion in ['me', 'mse']:
                P = batch[args.label_type] / 100
            elif args.criterion in ['crossentropy']:
                P = soft_target(batch[args.label_type])

            if torch.any(torch.isnan(P)):
                print("Nan encountered")
                continue

            l_me = criterion(logits, P)

            p_cls = torch.softmax(logits / 1.0, dim=-1)
            p_seg = torch.softmax(logits_heads, dim=3)

            p_moh = torch.sum(p_cls.view(p_cls.shape[1], p_cls.shape[0], 1, 1, 1, 1) * p_seg, dim=0)

            l_fl = focal_criterion(p_moh, gt)

            l_hs = l_me + args.hs_lambda * l_fl

            # Back pass
            optimizer.zero_grad()
            l_hs.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


            running_loss += l_hs.item()

            best_head = torch.argmax(logits, dim=-1)
            top5_target = torch.topk(P, k=5, dim=-1).indices

            top5_count = torch.any(top5_target == best_head.unsqueeze(-1), dim=-1)
            running_top5_acc += top5_count.sum().item()
            running_total_samples += P.size(0)  # Accumulate number of samples
            
            if batch_idx % 100 == 99:  # print every 100 batches
                top5_accuracy_percentage = (running_top5_acc / running_total_samples) * 100
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Top5 Head Accuracy: {top5_accuracy_percentage:.2f}%, Loss: {running_loss / 100:.4f}')
                wandb.log({
                    "train_loss": running_loss / 100,
                    "train_top5_acc": top5_accuracy_percentage
                })
                running_loss = 0.0
                running_top5_acc = 0.0
                running_total_samples = 0.0



        best_head_acc, top5_acc, val_loss, pixel_acc, miou = evaluate()
        log_dict = {"best_head_accuracy": best_head_acc, "top5_acc": top5_acc, "val_loss": val_loss, "epoch": epoch + 1}
        log_dict["learning_rate"] = args.learning_rate

        wandb.log(log_dict)
        print(f'Epoch {epoch+1} - Best Head Accuracy: {best_head_acc:.1f}%, Top5 Head Accuracy: {top5_acc:.1f}%,  Val Loss: {val_loss:.4f}')

        if miou > best_miou:
            # least_loss = val_loss
            best_miou = miou
            print(f"Saving checkpoint...")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f'best_{args.criterion}_{args.label_type}_{args.num_convs}_{args.feat_type}_{args.kernel_size}_{args.out_first}_classifier.ckpt'))

        print(f"Saving last checkpoint {args.criterion}_{args.feat_type}_{args.kernel_size}_{args.out_first}...")
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f'last_{args.criterion}_{args.label_type}_{args.num_convs}_{args.feat_type}_{args.kernel_size}_{args.out_first}_classifier.ckpt'))


def evaluate():
    model.eval()
    best_acc = 0
    least_loss = float('inf')
    best_head_correct = 0
    top5_acc = 0
    total_loss = 0.0
    num_classes = 2
    class_names = ["not building", "building"]
    ignore_index = 2

    metric_total = SCDMetric(
        num_classes, class_names,
        ignore_index
    ) 

    dataloader = val_dataloader

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) 
                     for key, value in batch.items()}
        
        outputs = model(batch)
        logits = outputs["hs_logits"].squeeze(dim=0)
        logits_heads = outputs['logits_heads']

        gt = batch['gt']
        if args.criterion in ['me', 'mse']:
            P = (batch[args.label_type] / 100).squeeze(dim=0)
        elif args.criterion in ['crossentropy']:
            P = soft_target(batch[args.label_type]).squeeze(dim=0)

        if torch.any(torch.isnan(P)):
            print("Nan encountered")

        best_head = torch.argmax(logits, dim=-1)
        best_pred = logits_heads[best_head].argmax(dim=2)

        top5_target = torch.topk(P, k=5, dim=-1).indices

        top5_count = torch.any(top5_target == best_head.unsqueeze(-1), dim=-1)

        loss = criterion(logits, P)
        total_loss += loss.item()

        best_head_correct += (best_head == torch.argmax(P, dim=-1)).sum().item()
        top5_acc += top5_count.sum().item()

        metric_total.update(best_pred, gt)

    mious = metric_total.compute()
    best_head_accuracy = 100 * best_head_correct / len(dataloader)
    top5_accuracy = 100 * top5_acc / len(dataloader)
    total_loss = total_loss / len(dataloader)
    
    print(f'Best head accuracy on val set: {best_head_accuracy:.1f}%')
    print(f'Top 5 head accuracy on val set: {top5_accuracy:.1f}%')
    print(f"Loss on val set: {total_loss:.2f}")
    return best_head_accuracy, top5_accuracy, total_loss, mious['acc'], mious['miou']

if __name__ == "__main__":
    train()