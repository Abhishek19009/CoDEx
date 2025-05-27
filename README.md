<div align="center">
<h2>
Spatial Domain Generalization for Semantic Segmentation of Satellite Image Time Series 

<a href="https://imagine.enpc.fr/~abhishek.kuriyal/">Abhishek Kuriyal</a>&emsp;
<a href="https://imagine.enpc.fr/~elliot.vincent/">Elliot Vincent</a>&emsp;
<a href="https://imagine.enpc.fr/~aubrym/">Mathieu Aubry</a>&emsp;
<a href="https://imagine.enpc.fr/~loic.landrieu/">Loic Landrieu</a>

<p></p>

</h2>
</div>

<i>This work is an extension of satellite image time series semantic change detection (SITS-SCD) task [1]. </i>

Checkout the official PyTorch implementation available here [**Satellite Image Time Series Semantic Change Detection: Novel Architecture and Analysis of Domain Shift**](https://github.com/ElliotVincent/SitsSCD).

## Installation & Instructions :gear:

### 1. Clone the repo

#### Recursive

```
git clone git@github.com:Abhishek19009/MultiHead-MultiUTAE.git
```

#### HTTPS

```
git clone https://github.com/Abhishek19009/MultiHead-MultiUTAE
```

### 2. Dataset Download
Similar to [1], this works benchmarks on processed versions of DynamicEarthNet [2] and MUDS [3]. 
Datasets can be downloaded using code below or following the links:
[DynamicEarthNet](https://drive.google.com/file/d/1cMP57SPQWYKMy8X60iK217C28RFBkd2z/view?usp=drive_link) (7.09G) and
[MUDS](https://drive.google.com/file/d/1RySuzHgQDSgHSw2cbriceY5gMqTsCs8I/view?usp=drive_link) (245M).


```
cd MultiHead-MultiUTAE
mkdir datasets
cd datasets
gdown 1RySuzHgQDSgHSw2cbriceY5gMqTsCs8I
unzip Muds.zip
gdown 1cMP57SPQWYKMy8X60iK217C28RFBkd2z
unzip DynamicEarthNet.zip
```

### 3. Create and activate virtual environment

```
conda create -n mmsits pytorch=2.0.1 torchvision=0.15.2 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate mmsits
pip install -r requirements.txt
```
Implementation uses PyTorch, PyTorch Lightning and Hydra.

## Usage :rocket:

Training is split into two steps.

### Step 1: Training MultiHead MultiUTAE
This step involves training the multihead architecture for different datasets.

For DynamicEarthNet:
```
python train_multiutae.py dataset=dynamicearthnet experiment=multihead_multiutae_de
```

For MUDS:
```
python train_multiutae.py dataset=muds experiment=multihead_multiutae_muds
```

### Step 2: Training Head Selector (Domain Generalization Network)

The head selector can be trained using two methods:

#### Method 1: Preprocessing before Training

This method involves preparing the data beforehand, which reduces computational requirements during training. 
Transforms are applied during head selector training.

Prepare the dataset (patches, ground truth SITS mask, and multihead performance metrics such as mIoU and Accuracy as targets):

For DynamicEarthNet:

```
sh scripts/train_head_selector_de.sh
```

For Muds:

```
sh scripts/train_head_selector_muds.sh
```

#### Method 2: Direct training (Experimental)

This method skips dataset preparation but computes multihead performance metrics in real-time. It is slower than Method 1.

Start direct training:
```
python train_head_selector.py dataset=dynamicearthnet experiment=multihead_selector_de
```

## Citing

```bibtex
@article{temp,
    title = {MultiHead MultiUTAE},
    author = {},
    journal = {arXiv},
    year = {2025},
}
```

## Bibliography

[1] Elliot Vincent et al. *Satellite image time series semantic change detection: Novel architecture and analysis of domain shift.* arXiv preprint arXiv:2407.07616, 2024.

[2] Aysim Toker et al. *Dynamicearthnet: Daily multi-spectral satellite dataset for semantic change segmentation*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21158–21167, 2022.

[3] Vivien Sainte Fare Garnot et al. *Panoptic segmentation of satellite image time series with convolutional
temporal attention networks*. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4872–4881, 2021

