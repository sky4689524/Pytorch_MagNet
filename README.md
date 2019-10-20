# Pytorch_MagNet

Pytorch implementation for [MagNet: a Two-Pronged Defense against Adversarial Examples](https://arxiv.org/pdf/1705.09064.pdf), by Meng, D., & Chen, H, at CCS 2017. Also, codes are referenced from 
https://github.com/Trevillie/MagNet. The main algorithms are included in 'defense.py' and 'worker.py' This repository is to defend segmentation models from adversarial attacks by using MagNet defense strategy.

## Usage

'train_autoencoder.py' : train autoencoder models for defense.
'defense.py' : test MagNet defense to segmentation model against adversarial attacks.

simple example

```
python train_autoencoder.py
```

```
python defense.py --model UNet --model_path "path" --reformer autoencoder1 --detector autoencoder1 \
--reformer_path checkpoints/autoencoder1.pth --detector_path checkpoints/autoencoder1.pth
```

You can see more detailed arguments.

```
python train_autoencoder.py -h
```
```
python defense.py -h
```
