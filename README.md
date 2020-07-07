# FastDVDnet: End-to-end video denoiser

A state-of-the-art, simple and fast network for Deep Video Denoising which uses no motion compensation.

Find the original project on: [FastDVDnet](https://github.com/m-tassano/fastdvdnet)

## Overview

This source code provides a PyTorch implementation of the FastDVDnet video denoising algorithm, as in
Tassano, Matias and Delon, Julie and Veit, Thomas. ["FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation", arXiv preprint arXiv:1907.01361 (2019).](https://arxiv.org/abs/1907.01361)

## Exammple

<img src="https://github.com/m-tassano/fastdvdnet/raw/master/img/9831-teaser.gif" width=256> <img src="https://github.com/m-tassano/fastdvdnet/raw/master/img/psnrs.png" width=600>

## Architecture

<img src="https://github.com/m-tassano/fastdvdnet/raw/master/img/arch.png" heigth=350>

## Code User Guide

### Dependencies

The code runs on Python +3.6. You can create a conda environment with all the dependecies by running (Thanks to Antoine Monod for the .yml file)
```
conda env create -f requirements.yml -n <env_name>
```

Note: this project needs the [NVIDIA DALI](https://github.com/NVIDIA/DALI) package for training. The tested version of DALI is 0.10.0. If you prefer to install it yourself (supposing you have CUDA 10.0), you need to run
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali==0.10.0
```

### Testing

If you want to denoise an image sequence using the pretrained model you can execute

```
./run.sh video/myvideo.mp4 25 25
```

**NOTES**
* 1st argument: Your video filename path
* 2nd argument: Noise level used on test set. The model has been trained for values of noise in [5, 55]
* 3rd argument: Max number of frames to load per sequence
* The output file is in the same folder with the same name and ending with [...]-DENOISED-[...].mkv

## ABOUT

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved. This file is offered as-is,
without any warranty.

* Author    : Matias Tassano `mtassano at gopro dot com`
* Copyright : (C) 2019 Matias Tassano
* Licence   : GPL v3+, see GPLv3.txt

The sequences are Copyright GoPro 2018
