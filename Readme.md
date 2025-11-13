# Assignment 3 – Computer Vision  
Author: Prajwal Bhandary (St125974)

## Overview
This repository contains three tasks covering classical and deep learning approaches to segmentation and generative modeling.

### Task 1 – Graph Cut Segmentation ([Task_1.ipynb](Task_1.ipynb))
- Detect persons using GroundingDINO (zero-shot object detection via Hugging Face `transformers`).
- Initialize and refine foreground masks with OpenCV GrabCut.
- Compare segmentation quality across 1, 3, 5 iterations and visualize masks, crops, and overlays.

### Task 2 – Fully Convolutional Networks for Semantic Segmentation ([Task_2.ipynb](Task_2.ipynb))
- Implement FCN-16s with VGG16 backbone (skip connection from pool4).
- Compare two upsampling strategies: transpose convolution (learnable) vs bilinear interpolation (fixed).
- Train on a subset of Pascal VOC; evaluate with Mean IoU and Pixel Accuracy; plot training curves and qualitative predictions.

### Task 3 – Variational Autoencoders on MNIST ([Task_3.ipynb](Task_3.ipynb))
- Build a convolutional VAE (encoder → μ, logσ²; reparameterization; transpose-conv decoder).
- Train models with latent dimensions 128 and 256.
- Visualize reconstructions, random generations, latent-space PCA (2D/3D), and interpolation between digit embeddings.
- Side-by-side comparison of reconstruction and sample quality for both latent sizes.

## Dependencies
See [requirements.txt](requirements.txt) for all required Python packages.

## How to Use
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebooks in order if desired; each is self-contained.
3. Assets used for figures are in `assets/`; sample images for Task 1 in `images/`.

## Notes
- GroundingDINO weights download on first run (internet required).
- Small Pascal VOC subset used for faster experimentation.
- VAE training stores latent snapshots periodically for visualization.
