# Deep Convolutional Generative Adversarial Network (DCGAN)

This repository contains an implementation of the DCGAN architecture described in the paper ["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"](https://arxiv.org/abs/1511.06434) by Radford et al. (2015).

## Overview

DCGANs are a class of Convolutional Neural Networks that learn to generate realistic images through adversarial training. This implementation follows the architectural guidelines specified in the paper, including:

- Using strided convolutions (discriminator) and fractional-strided convolutions (generator) instead of pooling layers
- Batch normalization in both the generator and discriminator
- Removing fully connected hidden layers
- Using ReLU activation in generator (except for output, which uses Tanh)
- Using LeakyReLU activation in the discriminator

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- tqdm
- PIL (Pillow)
- argparse

Install all dependencies with:

```bash
pip install torch torchvision tqdm matplotlib pillow argparse
```

## Datasets

This implementation supports training on two datasets:

### 1. CelebA Faces

The CelebA dataset contains over 200,000 celebrity face images with various attributes.

To prepare the CelebA dataset:
1. Download the dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Extract the images to a directory
3. Organize the directory structure as follows:
   ```
   celeba/
   └── img_align_celeba/
       ├── 000001.jpg
       ├── 000002.jpg
       └── ...
   ```

### 2. LSUN Bedrooms

The LSUN bedroom dataset contains around 3 million bedroom images.

To prepare the LSUN dataset:
1. Download the bedroom class data using the [LSUN download script](https://github.com/fyu/lsun)
2. Extract the data to a directory
3. Organize the directory structure as follows:
   ```
   lsun/
   └── bedroom_train/
       ├── data.mdb
       └── lock.mdb
   ```

## Dataset Preprocessing

Both datasets are preprocessed during training with the following steps:
1. Resize images to the desired size (default: 64x64)
2. Center crop to ensure square dimensions
3. Convert to tensors
4. Normalize to the range [-1, 1]

## Training the Model

To train the model on the CelebA dataset:

```bash
python train.py --dataset celeba --data_dir path/to/celeba --output_dir results_celeba
```

To train the model on the LSUN bedrooms dataset:

```bash
python train.py --dataset lsun --data_dir path/to/lsun --output_dir results_lsun
```

Additional training options:
```
--n_epochs: Number of training epochs (default: 200)
--batch_size: Batch size (default: 128)
--latent_dim: Dimension of the latent space (default: 100)
--lr: Learning rate (default: 0.0002)
--beta1: Beta1 parameter for Adam optimizer (default: 0.5)
--img_size: Image size (default: 64)
--sample_interval: Interval between image sampling (default: 500)
--save_interval: Interval between model saving (default: 10)
--seed: Random seed for reproducibility (default: 42)
--num_workers: Number of worker threads for dataloader (default: 4)
--no_cuda: Flag to avoid using CUDA even when available
```

## Generating Images

To generate images using a trained generator:

```bash
python generate.py --model_path path/to/generator_model.pth --output_dir generated_images
```

Generation options:
```
--latent_dim: Dimension of the latent space (default: 100)
--channels: Number of image channels (default: 3)
--img_size: Image size (default: 64)
--batch_size: Number of images to generate in each batch (default: 64)
--n_batches: Number of batches to generate (default: 1)
--seed: Random seed for reproducibility
--no_cuda: Flag to avoid using CUDA even when available
```

## Visualization Tools

### Visualizing Training Progress

To visualize the progress of the generator during training:

```bash
python visualization.py progress --image_dir path/to/sample/images --output_path progress.png
```

### Interpolating in Latent Space

To generate images by interpolating between points in the latent space:

```bash
python visualization.py interpolate --model_path path/to/generator_model.pth --output_path interpolation.png
```

## Expected Outputs

### Training Output
- Regular checkpoints of the generator and discriminator models
- Sample images showing training progress
- Runtime information about losses and training speed

### Generated Images
- Convincing bedroom scenes (if trained on LSUN) or facial images (if trained on CelebA)
- The generated images should have the following characteristics:
  - For LSUN: Recognizable bedroom layouts with furniture, windows, and other bedroom elements
  - For CelebA: Realistic faces with proper features, though they
