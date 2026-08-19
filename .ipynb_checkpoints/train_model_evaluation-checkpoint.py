import random
import cv2
from train_model import divide_train_test
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



def _image_distort_eva(img: np.ndarray,
                       mask: np.ndarray,
                       max_rotation_degree=45,
                       no_distortion=0.2,
                       scaling=True,
                       translating=True) -> np.ndarray:
    distorted = img.astype(np.float32, copy=True)
    height, width = img.shape

    # 1. Random noise (1% of max intensity)
    max_intensity = np.max(distorted)
    noise = np.random.normal(0, 0.01 * max_intensity, (height, width))
    distorted[mask] = np.clip(distorted[mask] + noise[mask], 0, None)
    if random.random() < no_distortion:
        return distorted

    # 3&4. Random rotation (-45 to 45 degrees) and scaling (90% to 110%), Random translation (up to 10% of image size)
    angle = random.uniform(-max_rotation_degree, max_rotation_degree)
    center = (width // 2, height // 2)
    if scaling:
        scale = random.uniform(0.90, 1.1)
    else:
        scale = 1
    if translating:
        max_trans_x = int(0.1 * width)
        max_trans_y = int(0.1 * height)
        trans_x = random.randint(-max_trans_x, max_trans_x)
        trans_y = random.randint(-max_trans_y, max_trans_y)
    else:
        trans_x = 0
        trans_y = 0

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:, 2] += (trans_x, trans_y)
    distorted = cv2.warpAffine(distorted, M, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 5. Random pixel dropout(5 % of pixels)
    dropout_mask = np.random.random((height, width)) < 0.05
    distorted[dropout_mask] = 0

    return distorted


def _augment_worker_eva(args):
    img, shape_mask, aug_per_sample, max_rot, no_d, sc, tr = args
    out = np.empty((aug_per_sample, *img.shape), dtype=np.float32)
    for j in range(aug_per_sample):
        out[j] = _image_distort_eva(
            img, shape_mask,
            max_rotation_degree=max_rot,
            no_distortion=no_d,
            scaling=sc,
            translating=tr
        )
    return out


def get_training_test_data_eva(inputs, aug_per_sample=10,
                               max_rotation_degree=45,
                               no_distortion=0.2,
                               scaling=True,
                               translating=True):
    train_inputs, test_inputs = divide_train_test(inputs)

    # Prepare train data
    train_data = []  # List: [[aug_intensity_array, corr_matrix]]
    for s_input in train_inputs:
        sample_id, shape_mask, _, intensity_array = s_input
        num_images = len(intensity_array)  # Shape: (n, w, h)
        intensity_array[:, ~shape_mask] = 0
        # Augmentation
        num_workers = min(os.cpu_count(), num_images)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = [
                (
                    intensity_array[i],
                    shape_mask,
                    aug_per_sample,
                    max_rotation_degree,
                    no_distortion,
                    scaling,
                    translating
                )
                for i in range(num_images)
            ]
            results = list(
                tqdm(
                    executor.map(_augment_worker_eva, tasks),
                    total=num_images,
                    desc=f"Training image augmentation in {sample_id}"
                )
            )
        aug_intensity_array = np.stack(results, axis=0)

        # Pearson similarity
        X = intensity_array[:, shape_mask]
        X = X.astype(np.float32, copy=False)
        X -= X.mean(axis=1, keepdims=True)
        X /= X.std(axis=1, keepdims=True)
        corr = X @ X.T / (X.shape[1] - 1)  # Pearson for fast evaluation
        corr = corr.astype(np.float32, copy=False)

        train_data.append({"augmented_images": aug_intensity_array,
                           "img_corr": corr})

    # Prepare test data
    test_data = []
    print('Preparing test data.')
    for s_input in test_inputs:
        sample_id, shape_mask, _, intensity_array = s_input
        num_images = len(intensity_array)  # Shape: (n, w, h)
        intensity_array[:, ~shape_mask] = 0

        num_workers = min(os.cpu_count(), num_images)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = [
                (
                    intensity_array[i],
                    shape_mask,
                    aug_per_sample,
                    max_rotation_degree,
                    no_distortion,
                    scaling,
                    translating
                )
                for i in range(num_images)
            ]
            results = list(
                tqdm(
                    executor.map(_augment_worker_eva, tasks),
                    total=num_images,
                    desc=f"Test image augmentation in {sample_id}"
                )
            )
        aug_intensity_array = np.stack(results, axis=0)

        X = intensity_array[:, shape_mask]
        X = X.astype(np.float64, copy=False)
        X -= X.mean(axis=1, keepdims=True)
        X /= X.std(axis=1, keepdims=True)
        img_corr = X @ X.T / (X.shape[1] - 1)  # Pearson for fast evaluation

        test_data.append({
            "augmented_images": aug_intensity_array,
            "intensity_array": intensity_array,
            "num_images": len(intensity_array),
            "img_corr": img_corr
        })

    print('Complete train and test data preparation.')
    return train_data, test_data


class ConvBlock(nn.Module):
    """
    Single-branch replacement for Inception block.
    Keeps similar parameter scale using stacked convs.
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()

        mid_channels = out_channels // 2  # bottleneck-style

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.block(x))


class MultiscaleEmbedding_baseline(nn.Module):
    def __init__(self,
                 embedding_dim: int = 28,
                 num_blocks: int = 4,
                 block_out_channels: int = 256,
                 dropout_p: float = 0.2,
                 projection_channels: int = 256,
                 in_channels: int = 1,
                 ):
        super().__init__()

        # Stem (same as before)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Conv stack (replace Inception)
        blocks = []
        current_channels = 64

        for i in range(num_blocks):
            blocks.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=block_out_channels,
                    dropout_p=dropout_p,
                )
            )
            current_channels = block_out_channels

            # Same downsampling strategy
            if i < num_blocks - 1:
                blocks.append(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )

        self.conv_stack = nn.Sequential(*blocks)

        # Projection (same)
        self.projection = nn.Sequential(
            nn.Conv2d(current_channels, projection_channels, kernel_size=1),
            nn.BatchNorm2d(projection_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding head (same)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_head = nn.Sequential(
            nn.Linear(projection_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_stack(x)
        x = self.projection(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding_head(x)

        return F.normalize(x, p=2, dim=1)