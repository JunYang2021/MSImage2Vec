import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import spearmanr

import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from embedding_models import ResNetEmbedding, ViTEmbedding, EfficientNetEmbedding, SimpleCNNEmbedding, \
    MultiscaleEmbedding
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


def divide_train_test(inputs: List, seed=None):
    """

    :param seed: Random seed
    :param inputs: list of data of all samples.
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :return:
    """
    if seed is not None:
        np.random.seed(seed)

    train_inputs, test_inputs = [], []
    for s_input in inputs:
        # generate random 80% of m/z to train, and 20% to test
        sample_id, shape_mask, mz_array, intensity_array = s_input
        num_images = len(mz_array)

        indices = np.random.permutation(num_images)
        split_idx = int(0.8 * num_images)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_mz = mz_array[train_indices]
        test_mz = mz_array[test_indices]
        train_intensity = intensity_array[train_indices]
        test_intensity = intensity_array[test_indices]

        train_s = [sample_id, shape_mask.copy(), train_mz, train_intensity]
        test_s = [sample_id, shape_mask.copy(), test_mz, test_intensity]
        train_inputs.append(train_s)
        test_inputs.append(test_s)

    return train_inputs, test_inputs


def _correlation(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """Calculate Pearson correlation between two images"""
    flat1 = img1[mask].flatten()
    flat2 = img2[mask].flatten()
    # return np.corrcoef(flat1, flat2)[0, 1]
    corr, _ = spearmanr(flat1, flat2)
    return corr


def _image_distort(img: np.ndarray,
                   mask: np.ndarray,
                   alpha_height_ratio=2,
                   sigma_height_ratio=0.12,
                   max_rotation_degree=45) -> np.ndarray:
    distorted = img.astype(np.float32, copy=True)
    height, width = img.shape

    # 1. Random noise (1% of max intensity)
    max_intensity = np.max(distorted)
    noise = np.random.normal(0, 0.01 * max_intensity, (height, width))
    distorted[mask] = np.clip(distorted[mask] + noise[mask], 0, None)

    # 2. Elastic deformations (non-linear distortions)
    if random.random() > 0.5:  # Apply to 50% of images
        # The strength of elastic deformation. Larger value means bigger deformation
        # alpha = random.uniform(height * 1.8, height * 2.8)
        # The smoothness of elastic deformation. Larger value means smoother deformation
        # sigma = random.uniform(height * 0.1, height * 0.15)
        alpha = alpha_height_ratio * height
        sigma = sigma_height_ratio * height

        # Displacement fields
        # np.random.rand range: [0, 1)  np.random.ran() * 2 -1 range: [-1, 1)
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x_distorted = np.clip(x + dx, 0, width - 1)
        y_distorted = np.clip(y + dy, 0, height - 1)

        distorted = map_coordinates(distorted, [y_distorted, x_distorted], order=1, mode='constant')

    # 3&4. Random rotation (-45 to 45 degrees) and scaling (90% to 110%), Random translation (up to 10% of image size)
    angle = random.uniform(-max_rotation_degree, max_rotation_degree)
    center = (width // 2, height // 2)
    scale = random.uniform(0.90, 1.1)
    max_trans_x = int(0.1 * width)
    max_trans_y = int(0.1 * height)
    trans_x = random.randint(-max_trans_x, max_trans_x)
    trans_y = random.randint(-max_trans_y, max_trans_y)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[: 2] += (trans_x, trans_y)
    distorted = cv2.warpAffine(distorted, M, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    """# 3. Random rotation (-45 to 45 degrees) and scaling (90% to 110%)
    angle = random.uniform(-max_rotation_degree, max_rotation_degree)
    center = (width // 2, height // 2)
    scale = random.uniform(0.90, 1.1)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    distorted = cv2.warpAffine(distorted, M, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 4. Random translation (up to 10% of image size)
    max_trans_x = int(0.1 * width)
    max_trans_y = int(0.1 * height)
    trans_x = random.randint(-max_trans_x, max_trans_x)
    trans_y = random.randint(-max_trans_y, max_trans_y)
    M = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    distorted = cv2.warpAffine(distorted, M, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)"""

    # 5. Random pixel dropout(5 % of pixels)
    dropout_mask = np.random.random((height, width)) < 0.05
    distorted[dropout_mask] = 0

    return distorted


class ImageDataset(Dataset):
    def __init__(self, train_data: List, pairs_per_sample: int = 2000):
        self.train_data = train_data
        self.pairs_per_sample = pairs_per_sample
        self.total_pairs = len(train_data) * pairs_per_sample

    def __len__(self) -> int:
        return self.total_pairs

    def __getitem__(self, idx: int):
        sample_idx = idx // self.pairs_per_sample
        s_train = self.train_data[sample_idx]
        aug_intensity_array = s_train["augmented_images"] # Shape: (n, aug_per_sample, w, h)
        corr_matrix = s_train["img_corr"] # Shape: (n, n)

        num_images, num_aug = aug_intensity_array.shape[0], aug_intensity_array.shape[1]
        i, j = random.sample(range(num_images), 2)
        i_a, j_a = random.sample(range(num_aug), 2)
        img1 = aug_intensity_array[i, i_a]
        img2 = aug_intensity_array[j, j_a]
        img1 = torch.from_numpy(img1).unsqueeze(0).float()
        img2 = torch.from_numpy(img2).unsqueeze(0).float()

        corr = corr_matrix[i, j]

        return (
            img1, # Shape: 1, height, width
            img2, # Shape: 1, height, width
            torch.tensor(corr, dtype=torch.float32)
        )


class TripletImageDataset(Dataset):
    def __init__(self, inputs: List, triplets_per_sample: int = 20000,
                 alpha_height_ratio=2,
                 sigma_height_ratio=0.12,
                 max_rotation_degree=45
                 ):
        """self.triplets = []
        for s_input in inputs:
            sample_id, shape_mask, _, intensity_array = s_input
            num_images = len(intensity_array)

            for _ in tqdm(range(triplets_per_sample),
                          desc=f"Sampling triplets from {sample_id}",
                          unit="triplet"):
                i1, i2, i3 = random.sample(range(num_images), 3)
                img1 = intensity_array[i1]
                img2 = intensity_array[i2]
                img3 = intensity_array[i3]

                corr12 = _correlation(img1, img2, shape_mask)
                corr13 = _correlation(img1, img3, shape_mask)

                if corr12 >= corr13:
                    pos_img, neg_img = img2, img3
                    corr_ap, corr_an = corr12, corr13
                else:
                    pos_img, neg_img = img3, img2
                    corr_ap, corr_an = corr13, corr12

                # Apply random image distort
                anchor_img = _image_distort(img1, shape_mask)
                pos_img = _image_distort(pos_img, shape_mask)
                neg_img = _image_distort(neg_img, shape_mask)
                self.triplets.append((anchor_img, pos_img, neg_img, corr_ap, corr_an))"""
        self.inputs = inputs
        self.triplets_per_sample = triplets_per_sample
        self.total_triplets = len(inputs) * triplets_per_sample
        self.alpha_height_ratio = alpha_height_ratio
        self.sigma_height_ratio = sigma_height_ratio
        self.max_ratation_degree = max_rotation_degree

    def __len__(self) -> int:
        return self.total_triplets

    def __getitem__(self, idx: int):
        """anchor_img, pos_img, neg_img, corr_ap, corr_an = self.triplets[idx]

        to_tensor = lambda img: torch.FloatTensor(img).unsqueeze(0)
        return (
            to_tensor(anchor_img),
            to_tensor(pos_img),
            to_tensor(neg_img),
            torch.tensor(corr_ap, dtype=torch.float32),
            torch.tensor(corr_an, dtype=torch.float32)
        )"""
        sample_idx = idx // self.triplets_per_sample
        s_input = self.inputs[sample_idx]
        sample_id, shape_mask, _, intensity_array = s_input

        num_images = len(intensity_array)
        i1, i2, i3 = random.sample(range(num_images), 3)
        img1, img2, img3 = intensity_array[i1], intensity_array[i2], intensity_array[i3]
        img1[~shape_mask] = 0
        img2[~shape_mask] = 0
        img3[~shape_mask] = 0

        corr12 = _correlation(img1, img2, shape_mask)
        corr13 = _correlation(img1, img3, shape_mask)

        if corr12 >= corr13:
            pos_img, neg_img = img2, img3
            corr_ap, corr_an = corr12, corr13
        else:
            pos_img, neg_img = img3, img2
            corr_ap, corr_an = corr13, corr12

        anchor_img = _image_distort(img1, shape_mask,
                                    alpha_height_ratio=self.alpha_height_ratio,
                                    sigma_height_ratio=self.sigma_height_ratio,
                                    max_rotation_degree=self.max_ratation_degree
                                    )
        pos_img = _image_distort(pos_img, shape_mask,
                                 alpha_height_ratio=self.alpha_height_ratio,
                                 sigma_height_ratio=self.sigma_height_ratio,
                                 max_rotation_degree=self.max_ratation_degree)
        neg_img = _image_distort(neg_img, shape_mask,
                                 alpha_height_ratio=self.alpha_height_ratio,
                                 sigma_height_ratio=self.sigma_height_ratio,
                                 max_rotation_degree=self.max_ratation_degree
                                 )

        to_tensor = lambda img: torch.FloatTensor(img).unsqueeze(0)
        return (
            to_tensor(anchor_img),
            to_tensor(pos_img),
            to_tensor(neg_img),
            torch.tensor(corr_ap, dtype=torch.float32),
            torch.tensor(corr_an, dtype=torch.float32)
        )


class CorrelationLoss(nn.Module):
    # Loss function that measures the difference between embedding cosine similarity
    # and actual image correlation

    def __init__(self):
        super().__init__()

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
                target_corr: torch.Tensor) -> torch.Tensor:
        
        # Args:
        #     emb1, emb2: Embeddings of the two images (batch_size x embedding_dim)
        #     target_corr: Target correlation coefficients (batch_size)
        # Margin-based MSE
        
        cos_sim = F.cosine_similarity(emb1, emb2)
        # diff = torch.abs(cos_sim - target_corr)
        # loss = F.mse_loss(cos_sim, target_corr)
        # loss = torch.mean(torch.clamp(diff - self.margin, min=0) ** 2)  # 仅惩罚超出margin的差异
        # return loss
        loss = F.mse_loss(cos_sim, target_corr.float())

        return loss


class JointLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, anchor, pos, neg, corr_ap, corr_an):
        sim_ap = F.cosine_similarity(anchor, pos)
        sim_an = F.cosine_similarity(anchor, neg)

        # Ranking loss (Triplet)
        rank_loss = F.relu(-(sim_ap - sim_an)).mean()
        reg_loss = (F.mse_loss(sim_ap, corr_ap.float()) +
                    F.mse_loss(sim_an, corr_an.float())) / 2

        total_loss = reg_loss + self.alpha * rank_loss

        return total_loss, reg_loss.detach(), rank_loss.detach()


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.X, self.Y = None, None
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.ion()
        self.fig.show()
        # plt.show()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for i, (x_vals, y_vals, fmt) in enumerate(zip(self.X, self.Y, self.fmts)):
            self.axes[0].plot(x_vals, y_vals, fmt)
            # 在最后一个点上添加文字标注
            if x_vals and y_vals:
                last_x = x_vals[-1]
                last_y = y_vals[-1]
                self.axes[0].text(last_x, last_y, f'{last_y:.4f}', fontsize=8,
                                  ha='left', va='bottom')
        self.config_axes()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def get_training_test_data(inputs, aug_per_sample=10,
                            alpha_height_ratio=2,
                             sigma_height_ratio=0.12,
                             max_rotation_degree=45):
    train_inputs, test_inputs = divide_train_test(inputs)

    # Prepare train data
    train_data = []  # List: [[aug_intensity_array, corr_matrix]]
    for s_input in train_inputs:
        sample_id, shape_mask, _, intensity_array = s_input
        num_images = len(intensity_array)   # Shape: (n, w, h)
        intensity_array[:, ~shape_mask] = 0
        # Augmentation
        aug_intensity_array = np.empty((num_images, aug_per_sample, intensity_array.shape[1], intensity_array.shape[2]),
                                       dtype=np.float32)  # Shape: (n, aug_per_sample, w, h)
        for i in tqdm(range(num_images),
                          desc=f"Training image augmentation in {sample_id}"):
            img = intensity_array[i]
            for j in range(aug_per_sample):
                dist_img = _image_distort(img, shape_mask,
                                          alpha_height_ratio=alpha_height_ratio,
                                          sigma_height_ratio=sigma_height_ratio,
                                          max_rotation_degree=max_rotation_degree
                                          )
                aug_intensity_array[i, j] = dist_img

        # Spearman similarity
        # corr = np.empty((num_images, num_images), dtype=np.float32)  # Shape: (n, n)
        # for i in tqdm(range(num_images),
        #                   desc=f"Training image correlation matrix in {sample_id}"):
        #     img1 = intensity_array[i]
        #     for j in range(num_images):
        #         img2 = intensity_array[j]
        #
        #         c = _correlation(img1, img2, shape_mask)  # Spearman
        #         corr[i, j] = c
        # Pearson similarity
        X = intensity_array[:, shape_mask]
        X = X.astype(np.float32, copy=False)
        X -= X.mean(axis=1, keepdims=True)
        X /= X.std(axis=1, keepdims=True)
        corr = X @ X.T / (X.shape[1] - 1) # Pearson for fast evaluation
        corr = corr.astype(np.float32, copy=False)

        train_data.append({"augmented_images": aug_intensity_array,
                           "img_corr": corr})

    # Prepare test data
    test_data = []
    print('Preparing test data.')
    for s_input in test_inputs:
        sample_id, shape_mask, _, intensity_array = s_input
        intensity_array[:, ~shape_mask] = 0

        # Vectorized tensor conversion (Slow)
        # imgs = torch.from_numpy(intensity_array).float().unsqueeze(1)  # (N, 1, H, W)
        # mask = torch.from_numpy(shape_mask).bool()
        #
        # imgs[:, 0, ~mask] = 0.0
        #
        # img_corr = np.zeros((imgs.shape[0], imgs.shape[0]))
        # print(f"Computing similarity matrix for {len(intensity_array)} ion images in sample: {sample_id}")
        # for i in range(imgs.shape[0]):
        #     for j in range(imgs.shape[0]):
        #         img_corr[i, j] = _correlation(
        #             intensity_array[i],
        #             intensity_array[j],
        #             shape_mask
        #         )
        X = intensity_array[:, shape_mask]
        X = X.astype(np.float64, copy=False)
        X -= X.mean(axis=1, keepdims=True)
        X /= X.std(axis=1, keepdims=True)
        img_corr = X @ X.T  # Pearson for fast evaluation

        test_data.append({
            "intensity_array": intensity_array,
            "num_images": len(intensity_array),
            "img_corr": img_corr
        })

    # save_data = (train_data, test_data)
    # with open(output_file, 'wb') as f:
    #     pickle.dump(save_data, f)
    print('Complete train and test data preparation.')
    return train_data, test_data


def train_embedding(inputs: List, model: nn.Module, args) -> tuple[
    list[float], list[float], list[float], list[float], list[float], list[float]]:
    """Train the embedding model

    Args:
        inputs: List of sample data
        model: The embedding model to train
        args: Training arguments
        (seed, train_pairs_per_sample, alpha_height_ratio, sigma_height_ratio, max_rotation_degree, batch_size, rank_reg_loss_ratio, optimizer, lr,
        max_epochs, early_stop_patience, early_stop_delta, output_path, model_data_file)

    Returns:
        Tuple of (train_loss_log, test_loss_log)
    """
    time0 = time.time()
    # Split data into train and test
    train_inputs, test_inputs = divide_train_test(inputs, seed=args.seed)

    # Create datasets
    train_dataset = TripletImageDataset(train_inputs,
                                        triplets_per_sample=args.train_pairs_per_sample,
                                        alpha_height_ratio=args.alpha_height_ratio,
                                        sigma_height_ratio=args.sigma_height_ratio,
                                        max_rotation_degree=args.max_rotation_degree
                                        )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)

    time1 = time.time()

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = JointLoss(alpha=args.rank_reg_loss_ratio)
    optimizerD = {'Adam': torch.optim.Adam(model.parameters(), lr=args.lr),
                  'SGD': torch.optim.SGD(model.parameters(), lr=args.lr),
                  'RMSprop': torch.optim.RMSprop(model.parameters(), lr=args.lr),
                  'Adagrad': torch.optim.Adagrad(model.parameters(), lr=args.lr),
                  'AdamW': torch.optim.AdamW(model.parameters(), lr=args.lr)}
    optimizer = optimizerD[args.optimizer]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1)

    train_loss_log = []
    test_loss_log = []
    train_reg_log = []
    train_rank_log = []
    test_sim_log = []
    test_class_log = []

    best_test_loss = float('inf')

    """animator = Animator(
        xlabel='Epoch', ylabel='Loss',
        legend=[
            'Train Total', 'Test Total',
            'Train Reg', 'Test Similarity',
            'Train Rank', 'Test Classification'
        ],
        xlim=[1, args.max_epochs],
        fmts=('-', 'm--', 'g-.', 'r', 'c:', 'b-.'),
        figsize=(8, 5)
    )"""

    # Prepare test images
    test_cache = []

    for s_input in test_inputs:
        sample_id, shape_mask, _, intensity_array = s_input

        # Vectorized tensor conversion (Slow)
        imgs = torch.from_numpy(intensity_array).float().unsqueeze(1)  # (N, 1, H, W)
        mask = torch.from_numpy(shape_mask).bool()
        #
        imgs[:, 0, ~mask] = 0.0
        #
        # img_corr = np.zeros((imgs.shape[0], imgs.shape[0]))
        # print(f"Computing similarity matrix for {len(intensity_array)} ion images in sample: {sample_id}")
        # for i in range(imgs.shape[0]):
        #     for j in range(imgs.shape[0]):
        #         img_corr[i, j] = _correlation(
        #             intensity_array[i],
        #             intensity_array[j],
        #             shape_mask
        #         )
        X = intensity_array[:, shape_mask]
        X = X.astype(np.float64, copy=False)
        X -= X.mean(axis=1, keepdims=True)
        X /= X.std(axis=1, keepdims=True)
        img_corr = X @ X.T

        test_cache.append({
            "images": imgs,
            "intensity_array": intensity_array,
            "shape_mask": shape_mask,
            "num_images": len(intensity_array),
            "img_corr": img_corr
        })

    # Training loop
    early_stop_counter = 0
    best_epoch = 0
    best_model_state = None
    for epoch in range(args.max_epochs):  # Change with early stopping, max epoch is args.epochs
        model.train()
        train_t0 = time.time()
        running_loss = 0.0
        running_reg = 0.0
        running_rank = 0.0

        # Training phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epochs} [Train]")
        for img1, img2, img3, corr12, corr13 in pbar:
            time1 = time.time()
            img1, img2, img3, corr12, corr13 = img1.to(device), \
                                               img2.to(device), img3.to(device), corr12.to(device), corr13.to(device)
            # print(img1.shape)
            optimizer.zero_grad()

            # Forward pass
            emb1 = model(img1)
            emb2 = model(img2)
            emb3 = model(img3)

            # Calculate loss
            loss, reg_loss, rank_loss = criterion(emb1, emb2, emb3, corr12, corr13)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
            running_reg += reg_loss.item() * img1.size(0)
            running_rank += rank_loss.item() * img1.size(0)
            pbar.set_postfix({'Training loss': loss.item()})

        train_loss_log.append(running_loss / len(train_loader.dataset))
        train_reg_log.append(running_reg / len(train_loader.dataset))
        train_rank_log.append(running_rank / len(train_loader.dataset))
        train_t1 = time.time()

        # Evaluation phase
        model.eval()
        test_t0 = time.time()
        test_spearman_loss = 0.0

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for sample_idx, cache in enumerate(test_cache):
                imgs = cache["images"].to(device)  # (Ni, 1, H, W)
                num_images = cache["num_images"]
                img_corr = cache["img_corr"]

                # --- Compute embeddings ---
                emb = model(imgs)  # (Ni, D)
                emb = F.normalize(emb, dim=1)
                emb_sim = (emb @ emb.T).cpu().numpy()

                # --- Spearman loss ---
                gt_vals = img_corr[np.triu_indices(num_images, k=1)]
                emb_vals = emb_sim[np.triu_indices(num_images, k=1)]

                spearman_corr, _ = spearmanr(gt_vals, emb_vals)
                spearman_loss = 1.0 - spearman_corr

                test_spearman_loss += spearman_loss

                all_embeddings.append(emb.cpu())
                all_labels.append(
                    torch.full((num_images,), sample_idx, dtype=torch.long)
                )

            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()

            clf = LogisticRegression(
                max_iter=500,
                multi_class="auto",
                n_jobs=1
            )

            try:
                clf.fit(all_embeddings, all_labels)
                preds = clf.predict(all_embeddings)
                classification_loss = accuracy_score(all_labels, preds)
            except Exception:
                classification_loss = 1.0  # worst-case penalty

            test_classification_loss = classification_loss
            test_loss = test_spearman_loss / len(test_cache) / 2 + test_classification_loss / 2

        test_loss_log.append(test_loss)
        test_sim_log.append(test_spearman_loss)
        test_class_log.append(test_classification_loss)
        test_t1 = time.time()
        print(f"Epoch {epoch + 1}/{args.max_epochs}: "
              f"Train Loss: {train_loss_log[-1]:.4f} (Time: {int(train_t1 - train_t0)} s), Test Loss: {test_loss_log[-1]:.4f} (Time: {int(test_t1 - test_t0)} s)")

        # Update learning rate
        scheduler.step(test_loss_log[-1])
        """animator.add(epoch + 1, [
            train_loss_log[-1], test_loss_log[-1],
            train_reg_log[-1], test_sim_log[-1],
            train_rank_log[-1], test_class_log[-1]
        ])"""

        # Save best model
        if test_loss_log[-1] < best_test_loss - args.early_stop_delta:
            best_test_loss = test_loss_log[-1]
            best_epoch = epoch
            early_stop_counter = 0

            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

            torch.save(best_model_state, os.path.join(args.output_path, args.model_data_file))
        else:
            early_stop_counter += 1
        if early_stop_counter >= args.early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch + 1}, "
                f"Best test loss: {best_test_loss:.4f}"
            )
            break
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    time2 = time.time()
    print(f'Time of preparing data: {time1 - time0}. Time of training model: {time2 - time1}')
    plt.ioff()
    plt.show()

    return train_loss_log, test_loss_log, train_reg_log, test_sim_log, train_rank_log, test_class_log

def train_embedding_pairs(train_data, test_data, model: nn.Module, args) -> tuple[
    list[float], list[float], list[float], list[float]]:
    """Train the embedding model

    Args:
        train_data: train_data from function: get_training_test_data
        test_data: test_data from function: get_training_test_data
        model: The embedding model to train
        args: Training arguments
        (seed, train_pairs_per_sample, alpha_height_ratio, sigma_height_ratio, max_rotation_degree, batch_size,  optimizer, lr,
        max_epochs, early_stop_patience, early_stop_delta, output_path, model_data_file)

    Returns:
        Tuple of (train_loss_log, test_loss_log)
    """
    time0 = time.time()

    # Create datasets
    train_dataset = ImageDataset(train_data,
                                pairs_per_sample=args.train_pairs_per_sample
                                )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    time1 = time.time()

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = CorrelationLoss()
    optimizerD = {'Adam': torch.optim.Adam(model.parameters(), lr=args.lr),
                  'SGD': torch.optim.SGD(model.parameters(), lr=args.lr),
                  'RMSprop': torch.optim.RMSprop(model.parameters(), lr=args.lr),
                  'Adagrad': torch.optim.Adagrad(model.parameters(), lr=args.lr),
                  'AdamW': torch.optim.AdamW(model.parameters(), lr=args.lr)}
    optimizer = optimizerD[args.optimizer]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1)

    train_loss_log = []
    test_loss_log = []
    test_sim_log = []
    test_class_log = []

    best_test_loss = float('inf')

    # animator = Animator(
    #     xlabel='Epoch', ylabel='Loss',
    #     legend=[
    #         'Train Loss', 'Test Total','Test Similarity','Test Classification'
    #     ],
    #     xlim=[1, args.max_epochs],
    #     fmts=('-', 'm--', 'g-.', 'r', 'c:', 'b-.'),
    #     figsize=(8, 5)
    # )

    # Training loop
    early_stop_counter = 0
    best_epoch = 0
    best_model_state = None
    for epoch in range(args.max_epochs):  # Change with early stopping, max epoch is args.epochs
        model.train()
        train_t0 = time.time()
        running_loss = 0.0

        # Training phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epochs} [Train]")
        for img1, img2, corr in pbar:
            img1, img2, corr = img1.to(device),img2.to(device),  corr.to(device)
            # print(img1.shape)
            optimizer.zero_grad()

            # Forward pass
            # print(img1.shape)
            emb1 = model(img1)
            emb2 = model(img2)

            # Calculate loss
            loss = criterion(emb1, emb2, corr)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
            pbar.set_postfix({'Training loss': loss.item()})

        train_loss_log.append(running_loss / len(train_loader.dataset))
        train_t1 = time.time()

        # Evaluation phase
        model.eval()
        test_t0 = time.time()
        test_spearman_loss = 0.0

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for sample_idx, cache in enumerate(test_data):
                imgs = torch.from_numpy(cache["intensity_array"]).float().unsqueeze(1)  # (N, 1, H, W)
                # mask = torch.from_numpy(cache["shape_mask"]).bool()
                # imgs[:, 0, ~mask] = 0.0

                imgs = imgs.to(device)  # (Ni, 1, H, W)
                num_images = cache["num_images"]
                img_corr = cache["img_corr"]

                # --- Compute embeddings ---
                emb = model(imgs)  # (Ni, D)
                emb = F.normalize(emb, dim=1)
                emb_sim = (emb @ emb.T).cpu().numpy()

                # --- Spearman loss ---
                gt_vals = img_corr[np.triu_indices(num_images, k=1)]
                emb_vals = emb_sim[np.triu_indices(num_images, k=1)]

                spearman_corr, _ = spearmanr(gt_vals, emb_vals)
                spearman_loss = 1.0 - spearman_corr

                test_spearman_loss += spearman_loss

                all_embeddings.append(emb.cpu())
                all_labels.append(
                    torch.full((num_images,), sample_idx, dtype=torch.long)
                )

            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()

            clf = LogisticRegression(
                max_iter=500,
                multi_class="auto",
                n_jobs=1
            )

            try:
                clf.fit(all_embeddings, all_labels)
                preds = clf.predict(all_embeddings)
                classification_loss = accuracy_score(all_labels, preds)
            except Exception:
                classification_loss = 1.0  # worst-case penalty

            test_classification_loss = classification_loss
            test_loss = test_spearman_loss / len(test_data) / 2 + test_classification_loss / 2

        test_loss_log.append(test_loss)
        test_sim_log.append(test_spearman_loss / len(test_data))
        test_class_log.append(test_classification_loss)
        test_t1 = time.time()
        print(f"Epoch {epoch + 1}/{args.max_epochs}: "
              f"Train Loss: {train_loss_log[-1]:.4f} (Time: {int(train_t1 - train_t0)} s), Test Loss: {test_loss_log[-1]:.4f} (Time: {int(test_t1 - test_t0)} s)")

        # Update learning rate
        scheduler.step(test_loss_log[-1])
        # animator.add(epoch + 1, [
        #     train_loss_log[-1], test_loss_log[-1],test_sim_log[-1],test_class_log[-1]
        # ])

        # Save best model
        if test_loss_log[-1] < best_test_loss - args.early_stop_delta:
            best_test_loss = test_loss_log[-1]
            best_epoch = epoch
            early_stop_counter = 0

            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

            torch.save(best_model_state, os.path.join(args.output_path, args.model_data_file))
        else:
            early_stop_counter += 1
        if early_stop_counter >= args.early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch + 1}, "
                f"Best test loss: {best_test_loss:.4f}"
            )
            break
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    time2 = time.time()
    print(f'Time of preparing data: {time1 - time0}. Time of training model: {time2 - time1}')
    plt.ioff()
    plt.show()

    return train_loss_log, test_loss_log,test_sim_log, test_class_log


def plot_loss_curve(train_loss, test_loss, train_reg, test_reg, train_rank, test_rank, output_path=None):
    """绘制静态的 loss 曲线"""
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="Train Loss", marker='o', markersize=3)
    plt.plot(test_loss, label="Test Loss", marker='s', markersize=3)
    plt.plot(train_reg, label="Train Reg Loss", marker='p', markersize=3)
    plt.plot(test_reg, label="Test Similarity Loss", marker='*', markersize=3)
    plt.plot(train_rank, label="Train Rank Loss", marker='d', markersize=3)
    plt.plot(test_rank, label="Test Classification Loss", marker='v', markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# Example usage
if __name__ == "__main__":
    r"""import pickle
    import time

    with open('./test_mice_brain_aging/input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)


    class Args:
        seed = 42
        lr = 1e-4
        epochs = 40
        batch_size = 200
        train_pairs_per_sample = 2000
        test_pairs_per_sample = 200
        margin = 0.1
        embedding_dim = 8
        output_path = './test_mice_brain_aging'
        model_data_file = 'temp_resnet.pth'


    args = Args()

    from image_preprocessing import *

    inputs_after = pre_alignment(inputs, sample_transform=['i', 'i', 'i', 'i'])
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print('Preprocessing finishing.')

    # Initialize model
    # model = ResNetEmbedding(embedding_dim=args.embedding_dim)
    # model = ViTEmbedding(embedding_dim=args.embedding_dim, img_size=(ih, iw), patch_size=args.patch_size,
    #                      dim=64, depth=1, heads=4, mlp_dim=28, dropout=0.1)
    # model = LightViTEmbedding()
    # model = EfficientNetEmbedding(embedding_dim=args.embedding_dim)

    # Train the model

    # train_losses, test_losses = train_embedding(inputs_after, model, args)


    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    for s in inputs:
        print(s[0], s[1].shape, s[2].shape, s[3].shape, s[3].dtype)

        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(s[3][88], cmap='magma')
        plt.colorbar(im1, ax=ax1)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(_image_distort(s[3][88], s[1]), cmap='magma')
        plt.colorbar(im2, ax=ax2)
        plt.show()"""

    # Test in five samples
    # import pickle
    # with open(r'E:\yangjun\msi\MSI_IIE_article\test_five_samples\input_data.pkl', 'rb') as f:
    #     inputs = pickle.load(f)
    #
    # class Args:
    #     seed = 42
    #     lr = 1e-4
    #     epochs = 40
    #     batch_size = 200
    #     train_pairs_per_sample = 4000
    #     test_pairs_per_sample = 400
    #     embedding_dim = 16
    #     output_path = r'E:\yangjun\msi\MSI_IIE_article\test_five_samples'
    #     model_data_file = 'cnn_16d.pth'
    #
    # args = Args()
    #
    # from image_preprocessing import *
    #
    # inputs_after = pre_alignment(inputs, sample_transform=['i', 'f', 'i', 'i', 'f'])
    # inputs_after = input_normalization(inputs_after)
    # ih, iw = get_input_size(inputs_after)
    # inputs_after = resize_images(inputs_after, ih, iw)
    # print('Preprocessing finishing.')

    # Initialize model
    # model = ResNetEmbedding(embedding_dim=args.embedding_dim)
    # model = ViTEmbedding(embedding_dim=args.embedding_dim, img_size=(ih, iw), patch_size=args.patch_size,
    #                      dim=64, depth=1, heads=4, mlp_dim=28, dropout=0.1)
    # model = LightViTEmbedding()
    # model = EfficientNetEmbedding(embedding_dim=args.embedding_dim)
    # model = SimpleCNNEmbedding(embedding_dim=args.embedding_dim)

    # Train the model

    # train_losses, test_losses = train_embedding(inputs_after, model, args)

    # Test in mcf-pos-neg
    r"""import pickle

    with open(r'E:\yangjun\msi\MSI_IIE_article\test_mcf_pos_neg\input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)


    class Args:
        seed = 42
        lr = 1e-4
        epochs = 40
        batch_size = 200
        train_pairs_per_sample = 20000
        test_pairs_per_sample = 1000
        embedding_dim = 32
        output_path = r'E:\yangjun\msi\MSI_IIE_article\test_mcf_pos_neg'
        model_data_file = 'multiscale_cnn_32d.pth'   # 2 inception blocks: 785k


    args = Args()

    from image_preprocessing import *

    inputs_after = pre_alignment(inputs, sample_transform=['i', 'i'])
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print('Preprocessing finishing.')

    # Initialize model
    # model = ResNetEmbedding(embedding_dim=args.embedding_dim)
    # model = ViTEmbedding(embedding_dim=args.embedding_dim, img_size=(ih, iw), patch_size=args.patch_size,
    #                      dim=64, depth=1, heads=4, mlp_dim=28, dropout=0.1)
    # model = LightViTEmbedding()
    # model = EfficientNetEmbedding(embedding_dim=args.embedding_dim)
    # model = SimpleCNNEmbedding(embedding_dim=args.embedding_dim)
    model = MultiscaleEmbedding(embedding_dim=args.embedding_dim)

    # Train the model

    train_losses, test_losses = train_embedding(inputs_after, model, args)"""

    r"""# Evaluation: application on ad data
    import pickle

    with open(r'E:\yangjun\msi\MSI_IIE_article\ad_study\input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)

    class Args:
        seed = 42
        lr = 1e-4
        epochs = 40
        batch_size = 200
        train_pairs_per_sample = 8000
        test_pairs_per_sample = 1000
        embedding_dim = 32
        output_path = r'E:\yangjun\msi\MSI_IIE_article\ad_study'
        model_data_file = 'multiscale_cnn_32d_ad.pth'


    args = Args()

    from image_preprocessing import *

    inputs_after = pre_alignment(inputs, sample_transform=['i', 'f', 'i', 'i', 'f',
                                                           'i', 'i', 'i', 'i', 'f',
                                                           'i', 'f', 'i', 'i', 'f',
                                                           'i', 'i', 'i', 'i', 'f'])
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print('Preprocessing finishing.')

    # Initialize model
    # model = ResNetEmbedding(embedding_dim=args.embedding_dim)
    # model = ViTEmbedding(embedding_dim=args.embedding_dim, img_size=(ih, iw), patch_size=args.patch_size,
    #                      dim=64, depth=1, heads=4, mlp_dim=28, dropout=0.1)
    # model = LightViTEmbedding()
    # model = EfficientNetEmbedding(embedding_dim=args.embedding_dim)
    # model = SimpleCNNEmbedding(embedding_dim=args.embedding_dim)
    model = MultiscaleEmbedding(embedding_dim=args.embedding_dim)

    # Train the model

    train_losses, test_losses = train_embedding(inputs_after, model, args)"""

    r"""# Evaluation: application on maldi-pre data
    import pickle

    with open(r'E:\yangjun\msi\MSI_IIE_article\maldi_pre_experiment\input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)


    class Args:
        seed = 42
        lr = 1e-4
        epochs = 40
        batch_size = 200
        train_pairs_per_sample = 8000
        test_pairs_per_sample = 1000
        embedding_dim = 32
        output_path = r'E:\yangjun\msi\MSI_IIE_article\maldi_pre_experiment'
        model_data_file = 'multiscale_cnn_32d_ad.pth'


    args = Args()

    from image_preprocessing import *

    inputs_after = pre_alignment(inputs, sample_transform=['f180', 'f180', 'i', 'i'], output_path=args.output_path)
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print('Preprocessing finishing.')

    model = MultiscaleEmbedding(embedding_dim=args.embedding_dim)
    train_losses, test_losses = train_embedding(inputs_after, model, args)"""

    r"""# Evaluation: application on ad whole brain preexperiment data
    import pickle

    with open(r'E:\yangjun\msi\MSI_IIE_article\ad_whole_brain_pre\input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)


    class Args:
        seed = 42
        lr = 1e-4
        epochs = 40
        batch_size = 200
        train_pairs_per_sample = 8000
        test_pairs_per_sample = 1000
        embedding_dim = 32
        output_path = r'E:\yangjun\msi\MSI_IIE_article\ad_whole_brain_pre'
        model_data_file = 'multiscale_cnn_32d_ad.pth'


    args = Args()

    from image_preprocessing import *

    inputs_after = pre_alignment(inputs, sample_transform=['i', 'i', 'i', 'i'], output_path=args.output_path)
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print(f'Preprocessing finishing. ({ih}, {iw})')

    model = MultiscaleEmbedding(embedding_dim=args.embedding_dim)
    train_losses, test_losses = train_embedding(inputs_after, model, args)"""


    class Args:
        seed = 42
        train_pairs_per_sample = 5000
        alpha_height_ratio = 2  # (0, 3)
        sigma_height_ratio = 0.12  # (0, 1)
        max_rotation_degree = 45  # (0, 60)
        batch_size = 200  # (100, 800)
        rank_reg_loss_ratio = 1   # (0, 10)
        optimizer = "Adam"  # ['Adam','SGD','RMSprop','Adagrad','AdamW']
        lr = 1e-4
        max_epochs = 60
        early_stop_patience = 8
        early_stop_delta = 1e-4
        output_path = r'E:\yangjun\msi\MSI_IIE_article\ad_whole_brain_pre'
        model_data_file = 'multiscale_cnn_32d_ad.pth'

        embedding_dim = 28  # (28, 256)
        num_inception_blocks = 4  # (1, 5)
        dropout_p = 0.2 # (0, 0.4)


    args = Args()
    model = MultiscaleEmbedding(embedding_dim=args.embedding_dim,
                                num_inception_blocks=args.num_inception_blocks,
                                dropout_p=args.dropout_p)
    train_loss_log, test_loss_log, train_reg_log, test_sim_log, train_rank_log, test_class_log = train_embedding(
        inputs_after, model, args)


    # Bayes optimization
    r'''import optuna

    # Optuna Objective Function
    def objective(trial):

        class Args:
            # Fixed
            seed = 42
            train_pairs_per_sample = 5000
            output_path = r'E:\yangjun\msi\MSI_IIE_article\ad_whole_brain_pre'
            model_data_file = f'multiscale_cnn_trial_{trial.number}.pth'
            max_epochs = 60
            early_stop_patience = 8
            early_stop_delta = 1e-4

            # Data augmentation
            alpha_height_ratio = trial.suggest_float("alpha_height_ratio", 0.0, 3.0)
            sigma_height_ratio = trial.suggest_float("sigma_height_ratio", 0.0, 1.0)
            max_rotation_degree = trial.suggest_int("max_rotation_degree", 0, 60)

            # Training
            batch_size = trial.suggest_int("batch_size", 100, 800, step=50)
            rank_reg_loss_ratio = trial.suggest_float("rank_reg_loss_ratio", 0.0, 10.0)
            optimizer = trial.suggest_categorical(
                "optimizer", ["Adam", "SGD", "RMSprop", "Adagrad", "AdamW"]
            )
            lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

            # Architecture
            embedding_dim = trial.suggest_int("embedding_dim", 28, 256, step=16)
            num_inception_blocks = trial.suggest_int("num_inception_blocks", 1, 5)
            dropout_p = trial.suggest_float("dropout_p", 0.0, 0.4)

        args = Args()

        # Reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        model = MultiscaleEmbedding(
            embedding_dim=args.embedding_dim,
            num_inception_blocks=args.num_inception_blocks,
            dropout_p=args.dropout_p,
        )

        try:
            _, test_loss_log, _, _, _, _ = train_embedding(
                inputs_after, model, args
            )

            final_test_loss = test_loss_log[-1]

        except RuntimeError as e:
            print(f"Trial {trial.number} failed: {e}")
            return float("inf")

        return final_test_loss

    # Run Bayesian Optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=50)

    # Inspect Best Results
    print("Best trial:")
    print(f"  Value: {study.best_trial.value:.6f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")'''


    # Bayesian optimization based on bayes_opt
    from bayes_opt import BayesianOptimization


    def bayes_objective(
            embedding_dim,
            num_inception_blocks,
            dropout_p,
            batch_size,
            rank_reg_loss_ratio,
            lr
    ):
        # ----------------------------
        # Convert types
        # ----------------------------
        embedding_dim = int(round(embedding_dim))
        num_inception_blocks = int(round(num_inception_blocks))
        batch_size = int(round(batch_size))
        rank_reg_loss_ratio = float(rank_reg_loss_ratio)
        lr = float(lr)

        # ----------------------------
        # Update args
        # ----------------------------
        args.embedding_dim = embedding_dim
        args.num_inception_blocks = num_inception_blocks
        args.dropout_p = dropout_p
        args.batch_size = batch_size
        args.rank_reg_loss_ratio = rank_reg_loss_ratio
        args.lr = lr

        # Fix optimizer (can be extended later)
        args.optmizer = "Adam"

        # ----------------------------
        # Reproducibility
        # ----------------------------
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # ----------------------------
        # Build model
        # ----------------------------
        model = MultiscaleEmbedding(
            embedding_dim=args.embedding_dim,
            num_inception_blocks=args.num_inception_blocks,
            dropout_p=args.dropout_p
        )

        # ----------------------------
        # Train model
        # ----------------------------
        (
            train_loss_log,
            test_loss_log,
            *_,
        ) = train_embedding(inputs_after, model, args)

        final_test_loss = test_loss_log[-1]

        # BO maximizes → negate loss
        return -final_test_loss


    pbounds = {
        "embedding_dim": (28, 256),
        "num_inception_blocks": (1, 5),
        "dropout_p": (0.0, 0.4),
        "batch_size": (100, 800),
        "rank_reg_loss_ratio": (0.0, 10.0),
        "lr": (1e-5, 5e-4),
    }

    optimizer = BayesianOptimization(
        f=bayes_objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2,
    )

    optimizer.maximize(
        init_points=5,  # random initialization
        n_iter=20,  # Bayesian steps
    )

    best_params = optimizer.max["params"]

    # Convert discrete params properly
    best_params["embedding_dim"] = int(round(best_params["embedding_dim"]))
    best_params["num_inception_blocks"] = int(round(best_params["num_inception_blocks"]))
    best_params["batch_size"] = int(round(best_params["batch_size"]))

    print("Best hyperparameters found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    print(f"Best test loss: {-optimizer.max['target']:.6f}")

    import pandas as pd

    results = pd.DataFrame(optimizer.res)
    results.to_csv("bayes_optimization_results.csv", index=False)




