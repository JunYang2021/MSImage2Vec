import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from scipy.ndimage import gaussian_filter, map_coordinates

import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from embedding_models import ResNetEmbedding, ViTEmbedding, EfficientNetEmbedding, SimpleCNNEmbedding, MultiscaleEmbedding
import time


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
    return np.corrcoef(flat1, flat2)[0, 1]


def _image_distort(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    distorted = img.copy()
    height, width = img.shape

    # 1. Random noise (1% of max intensity)
    max_intensity = np.max(distorted)
    noise = np.random.normal(0, 0.01 * max_intensity, (height, width))
    distorted[mask] = np.clip(distorted[mask] + noise[mask], 0, None)

    # 2. Elastic deformations (non-linear distortions)
    if random.random() > 0.5:  # Apply to 50% of images
        # The strength of elastic deformation. Larger value means bigger deformation
        alpha = random.uniform(height * 0.8, height * 1.8)
        # The smoothness of elastic deformation. Larger value means smoother deformation
        sigma = random.uniform(height * 0.1, height * 0.15)
        # The strength of elastic deformation. Larger value means bigger deformation
        alpha = random.uniform(height * 1.8, height * 2.8)
        # The smoothness of elastic deformation. Larger value means smoother deformation
        sigma = random.uniform(height * 0.1, height * 0.15)

        # Displacement fields
        # np.random.rand range: [0, 1)  np.random.ran() * 2 -1 range: [-1, 1)
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x_distorted = np.clip(x + dx, 0, width - 1)
        y_distorted = np.clip(y + dy, 0, height - 1)

        distorted = map_coordinates(distorted, [y_distorted, x_distorted], order=1, mode='constant')

    # 3. Random rotation (-15 to 15 degrees) and scaling (95% to 105%)
    angle = random.uniform(-15, 15)
    angle = random.uniform(-30, 30)
    center = (width // 2, height // 2)
    scale = random.uniform(0.95, 1.05)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    distorted = cv2.warpAffine(distorted, M, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 4. Random translation (up to 5% of image size)
    max_trans_x = int(0.05 * width)
    max_trans_y = int(0.05 * height)
    trans_x = random.randint(-max_trans_x, max_trans_x)
    trans_y = random.randint(-max_trans_y, max_trans_y)
    M = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    distorted = cv2.warpAffine(distorted, M, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 5. Random pixel dropout(1 % of pixels)
    dropout_mask = np.random.random((height, width)) < 0.01
    distorted[dropout_mask] = 0

    return distorted


class ImageDataset(Dataset):
    def __init__(self, inputs: List, pairs_per_sample: int = 20000):
        self.pairs = []
        for s_input in inputs:
            sample_id, shape_mask, _, intensity_array = s_input
            num_images = len(intensity_array)

            for _ in tqdm(range(pairs_per_sample),
                          desc=f"Constructing dataset in {sample_id}",
                          unit="pair"):
                i, j = random.sample(range(num_images), 2)
                img1 = intensity_array[i]
                img2 = intensity_array[j]

                corr = _correlation(img1, img2, shape_mask)
                # Apply random image distort
                dist_img1 = _image_distort(img1, shape_mask)
                dist_img2 = _image_distort(img2, shape_mask)
                self.pairs.append((dist_img1, dist_img2, corr))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        img1, img2, corr = self.pairs[idx]

        # Convert to tensors and add channel dimension
        img1_tensor = torch.FloatTensor(img1).unsqueeze(0)  # Shape: 1, height, width
        img2_tensor = torch.FloatTensor(img2).unsqueeze(0)  # Shape: 1, height, width

        return img1_tensor, img2_tensor, corr


class TripletImageDataset(Dataset):
    def __init__(self, inputs: List, triplets_per_sample: int = 20000):
        self.triplets = []
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
                self.triplets.append((anchor_img, pos_img, neg_img, corr_ap, corr_an))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        anchor_img, pos_img, neg_img, corr_ap, corr_an = self.triplets[idx]

        to_tensor = lambda img: torch.FloatTensor(img).unsqueeze(0)
        return (
            to_tensor(anchor_img),
            to_tensor(pos_img),
            to_tensor(neg_img),
            torch.tensor(corr_ap, dtype=torch.float32),
            torch.tensor(corr_an, dtype=torch.float32)
        )


class CorrelationLoss(nn.Module):
    """Loss function that measures the difference between embedding cosine similarity
    and actual image correlation"""

    def __init__(self):
        super().__init__()

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
                target_corr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1, emb2: Embeddings of the two images (batch_size x embedding_dim)
            target_corr: Target correlation coefficients (batch_size)
        # Margin-based MSE
        """
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


def train_embedding(inputs: List, model: nn.Module, args) -> Tuple[List[float], List[float]]:
    """Train the embedding model

    Args:
        inputs: List of sample data
        model: The embedding model to train
        args: Training arguments (should contain lr, epochs, batch_size, etc.)

    Returns:
        Tuple of (train_loss_log, test_loss_log)
    """
    time0 = time.time()
    # Split data into train and test
    train_inputs, test_inputs = divide_train_test(inputs, seed=args.seed)

    # Create datasets
    train_dataset = TripletImageDataset(train_inputs, triplets_per_sample=args.train_pairs_per_sample)
    test_dataset = TripletImageDataset(test_inputs, triplets_per_sample=args.test_pairs_per_sample)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    time1 = time.time()

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = JointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1)

    train_loss_log = []
    test_loss_log = []
    train_reg_log = []
    train_rank_log = []
    test_reg_log = []
    test_rank_log = []

    best_test_loss = float('inf')

    animator = Animator(
        xlabel='Epoch', ylabel='Loss',
        legend=[
            'Train Total', 'Test Total',
            'Train Reg', 'Test Reg',
            'Train Rank', 'Test Rank'
        ],
        xlim=[1, args.epochs],
        fmts=('-', 'm--', 'g-.', 'r', 'c:', 'b-.'),
        figsize=(8, 5)
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_reg = 0.0
        running_rank = 0.0

        # Training phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for img1, img2, img3, corr12, corr13 in pbar:
            img1, img2, img3, corr12, corr13 = img1.to(device),\
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
            pbar.set_postfix({'loss': loss.item()})

        train_loss_log.append(running_loss / len(train_loader.dataset))
        train_reg_log.append(running_reg / len(train_loader.dataset))
        train_rank_log.append(running_rank / len(train_loader.dataset))

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_reg = 0.0
        test_rank = 0.0
        with torch.no_grad():
            for img1, img2, img3, corr12, corr13 in test_loader:
                img1, img2, img3, corr12, corr13 = img1.to(device), \
                    img2.to(device), img3.to(device), corr12.to(device), corr13.to(device)

                emb1 = model(img1)
                emb2 = model(img2)
                emb3 = model(img3)

                loss, reg_loss, rank_loss = criterion(emb1, emb2, emb3, corr12, corr13)
                test_loss += loss.item() * img1.size(0)
                test_reg += reg_loss.item() * img1.size(0)
                test_rank += rank_loss.item() * img1.size(0)

        test_loss_log.append(test_loss / len(test_loader.dataset))
        test_reg_log.append(test_reg / len(test_loader.dataset))
        test_rank_log.append(test_rank / len(test_loader.dataset))
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"Train Loss: {train_loss_log[-1]:.4f}, Test Loss: {test_loss_log[-1]:.4f}")

        # Update learning rate
        scheduler.step(test_loss_log[-1])
        animator.add(epoch + 1, [
            train_loss_log[-1], test_loss_log[-1],
            train_reg_log[-1], test_reg_log[-1],
            train_rank_log[-1], test_rank_log[-1]
        ])

        # Save best model
        if test_loss_log[-1] < best_test_loss:
            best_test_loss = test_loss_log[-1]
            torch.save(model.state_dict(), os.path.join(args.output_path, args.model_data_file))

    time2 = time.time()
    print(f'Time of preparing data: {time1 - time0}. Time of training model: {time2 - time1}')
    plt.ioff()
    plt.show()

    return train_loss_log, test_loss_log


# Example usage
if __name__ == "__main__":
    """import pickle
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
    import pickle

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

    train_losses, test_losses = train_embedding(inputs_after, model, args)
