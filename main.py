import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import _T_co
from typing import List, Tuple


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


class ImageDataset(Dataset):
    def __init__(self, inputs: List, pairs_per_sample: int = 20000):
        self.pairs = []
        for s_input in inputs:
            _, shape_mask, _, intensity_array = s_input
            num_images = len(intensity_array)

            for _ in range(pairs_per_sample):
                i, j = random.sample(range(num_images), 2)
                img1 = intensity_array[i]
                img2 = intensity_array[j]

                corr = self._correlation(img1, img2, shape_mask)
                # Apply random image distort
                dist_img1 = self._image_distort(img1)
                dist_img2 = self._image_distort(img2)
                self.pairs.append((dist_img1, dist_img2, corr))

    def _correlation(self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
        """Calculate Pearson correlation between two images"""
        flat1 = img1[mask].flatten()
        flat2 = img2[mask].flatten()
        return np.corrcoef(flat1, flat2)[0, 1]

    def _image_distort(self, img1: np.ndarray):
        pass

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        img1, img2, corr = self.pairs[idx]

        # Convert to tensors and add channel dimension
        img1_tensor = torch.FloatTensor(img1).unsqueeze(0)  # Shape: 1, height, width
        img2_tensor = torch.FloatTensor(img2).unsqueeze(0)  # Shape: 1, height, width

        return img1_tensor, img2_tensor, corr


def train_embedding(inputs, model, args):
    input_height, input_width = inputs[0][1].shape[0], inputs[0][1].shape[1]

    train_loss_log = []
    test_loss_log = []

    model.tranin()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        pass


class ResNetEmbedding(nn.Module):
    """ResNet-based embedding model for ion images"""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        # Basic ResNet blocks
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_residual_layer(64, 64, 2)
        self.layer2 = self._make_residual_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_residual_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_residual_layer(256, 512, 2, stride=2)

        # Final embedding layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)

    def _make_residual_layer(self, in_channels: int, out_channels: int,
                             blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # L2 normalize the embeddings
        x = F.normalize(x, p=2, dim=1)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block for our ResNet"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)
        return out


class CorrelationLoss(nn.Module):
    """Loss function that measures the difference between embedding cosine similarity
    and actual image correlation"""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
                target_corr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1, emb2: Embeddings of the two images (batch_size x embedding_dim)
            target_corr: Target correlation coefficients (batch_size)
        """
        # Calculate cosine similarity between embeddings
        cos_sim = F.cosine_similarity(emb1, emb2)

        # Calculate MSE between cosine similarity and target correlation
        loss = F.mse_loss(cos_sim, target_corr)

        return loss


def train_embedding(inputs: List, model: nn.Module, args) -> Tuple[List[float], List[float]]:
    """Train the embedding model

    Args:
        inputs: List of sample data
        model: The embedding model to train
        args: Training arguments (should contain lr, epochs, batch_size, etc.)

    Returns:
        Tuple of (train_loss_log, test_loss_log)
    """
    # Split data into train and test
    train_inputs, test_inputs = divide_train_test(inputs, seed=args.seed)

    # Create datasets
    train_dataset = IonImageDataset(train_inputs, pairs_per_sample=args.pairs_per_sample)
    test_dataset = IonImageDataset(test_inputs, pairs_per_sample=args.pairs_per_sample)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = CorrelationLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    train_loss_log = []
    test_loss_log = []

    best_test_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for img1, img2, corr, _ in pbar:
            img1, img2, corr = img1.to(device), img2.to(device), corr.to(device)

            optimizer.zero_grad()

            # Forward pass
            emb1 = model(img1)
            emb2 = model(img2)

            # Calculate loss
            loss = criterion(emb1, emb2, corr)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_log.append(epoch_loss)

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for img1, img2, corr, _ in test_loader:
                img1, img2, corr = img1.to(device), img2.to(device), corr.to(device)

                emb1 = model(img1)
                emb2 = model(img2)

                loss = criterion(emb1, emb2, corr)
                test_loss += loss.item() * img1.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        test_loss_log.append(test_loss)

        # Update learning rate
        scheduler.step(test_loss)

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

    return train_loss_log, test_loss_log


# Example usage
if __name__ == "__main__":
    # Define your arguments (in practice, you might use argparse)
    class Args:
        seed = 42
        lr = 1e-4
        epochs = 20
        batch_size = 32
        pairs_per_sample = 20
        margin = 0.1
        embedding_dim = 128


    args = Args()

    # Load your input data (replace with actual data loading)
    # inputs = load_your_data_here()

    # Initialize model
    model = ResNetEmbedding(embedding_dim=args.embedding_dim)

    # Train the model
    train_losses, test_losses = train_embedding(inputs, model, args)