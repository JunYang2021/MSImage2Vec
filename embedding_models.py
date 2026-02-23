import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from typing import Tuple


class ResNetEmbedding(nn.Module):
    """ResNet-based embedding model for ion images"""

    def __init__(self, embedding_dim: int = 28):
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
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

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
        # x = self.bn(x)  # 添加 BN 层
        # x = x - x.mean(dim=1, keepdim=True)  # 显式中心化（可选）
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


class ViTEmbedding(nn.Module):   # 编码效果不好
    """Vision Transformer-based embedding model for ion images"""

    def __init__(self, embedding_dim: int = 28, img_size=(64, 64), patch_size=8,
                 dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        patch_dim = patch_size * patch_size  # For 1 channel input

        self.patch_embedding = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
                                                     dropout=dropout),
            num_layers=depth
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embedding(x)  # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, dim)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])  # Take the CLS token
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x


class EfficientNetEmbedding(nn.Module):
    """EfficientNet-based embedding model for ion images."""

    def __init__(self, embedding_dim: int = 28):
        super().__init__()

        # Load pretrained EfficientNet model
        base_model = efficientnet_b0(weights=None)

        # Modify first conv layer to accept 1-channel grayscale input instead of 3-channel RGB
        first_conv = base_model.features[0][0]  # Conv2d layer
        new_first_conv = nn.Conv2d(
            1, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )

        base_model.features[0][0] = new_first_conv

        self.features = base_model.features  # EfficientNet feature extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # To handle any input size
        self.embedding = nn.Linear(base_model.classifier[1].in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)  # L2-normalize embeddings
        return x


class SimpleCNNEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 28):
        super().__init__()

        self.features = nn.Sequential(
            # 下采样部分
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # 1/2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 1/4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 1/8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 特征提取部分
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# Inception
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        # Path 1
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)


class Inception_p(nn.Module):
    """
        Ratio-based Inception block.
        Branch output channels are computed from out_channels * ratios.
        """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 branch_ratios: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
                 reduction_ratio: float = 0.5,
                 dropout_p: float = 0.0,
                 ):
        super().__init__()
        assert abs(sum(branch_ratios) - 1.0) < 1e-6, "Branch ratios must sum to 1."

        c1 = int(out_channels * branch_ratios[0])
        c2 = int(out_channels * branch_ratios[1])
        c3 = int(out_channels * branch_ratios[2])
        c4 = out_channels - (c1 + c2 + c3)  # avoid rounding loss

        c2r = max(1, int(c2 * reduction_ratio))
        c3r = max(1, int(c3 * reduction_ratio))

        # Path 1: 1x1
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Path 2: 1x1 -> 3x3
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2r, kernel_size=1),
            nn.BatchNorm2d(c2r),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2r, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        # Path 3: 1x1 -> 5x5
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3r, kernel_size=1),
            nn.BatchNorm2d(c3r),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3r, c3, kernel_size=5, padding=2),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        # Path 4: Pool -> 1x1
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            (self.p1(x), self.p2(x), self.p3(x), self.p4(x)),
            dim=1,
        )
        return self.dropout(x)


class MultiscaleEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 28):
        super().__init__()
        # 初始特征提取 + 下采样

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 输出大小约为 1/2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出大小约为 1/4
        )

        # 添加多个 Inception 块来捕捉多尺度信息
        self.inception_block1 = Inception(64, 32, (48, 64), (8, 16), 16)  # -> 128 channels
        self.inception_block2 = Inception(128, 64, (64, 96), (16, 32), 32)  # -> 224 channels
        self.inception_block3 = Inception(224, 64, (64, 96), (16, 32), 32)  # -> 224 channels
        self.inception_block4 = Inception(224, 64, (96, 128), (32, 64), 32)  # -> 288 channels

        self.mid_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 空间减半
        # 使用 1x1 卷积调整通道数（可选）
        self.conv_projection = nn.Conv2d(288, 256, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception_block1(x)
        x = self.mid_pool(x)
        x = self.inception_block2(x)
        x = self.inception_block3(x)
        x = self.inception_block4(x)
        x = F.relu(self.conv_projection(x))  # 调整为统一通道数

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # 单位化嵌入


class MultiscaleEmbedding_p(nn.Module):
    def __init__(self,
                 embedding_dim: int = 28,
                 num_inception_blocks: int = 4,
                 inception_out_channels: int = 256,
                 branch_ratios: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
                 reduction_ratio: float = 0.5,
                 dropout_p: float = 0.2,
                 projection_channels: int = 256,
                 in_channels: int = 1,
                 ):
        super().__init__()
        # 初始特征提取 + 下采样

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # 输出大小约为 1/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出大小约为 1/4
        )

        # Inception stack
        inception_blocks = []
        current_channels = 64

        for i in range(num_inception_blocks):
            inception_blocks.append(
                Inception_p(
                    in_channels=current_channels,
                    out_channels=inception_out_channels,
                    branch_ratios=branch_ratios,
                    reduction_ratio=reduction_ratio,
                    dropout_p=dropout_p,
                )
            )
            current_channels = inception_out_channels
            # Optional downsampling between blocks (except last)
            if i < num_inception_blocks - 1:
                inception_blocks.append(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
        self.inception_stack = nn.Sequential(*inception_blocks)

        # Projection
        self.projection = nn.Sequential(
            nn.Conv2d(current_channels, projection_channels, kernel_size=1),
            nn.BatchNorm2d(projection_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_head = nn.Sequential(
            nn.Linear(projection_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception_stack(x)
        x = self.projection(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding_head(x)

        return F.normalize(x, p=2, dim=1)


if __name__ == '__main__':
    model = SimpleCNNEmbedding(embedding_dim=32)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, params: {param.numel()}")

    print('-'*50)

    model = MultiscaleEmbedding(embedding_dim=32)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, params: {param.numel()}")

