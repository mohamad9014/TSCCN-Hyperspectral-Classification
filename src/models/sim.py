import math

import torch
import torch.nn as nn


# ============================================================
# ECA FOR 1D FEATURES
# ============================================================

class ECA1D(nn.Module):
    """
    Efficient Channel Attention for 1D features.

    Input:
        (B, C, L)

    Output:
        (B, C, L)

    The feature is globally averaged over the spectral
    dimension L. Then a small Conv1D learns interactions
    between neighboring channels.
    """

    def __init__(
        self,
        channels,
        gamma=2,
        b=1,
    ):
        super().__init__()

        # Standard ECA adaptive kernel-size rule.
        #
        # It produces a small odd kernel size.

        t = int(
            abs(
                (
                    math.log2(channels)
                    + b
                )
                / gamma
            )
        )

        kernel_size = (
            t
            if t % 2 == 1
            else t + 1
        )

        kernel_size = max(
            kernel_size,
            1,
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x:
        # B, C, L

        # Global average over L:
        #
        # B, C, 1

        y = self.avg_pool(x)

        # Change to:
        #
        # B, 1, C

        y = y.transpose(
            1,
            2,
        )

        # Learn local interactions
        # between channels.

        y = self.conv(y)

        y = self.sigmoid(y)

        # Back to:
        #
        # B, C, 1

        y = y.transpose(
            1,
            2,
        )

        # Reweight original features.

        return x * y


# ============================================================
# ECA FOR 2D FEATURES
# ============================================================

class ECA2D(nn.Module):
    """
    Efficient Channel Attention for 2D feature maps.

    Input:
        (B, C, H, W)

    Output:
        (B, C, H, W)
    """

    def __init__(
        self,
        channels,
        gamma=2,
        b=1,
    ):
        super().__init__()

        t = int(
            abs(
                (
                    math.log2(channels)
                    + b
                )
                / gamma
            )
        )

        kernel_size = (
            t
            if t % 2 == 1
            else t + 1
        )

        kernel_size = max(
            kernel_size,
            1,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x:
        # B, C, H, W

        # Global average pooling:
        #
        # B, C, 1, 1

        y = self.avg_pool(x)

        # B, C

        y = y.squeeze(-1).squeeze(-1)

        # B, 1, C

        y = y.unsqueeze(1)

        y = self.conv(y)

        y = self.sigmoid(y)

        # B, C, 1, 1

        y = y.squeeze(1)
        y = y.unsqueeze(-1).unsqueeze(-1)

        return x * y


# ============================================================
# 1D SIM SUB-MODULE
# ============================================================

class SIM1DBlock(nn.Module):
    """
    One sub-module of the paper's 1D SIM.

    Structure:

        Conv1D kernel 1
        32 filters
        ReLU

        -> Conv1D kernel 5
        -> ECA
        -> residual addition

        -> Conv1D kernel 5
        -> ECA
        -> residual addition

    Input:
        (B, in_channels, spectral_subspace_length)

    Output:
        (B, 32, spectral_subspace_length)
    """

    def __init__(
        self,
        in_channels=32,
    ):
        super().__init__()

        # First convolution:
        #
        # kernel = 1
        # output channels = 32

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        # First refinement block.

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        self.eca1 = ECA1D(
            channels=32
        )

        # Second refinement block.

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        self.eca2 = ECA1D(
            channels=32
        )

    def forward(self, x):

        # Initial transform.

        x = self.conv1(x)
        x = self.relu(x)

        # --------------------------------
        # Residual block 1
        # --------------------------------

        residual = x

        y = self.conv2(x)
        y = self.eca1(y)

        x = residual + y

        # --------------------------------
        # Residual block 2
        # --------------------------------

        residual = x

        y = self.conv3(x)
        y = self.eca2(y)

        x = residual + y

        return x


# ============================================================
# CORRECTED 1D SIM
# ============================================================

class SIM1D(nn.Module):
    """
    1D Structural Information Mining module.

    IMPORTANT:

    The paper divides the SPECTRAL FEATURE DIMENSION
    into g1 exclusive groups.

    Input:
        (B, C, P)

    Example:
        (B, 32, 100)

    With groups=4:

        four groups of:
        (B, 32, 25)

    Each group is independently processed.

    Then the outputs are concatenated again along
    the spectral dimension.

    Final output:

        (B, 32, 100)
    """

    def __init__(
        self,
        in_channels=32,
        groups=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.groups = groups

        self.blocks = nn.ModuleList(
            [
                SIM1DBlock(
                    in_channels=in_channels
                )
                for _ in range(groups)
            ]
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "SIM1D expects input shape "
                "(batch, channels, spectral_length)"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"SIM1D expected {self.in_channels} channels, "
                f"but received {x.size(1)}."
            )

        spectral_length = x.size(2)

        if spectral_length % self.groups != 0:
            raise ValueError(
                f"Spectral length {spectral_length} "
                f"must be divisible by groups={self.groups}."
            )

        # ------------------------------------------------
        # CORRECT PAPER INTERPRETATION
        #
        # Split spectral dimension P.
        #
        # NOT channel dimension.
        # ------------------------------------------------

        spectral_groups = torch.chunk(
            x,
            chunks=self.groups,
            dim=2,
        )

        outputs = []

        for block, group in zip(
            self.blocks,
            spectral_groups,
        ):
            outputs.append(
                block(group)
            )

        # Reconnect spectral subspaces.

        output = torch.cat(
            outputs,
            dim=2,
        )

        return output


# ============================================================
# 2D SIM SUB-MODULE
# ============================================================

class SIM2DBlock(nn.Module):
    """
    One sub-module of the paper's 2D SIM.

    Structure:

        1x1 Conv2D -> 32 channels -> ReLU

        5x5 Conv2D
        ECA
        residual addition

        5x5 Conv2D
        ECA
        residual addition
    """

    def __init__(
        self,
        in_channels,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        self.eca1 = ECA2D(
            channels=32
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        self.eca2 = ECA2D(
            channels=32
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)

        # First residual block.

        residual = x

        y = self.conv2(x)
        y = self.eca1(y)

        x = residual + y

        # Second residual block.

        residual = x

        y = self.conv3(x)
        y = self.eca2(y)

        x = residual + y

        return x


# ============================================================
# 2D SIM
# ============================================================

class SIM2D(nn.Module):
    """
    2D Structural Information Mining module.

    Unlike SIM1D, the paper divides the FEATURE MAP /
    CHANNEL dimension for the 2D SIM.

    Input example:

        (B, 32, H, W)

    groups = 4

    Each group receives:

        (B, 8, H, W)

    Each group produces:

        (B, 32, H, W)

    Concatenation gives:

        (B, 128, H, W)
    """

    def __init__(
        self,
        in_channels=32,
        groups=4,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError(
                "in_channels must be divisible by groups"
            )

        self.in_channels = in_channels
        self.groups = groups

        channels_per_group = (
            in_channels // groups
        )

        self.blocks = nn.ModuleList(
            [
                SIM2DBlock(
                    in_channels=channels_per_group
                )
                for _ in range(groups)
            ]
        )

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                "SIM2D expects input shape "
                "(batch, channels, height, width)"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"SIM2D expected {self.in_channels} channels, "
                f"but received {x.size(1)}."
            )

        # Paper's 2D SIM groups feature maps,
        # therefore split channel dimension.

        groups = torch.chunk(
            x,
            chunks=self.groups,
            dim=1,
        )

        outputs = []

        for block, group in zip(
            self.blocks,
            groups,
        ):
            outputs.append(
                block(group)
            )

        # 4 groups x 32 channels = 128.

        output = torch.cat(
            outputs,
            dim=1,
        )

        return output