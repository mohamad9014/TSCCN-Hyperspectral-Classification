import torch
import torch.nn as nn
import torch.nn.functional as F

from .capsule import (
    squash,
    DynamicRouting,
    capsule_lengths,
)

from .sim import SIM2D


class PrimaryCapsule2D(nn.Module):
    """
    Primary capsule layer for the spectral-spatial branch.

    Paper configuration:
        kernel size      = 4 x 4
        capsule channels = 4
        capsule dim      = 8
        stride           = 2
        padding          = No
        BN               = Yes
        activation       = ReLU + Squash

    Input:
        (B, 64, H, W)

    Output:
        (B, 4, H_out, W_out, 8)
    """

    def __init__(
        self,
        in_channels=64,
        capsule_channels=4,
        capsule_dim=8,
    ):
        super().__init__()

        self.capsule_channels = capsule_channels
        self.capsule_dim = capsule_dim

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=capsule_channels * capsule_dim,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            capsule_channels * capsule_dim
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)

        # Current:
        # (B, capsule_channels * capsule_dim, H, W)

        x = x.view(
            batch_size,
            self.capsule_channels,
            self.capsule_dim,
            height,
            width,
        )

        # Desired:
        # (B, capsule_channels, H, W, capsule_dim)

        x = x.permute(
            0,
            1,
            3,
            4,
            2,
        ).contiguous()

        x = squash(
            x,
            dim=-1,
        )

        return x


class ConvCapsule2D(nn.Module):
    """
    2D convolutional capsule layer.

    Paper configuration:
        kernel size      = 3 x 3
        input capsules   = 4
        input dim        = 8
        output capsules  = 8
        output dim       = 4
        stride           = 2
        padding          = No
        activation       = Squash

    Input:
        (B, 4, H, W, 8)

    Output:
        (B, 8, H_out, W_out, 4)
    """

    def __init__(
        self,
        in_capsules=4,
        in_dim=8,
        out_capsules=8,
        out_dim=4,
        kernel_size=3,
        stride=2,
        routing_iterations=3,
    ):
        super().__init__()

        self.in_capsules = in_capsules
        self.in_dim = in_dim

        self.out_capsules = out_capsules
        self.out_dim = out_dim

        self.kernel_size = kernel_size
        self.stride = stride

        self.routing_iterations = (
            routing_iterations
        )

        # Transformation matrices shared
        # over all spatial locations.
        #
        # Shape:
        # kernel_y,
        # kernel_x,
        # input capsule,
        # output capsule,
        # input dim,
        # output dim

        self.weight = nn.Parameter(
            torch.empty(
                kernel_size,
                kernel_size,
                in_capsules,
                out_capsules,
                in_dim,
                out_dim,
            )
        )

        nn.init.xavier_uniform_(
            self.weight
        )

    def forward(self, x):

        if x.ndim != 5:
            raise ValueError(
                "ConvCapsule2D expects "
                "(B, capsules, H, W, dim)"
            )

        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)

        if x.size(1) != self.in_capsules:
            raise ValueError(
                "Incorrect number of input capsules"
            )

        if x.size(4) != self.in_dim:
            raise ValueError(
                "Incorrect input capsule dimension"
            )

        # ------------------------------------------------
        # Prepare for torch.nn.functional.unfold
        # ------------------------------------------------

        # Current:
        # B, capsules, H, W, dim
        #
        # Change to:
        # B, capsules, dim, H, W

        x = x.permute(
            0,
            1,
            4,
            2,
            3,
        ).contiguous()

        # Merge capsule and vector dimension:
        #
        # B, capsules * dim, H, W

        x = x.view(
            batch_size,
            self.in_capsules * self.in_dim,
            height,
            width,
        )

        # Extract 3x3 windows.
        #
        # Output:
        # B,
        # (capsules * dim * kernel * kernel),
        # number_of_windows

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )

        output_height = (
            (height - self.kernel_size)
            // self.stride
            + 1
        )

        output_width = (
            (width - self.kernel_size)
            // self.stride
            + 1
        )

        locations = (
            output_height
            * output_width
        )

        # Reshape:
        #
        # B,
        # input_capsules,
        # input_dim,
        # kernel_y,
        # kernel_x,
        # locations

        patches = patches.view(
            batch_size,
            self.in_capsules,
            self.in_dim,
            self.kernel_size,
            self.kernel_size,
            locations,
        )

        # Desired:
        #
        # B,
        # locations,
        # kernel_y,
        # kernel_x,
        # input_capsules,
        # input_dim

        patches = patches.permute(
            0,
            5,
            3,
            4,
            1,
            2,
        ).contiguous()

        # ------------------------------------------------
        # Prediction vectors
        # ------------------------------------------------
        #
        # patches:
        # B L Y X I D
        #
        # weight:
        # Y X I O D H
        #
        # predictions:
        # B L Y X I O H

        predictions = torch.einsum(
            "blyxid,yxiodh->blyxioh",
            patches,
            self.weight,
        )

        # Routing logits:
        #
        # B L Y X I O

        logits = torch.zeros(
            batch_size,
            locations,
            self.kernel_size,
            self.kernel_size,
            self.in_capsules,
            self.out_capsules,
            device=x.device,
            dtype=x.dtype,
        )

        for iteration in range(
            self.routing_iterations
        ):

            coupling = F.softmax(
                logits,
                dim=-1,
            )

            # Sum over:
            #
            # kernel_y
            # kernel_x
            # input capsules
            #
            # Result:
            # B L O H

            s = torch.sum(
                coupling.unsqueeze(-1)
                * predictions,
                dim=(2, 3, 4),
            )

            outputs = squash(
                s,
                dim=-1,
            )

            if (
                iteration
                < self.routing_iterations - 1
            ):

                agreement = torch.sum(
                    predictions
                    * outputs
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .unsqueeze(4),
                    dim=-1,
                )

                logits = (
                    logits
                    + agreement
                )

        # outputs:
        # B L O H
        #
        # Convert locations back into
        # spatial dimensions.

        outputs = outputs.view(
            batch_size,
            output_height,
            output_width,
            self.out_capsules,
            self.out_dim,
        )

        # Desired:
        # B O H W D

        outputs = outputs.permute(
            0,
            3,
            1,
            2,
            4,
        ).contiguous()

        return outputs


class SpatialStream(nn.Module):
    """
    TSCCN spectral-spatial stream.

    Input:
        (B, 100, 27, 27)

    Entity capsule output:
        (B, num_classes, entity_dim)

    Scores:
        (B, num_classes)
    """

    def __init__(
        self,
        num_bands=100,
        num_classes=4,
        sim_groups=4,
        entity_dim=8,
        routing_iterations=3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.entity_dim = entity_dim

        # -----------------------------------------
        # Layer 1
        #
        # 3x3x32
        # BN = Yes
        # stride = 1
        # padding = No
        # ReLU
        # -----------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=num_bands,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            32
        )

        self.relu1 = nn.ReLU(
            inplace=True
        )

        # -----------------------------------------
        # Layer 2
        # 2D SIM
        # -----------------------------------------

        self.sim = SIM2D(
            in_channels=32,
            groups=sim_groups,
        )

        # Each group currently produces
        # 32 output channels.

        sim_out_channels = (
            sim_groups * 32
        )

        # -----------------------------------------
        # Layer 3
        #
        # 3x3x64
        # BN = Yes
        # stride = 1
        # padding = No
        # ReLU
        # -----------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=sim_out_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            64
        )

        self.relu2 = nn.ReLU(
            inplace=True
        )

        # -----------------------------------------
        # Layer 4
        #
        # Primary capsules
        # (4x4x8) x 4
        # -----------------------------------------

        self.primary_caps = (
            PrimaryCapsule2D(
                in_channels=64,
                capsule_channels=4,
                capsule_dim=8,
            )
        )

        # -----------------------------------------
        # Layer 5
        #
        # Conv capsules
        # (3x3x4) x 8
        # -----------------------------------------

        self.conv_caps = (
            ConvCapsule2D(
                in_capsules=4,
                in_dim=8,
                out_capsules=8,
                out_dim=4,
                kernel_size=3,
                stride=2,
                routing_iterations=(
                    routing_iterations
                ),
            )
        )

        # Determine the number of capsules
        # entering the entity layer.

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                num_bands,
                27,
                27,
            )

            features = (
                self._forward_features(
                    dummy
                )
            )

            # features:
            #
            # B, capsule_channels,
            # H, W, capsule_dim

            num_entity_inputs = (
                features.size(1)
                * features.size(2)
                * features.size(3)
            )

            entity_input_dim = (
                features.size(4)
            )

        # -----------------------------------------
        # Layer 6
        #
        # Entity capsules
        # -----------------------------------------

        self.entity_caps = DynamicRouting(
            num_input_capsules=(
                num_entity_inputs
            ),
            input_dim=(
                entity_input_dim
            ),
            num_output_capsules=(
                num_classes
            ),
            output_dim=(
                entity_dim
            ),
            routing_iterations=(
                routing_iterations
            ),
        )

    def _forward_features(
        self,
        patch,
    ):

        if patch.ndim != 4:
            raise ValueError(
                "SpatialStream expects "
                "(batch, bands, height, width)"
            )

        # Layer 1

        x = self.conv1(patch)
        x = self.bn1(x)
        x = self.relu1(x)

        # Layer 2

        x = self.sim(x)

        # Layer 3

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Layer 4

        x = self.primary_caps(x)

        # Layer 5

        x = self.conv_caps(x)

        return x

    def forward(
        self,
        patch,
    ):

        x = self._forward_features(
            patch
        )

        batch_size = x.size(0)

        # Current:
        #
        # B, capsule_channels,
        # H, W, capsule_dim

        # Flatten capsule channel +
        # spatial positions.

        x = x.reshape(
            batch_size,
            -1,
            x.size(-1),
        )

        # Final entity capsules:
        #
        # B, 4, entity_dim

        capsules = self.entity_caps(
            x
        )

        # L2 class scores:
        #
        # B, 4

        scores = capsule_lengths(
            capsules
        )

        return {
            "capsules": capsules,
            "scores": scores,
        }