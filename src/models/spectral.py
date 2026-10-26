import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .capsule import (
    squash,
    DynamicRouting,
    capsule_lengths,
)

from .sim import SIM1D


# ============================================================
# PRIMARY 1D CAPSULE LAYER
# ============================================================

class PrimaryCapsule1D(nn.Module):
    """
    Paper Layer 4 of the spectral stream.

    Configuration:

        kernel size      = 12
        stride           = 2
        padding          = No

        capsule channels = 16
        capsule dim      = 10

        BN               = Yes
        activation       = ReLU + Squash

    Input:
        (B, 64, L)

    For L = 92:

        output length = 41

    Output:
        (B, 16, 41, 10)
    """

    def __init__(
        self,
        in_channels=64,
        capsule_channels=16,
        capsule_dim=10,
    ):
        super().__init__()

        self.capsule_channels = (
            capsule_channels
        )

        self.capsule_dim = (
            capsule_dim
        )

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=(
                capsule_channels
                * capsule_dim
            ),
            kernel_size=12,
            stride=2,
            padding=0,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(
            capsule_channels
            * capsule_dim
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        batch_size = x.size(0)
        length = x.size(2)

        # Current:
        #
        # B,
        # capsule_channels * capsule_dim,
        # L

        x = x.view(
            batch_size,
            self.capsule_channels,
            self.capsule_dim,
            length,
        )

        # Desired:
        #
        # B,
        # capsule_channels,
        # L,
        # capsule_dim

        x = x.permute(
            0,
            1,
            3,
            2,
        ).contiguous()

        x = squash(
            x,
            dim=-1,
        )

        return x


# ============================================================
# 1D CONVOLUTIONAL CAPSULE LAYER
# ============================================================

class ConvCapsule1D(nn.Module):
    """
    Paper Layer 5 of the spectral stream.

    Configuration:

        kernel size      = 4
        stride           = 2
        padding          = Yes

        input capsules   = 16
        input dim        = 10

        output capsules  = 8
        output dim       = 8

        activation       = Squash

    Input:
        (B, 16, L, 10)

    With L=41 and TensorFlow-style SAME padding:

        output length = ceil(41 / 2) = 21

    Output:
        (B, 8, 21, 8)

    Local dynamic routing is performed independently
    at every output spectral position.
    """

    def __init__(
        self,
        in_capsules=16,
        in_dim=10,
        out_capsules=8,
        out_dim=8,
        kernel_size=4,
        stride=2,
        routing_iterations=3,
    ):
        super().__init__()

        self.in_capsules = (
            in_capsules
        )

        self.in_dim = (
            in_dim
        )

        self.out_capsules = (
            out_capsules
        )

        self.out_dim = (
            out_dim
        )

        self.kernel_size = (
            kernel_size
        )

        self.stride = (
            stride
        )

        self.routing_iterations = (
            routing_iterations
        )

        # Shared transformation matrix.
        #
        # Dimensions:
        #
        # kernel position
        # input capsule
        # output capsule
        # input dimension
        # output dimension

        self.weight = nn.Parameter(
            torch.empty(
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

    def _same_padding(
        self,
        length,
    ):
        """
        TensorFlow-style SAME padding.

        output_length = ceil(length / stride)
        """

        output_length = math.ceil(
            length / self.stride
        )

        total_padding = max(
            (
                (output_length - 1)
                * self.stride
                + self.kernel_size
                - length
            ),
            0,
        )

        padding_left = (
            total_padding // 2
        )

        padding_right = (
            total_padding
            - padding_left
        )

        return (
            padding_left,
            padding_right,
            output_length,
        )

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                "ConvCapsule1D expects "
                "(batch, capsules, length, dim)"
            )

        batch_size = x.size(0)

        if x.size(1) != self.in_capsules:
            raise ValueError(
                f"Expected {self.in_capsules} "
                f"input capsules, got {x.size(1)}."
            )

        if x.size(3) != self.in_dim:
            raise ValueError(
                f"Expected capsule dimension "
                f"{self.in_dim}, got {x.size(3)}."
            )

        input_length = x.size(2)

        (
            pad_left,
            pad_right,
            output_length,
        ) = self._same_padding(
            input_length
        )

        # Current:
        #
        # B, I, L, D
        #
        # Change to:
        #
        # B, I, D, L

        x = x.permute(
            0,
            1,
            3,
            2,
        ).contiguous()

        # Pad only final spectral dimension.

        x = F.pad(
            x,
            (
                pad_left,
                pad_right,
            ),
        )

        # Extract local windows along spectral length.
        #
        # Result:
        #
        # B, I, D, output_length, kernel_size

        patches = x.unfold(
            dimension=3,
            size=self.kernel_size,
            step=self.stride,
        )

        # Change to:
        #
        # B,
        # output_length,
        # kernel_size,
        # input_capsules,
        # input_dim

        patches = patches.permute(
            0,
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
        # B L K I D
        #
        # weights:
        # K I O D H
        #
        # predictions:
        # B L K I O H

        predictions = torch.einsum(
            "blkid,kiodh->blkioh",
            patches,
            self.weight,
        )

        # Routing logits:
        #
        # B L K I O

        logits = torch.zeros(
            batch_size,
            output_length,
            self.kernel_size,
            self.in_capsules,
            self.out_capsules,
            device=x.device,
            dtype=x.dtype,
        )

        for iteration in range(
            self.routing_iterations
        ):

            # Each lower capsule distributes
            # its coupling across output capsules.

            coupling = F.softmax(
                logits,
                dim=-1,
            )

            # Weighted sum over:
            #
            # kernel positions
            # input capsule channels
            #
            # B L O H

            s = torch.sum(
                coupling.unsqueeze(-1)
                * predictions,
                dim=(2, 3),
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
                    .unsqueeze(3),
                    dim=-1,
                )

                logits = (
                    logits
                    + agreement
                )

        # Current:
        #
        # B, L, O, H
        #
        # Desired:
        #
        # B, O, L, H

        outputs = outputs.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        return outputs


# ============================================================
# COMPLETE SPECTRAL STREAM
# ============================================================

class SpectralStream(nn.Module):
    """
    TSCCN spectral sub-network.

    Paper architecture:

        Layer 1
        Conv1D:
            kernel 9
            32 filters
            stride 1
            padding yes
            BN
            ReLU

        Layer 2
        1D SIM

        Layer 3
        Conv1D:
            kernel 9
            64 filters
            stride 1
            no padding
            BN
            ReLU

        Layer 4
        Primary capsules:
            (12 x 1 x 10) x 16
            stride 2

        Layer 5
        Conv capsules:
            (4 x 1 x 8) x 8
            stride 2
            padding yes

        Layer 6
        Entity capsules:
            one capsule per class


    Input:
        (B, 100)

    Output dictionary:

        capsules:
            (B, num_classes, entity_dim)

        scores:
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

        self.num_bands = (
            num_bands
        )

        self.num_classes = (
            num_classes
        )

        self.entity_dim = (
            entity_dim
        )

        # ==================================================
        # Layer 1
        # ==================================================

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(
            32
        )

        self.relu1 = nn.ReLU(
            inplace=True
        )

        # ==================================================
        # Layer 2
        #
        # Corrected SIM1D:
        #
        # input  = B,32,100
        # output = B,32,100
        # ==================================================

        self.sim = SIM1D(
            in_channels=32,
            groups=sim_groups,
        )

        # ==================================================
        # Layer 3
        #
        # IMPORTANT CORRECTION:
        #
        # SIM1D now produces only 32 channels.
        #
        # Therefore conv2 input channels = 32.
        # ==================================================

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(
            64
        )

        self.relu2 = nn.ReLU(
            inplace=True
        )

        # ==================================================
        # Layer 4
        # ==================================================

        self.primary_caps = PrimaryCapsule1D(
            in_channels=64,
            capsule_channels=16,
            capsule_dim=10,
        )

        # ==================================================
        # Layer 5
        # ==================================================

        self.conv_caps = ConvCapsule1D(
            in_capsules=16,
            in_dim=10,
            out_capsules=8,
            out_dim=8,
            kernel_size=4,
            stride=2,
            routing_iterations=(
                routing_iterations
            ),
        )

        # ==================================================
        # Determine input count for entity capsules.
        # ==================================================

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                num_bands,
            )

            features = (
                self._forward_features(
                    dummy
                )
            )

            # Expected:
            #
            # B, 8, 21, 8

            num_entity_inputs = (
                features.size(1)
                * features.size(2)
            )

            entity_input_dim = (
                features.size(3)
            )

        # ==================================================
        # Layer 6
        # ==================================================

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
        spectral,
    ):

        if spectral.ndim != 2:
            raise ValueError(
                "SpectralStream expects "
                "(batch, spectral_bands)"
            )

        if spectral.size(1) != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} bands, "
                f"got {spectral.size(1)}."
            )

        # Input:
        #
        # B, 100

        # Add channel dimension:
        #
        # B, 1, 100

        x = spectral.unsqueeze(1)

        # --------------------------------------------------
        # Layer 1
        #
        # B,1,100
        # ->
        # B,32,100
        # --------------------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # --------------------------------------------------
        # Layer 2
        #
        # B,32,100
        # ->
        # B,32,100
        # --------------------------------------------------

        x = self.sim(x)

        # --------------------------------------------------
        # Layer 3
        #
        # B,32,100
        # ->
        # B,64,92
        # --------------------------------------------------

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # --------------------------------------------------
        # Layer 4
        #
        # B,64,92
        # ->
        # B,16,41,10
        # --------------------------------------------------

        x = self.primary_caps(x)

        # --------------------------------------------------
        # Layer 5
        #
        # B,16,41,10
        # ->
        # B,8,21,8
        # --------------------------------------------------

        x = self.conv_caps(x)

        return x

    def forward(
        self,
        spectral,
    ):

        x = self._forward_features(
            spectral
        )

        batch_size = x.size(0)

        # Current:
        #
        # B,
        # capsule channels,
        # spectral positions,
        # capsule dimension

        # Flatten capsule channel and position:
        #
        # B, 8*21, 8
        #
        # =
        #
        # B, 168, 8

        x = x.reshape(
            batch_size,
            -1,
            x.size(-1),
        )

        # Final class/entity capsules:
        #
        # B,4,entity_dim

        capsules = self.entity_caps(
            x
        )

        # L2 lengths:
        #
        # B,4

        scores = capsule_lengths(
            capsules
        )

        return {
            "capsules": capsules,
            "scores": scores,
        }