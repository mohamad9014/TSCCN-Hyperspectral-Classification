import torch
import torch.nn as nn

from .capsule import capsule_lengths
from .spectral import SpectralStream
from .spatial import SpatialStream


class TSCCN(nn.Module):
    """
    Two-Stream Spectral-Spatial Convolutional Capsule Network.

    The model has two parallel streams:

    1. Spectral stream
       Input:
           (B, 100)

       Output entity capsules:
           (B, num_classes, entity_dim)


    2. Spectral-spatial stream
       Input:
           (B, 100, 27, 27)

       Output entity capsules:
           (B, num_classes, entity_dim)


    Fusion:
        Concatenate corresponding class capsules
        from the two streams.

        Example with entity_dim=8:

            spectral:
                (B, 4, 8)

            spatial:
                (B, 4, 8)

            fused:
                (B, 4, 16)


    Classification:
        Compute L2 norm of each fused class vector.

        Result:
            (B, 4)

    The largest score gives the predicted class.
    """

    def __init__(
        self,
        num_bands=100,
        num_classes=4,
        patch_size=27,
        sim_groups=4,
        entity_dim=8,
        routing_iterations=3,
    ):
        super().__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.entity_dim = entity_dim

        # ====================================================
        # STREAM 1
        # Spectral Conv-CapsNet
        # ====================================================

        self.spectral_stream = SpectralStream(
            num_bands=num_bands,
            num_classes=num_classes,
            sim_groups=sim_groups,
            entity_dim=entity_dim,
            routing_iterations=routing_iterations,
        )

        # ====================================================
        # STREAM 2
        # Spectral-spatial Conv-CapsNet
        # ====================================================

        self.spatial_stream = SpatialStream(
         num_bands=num_bands,
         num_classes=num_classes,
         sim_groups=sim_groups,
         entity_dim=entity_dim,
         routing_iterations=routing_iterations,
        )

    def forward(
        self,
        spectral,
        patch,
    ):
        """
        Parameters
        ----------
        spectral:
            Tensor with shape:

                (B, 100)

        patch:
            Tensor with shape:

                (B, 100, 27, 27)


        Returns
        -------
        Dictionary containing:

            scores
            fused_capsules
            spectral_capsules
            spatial_capsules
            spectral_scores
            spatial_scores
        """

        # ====================================================
        # BASIC INPUT CHECKS
        # ====================================================

        if spectral.ndim != 2:
            raise ValueError(
                "spectral input must have shape "
                "(batch, spectral_bands)"
            )

        if patch.ndim != 4:
            raise ValueError(
                "patch input must have shape "
                "(batch, spectral_bands, height, width)"
            )

        # Batch sizes must match.

        if spectral.size(0) != patch.size(0):
            raise ValueError(
                "Spectral and spatial inputs must "
                "have the same batch size."
            )

        # Check spectral bands.

        if spectral.size(1) != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} spectral bands "
                f"but got {spectral.size(1)}."
            )

        if patch.size(1) != self.num_bands:
            raise ValueError(
                f"Expected patch to contain "
                f"{self.num_bands} spectral bands "
                f"but got {patch.size(1)}."
            )

        # Check spatial patch dimensions.

        if (
            patch.size(2) != self.patch_size
            or
            patch.size(3) != self.patch_size
        ):
            raise ValueError(
                f"Expected spatial patch size "
                f"{self.patch_size}x{self.patch_size}, "
                f"but got "
                f"{patch.size(2)}x{patch.size(3)}."
            )

        # ====================================================
        # STREAM 1
        # ====================================================

        spectral_output = self.spectral_stream(
            spectral
        )

        spectral_capsules = spectral_output[
            "capsules"
        ]

        spectral_scores = spectral_output[
            "scores"
        ]

        # Expected:
        #
        # spectral_capsules
        #
        # B, 4, 8

        # ====================================================
        # STREAM 2
        # ====================================================

        spatial_output = self.spatial_stream(
            patch
        )

        spatial_capsules = spatial_output[
            "capsules"
        ]

        spatial_scores = spatial_output[
            "scores"
        ]

        # Expected:
        #
        # spatial_capsules
        #
        # B, 4, 8

        # ====================================================
        # CHECK THAT BOTH STREAMS MATCH
        # ====================================================

        if (
            spectral_capsules.size(0)
            != spatial_capsules.size(0)
        ):
            raise RuntimeError(
                "The two streams produced different "
                "batch sizes."
            )

        if (
            spectral_capsules.size(1)
            != spatial_capsules.size(1)
        ):
            raise RuntimeError(
                "The two streams produced different "
                "numbers of class capsules."
            )

        # ====================================================
        # FEATURE FUSION
        # ====================================================
        #
        # Paper:
        #
        # The separated entity-related features from
        # the two streams are fused by concatenation.
        #
        #
        # spectral:
        #
        #     B,4,8
        #
        # spatial:
        #
        #     B,4,8
        #
        # concatenate last dimension:
        #
        #     B,4,16
        # ====================================================

        fused_capsules = torch.cat(
            [
                spectral_capsules,
                spatial_capsules,
            ],
            dim=-1,
        )

        # ====================================================
        # L2-NORM CLASSIFIER
        # ====================================================
        #
        # Calculate length of every fused class vector.
        #
        # fused:
        #
        #     B,4,16
        #
        # scores:
        #
        #     B,4
        # ====================================================

        scores = capsule_lengths(
            fused_capsules,
            dim=-1,
        )

        # ====================================================
        # RETURN EVERYTHING USEFUL
        # ====================================================

        return {
            "scores": scores,

            "fused_capsules": fused_capsules,

            "spectral_capsules": spectral_capsules,

            "spatial_capsules": spatial_capsules,

            "spectral_scores": spectral_scores,

            "spatial_scores": spatial_scores,
        }

    def predict(
        self,
        spectral,
        patch,
    ):
        """
        Return predicted class index.

        Example:

            0 = kidney
            1 = liver
            2 = spleen
            3 = stomach
        """

        output = self.forward(
            spectral,
            patch,
        )

        predictions = torch.argmax(
            output["scores"],
            dim=1,
        )

        return predictions