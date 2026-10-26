import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    """
    Capsule margin loss.

    Paper equation:

        L_j =
            I_j * max(0, tau_plus - ||v_j||)^2
            +
            lambda * (1 - I_j)
            * max(0, ||v_j|| - tau_minus)^2

    Total loss is summed over classes and
    averaged over the batch.

    Notes
    -----
    lambda_value = 0.5
        This value is explicitly stated in the TSCCN paper.

    tau_plus = 0.9
    tau_minus = 0.1
        These numerical values are NOT explicitly reported
        in the TSCCN paper. We use common CapsNet values
        as an implementation choice.
    """

    def __init__(
        self,
        tau_plus=0.9,
        tau_minus=0.1,
        lambda_value=0.5,
    ):
        super().__init__()

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.lambda_value = lambda_value

    def forward(
        self,
        scores,
        labels,
    ):
        """
        Parameters
        ----------
        scores:
            Capsule lengths.

            Shape:
                (B, num_classes)

        labels:
            Integer class labels.

            Shape:
                (B,)

            Example:
                [0, 2, 1, 3]


        Returns
        -------
        Scalar loss.
        """

        if scores.ndim != 2:
            raise ValueError(
                "scores must have shape "
                "(batch, num_classes)"
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape (batch,)"
            )

        if scores.size(0) != labels.size(0):
            raise ValueError(
                "scores and labels must have "
                "the same batch size"
            )

        num_classes = scores.size(1)

        # ----------------------------------------------------
        # Convert labels to one-hot representation
        #
        # Example:
        #
        # label = 2
        #
        # becomes:
        #
        # [0, 0, 1, 0]
        # ----------------------------------------------------

        targets = F.one_hot(
            labels,
            num_classes=num_classes,
        ).to(
            dtype=scores.dtype,
            device=scores.device,
        )

        # ----------------------------------------------------
        # Positive-class loss
        #
        # We want the correct class score to become
        # larger than tau_plus.
        #
        # Example:
        #
        # correct score = 0.4
        # tau_plus = 0.9
        #
        # penalty:
        #
        # (0.9 - 0.4)^2
        # ----------------------------------------------------

        positive_loss = targets * torch.clamp(
            self.tau_plus - scores,
            min=0.0,
        ).pow(2)

        # ----------------------------------------------------
        # Negative-class loss
        #
        # We want incorrect class scores to become
        # smaller than tau_minus.
        # ----------------------------------------------------

        negative_loss = (
            self.lambda_value
            * (1.0 - targets)
            * torch.clamp(
                scores - self.tau_minus,
                min=0.0,
            ).pow(2)
        )

        # ----------------------------------------------------
        # Add positive + negative losses.
        # ----------------------------------------------------

        class_loss = (
            positive_loss
            + negative_loss
        )

        # Sum over classes.
        #
        # Shape:
        # (B,)

        sample_loss = class_loss.sum(
            dim=1
        )

        # Average over batch.

        loss = sample_loss.mean()

        return loss