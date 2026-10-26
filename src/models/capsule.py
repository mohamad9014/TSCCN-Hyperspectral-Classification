import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x, dim=-1, eps=1e-8):
    """
    Capsule squashing function.

    Parameters
    ----------
    x : torch.Tensor
        Input capsule vectors.

    dim : int
        Dimension containing the capsule vector.

    eps : float
        Small value for numerical stability.

    Returns
    -------
    torch.Tensor
        Squashed capsule vectors with the same shape as x.
    """

    squared_norm = torch.sum(
        x ** 2,
        dim=dim,
        keepdim=True,
    )

    norm = torch.sqrt(
        squared_norm + eps
    )

    scale = (
        squared_norm
        / (1.0 + squared_norm)
    )

    return scale * (
        x / norm
    )


class DynamicRouting(nn.Module):
    """
    Fully connected capsule routing.

    Input shape:
        (batch, num_input_capsules, input_dim)

    Output shape:
        (batch, num_output_capsules, output_dim)
    """

    def __init__(
        self,
        num_input_capsules,
        input_dim,
        num_output_capsules,
        output_dim,
        routing_iterations=3,
    ):
        super().__init__()

        self.num_input_capsules = (
            num_input_capsules
        )

        self.input_dim = input_dim

        self.num_output_capsules = (
            num_output_capsules
        )

        self.output_dim = output_dim

        self.routing_iterations = (
            routing_iterations
        )

        # Transformation matrices:
        #
        # one matrix for every
        # input capsule -> output capsule pair
        #
        # Shape:
        # (
        #   num_input_capsules,
        #   num_output_capsules,
        #   input_dim,
        #   output_dim
        # )

        self.weight = nn.Parameter(
            torch.empty(
                num_input_capsules,
                num_output_capsules,
                input_dim,
                output_dim,
            )
        )

        nn.init.xavier_uniform_(
            self.weight
        )

    def forward(self, x):
        """
        Perform dynamic routing.
        """

        if x.ndim != 3:
            raise ValueError(
                "DynamicRouting expects input "
                "shape "
                "(batch, num_capsules, capsule_dim)"
            )

        batch_size = x.size(0)

        if (
            x.size(1)
            != self.num_input_capsules
        ):
            raise ValueError(
                "Incorrect number of "
                "input capsules. "
                f"Expected "
                f"{self.num_input_capsules}, "
                f"got {x.size(1)}"
            )

        if x.size(2) != self.input_dim:
            raise ValueError(
                "Incorrect capsule dimension. "
                f"Expected {self.input_dim}, "
                f"got {x.size(2)}"
            )

        # -------------------------------------------------
        # Prediction vectors
        # -------------------------------------------------
        #
        # x:
        # (B, I, D_in)
        #
        # weight:
        # (I, O, D_in, D_out)
        #
        # predictions:
        # (B, I, O, D_out)

        predictions = torch.einsum(
            "bid,iodh->bioh",
            x,
            self.weight,
        )

        # Routing logits b_ij.
        #
        # Initially all zeros as described
        # in the dynamic routing algorithm.
        #
        # Shape:
        # (B, I, O)

        routing_logits = torch.zeros(
            batch_size,
            self.num_input_capsules,
            self.num_output_capsules,
            device=x.device,
            dtype=x.dtype,
        )

        for iteration in range(
            self.routing_iterations
        ):

            # Coupling coefficients c_ij.
            #
            # For every input capsule,
            # coefficients over output
            # capsules sum to one.

            coupling = F.softmax(
                routing_logits,
                dim=2,
            )

            # Weighted sum of predictions.
            #
            # output_input:
            # (B, O, D_out)

            output_input = torch.sum(
                coupling.unsqueeze(-1)
                * predictions,
                dim=1,
            )

            # Squashing activation.

            outputs = squash(
                output_input,
                dim=-1,
            )

            # Update routing logits except
            # after the final iteration.

            if (
                iteration
                < self.routing_iterations - 1
            ):

                # Agreement:
                #
                # prediction dot output
                #
                # Shape:
                # (B, I, O)

                agreement = torch.sum(
                    predictions
                    * outputs.unsqueeze(1),
                    dim=-1,
                )

                routing_logits = (
                    routing_logits
                    + agreement
                )

        return outputs


def capsule_lengths(
    capsules,
    dim=-1,
):
    return torch.linalg.vector_norm(
        capsules,
        ord=2,
        dim=dim,
    )
    """
    Calculate L2 lengths of capsule vectors.

    These lengths are later used as
    class scores/probabilities.

    Example:

        capsules shape:
            (B, 4, 8)

        output shape:
            (B, 4)
    """

    return torch.sqrt(
        torch.sum(
            capsules ** 2,
            dim=dim,
        )
        + eps
    )