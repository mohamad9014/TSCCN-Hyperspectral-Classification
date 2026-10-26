from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.models.loss import MarginLoss


print("Creating margin loss...")


criterion = MarginLoss(
    tau_plus=0.9,
    tau_minus=0.1,
    lambda_value=0.5,
)


# ============================================================
# TEST 1
# RANDOM-LIKE SCORES
# ============================================================

print("\nTest 1: ordinary scores")


scores = torch.tensor(
    [
        [0.20, 0.30, 0.60, 0.10],
        [0.70, 0.20, 0.10, 0.10],
    ],
    dtype=torch.float32,
)


labels = torch.tensor(
    [
        2,
        0,
    ],
    dtype=torch.long,
)


loss = criterion(
    scores,
    labels,
)


print("Scores:")
print(scores)

print("\nLabels:")
print(labels)

print("\nLoss:")
print(loss)


assert loss.ndim == 0

assert torch.isfinite(
    loss
)

assert loss.item() >= 0


# ============================================================
# TEST 2
# VERY GOOD PREDICTIONS
# ============================================================

print(
    "\nTest 2: good predictions"
)


good_scores = torch.tensor(
    [
        [0.95, 0.02, 0.03, 0.01],
        [0.01, 0.03, 0.95, 0.02],
    ],
    dtype=torch.float32,
)


good_labels = torch.tensor(
    [
        0,
        2,
    ],
    dtype=torch.long,
)


good_loss = criterion(
    good_scores,
    good_labels,
)


print("Good scores:")
print(good_scores)

print("\nGood loss:")
print(good_loss)


# A very good prediction should give
# almost zero margin loss.

assert good_loss.item() < 1e-6


# ============================================================
# TEST 3
# BAD PREDICTIONS
# ============================================================

print(
    "\nTest 3: bad predictions"
)


bad_scores = torch.tensor(
    [
        [0.05, 0.90, 0.80, 0.70],
        [0.80, 0.70, 0.05, 0.90],
    ],
    dtype=torch.float32,
)


bad_labels = torch.tensor(
    [
        0,
        2,
    ],
    dtype=torch.long,
)


bad_loss = criterion(
    bad_scores,
    bad_labels,
)


print("Bad scores:")
print(bad_scores)

print("\nBad loss:")
print(bad_loss)


assert bad_loss > good_loss


print(
    "\nMARGIN LOSS TEST PASSED"
)