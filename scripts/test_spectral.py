from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.models.spectral import (
    SpectralStream,
)


print(
    "Creating spectral stream..."
)

model = SpectralStream(
    num_bands=100,
    num_classes=4,
    sim_groups=4,
    entity_dim=8,
    routing_iterations=3,
)


print(
    "\nTesting random input..."
)

x = torch.randn(
    8,
    100,
)

result = model(x)

capsules = result["capsules"]
scores = result["scores"]


print("\nInput shape:")
print(x.shape)

print("\nEntity capsule shape:")
print(capsules.shape)

print("\nScore shape:")
print(scores.shape)

print("\nExample scores:")
print(scores[0])


assert capsules.shape == (
    8,
    4,
    8,
)

assert scores.shape == (
    8,
    4,
)

assert torch.isfinite(
    capsules
).all()

assert torch.isfinite(
    scores
).all()

assert torch.all(
    scores >= 0
)

assert torch.all(
    scores < 1
)


print("\nTEST PASSED")