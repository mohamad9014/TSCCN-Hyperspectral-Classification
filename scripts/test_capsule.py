from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.models.capsule import (
    squash,
    DynamicRouting,
    capsule_lengths,
)


print("Testing squash...")

x = torch.randn(
    8,
    20,
    8,
)

y = squash(x)

print("Input shape:")
print(x.shape)

print("Output shape:")
print(y.shape)

lengths = torch.linalg.norm(
    y,
    dim=-1,
)

print("Maximum squashed length:")
print(lengths.max())

assert y.shape == x.shape

assert torch.all(
    lengths < 1.0
)


print("\nTesting dynamic routing...")

routing = DynamicRouting(
    num_input_capsules=20,
    input_dim=8,
    num_output_capsules=4,
    output_dim=8,
    routing_iterations=3,
)

capsules = routing(x)

print("Entity capsule shape:")
print(capsules.shape)

assert capsules.shape == (
    8,
    4,
    8,
)


print("\nTesting capsule lengths...")

scores = capsule_lengths(
    capsules
)

print("Score shape:")
print(scores.shape)

print("Example scores:")
print(scores[0])

assert scores.shape == (
    8,
    4,
)

assert torch.all(
    scores >= 0
)

assert torch.all(
    scores < 1
)


print("\nTEST PASSED")