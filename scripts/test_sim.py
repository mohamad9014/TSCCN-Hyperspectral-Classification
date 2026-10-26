from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.models.sim import SIM1D, SIM2D


# ============================================================
# TEST SIM1D
# ============================================================

print("Testing SIM1D...")

sim1d = SIM1D(
    in_channels=32,
    groups=4,
)

x1 = torch.randn(
    8,
    32,
    100,
)

y1 = sim1d(x1)

print("Input shape:")
print(x1.shape)

print("Output shape:")
print(y1.shape)


# Correct paper-faithful output:
# [8, 32, 100]

assert y1.shape == torch.Size(
    [8, 32, 100]
)

assert torch.isfinite(
    y1
).all()


print("SIM1D PASSED")


# ============================================================
# TEST SIM2D
# ============================================================

print("\nTesting SIM2D...")

sim2d = SIM2D(
    in_channels=32,
    groups=4,
)

x2 = torch.randn(
    8,
    32,
    25,
    25,
)

y2 = sim2d(x2)

print("Input shape:")
print(x2.shape)

print("Output shape:")
print(y2.shape)


# 4 groups x 32 output channels = 128

assert y2.shape == torch.Size(
    [8, 128, 25, 25]
)

assert torch.isfinite(
    y2
).all()


print("SIM2D PASSED")

print("\nTEST PASSED")