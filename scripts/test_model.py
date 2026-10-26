from pathlib import Path
import sys

import torch


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.models.tsccn import TSCCN


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 2

NUM_BANDS = 100

NUM_CLASSES = 4

PATCH_SIZE = 27

ENTITY_DIM = 8


# ============================================================
# CREATE MODEL
# ============================================================

print("Creating full TSCCN model...")


model = TSCCN(
    num_bands=NUM_BANDS,
    num_classes=NUM_CLASSES,
    patch_size=PATCH_SIZE,
    sim_groups=4,
    entity_dim=ENTITY_DIM,
    routing_iterations=3,
)


print("Model created successfully.")


# ============================================================
# CREATE RANDOM TEST DATA
# ============================================================

print(
    "\nCreating random input..."
)


# Spectral input:
#
# B × 100

spectral = torch.randn(
    BATCH_SIZE,
    NUM_BANDS,
)


# Spatial input:
#
# B × 100 × 27 × 27

patch = torch.randn(
    BATCH_SIZE,
    NUM_BANDS,
    PATCH_SIZE,
    PATCH_SIZE,
)


print(
    "\nSpectral input shape:"
)

print(
    spectral.shape
)


print(
    "\nPatch input shape:"
)

print(
    patch.shape
)


# ============================================================
# FORWARD PASS
# ============================================================

print(
    "\nRunning full TSCCN..."
)


output = model(
    spectral,
    patch,
)


# ============================================================
# GET OUTPUTS
# ============================================================

spectral_capsules = output[
    "spectral_capsules"
]

spatial_capsules = output[
    "spatial_capsules"
]

fused_capsules = output[
    "fused_capsules"
]

scores = output[
    "scores"
]


# ============================================================
# PRINT SHAPES
# ============================================================

print(
    "\nSpectral entity capsule shape:"
)

print(
    spectral_capsules.shape
)


print(
    "\nSpatial entity capsule shape:"
)

print(
    spatial_capsules.shape
)


print(
    "\nFused capsule shape:"
)

print(
    fused_capsules.shape
)


print(
    "\nFinal score shape:"
)

print(
    scores.shape
)


print(
    "\nExample final scores:"
)

print(
    scores[0]
)


# ============================================================
# EXPECTED SHAPES
# ============================================================

assert spectral_capsules.shape == (
    BATCH_SIZE,
    NUM_CLASSES,
    ENTITY_DIM,
)


assert spatial_capsules.shape == (
    BATCH_SIZE,
    NUM_CLASSES,
    ENTITY_DIM,
)


# Two 8-dimensional vectors are concatenated:
#
# 8 + 8 = 16

assert fused_capsules.shape == (
    BATCH_SIZE,
    NUM_CLASSES,
    ENTITY_DIM * 2,
)


assert scores.shape == (
    BATCH_SIZE,
    NUM_CLASSES,
)


# ============================================================
# CHECK NUMBERS
# ============================================================

assert torch.isfinite(
    spectral_capsules
).all()


assert torch.isfinite(
    spatial_capsules
).all()


assert torch.isfinite(
    fused_capsules
).all()


assert torch.isfinite(
    scores
).all()


assert torch.all(
    scores >= 0
)


# ============================================================
# TEST PREDICTION
# ============================================================

predictions = model.predict(
    spectral,
    patch,
)


print(
    "\nPredicted class indices:"
)

print(
    predictions
)


assert predictions.shape == (
    BATCH_SIZE,
)


assert torch.all(
    predictions >= 0
)


assert torch.all(
    predictions < NUM_CLASSES
)


# ============================================================
# CLASS NAME EXAMPLE
# ============================================================

class_names = [
    "kidney",
    "liver",
    "spleen",
    "stomach",
]


predicted_names = [
    class_names[index]
    for index in predictions.tolist()
]


print(
    "\nPredicted class names:"
)

print(
    predicted_names
)


# ============================================================
# SUCCESS
# ============================================================

print(
    "\nFULL TSCCN TEST PASSED"
)