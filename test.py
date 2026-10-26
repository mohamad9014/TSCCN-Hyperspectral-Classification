import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# IMPORT PROJECT CODE
# ============================================================

from src.data.dataset import (
    HeiPorTSCCNDataset,
    CLASS_NAMES,
)

from src.models.tsccn import TSCCN


# ============================================================
# PATHS
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "cubes_expanded.yaml"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "heiporspectral"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "tsccn_best.pt"
)


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = 27

BATCH_SIZE = 100

# For a quick preliminary test.
#
# You have 4 test cubes.
# 100 pixels/cube -> 400 total test samples.
#
# Later, for final evaluation, we can increase this.
TEST_SAMPLES_PER_CUBE = 100

CACHE_SIZE = 4

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TEST FUNCTION
# ============================================================

@torch.no_grad()
def test_model(
    model,
    loader,
    device,
):

    model.eval()

    all_predictions = []
    all_labels = []

    total_correct = 0
    total_samples = 0

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):

        spectral = batch[
            "spectral"
        ].to(
            device
        )

        patch = batch[
            "patch"
        ].to(
            device
        )

        labels = batch[
            "label"
        ].to(
            device
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        output = model(
            spectral,
            patch,
        )

        scores = output[
            "scores"
        ]

        predictions = torch.argmax(
            scores,
            dim=1,
        )

        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        correct = (
            predictions
            == labels
        ).sum().item()

        total_correct += correct

        total_samples += labels.size(
            0
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        batch_accuracy = (
            correct
            / labels.size(0)
        )

        print(
            f"Batch "
            f"{batch_number:3d}/"
            f"{len(loader):3d}"
            f" | "
            f"Accuracy: "
            f"{batch_accuracy * 100:.2f}%"
        )

    overall_accuracy = (
        total_correct
        / total_samples
    )

    return (
        overall_accuracy,
        all_labels,
        all_predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "========================================"
    )

    print(
        "TSCCN TESTING"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Device:"
    )

    print(
        DEVICE
    )

    print()

    print(
        "Classes:"
    )

    print(
        CLASS_NAMES
    )

    # ========================================================
    # CHECK CHECKPOINT
    # ========================================================

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    print()

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    # ========================================================
    # TEST DATASET
    # ========================================================

    print()

    print(
        "Creating test dataset..."
    )

    test_dataset = HeiPorTSCCNDataset(

        config_path=CONFIG_PATH,

        data_root=DATA_ROOT,

        split="test",

        patch_size=PATCH_SIZE,

        samples_per_cube=(
            TEST_SAMPLES_PER_CUBE
        ),

        seed=SEED,

        cache_size=CACHE_SIZE,

    )

    print(
        "Test samples:"
    )

    print(
        len(test_dataset)
    )

    # ========================================================
    # TEST DATALOADER
    # ========================================================

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=(
            torch.cuda.is_available()
        ),

    )

    print()

    print(
        "Test batches:"
    )

    print(
        len(test_loader)
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()

    print(
        "Creating TSCCN model..."
    )

    model = TSCCN(

        num_classes=len(
            CLASS_NAMES
        )

    )

    model = model.to(
        DEVICE
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()

    print(
        "Loading checkpoint..."
    )

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE,

    )

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )

    print(
        "Checkpoint loaded."
    )

    if "epoch" in checkpoint:

        print()

        print(
            "Checkpoint epoch:"
        )

        print(
            checkpoint[
                "epoch"
            ]
        )

    if "val_accuracy" in checkpoint:

        print()

        print(
            "Saved validation accuracy:"
        )

        print(
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

    # ========================================================
    # TEST
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "RUNNING TEST"
    )

    print(
        "========================================"
    )

    print()

    (
        test_accuracy,
        true_labels,
        predictions,
    ) = test_model(

        model=model,

        loader=test_loader,

        device=DEVICE,

    )

    # ========================================================
    # FINAL ACCURACY
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "TEST RESULTS"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Overall Test Accuracy:"
    )

    print(
        f"{test_accuracy * 100:.2f}%"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(

        true_labels,

        predictions,

        labels=list(
            range(
                len(CLASS_NAMES)
            )
        ),

    )

    print()

    print(
        "Confusion Matrix:"
    )

    print()

    print(
        matrix
    )

    # ========================================================
    # PER-CLASS ACCURACY
    # ========================================================

    print()

    print(
        "Per-Class Accuracy:"
    )

    print()

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_total = matrix[
            class_index
        ].sum()

        class_correct = matrix[
            class_index,
            class_index,
        ]

        if class_total > 0:

            class_accuracy = (
                class_correct
                / class_total
            )

        else:

            class_accuracy = 0.0

        print(
            f"{class_name:10s}: "
            f"{class_accuracy * 100:.2f}% "
            f"("
            f"{class_correct}/"
            f"{class_total}"
            f")"
        )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()

    print(
        "Classification Report:"
    )

    print()

    report = classification_report(

        true_labels,

        predictions,

        labels=list(
            range(
                len(CLASS_NAMES)
            )
        ),

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0,

    )

    print(
        report
    )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print(
        "Prediction Distribution:"
    )

    print()

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        count = predictions.count(
            class_index
        )

        print(
            f"{class_name:10s}: "
            f"{count}"
        )

    print()

    print(
        "========================================"
    )

    print(
        "TESTING FINISHED"
    )

    print(
        "========================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()