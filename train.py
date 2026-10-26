import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


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
from src.models.loss import MarginLoss


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

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 100

LEARNING_RATE = 0.001

# Keep this at 2 first.
# If learning looks correct, later change it to 100.
NUM_EPOCHS = 20

PATCH_SIZE = 27

TRAIN_SAMPLES_PER_CUBE = 500

VAL_SAMPLES_PER_CUBE = 500

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
# RANDOM SEED
# ============================================================

def set_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(
    scores,
    labels,
):

    predictions = torch.argmax(
        scores,
        dim=1,
    )

    correct = (
        predictions
        == labels
    ).sum().item()

    total = labels.size(0)

    return correct, total


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):

        # ----------------------------------------------------
        # Move data to CPU / GPU
        # ----------------------------------------------------

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
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(
            spectral,
            patch,
        )

        scores = output[
            "scores"
        ]

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            scores,
            labels,
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model weights
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        correct, total = calculate_accuracy(
            scores,
            labels,
        )

        total_correct += correct

        total_samples += total

        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        if (
            batch_number % 10 == 0
            or
            batch_number == len(loader)
        ):

            batch_accuracy = (
                correct
                / total
            )

            print(
                f"    Batch "
                f"{batch_number:3d}/"
                f"{len(loader):3d}"
                f" | "
                f"Loss: "
                f"{loss.item():.4f}"
                f" | "
                f"Accuracy: "
                f"{batch_accuracy * 100:.2f}%"
            )

    epoch_loss = (
        total_loss
        / total_samples
    )

    epoch_accuracy = (
        total_correct
        / total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for batch in loader:

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

        output = model(
            spectral,
            patch,
        )

        scores = output[
            "scores"
        ]

        loss = criterion(
            scores,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        correct, total = calculate_accuracy(
            scores,
            labels,
        )

        total_correct += correct

        total_samples += total

    validation_loss = (
        total_loss
        / total_samples
    )

    validation_accuracy = (
        total_correct
        / total_samples
    )

    return (
        validation_loss,
        validation_accuracy,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    print()

    print(
        "========================================"
    )

    print(
        "TSCCN TRAINING"
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
    # CREATE TRAIN DATASET
    # ========================================================

    print()

    print(
        "Creating training dataset..."
    )

    train_dataset = HeiPorTSCCNDataset(
        config_path=CONFIG_PATH,
        data_root=DATA_ROOT,
        split="train",
        patch_size=PATCH_SIZE,
        samples_per_cube=(
            TRAIN_SAMPLES_PER_CUBE
        ),
        seed=SEED,
        cache_size=CACHE_SIZE,
    )

    print(
        "Training samples:"
    )

    print(
        len(train_dataset)
    )

    # ========================================================
    # CREATE VALIDATION DATASET
    # ========================================================

    print()

    print(
        "Creating validation dataset..."
    )

    val_dataset = HeiPorTSCCNDataset(
        config_path=CONFIG_PATH,
        data_root=DATA_ROOT,
        split="val",
        patch_size=PATCH_SIZE,
        samples_per_cube=(
            VAL_SAMPLES_PER_CUBE
        ),
        seed=SEED,
        cache_size=CACHE_SIZE,
    )

    print(
        "Validation samples:"
    )

    print(
        len(val_dataset)
    )

    # ========================================================
    # DATALOADERS
    # ========================================================
    #
    # IMPORTANT:
    #
    # We now use normal shuffle=True.
    #
    # This means batches can contain samples from different
    # classes instead of only one cube / one class.
    #
    # This is better for testing whether the model can learn
    # the four classes correctly.
    #
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,

        # Keep 0 on Windows for now.
        num_workers=0,

        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    print()

    print(
        "Training batches per epoch:"
    )

    print(
        len(train_loader)
    )

    print()

    print(
        "Validation batches:"
    )

    print(
        len(val_loader)
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
        ),
    )

    model = model.to(
        DEVICE
    )

    # ========================================================
    # LOSS
    # ========================================================
    #
    # lambda = 0.5 comes from the paper.
    #
    # tau_plus = 0.9 and tau_minus = 0.1 are common CapsNet
    # implementation choices because the paper does not state
    # their numerical values explicitly.
    #
    # ========================================================

    criterion = MarginLoss(
        tau_plus=0.9,
        tau_minus=0.1,
        lambda_value=0.5,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ========================================================
    # CHECKPOINT SETTINGS
    # ========================================================

    best_val_accuracy = -1.0

    best_val_loss = float(
        "inf"
    )

    best_checkpoint_path = (
        CHECKPOINT_DIR
        / "tsccn_best.pt"
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()

        print(
            "========================================"
        )

        print(
            f"Epoch "
            f"{epoch}/"
            f"{NUM_EPOCHS}"
        )

        print(
            "========================================"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=DEVICE,
            )
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        val_loss, val_accuracy = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print()

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        #
        # Primary criterion:
        #     validation accuracy
        #
        # Tie:
        #     smaller validation loss
        # ----------------------------------------------------

        is_better = False

        if (
            val_accuracy
            > best_val_accuracy
        ):

            is_better = True

        elif (
            val_accuracy
            == best_val_accuracy
            and
            val_loss
            < best_val_loss
        ):

            is_better = True

        if is_better:

            best_val_accuracy = (
                val_accuracy
            )

            best_val_loss = (
                val_loss
            )

            checkpoint = {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "val_loss":
                    val_loss,

                "train_accuracy":
                    train_accuracy,

                "train_loss":
                    train_loss,

                "class_names":
                    CLASS_NAMES,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "patch_size":
                    PATCH_SIZE,

                "samples_per_train_cube":
                    TRAIN_SAMPLES_PER_CUBE,

                "samples_per_val_cube":
                    VAL_SAMPLES_PER_CUBE,
            }

            torch.save(
                checkpoint,
                best_checkpoint_path,
            )

            print()

            print(
                "Best model saved:"
            )

            print(
                best_checkpoint_path
            )

    # ========================================================
    # FINISHED
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "TRAINING FINISHED"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Best validation accuracy:"
    )

    print(
        f"{best_val_accuracy * 100:.2f}%"
    )

    print()

    print(
        "Best validation loss:"
    )

    print(
        f"{best_val_loss:.4f}"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        best_checkpoint_path
    )


if __name__ == "__main__":
    main()
