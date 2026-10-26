import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.data.dataset import (
    HeiPorTSCCNDataset,
    CLASS_NAMES,
)


CONFIG = (
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


def main():

    print(
        "Creating training dataset..."
    )

    train_dataset = HeiPorTSCCNDataset(
        config_path=CONFIG,
        data_root=DATA_ROOT,
        split="train",
        patch_size=27,

        # Use 500 labeled pixels
        # from each cube for testing.
        samples_per_cube=500,

        seed=42,

        # New multi-cube cache.
        cache_size=4,
    )

    print()
    print(
        "Number of training samples:"
    )
    print(
        len(train_dataset)
    )

    print()
    print(
        "Classes:"
    )
    print(
        CLASS_NAMES
    )

    print()
    print(
        "Testing one sample..."
    )

    sample = train_dataset[0]

    print()
    print(
        "Spectral shape:"
    )
    print(
        sample["spectral"].shape
    )

    print()
    print(
        "Patch shape:"
    )
    print(
        sample["patch"].shape
    )

    print()
    print(
        "Label:"
    )
    print(
        sample["label"]
    )

    print()
    print(
        "Position:"
    )
    print(
        sample["metadata"]["row"],
        sample["metadata"]["col"],
    )

    print()
    print(
        "Class name:"
    )
    print(
        sample["metadata"]["class_name"]
    )

    print()
    print(
        "Spectral L1 sum:"
    )
    print(
        torch.abs(
            sample["spectral"]
        ).sum()
    )

    print()
    print(
        "Patch finite:"
    )
    print(
        torch.isfinite(
            sample["patch"]
        ).all()
    )

    print()
    print(
        "Cached cube IDs:"
    )
    print(
        train_dataset.cached_cube_ids()
    )

    # ========================================================
    # TEST DATALOADER
    # ========================================================

    print()
    print(
        "Testing DataLoader..."
    )

    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch = next(
        iter(loader)
    )

    print()
    print(
        "Batch spectral shape:"
    )
    print(
        batch["spectral"].shape
    )

    print()
    print(
        "Batch patch shape:"
    )
    print(
        batch["patch"].shape
    )

    print()
    print(
        "Batch labels shape:"
    )
    print(
        batch["label"].shape
    )

    print()
    print(
        "Batch labels:"
    )
    print(
        batch["label"]
    )

    # ========================================================
    # ASSERTIONS
    # ========================================================

    assert (
        sample["spectral"].shape
        == torch.Size([100])
    )

    assert (
        sample["patch"].shape
        == torch.Size(
            [100, 27, 27]
        )
    )

    assert (
        batch["spectral"].shape
        == torch.Size([8, 100])
    )

    assert (
        batch["patch"].shape
        == torch.Size(
            [8, 100, 27, 27]
        )
    )

    assert (
        batch["label"].shape
        == torch.Size([8])
    )

    assert torch.isfinite(
        batch["spectral"]
    ).all()

    assert torch.isfinite(
        batch["patch"]
    ).all()

    print()
    print(
        "TEST PASSED"
    )


if __name__ == "__main__":
    main()