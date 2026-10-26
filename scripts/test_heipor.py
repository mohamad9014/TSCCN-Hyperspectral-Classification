import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.data.heipor import (
    read_cube,
    read_mask,
    get_labeled_coordinates,
    get_spectrum,
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
    with open(
        CONFIG,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    # Just test the first cube
    item = config["cubes"][0]

    timestamp = item["timestamp"]
    image_name = item["image_name"]
    label = item["label"]

    folder = DATA_ROOT / image_name.replace(
        "#",
        "_",
    )

    cube_path = (
        folder
        / f"{timestamp}_SpecCube.dat"
    )

    mask_path = (
        folder
        / (
            f"{timestamp}"
            f"#polygon#annotator1"
            f"#{label}#binary.png"
        )
    )

    print("Loading:")
    print(" Cube:", cube_path)
    print(" Mask:", mask_path)
    print()

    cube = read_cube(
        cube_path,
        normalize=True,
    )

    mask = read_mask(mask_path)

    coords = get_labeled_coordinates(mask)

    row, col = coords[0]

    spectrum = get_spectrum(
        cube,
        row,
        col,
    )

    print("Cube shape:")
    print(cube.shape)

    print("\nCube dtype:")
    print(cube.dtype)

    print("\nMask shape:")
    print(mask.shape)

    print("\nNumber of labeled pixels:")
    print(len(coords))

    print("\nFirst labeled coordinate:")
    print((row, col))

    print("\nSpectrum shape:")
    print(spectrum.shape)

    print("\nSpectrum:")
    print(spectrum)

    print("\nSpectrum L1 sum:")
    print(abs(spectrum).sum())

    print("\nTEST PASSED")


if __name__ == "__main__":
    main()