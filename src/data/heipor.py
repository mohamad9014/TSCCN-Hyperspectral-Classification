from pathlib import Path

import numpy as np
from PIL import Image


# HeiPorSPECTRAL / TIVITA dimensions
HEIGHT = 480
WIDTH = 640
BANDS = 100

# .dat structure:
# 12-byte header = 3 big-endian int32 values
# followed by big-endian float32 spectral values
HEADER_BYTES = 12
RAW_DTYPE = np.dtype(">f4")


def read_header(cube_path):
    """
    Read the 12-byte TIVITA SpecCube header.

    Expected values:
        (640, 480, 100)
        = width, height, bands
    """
    cube_path = Path(cube_path)

    with open(cube_path, "rb") as f:
        header = np.fromfile(f, dtype=">i4", count=3)

    if len(header) != 3:
        raise ValueError(
            f"Invalid SpecCube header in {cube_path}"
        )

    return tuple(int(x) for x in header)


def read_cube(cube_path, normalize=True):
    """
    Read a HeiPorSPECTRAL SpecCube.dat file.

    Returns
    -------
    cube : np.ndarray
        Shape: (480, 640, 100)
        dtype: float32
    """
    cube_path = Path(cube_path)

    if not cube_path.exists():
        raise FileNotFoundError(
            f"Cube not found: {cube_path}"
        )

    header = read_header(cube_path)

    expected_header = (WIDTH, HEIGHT, BANDS)

    if header != expected_header:
        raise ValueError(
            f"Unexpected cube header {header}. "
            f"Expected {expected_header}"
        )

    # Read big-endian float32 data after the 12-byte header
    data = np.memmap(
        cube_path,
        dtype=RAW_DTYPE,
        mode="r",
        offset=HEADER_BYTES,
        shape=(HEIGHT * WIDTH * BANDS,),
    )

    # Convert to normal/native float32
    cube = np.asarray(data, dtype=np.float32)

    # BIP format:
    # each pixel contains its 100 spectral bands consecutively
    cube = cube.reshape(
        HEIGHT,
        WIDTH,
        BANDS,
    )

    if normalize:
        cube = l1_normalize(cube)

    return cube


def l1_normalize(cube):
    """
    L1-normalize every pixel spectrum independently.

    For each pixel:
        spectrum = spectrum / sum(abs(spectrum))

    NaN and Inf values are replaced with zero.
    """
    cube = np.asarray(cube, dtype=np.float32)

    denominator = np.linalg.norm(
        cube,
        ord=1,
        axis=-1,
        keepdims=True,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        cube = cube / denominator

    cube = np.nan_to_num(
        cube,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return cube.astype(
        np.float32,
        copy=False,
    )


def read_mask(mask_path):
    """
    Read a binary organ annotation mask.

    Returns
    -------
    mask : np.ndarray
        Shape: (480, 640)
        dtype: bool
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    if mask.shape != (HEIGHT, WIDTH):
        raise ValueError(
            f"Unexpected mask shape {mask.shape}. "
            f"Expected {(HEIGHT, WIDTH)}"
        )

    return mask > 0


def get_labeled_coordinates(mask):
    """
    Return coordinates of all labeled pixels.

    Returns
    -------
    coords : np.ndarray
        Shape: (N, 2)

        Each row is:
            [row, column]
    """
    mask = np.asarray(mask, dtype=bool)

    rows, cols = np.nonzero(mask)

    coords = np.column_stack(
        (rows, cols)
    )

    return coords.astype(np.int32)


def get_spectrum(cube, row, col):
    """
    Get the 100-band spectrum for one pixel.

    Returns shape:
        (100,)
    """
    return np.asarray(
        cube[row, col, :],
        dtype=np.float32,
    )


if __name__ == "__main__":
    print("HeiPorSPECTRAL loader module")
    print(
        f"Expected cube shape: "
        f"{HEIGHT} x {WIDTH} x {BANDS}"
    )