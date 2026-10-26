"""Verify the downloaded HeiPorSPECTRAL subset.

Raw TIVITA SpecCube.dat format:
- 12-byte dimension header
- hyperspectral payload stored as BIG-ENDIAN float32
- band-interleaved-by-pixel (BIP), giving logical shape (480, 640, 100)

Usage:
    python scripts/verify_dataset.py
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

H, W, B = 480, 640, 100
HEADER_BYTES = 12
N_VALUES = H * W * B

# IMPORTANT: TIVITA SpecCube samples are big-endian float32.
RAW_DTYPE = np.dtype(">f4")
PAYLOAD_BYTES = N_VALUES * RAW_DTYPE.itemsize
EXPECTED_FILE_BYTES = HEADER_BYTES + PAYLOAD_BYTES

VALID_SPLITS = {"train", "val", "test"}
VALID_RGB_SIZES = {(646, 513), (640, 480)}
MASK_SIZE = (640, 480)


def local_paths(data_root: Path, cube: dict, annotator: str) -> dict:
    timestamp = cube["timestamp"]
    folder = data_root / cube["image_name"].replace("#", "_")

    return {
        "cube": folder / f"{timestamp}_SpecCube.dat",
        "rgb": folder / f"{timestamp}_RGB-Image.png",
        "mask": folder / f"{timestamp}#polygon#{annotator}#{cube['label']}#binary.png",
        "meta": folder / f"{timestamp}_meta.json",
    }


def inspect_header(path: Path):
    """Read the first 12 bytes and report both endian interpretations."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)

    if len(raw) != HEADER_BYTES:
        return None, None

    be = np.frombuffer(raw, dtype=">i4").tolist()
    le = np.frombuffer(raw, dtype="<i4").tolist()
    return be, le


def verify_cube(path: Path):
    size = path.stat().st_size

    if size != EXPECTED_FILE_BYTES:
        return False, (
            f"unexpected size {size:,} bytes; "
            f"expected {EXPECTED_FILE_BYTES:,}"
        ), None

    try:
        header_be, header_le = inspect_header(path)

        # Read the hyperspectral payload correctly as big-endian float32.
        flat = np.memmap(
            path,
            dtype=RAW_DTYPE,
            mode="r",
            offset=HEADER_BYTES,
            shape=(N_VALUES,),
        )

        # BIP storage: 100 consecutive bands belong to one spatial pixel.
        cube = flat.reshape(H, W, B)

        # Sample several pixels across the image.
        coords = [
            (0, 0),
            (H // 4, W // 4),
            (H // 2, W // 2),
            (3 * H // 4, 3 * W // 4),
            (H - 1, W - 1),
        ]
        sample = np.stack(
            [np.asarray(cube[r, c, :], dtype=np.float32) for r, c in coords]
        )

        finite_count = int(np.isfinite(sample).sum())
        total_count = int(sample.size)

        if finite_count == 0:
            return False, (
                "sample contains no finite reflectance values "
                "(possible format/endian problem)"
            ), (header_be, header_le)

        # Raw cubes can contain invalid/background values. We report them,
        # but do not reject an otherwise correctly decoded cube solely for that.
        nonfinite = total_count - finite_count
        finite_values = sample[np.isfinite(sample)]

        # Reflectance data should not decode to absurd float magnitudes.
        max_abs = float(np.max(np.abs(finite_values)))
        if max_abs > 1e6:
            return False, (
                f"decoded values have implausible magnitude (max abs={max_abs:.3g}); "
                "possible format/endian problem"
            ), (header_be, header_le)

        msg = (
            f"shape=({H},{W},{B}), dtype=>f4, BIP, "
            f"finite={finite_count}/{total_count}, "
            f"finite range=[{finite_values.min():.6g}, {finite_values.max():.6g}], "
            f"header(be)={header_be}"
        )

        if nonfinite:
            msg += f", nonfinite_sample_values={nonfinite}"

        return True, msg, (header_be, header_le)

    except Exception as exc:
        return False, f"cannot decode cube: {exc}", None


def verify_rgb(path: Path):
    try:
        with Image.open(path) as image:
            image.load()
            size = image.size
            mode = image.mode

        if size not in VALID_RGB_SIZES:
            return False, (
                f"unexpected RGB size {size}; "
                f"accepted: {sorted(VALID_RGB_SIZES)}"
            )

        return True, f"size={size}, mode={mode}"

    except Exception as exc:
        return False, f"cannot read RGB: {exc}"


def verify_mask(path: Path):
    try:
        with Image.open(path) as image:
            image.load()
            size = image.size
            arr = np.asarray(image)

        if size != MASK_SIZE:
            return False, (
                f"wrong mask size {size}; expected {MASK_SIZE}"
            ), arr

        return True, f"size={size}", arr

    except Exception as exc:
        return False, f"cannot read mask: {exc}", None


def verify_meta(path: Path, label: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        label_meta = meta.get("label_meta", {})
        if label not in label_meta:
            return False, f"label '{label}' missing from label_meta"

        info = label_meta[label]
        return True, (
            f"situs={info.get('situs')}, "
            f"angle={info.get('angle')}, "
            f"repetition={info.get('repetition')}"
        )

    except Exception as exc:
        return False, f"cannot read metadata: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "cubes_expanded.yaml"),
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "raw" / "heiporspectral"),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    data_root = Path(args.data_root)

    if not config_path.exists():
        raise SystemExit(f"ERROR: config not found: {config_path}")
    if not data_root.exists():
        raise SystemExit(f"ERROR: data root not found: {data_root}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cubes = cfg.get("cubes", [])
    annotator = cfg.get("annotator", "annotator1")

    if not cubes:
        raise SystemExit("ERROR: no cubes found in config")

    errors = []
    warnings = []
    split_counts = Counter()
    class_split = defaultdict(Counter)
    split_subjects = defaultdict(set)
    pixel_counts = defaultdict(list)
    rgb_size_counts = Counter()
    header_counts = Counter()

    print("=" * 76)
    print("HeiPorSPECTRAL DATASET VERIFICATION")
    print("=" * 76)
    print(f"Config: {config_path}")
    print(f"Data:   {data_root}")
    print(f"Cubes:  {len(cubes)}")
    print("Raw cube decoding: 12-byte header + big-endian float32 BIP")
    print()

    for i, cube_cfg in enumerate(cubes, 1):
        name = cube_cfg.get("image_name", "<missing>")
        label = cube_cfg.get("label", "<missing>")
        subject = cube_cfg.get("subject", "<missing>")
        split = cube_cfg.get("split")

        print(f"[{i:02d}/{len(cubes):02d}] {name} | {label} | {split}")

        if split not in VALID_SPLITS:
            errors.append(f"{name}: invalid/missing split {split!r}")
        else:
            split_counts[split] += 1
            class_split[label][split] += 1
            split_subjects[split].add(subject)

        try:
            paths = local_paths(data_root, cube_cfg, annotator)
        except KeyError as exc:
            errors.append(f"{name}: missing config key {exc}")
            print(f"   FAIL config: {exc}")
            continue

        missing = [key for key, path in paths.items() if not path.exists()]
        if missing:
            print("   FAIL missing: " + ", ".join(missing))
            for key in missing:
                errors.append(f"{name}: missing {key}: {paths[key]}")
            continue

        ok, msg, header = verify_cube(paths["cube"])
        print(f"   {'OK  ' if ok else 'FAIL'} cube: {msg}")
        if not ok:
            errors.append(f"{name}: cube: {msg}")
        elif header:
            header_counts[tuple(header[0])] += 1

        try:
            with Image.open(paths["rgb"]) as img:
                rgb_size_counts[img.size] += 1
        except Exception:
            pass

        ok, msg = verify_rgb(paths["rgb"])
        print(f"   {'OK  ' if ok else 'FAIL'} rgb : {msg}")
        if not ok:
            errors.append(f"{name}: RGB: {msg}")

        ok, msg, mask = verify_mask(paths["mask"])
        if ok:
            labeled = int(np.count_nonzero(mask))
            unique = np.unique(mask)
            pixel_counts[label].append(labeled)
            print(
                f"   OK   mask: {msg}, labeled_pixels={labeled:,}, "
                f"values={unique.tolist()[:10]}"
                + ("..." if len(unique) > 10 else "")
            )
            if labeled == 0:
                errors.append(f"{name}: mask has zero labeled pixels")
            if len(unique) > 2:
                warnings.append(
                    f"{name}: mask has >2 unique values: {unique.tolist()}"
                )
        else:
            print(f"   FAIL mask: {msg}")
            errors.append(f"{name}: mask: {msg}")

        ok, msg = verify_meta(paths["meta"], label)
        print(f"   {'OK  ' if ok else 'FAIL'} meta: {msg}")
        if not ok:
            errors.append(f"{name}: metadata: {msg}")

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_subjects[a] & split_subjects[b]
        if overlap:
            errors.append(
                f"SUBJECT LEAKAGE {a}/{b}: {', '.join(sorted(overlap))}"
            )

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(
        f"Split counts: train={split_counts['train']}, "
        f"val={split_counts['val']}, test={split_counts['test']}"
    )

    print("\nSubjects:")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {', '.join(sorted(split_subjects[split])) or '-'}")

    no_overlap = (
        not (split_subjects["train"] & split_subjects["val"])
        and not (split_subjects["train"] & split_subjects["test"])
        and not (split_subjects["val"] & split_subjects["test"])
    )
    print(f"  subject leakage: {'NONE' if no_overlap else 'FOUND'}")

    print("\nClass counts:")
    for label in sorted(class_split):
        c = class_split[label]
        print(
            f"  {label:10s} train={c['train']} val={c['val']} "
            f"test={c['test']} total={sum(c.values())}"
        )

    print("\nLabeled pixels:")
    for label in sorted(pixel_counts):
        vals = pixel_counts[label]
        if vals:
            print(
                f"  {label:10s} cubes={len(vals)}, "
                f"min={min(vals):,}, max={max(vals):,}, "
                f"mean={sum(vals)/len(vals):,.0f}"
            )

    print("\nSpecCube header values (big-endian int32 interpretation):")
    for header, count in header_counts.items():
        print(f"  {header}: {count} cube(s)")

    print("\nRGB sizes:")
    for size, count in sorted(rgb_size_counts.items()):
        print(f"  {size}: {count} image(s)")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print("  -", w)

    if errors:
        print("\nRESULT: FAILED")
        print(f"{len(errors)} problem(s) found:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)

    print("\nRESULT: PASSED")
    print("All configured recordings are present and readable.")
    print("SpecCube files decode as big-endian float32 BIP data.")
    print("All masks align with the 480 x 640 HSI grid.")
    print("Train/validation/test subjects are disjoint.")
    print("Dataset is ready for preprocessing and PyTorch Dataset construction.")


if __name__ == "__main__":
    main()
