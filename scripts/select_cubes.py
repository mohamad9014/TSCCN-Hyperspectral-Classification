"""
Select a balanced, leakage-safe subset of HeiPorSPECTRAL recordings.

Strategy
--------
1. Read the small precomputed annotator-1 median-spectra table directly
   from the remote HeiPorSPECTRAL ZIP using HTTP range requests.
2. Keep the requested organ classes.
3. Prefer subject-disjoint train/val/test splits (pig-level split).
4. Sample a fixed number of cubes per organ.
5. Write configs/cubes_expanded.yaml, which is compatible with the
   existing download_data.py (the extra `split` field is simply ignored).

Example
-------
python scripts/select_cubes.py --per-class 8 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from collections import Counter, defaultdict
from pathlib import Path

import remotezip
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_URL = (
    "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/"
    "data/HeiPorSPECTRAL.zip"
)
ARCHIVE_ROOT = "HeiPorSPECTRAL"

# This table has one row per annotated label/image and includes image metadata.
TABLE_CANDIDATES = [
    f"{ARCHIVE_ROOT}/intermediates/tables/"
    "HeiPorSPECTRAL@median_spectra@polygon#annotator1.csv",
    # Fallback in case the ZIP contains a slightly different top-level layout.
    "intermediates/tables/"
    "HeiPorSPECTRAL@median_spectra@polygon#annotator1.csv",
]

DEFAULT_ORGANS = ["liver", "spleen", "kidney", "stomach"]


def norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def first_present(fieldnames, candidates):
    normalized = {norm(f): f for f in fieldnames}
    for candidate in candidates:
        if norm(candidate) in normalized:
            return normalized[norm(candidate)]
    return None


def read_remote_table(zf: remotezip.RemoteZip):
    names = set(zf.namelist())
    table_path = next((p for p in TABLE_CANDIDATES if p in names), None)

    if table_path is None:
        # Be forgiving about the archive root.
        matches = [
            n for n in names
            if n.endswith(
                "intermediates/tables/"
                "HeiPorSPECTRAL@median_spectra@polygon#annotator1.csv"
            )
        ]
        if not matches:
            raise FileNotFoundError(
                "Could not find the annotator-1 median-spectra table in the archive."
            )
        table_path = matches[0]

    print(f"Reading remote metadata table:\n  {table_path}")
    with zf.open(table_path) as src:
        text = io.TextIOWrapper(src, encoding="utf-8-sig")
        reader = csv.DictReader(text)
        rows = list(reader)

    if not rows:
        raise RuntimeError("Metadata table was empty.")

    return rows


def infer_columns(rows):
    fields = rows[0].keys()

    image_col = first_present(
        fields,
        [
            "image_name",
            "image",
            "image_id",
            "image_name_annotations",
            "timestamp",
        ],
    )
    label_col = first_present(
        fields,
        [
            "label_name",
            "label",
            "organ",
            "label_name_mapped",
        ],
    )
    subject_col = first_present(
        fields,
        [
            "subject_name",
            "subject",
            "subject_id",
            "pig",
            "pig_id",
        ],
    )

    if image_col is None:
        # Many HTC tables use a path/image identifier rather than a literal
        # "image_name" column. Find a column whose values look like P0xx#...
        for field in fields:
            vals = [str(r.get(field, "")) for r in rows[:100]]
            if any(v.startswith("P0") and "#" in v for v in vals):
                image_col = field
                break

    if label_col is None:
        raise RuntimeError(
            "Could not infer the organ/label column.\n"
            f"Available columns:\n{list(fields)}"
        )

    return image_col, label_col, subject_col


def parse_image_and_subject(row, image_col, subject_col):
    raw_image = str(row.get(image_col, "")).strip() if image_col else ""

    # HTC image names normally look like P087#2021_04_16_11_39_47.
    if "#" in raw_image:
        image_name = raw_image.split("/")[-1]
        subject, timestamp = image_name.split("#", 1)
        return image_name, subject, timestamp

    # If only timestamp is present, use the subject column.
    timestamp = raw_image.split("/")[-1]
    subject = str(row.get(subject_col, "")).strip() if subject_col else ""

    if subject and timestamp:
        return f"{subject}#{timestamp}", subject, timestamp

    raise ValueError(f"Could not parse image identity from row: {row}")


def build_catalog(rows, organs):
    image_col, label_col, subject_col = infer_columns(rows)
    print(
        "Detected columns:",
        f"image={image_col!r}, label={label_col!r}, subject={subject_col!r}",
    )

    wanted = {norm(o): o for o in organs}
    catalog = defaultdict(list)
    seen = set()

    for row in rows:
        label_raw = norm(row.get(label_col, ""))
        if label_raw not in wanted:
            continue

        organ = wanted[label_raw]

        try:
            image_name, subject, timestamp = parse_image_and_subject(
                row, image_col, subject_col
            )
        except ValueError:
            continue

        key = (organ, image_name)
        if key in seen:
            continue
        seen.add(key)

        catalog[organ].append(
            {
                "image_name": image_name,
                "subject": subject,
                "timestamp": timestamp,
                "label": organ,
            }
        )

    return catalog


def choose_split_subjects(catalog, organs, rng):
    """
    Choose one validation pig and one test pig that maximize class coverage.
    Remaining subjects become training subjects.

    We score pairs by the number of requested organs available in both pigs.
    """
    coverage = defaultdict(set)
    for organ in organs:
        for item in catalog[organ]:
            coverage[item["subject"]].add(organ)

    subjects = sorted(coverage)
    if len(subjects) < 3:
        raise RuntimeError("Need at least 3 subjects for train/val/test pig-level splits.")

    candidates = []
    for val_subject in subjects:
        for test_subject in subjects:
            if val_subject == test_subject:
                continue
            shared = len(
                coverage[val_subject].intersection(coverage[test_subject], organs)
            )
            val_cov = len(coverage[val_subject].intersection(organs))
            test_cov = len(coverage[test_subject].intersection(organs))
            score = (shared, val_cov + test_cov)
            candidates.append((score, val_subject, test_subject))

    best_score = max(x[0] for x in candidates)
    best = [x for x in candidates if x[0] == best_score]
    _, val_subject, test_subject = rng.choice(best)

    train_subjects = [s for s in subjects if s not in {val_subject, test_subject}]
    return train_subjects, val_subject, test_subject, coverage


def sample_balanced(catalog, organs, per_class, rng):
    train_subjects, val_subject, test_subject, coverage = choose_split_subjects(
        catalog, organs, rng
    )

    print("\nChosen subject split")
    print(f"  validation pig: {val_subject}")
    print(f"  test pig:       {test_subject}")
    print(f"  training pigs:  {', '.join(train_subjects)}")

    selected = []

    # Reserve one cube/class for validation and one cube/class for test.
    # All remaining requested cubes come from training subjects.
    for organ in organs:
        items = catalog[organ]
        by_subject = defaultdict(list)
        for x in items:
            by_subject[x["subject"]].append(x)

        for values in by_subject.values():
            rng.shuffle(values)

        val_pool = by_subject.get(val_subject, [])
        test_pool = by_subject.get(test_subject, [])

        if not val_pool or not test_pool:
            raise RuntimeError(
                f"Cannot make a fully subject-disjoint split for {organ!r}: "
                f"val pig {val_subject} has {len(val_pool)}, "
                f"test pig {test_subject} has {len(test_pool)} recording(s).\n"
                "Try another seed, fewer classes, or inspect class availability."
            )

        val_item = dict(val_pool[0], split="val")
        test_item = dict(test_pool[0], split="test")

        n_train = per_class - 2
        if n_train < 1:
            raise ValueError("--per-class must be at least 3.")

        # Diverse training: round-robin across training pigs before reusing pigs.
        pools = {
            s: list(by_subject.get(s, []))
            for s in train_subjects
            if by_subject.get(s)
        }
        train_items = []
        train_order = list(pools)
        rng.shuffle(train_order)

        while len(train_items) < n_train and train_order:
            made_progress = False
            for subject in list(train_order):
                if pools[subject]:
                    train_items.append(dict(pools[subject].pop(), split="train"))
                    made_progress = True
                    if len(train_items) == n_train:
                        break
            if not made_progress:
                break

        if len(train_items) < n_train:
            raise RuntimeError(
                f"Not enough training recordings for {organ!r}: "
                f"needed {n_train}, found {len(train_items)}."
            )

        selected.extend(train_items + [val_item, test_item])

    return selected


def print_summary(selected, organs):
    print("\nSelection summary")
    for organ in organs:
        rows = [x for x in selected if x["label"] == organ]
        counts = Counter(x["split"] for x in rows)
        subjects = sorted({x["subject"] for x in rows})
        print(
            f"  {organ:8s}: {len(rows):2d} cubes | "
            f"train={counts['train']} val={counts['val']} test={counts['test']} | "
            f"pigs={', '.join(subjects)}"
        )

    print("\nGlobal split subjects")
    for split in ("train", "val", "test"):
        subjects = sorted({x["subject"] for x in selected if x["split"] == split})
        print(f"  {split:5s}: {', '.join(subjects)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--organs",
        nargs="+",
        default=DEFAULT_ORGANS,
        help="Organ classes to include.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=8,
        help="Total cubes per organ, including validation and test (default: 8).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "configs" / "cubes_expanded.yaml"),
    )
    args = parser.parse_args()

    organs = [norm(x) for x in args.organs]
    rng = random.Random(args.seed)

    print("Opening remote HeiPorSPECTRAL archive index...")
    zf = remotezip.RemoteZip(ARCHIVE_URL)

    rows = read_remote_table(zf)
    catalog = build_catalog(rows, organs)

    print("\nAvailable recordings")
    for organ in organs:
        items = catalog.get(organ, [])
        subjects = sorted({x["subject"] for x in items})
        print(
            f"  {organ:8s}: {len(items):4d} recordings across "
            f"{len(subjects):2d} pigs: {', '.join(subjects)}"
        )

    missing = [o for o in organs if not catalog.get(o)]
    if missing:
        raise RuntimeError(f"No recordings found for: {missing}")

    selected = sample_balanced(
        catalog=catalog,
        organs=organs,
        per_class=args.per_class,
        rng=rng,
    )
    print_summary(selected, organs)

    cfg = {
        "archive_url": ARCHIVE_URL,
        "archive_root": ARCHIVE_ROOT,
        "annotator": "annotator1",
        "selection": {
            "strategy": "subject_disjoint",
            "seed": args.seed,
            "organs": organs,
            "cubes_per_class": args.per_class,
        },
        "cubes": selected,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"\nWrote: {out}")
    print(
        "\nNext step:\n"
        f"  python scripts/download_data.py --config {out}\n"
        "\nImportant: keep train/val/test separated by the `split` field "
        "when building the PyTorch Dataset."
    )


if __name__ == "__main__":
    main()
