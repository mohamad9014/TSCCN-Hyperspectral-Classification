"""Download the SpecCube, RGB reference image, and annotation mask for each
cube listed in configs/cubes.yaml, using HTTP range requests so we never
pull the full 845 GiB HeiPorSPECTRAL archive.
"""
import argparse
from pathlib import Path

import remotezip
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def files_for_cube(archive_root: str, subject: str, timestamp: str, annotator: str, label: str):
    base = f"{archive_root}/data/subjects/{subject}/{timestamp}"
    return {
        "cube": f"{base}/{timestamp}_SpecCube.dat",
        "rgb": f"{base}/{timestamp}_RGB-Image.png",
        "mask": f"{base}/annotations/{timestamp}#polygon#{annotator}#{label}#binary.png",
        "meta": f"{base}/annotations/{timestamp}_meta.json",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "cubes.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "raw" / "heiporspectral"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Opening remote archive index: {cfg['archive_url']}")
    zf = remotezip.RemoteZip(cfg["archive_url"])

    for cube in cfg["cubes"]:
        dest = out_root / cube["image_name"].replace("#", "_")
        dest.mkdir(parents=True, exist_ok=True)
        entries = files_for_cube(cfg["archive_root"], cube["subject"], cube["timestamp"], cfg["annotator"], cube["label"])

        for key, zip_path in entries.items():
            out_path = dest / Path(zip_path).name
            if out_path.exists():
                print(f"  [skip] {out_path.name} already exists")
                continue
            print(f"  [get]  {cube['image_name']} :: {key} -> {out_path.name}")
            with zf.open(zip_path) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

    print("Done.")


if __name__ == "__main__":
    main()
