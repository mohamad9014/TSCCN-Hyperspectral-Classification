from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from .heipor import read_cube, read_mask


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "kidney",
    "liver",
    "spleen",
    "stomach",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# DATASET
# ============================================================

class HeiPorTSCCNDataset(Dataset):
    """
    Dataset for the TSCCN project.

    Each sample returns:

        spectral:
            Tensor (100,)

        patch:
            Tensor (100, 27, 27)

        label:
            integer class index

        metadata:
            dictionary with cube/pixel information

    This version uses a small LRU cache so several
    hyperspectral cubes can stay in RAM.
    """

    def __init__(
        self,
        config_path,
        data_root,
        split="train",
        patch_size=27,
        samples_per_cube=None,
        seed=42,
        cache_size=4,
    ):
        super().__init__()

        self.config_path = Path(config_path)
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.samples_per_cube = samples_per_cube
        self.seed = seed
        self.cache_size = cache_size

        # ----------------------------------------------------
        # Validate patch size
        # ----------------------------------------------------

        if patch_size % 2 == 0:
            raise ValueError(
                "patch_size must be odd. "
                "For this project use 27."
            )

        if cache_size < 1:
            raise ValueError(
                "cache_size must be at least 1."
            )

        self.radius = patch_size // 2

        # ----------------------------------------------------
        # Load YAML
        # ----------------------------------------------------

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        # ----------------------------------------------------
        # Support different YAML layouts
        # ----------------------------------------------------

        if isinstance(config, list):
            all_cubes = config

        elif isinstance(config, dict):

            if "cubes" in config:
                all_cubes = config["cubes"]

            elif "selected_cubes" in config:
                all_cubes = config["selected_cubes"]

            elif "data" in config:
                all_cubes = config["data"]

            else:
                raise ValueError(
                    "Could not find cube list in YAML. "
                    "Expected 'cubes', 'selected_cubes', "
                    "or 'data'."
                )

        else:
            raise ValueError(
                "Unexpected YAML configuration format."
            )

        # ----------------------------------------------------
        # Keep only requested split
        # ----------------------------------------------------

        self.cubes = [
            cube
            for cube in all_cubes
            if cube["split"] == split
        ]

        if len(self.cubes) == 0:
            raise ValueError(
                f"No cubes found for split='{split}'."
            )

        # ----------------------------------------------------
        # LRU cache
        # ----------------------------------------------------

        self._cube_cache = OrderedDict()

        # ----------------------------------------------------
        # Build index
        # ----------------------------------------------------

        self.index = []

        self._build_index()

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    def __len__(self):
        return len(self.index)

    # ========================================================
    # YAML FIELD HELPERS
    # ========================================================

    def _class_name(
        self,
        cube_info,
    ):
        """
        Read class name from YAML.

        Your current cubes_expanded.yaml uses:

            label: liver

        but this also supports other common field names.
        """

        for key in (
            "class_name",
            "class",
            "organ",
            "label",
        ):
            if key in cube_info:
                return str(
                    cube_info[key]
                ).lower()

        raise KeyError(
            "Cube configuration has no class name. "
            "Expected class_name/class/organ/label."
        )

    def _image_name(
        self,
        cube_info,
    ):
        for key in (
            "image_name",
            "image",
            "subject_image",
        ):
            if key in cube_info:
                return str(
                    cube_info[key]
                )

        raise KeyError(
            "Cube configuration has no image_name."
        )

    def _timestamp(
        self,
        cube_info,
    ):
        for key in (
            "timestamp",
            "cube_timestamp",
            "time",
        ):
            if key in cube_info:
                return str(
                    cube_info[key]
                )

        raise KeyError(
            "Cube configuration has no timestamp."
        )

    # ========================================================
    # PATH RESOLUTION
    # ========================================================

    def _paths_for_cube(
        self,
        cube_info,
    ):
        """
        Resolve cube and mask paths.
        """

        class_name = self._class_name(
            cube_info
        )

        image_name = self._image_name(
            cube_info
        )

        timestamp = self._timestamp(
            cube_info
        )

        # ----------------------------------------------------
        # Explicit cube path
        # ----------------------------------------------------

        cube_path = None

        for key in (
            "cube_path",
            "spec_cube",
            "dat_path",
        ):

            if key in cube_info:

                candidate = Path(
                    cube_info[key]
                )

                if not candidate.is_absolute():
                    candidate = (
                        self.data_root
                        / candidate
                    )

                cube_path = candidate

                break

        # ----------------------------------------------------
        # Explicit mask path
        # ----------------------------------------------------

        mask_path = None

        for key in (
            "mask_path",
            "mask",
            "label_path",
        ):

            if key in cube_info:

                candidate = Path(
                    cube_info[key]
                )

                if not candidate.is_absolute():
                    candidate = (
                        self.data_root
                        / candidate
                    )

                mask_path = candidate

                break

        # ----------------------------------------------------
        # Dataset folder
        # ----------------------------------------------------

        image_folder_name = (
            image_name.replace(
                "#",
                "_",
            )
        )

        image_folder = (
            self.data_root
            / image_folder_name
        )

        # ----------------------------------------------------
        # Search cube
        # ----------------------------------------------------

        if cube_path is None:

            patterns = [
                f"*{timestamp}*SpecCube.dat",
                f"*{timestamp}*.dat",
            ]

            candidates = []

            if image_folder.exists():

                for pattern in patterns:
                    candidates.extend(
                        image_folder.rglob(
                            pattern
                        )
                    )

            if len(candidates) == 0:

                for pattern in patterns:
                    candidates.extend(
                        self.data_root.rglob(
                            pattern
                        )
                    )

            candidates = list(
                dict.fromkeys(
                    candidates
                )
            )

            spec_candidates = [
                path
                for path in candidates
                if "speccube" in path.name.lower()
            ]

            if len(spec_candidates) > 0:
                cube_path = spec_candidates[0]

            elif len(candidates) > 0:
                cube_path = candidates[0]

            else:
                raise FileNotFoundError(
                    "Could not find cube for:\n"
                    f"image={image_name}\n"
                    f"timestamp={timestamp}"
                )

        # ----------------------------------------------------
        # Search mask
        # ----------------------------------------------------

        if mask_path is None:

            extensions = (
                "*.png",
                "*.tif",
                "*.tiff",
                "*.bmp",
            )

            candidates = []

            search_folder = (
                image_folder
                if image_folder.exists()
                else self.data_root
            )

            # First try:
            # class + timestamp

            for extension in extensions:

                for path in search_folder.rglob(
                    extension
                ):

                    name = path.name.lower()

                    if (
                        class_name in name
                        and
                        timestamp.lower() in name
                    ):
                        candidates.append(
                            path
                        )

            # Fallback:
            # class + mask/label/seg

            if len(candidates) == 0:

                for extension in extensions:

                    for path in search_folder.rglob(
                        extension
                    ):

                        name = path.name.lower()

                        if (
                            class_name in name
                            and
                            (
                                "mask" in name
                                or
                                "label" in name
                                or
                                "seg" in name
                            )
                        ):
                            candidates.append(
                                path
                            )

            if len(candidates) == 0:
                raise FileNotFoundError(
                    "Could not find mask for:\n"
                    f"image={image_name}\n"
                    f"timestamp={timestamp}\n"
                    f"class={class_name}"
                )

            mask_path = candidates[0]

        # ----------------------------------------------------
        # Validate paths
        # ----------------------------------------------------

        if not cube_path.exists():
            raise FileNotFoundError(
                f"Cube does not exist:\n"
                f"{cube_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask does not exist:\n"
                f"{mask_path}"
            )

        return (
            cube_path,
            mask_path,
        )

    # ========================================================
    # BUILD PIXEL INDEX
    # ========================================================

    def _build_index(
        self,
    ):
        """
        Build list of selected labeled pixels.

        Each item:

            (
                cube_id,
                row,
                column,
                label_index,
            )
        """

        rng = np.random.default_rng(
            self.seed
        )

        for cube_id, cube_info in enumerate(
            self.cubes
        ):

            class_name = self._class_name(
                cube_info
            )

            if class_name not in CLASS_TO_INDEX:
                raise ValueError(
                    f"Unknown class: {class_name}"
                )

            label_index = (
                CLASS_TO_INDEX[
                    class_name
                ]
            )

            (
                cube_path,
                mask_path,
            ) = self._paths_for_cube(
                cube_info
            )

            # Save resolved paths
            # so we do not search again later.

            cube_info[
                "_resolved_cube_path"
            ] = str(
                cube_path
            )

            cube_info[
                "_resolved_mask_path"
            ] = str(
                mask_path
            )

            mask = read_mask(
                mask_path
            )

            coordinates = np.argwhere(
                mask
            )

            if len(coordinates) == 0:
                raise ValueError(
                    "Mask contains no labeled pixels:\n"
                    f"{mask_path}"
                )

            # ------------------------------------------------
            # Optional fixed sampling
            # ------------------------------------------------

            if (
                self.samples_per_cube
                is not None
            ):

                number_to_sample = min(
                    self.samples_per_cube,
                    len(coordinates),
                )

                selected = rng.choice(
                    len(coordinates),
                    size=number_to_sample,
                    replace=False,
                )

                coordinates = (
                    coordinates[selected]
                )

            # ------------------------------------------------
            # Add coordinates
            # ------------------------------------------------

            for row, column in coordinates:

                self.index.append(
                    (
                        cube_id,
                        int(row),
                        int(column),
                        label_index,
                    )
                )

    # ========================================================
    # LRU CACHE
    # ========================================================

    def _get_cube(
        self,
        cube_id,
    ):
        """
        Get normalized cube using an LRU cache.
        """

        # ----------------------------------------------------
        # Cache hit
        # ----------------------------------------------------

        if cube_id in self._cube_cache:

            cube = self._cube_cache.pop(
                cube_id
            )

            self._cube_cache[
                cube_id
            ] = cube

            return cube

        # ----------------------------------------------------
        # Cache miss
        # ----------------------------------------------------

        cube_info = (
            self.cubes[
                cube_id
            ]
        )

        cube_path = Path(
            cube_info[
                "_resolved_cube_path"
            ]
        )

        cube = read_cube(
            cube_path,
            normalize=True,
        )

        self._cube_cache[
            cube_id
        ] = cube

        # ----------------------------------------------------
        # Remove least recently used cube
        # ----------------------------------------------------

        while (
            len(self._cube_cache)
            > self.cache_size
        ):

            self._cube_cache.popitem(
                last=False
            )

        return cube

    # ========================================================
    # PATCH EXTRACTION
    # ========================================================

    def _extract_patch(
        self,
        cube,
        row,
        column,
    ):
        """
        Extract local patch.

        Output:

            (patch_size, patch_size, 100)

        Reflect padding is applied only when the
        center pixel is close to an image border.
        """

        height = cube.shape[0]
        width = cube.shape[1]

        r0 = (
            row
            - self.radius
        )

        r1 = (
            row
            + self.radius
            + 1
        )

        c0 = (
            column
            - self.radius
        )

        c1 = (
            column
            + self.radius
            + 1
        )

        # ----------------------------------------------------
        # Crop valid image region
        # ----------------------------------------------------

        crop_r0 = max(
            r0,
            0,
        )

        crop_r1 = min(
            r1,
            height,
        )

        crop_c0 = max(
            c0,
            0,
        )

        crop_c1 = min(
            c1,
            width,
        )

        patch = cube[
            crop_r0:crop_r1,
            crop_c0:crop_c1,
            :
        ]

        # ----------------------------------------------------
        # Required padding
        # ----------------------------------------------------

        pad_top = max(
            0,
            -r0,
        )

        pad_bottom = max(
            0,
            r1 - height,
        )

        pad_left = max(
            0,
            -c0,
        )

        pad_right = max(
            0,
            c1 - width,
        )

        # ----------------------------------------------------
        # Reflect padding
        # ----------------------------------------------------

        if (
            pad_top > 0
            or
            pad_bottom > 0
            or
            pad_left > 0
            or
            pad_right > 0
        ):

            patch = np.pad(
                patch,
                pad_width=(
                    (
                        pad_top,
                        pad_bottom,
                    ),
                    (
                        pad_left,
                        pad_right,
                    ),
                    (
                        0,
                        0,
                    ),
                ),
                mode="reflect",
            )

        if patch.shape[:2] != (
            self.patch_size,
            self.patch_size,
        ):
            raise RuntimeError(
                "Patch extraction produced "
                "incorrect shape: "
                f"{patch.shape}"
            )

        return patch

    # ========================================================
    # GET SAMPLE
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        (
            cube_id,
            row,
            column,
            label_index,
        ) = self.index[
            index
        ]

        cube_info = (
            self.cubes[
                cube_id
            ]
        )

        # ----------------------------------------------------
        # Load/reuse cube
        # ----------------------------------------------------

        cube = self._get_cube(
            cube_id
        )

        # ----------------------------------------------------
        # Spectral input
        #
        # Shape:
        # (100,)
        # ----------------------------------------------------

        spectral = cube[
            row,
            column,
            :
        ]

        # ----------------------------------------------------
        # Spatial-spectral patch
        #
        # Cube patch:
        # (27, 27, 100)
        #
        # PyTorch:
        # (100, 27, 27)
        # ----------------------------------------------------

        patch = self._extract_patch(
            cube,
            row,
            column,
        )

        patch = np.transpose(
            patch,
            (
                2,
                0,
                1,
            ),
        )

        # ----------------------------------------------------
        # Convert to contiguous float32
        # ----------------------------------------------------

        spectral = np.ascontiguousarray(
            spectral,
            dtype=np.float32,
        )

        patch = np.ascontiguousarray(
            patch,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # PyTorch tensors
        # ----------------------------------------------------

        spectral = torch.from_numpy(
            spectral
        )

        patch = torch.from_numpy(
            patch
        )

        label = torch.tensor(
            label_index,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "cube_id": cube_id,
            "row": row,
            "col": column,
            "class_name": CLASS_NAMES[
                label_index
            ],
            "image_name": self._image_name(
                cube_info
            ),
            "timestamp": self._timestamp(
                cube_info
            ),
        }

        return {
            "spectral": spectral,
            "patch": patch,
            "label": label,
            "metadata": metadata,
        }

    # ========================================================
    # CACHE UTILITIES
    # ========================================================

    def clear_cache(
        self,
    ):
        """
        Remove all cached cubes from RAM.
        """

        self._cube_cache.clear()

    def cached_cube_ids(
        self,
    ):
        """
        Return currently cached cube IDs.
        """

        return list(
            self._cube_cache.keys()
        )