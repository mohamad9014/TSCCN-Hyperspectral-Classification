import random

from torch.utils.data import Sampler


class CubeBatchSampler(Sampler):
    """
    Batch sampler that keeps samples from the same cube together.

    Why:
        Hyperspectral cubes are large.

        If normal global shuffle is used, training may jump between
        many cubes and constantly reload them.

        This sampler:
            1. groups indices by cube
            2. shuffles cube order
            3. shuffles samples inside each cube
            4. creates batches from one cube at a time
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=False,
        seed=42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.epoch = 0

        self.indices_by_cube = {}

        for dataset_index, item in enumerate(
            dataset.index
        ):
            cube_id = item[0]

            if cube_id not in self.indices_by_cube:
                self.indices_by_cube[cube_id] = []

            self.indices_by_cube[cube_id].append(
                dataset_index
            )

    def set_epoch(
        self,
        epoch,
    ):
        self.epoch = epoch

    def __iter__(
        self,
    ):
        rng = random.Random(
            self.seed + self.epoch
        )

        cube_ids = list(
            self.indices_by_cube.keys()
        )

        if self.shuffle:
            rng.shuffle(
                cube_ids
            )

        for cube_id in cube_ids:

            indices = list(
                self.indices_by_cube[
                    cube_id
                ]
            )

            if self.shuffle:
                rng.shuffle(
                    indices
                )

            for start in range(
                0,
                len(indices),
                self.batch_size,
            ):
                batch = indices[
                    start:
                    start + self.batch_size
                ]

                if (
                    len(batch) < self.batch_size
                    and self.drop_last
                ):
                    continue

                yield batch

    def __len__(
        self,
    ):
        total_batches = 0

        for indices in self.indices_by_cube.values():

            if self.drop_last:
                total_batches += (
                    len(indices)
                    // self.batch_size
                )

            else:
                total_batches += (
                    len(indices)
                    + self.batch_size
                    - 1
                ) // self.batch_size

        return total_batches