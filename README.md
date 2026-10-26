# TSCCN for Hyperspectral Organ Classification (HeiPorSPECTRAL)

PyTorch implementation of Zhai & Zhao (2024), *"Two-Stream spectral-spatial
convolutional capsule network for Hyperspectral image classification"*
(IJAEOG 127, 103614), adapted to the HeiPorSPECTRAL porcine organ dataset.

Task: pixel-wise classification of 4 organs (kidney, liver, spleen, stomach).
Evaluation is subject-disjoint (split by pig): 8 pigs train, P094 validation,
P087 test. 32 cubes in total (8 per organ).

## Folder layout

```
configs/cubes_expanded.yaml   the 32 cubes used (organ, pig, split)
configs/validation_folds/     3 alternative validation pigs (test pig stays P087)
scripts/                      data selection/download/verification + unit tests
src/data/                     heipor.py (file reader), dataset.py, sampler.py
src/models/                   capsule.py, sim.py, spectral.py, spatial.py, tsccn.py, loss.py
train.py                      training (20 epochs, keeps best validation model)
test.py                       evaluation of a checkpoint on the held-out test pig
checkpoints/tsccn_best.pt     trained model (epoch 20, 71.90% validation accuracy)
docs/data_notes.md            notes on the dataset (written for the initial 4-cube design)
```

## Run order

```
pip install -r requirements.txt

# 1. (optional) re-create the cube selection -> configs/cubes_expanded.yaml
python scripts/select_cubes.py --per-class 8 --seed 42

# 2. download only the 32 needed cubes (not the 845 GiB archive) -> data/raw/heiporspectral/
python scripts/download_data.py --config configs/cubes_expanded.yaml

# 3. check the download and that no pig appears in two splits
python scripts/verify_dataset.py

# 4. train (writes checkpoints/tsccn_best.pt)
python train.py

# 5. evaluate on the unseen test pig (accuracy, confusion matrix, per-class report)
python test.py
```

`scripts/test_*.py` are unit/sanity tests for individual parts (capsule, SIM,
spectral stream, spatial stream, full model, loss, loader, dataset).
Settings (epochs, samples per cube, paths) are constants at the top of
`train.py` and `test.py`.

The raw data is not included (large); step 2 downloads it. The provided
checkpoint allows running steps 3 and 5 without retraining.

## Results (epoch-20 checkpoint)

| Split | Accuracy |
|---|---|
| Validation (pig P094, 2,000 samples) | 71.90% |
| Test (pig P087, 400 samples, 100/class) | 50.50% (macro F1 49.67%) |

Chance level is 25%. The validation-test gap indicates that inter-subject
variability is the main limitation. Results come from a single test pig.
