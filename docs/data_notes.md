# Data notes: HeiPorSPECTRAL

- Full archive: `data/HeiPorSPECTRAL.zip` on the DKFZ S3 bucket, 845 GiB, single zip.
  Individual files are pulled via HTTP range requests (`remotezip`) instead of
  downloading the whole archive.
- Cube shape: 480 x 640 x 100 (height, width, spectral channels), 500-1000nm.
- Unlike the paper's benchmark scenes (Indian Pines/Salinas/KSC - one scene,
  many land-cover classes), HeiPorSPECTRAL annotates **one organ per image**:
  across all 5,756 images and all 3 annotators, only 2 images have more than
  one labeled organ. The rest of each image is unlabeled background/tissue.
- Task adaptation: we pool 4 cubes, each a different organ, into one
  multi-class dataset (organ pixels from each cube + unlabeled background),
  rather than expecting multiple classes within a single cube.
- Chosen organs: liver + spleen (both blood-rich, spectrally similar - a
  good test of whether spatial context/capsules help disambiguate) plus
  kidney and stomach (more spectrally distinct) for contrast.
- Ground truth: annotator1's binary polygon mask per image
  (`annotations/*_binary.png`), not the combined multi-annotator `.blosc`
  segmentation, to keep the label pipeline simple and transparent.
