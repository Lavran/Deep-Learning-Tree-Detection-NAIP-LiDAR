# Fine-tuning DeepForest with LiDAR crowns

Individual tree detection in 0.6 m NAIP imagery with
[DeepForest](https://deepforest.readthedocs.io/), fine-tuned on tree crowns segmented
from airborne LiDAR instead of hand-drawn labels. The fine-tuned model is compared with
the released DeepForest model by five-fold spatial cross-validation across 40
one-kilometer LiDAR tiles in eastern Mount Rainier National Park (194,055 reference
trees), and detection is related to canopy surface complexity (rumple) at a fixed crown
size.

**Manuscript:** [Fine-tuning-DeepForest-with-LiDAR-crowns.html](Fine-tuning-DeepForest-with-LiDAR-crowns.html)
(source: `Fine-tuning-DeepForest-with-LiDAR-crowns.qmd`)

## Main results

- Fine-tuning raised cross-validated F1 from 0.42 to 0.75 when a tree counts as detected
  if its LiDAR treetop falls inside a predicted box (precision 0.48 to 0.76, recall 0.37
  to 0.74), and from 0.06 to 0.44 under the stricter box-overlap criterion (IoU ≥ 0.5).
- Averaged over all trees, the fine-tuned model was more accurate in rougher canopy (F1
  0.65 in the smoothest fifth of trees, 0.80 in the roughest), but only because rough
  canopy holds larger crowns. At a given crown area and tree height, each standard
  deviation of rumple lowered the odds of detection (odds ratio 0.83 for the fine-tuned
  model, 0.81 for the released model).

## Pipeline

Run the scripts in numeric order. All paths are set in `config.py`.

| Script | What it does |
|---|---|
| `00_download_naip.py` | Downloads the 2019 NAIP RGB mosaic from Google Earth Engine onto a 0.6 m grid aligned with the canopy height model |
| `01_watershed_crowns.py` | Reference trees from the 1 m canopy height model: treetops by variable-window local maxima, crowns by marker-controlled watershed |
| `02_prepare_training_data.py` | Keeps trees at least 5 m tall, assigns the 40 tiles to five folds (sorted by mean rumple, dealt in rotation), and cuts NAIP into 200-pixel patches with the crown boxes as labels |
| `03_train_deepforest.py` | Fine-tunes DeepForest on each fold's training tiles for 10 epochs, keeping the epoch with the highest validation accuracy |
| `04_predict_tiles.py` | Predicts each fold's test tiles with its fine-tuned model and with the released model, matches predictions to reference trees under five criteria, and writes precision, recall and F1 |
| `05_rumple_analysis.py` | Relates detection to rumple: accuracy by rumple bin and by tile, recall by crown size, tree-level logistic regressions, robustness checks and the figures |

`config.py` takes the project folder to be the folder it sits in, so the scripts run
wherever the repository is cloned, with no path to edit: put the data in the folders it
names, or change the paths there to point at your own copies. `00` also needs your own
Google Earth Engine project ID, set in the same file.

`03` and `04` skip folds that are already finished, so they can be stopped and
restarted; set `redo = True` to run everything again. To rebuild the manuscript, run
`quarto render Fine-tuning-DeepForest-with-LiDAR-crowns.qmd`; it reads the NAIP mosaic
and the outputs of 02, 04 and 05.

## Outputs

| Folder | Contents |
|---|---|
| `watershed_outputs/` | Reference crowns and treetops (01) |
| `deepforest_training_data_cv/` | NAIP patches, the training and validation tables of each fold, and the tiles with their folds, `tile_folds.shp` (02) |
| `deepforest_models/`, `lightning_logs/` | Fine-tuned model of each fold; loss and validation accuracy by epoch (03) |
| `deepforest_predictions_cv/` | Predictions and scored reference trees of each fold; cross-validated scores in `cv_scores.csv`, `cv_scores_by_fold.csv` and `cv_tile_scores.csv` (04) |
| `rumple_analysis_cv/` | Tables, regression results and figures of the rumple analysis: one subfolder per model, plus the comparison of the two models (05) |

## Setup

```
conda env create -f environment.yml
conda activate tree-detection
```

The environment pins the package versions the results were produced with (Python 3.12,
DeepForest 2.1.0, PyTorch 2.9.1). conda-forge installs the GPU build of PyTorch on a
machine with an NVIDIA driver and the CPU build otherwise; fine-tuning (03) was run on a
GPU. The environment solves on 64-bit Windows and Linux, but not on Apple-silicon Macs,
where conda-forge lacks one of DeepForest's dependencies. `00_download_naip.py` also
needs a Google Earth Engine account.

## Data

- **Imagery:** 2019 NAIP, 0.6 m, red-green-blue, from the Google Earth Engine collection
  `USDA/NAIP/DOQQ` (`00_download_naip.py`).
- **LiDAR:** USGS 3DEP *WA Eastern Cascades 2019 B19* (EPSG:6339), downloaded from the
  [Washington State DNR LiDAR Portal](https://lidarportal.dnr.wa.gov/).
- **Canopy height model and rumple:** the 1 m pit-free canopy height model and the 30 m
  rumple raster are built by the companion repository,
  [LiDAR-Canopy-Surface-Complexity-ICAR](https://github.com/Lavran/LiDAR-Canopy-Surface-Complexity-ICAR)
  (its steps 01, 04 and 05). This pipeline reads them from `Rainier_lidar_2019/`, under
  the same folder names as that repository (see `config.py`).

The imagery, LiDAR products, training patches, model checkpoints (3.6 GB) and
prediction shapefiles are too large for the repo; the scripts regenerate them. The repo
includes the scripts, the manuscript, and the summary tables and figures of steps 04 and
05 (`deepforest_predictions_cv/*.csv` and `rumple_analysis_cv/`).

## Reproducibility

Fold assignment (02) depends only on the tiles' mean rumple, so it is the same on every
run. Fine-tuning (03) is not seeded: retrained models, and the scores that follow from
them, will differ slightly from those in the manuscript. In the author's run, fold 2
reused the checkpoint of an earlier run with the same tiles and settings; on a fresh
clone, 03 trains all five folds. Given the same models, 05 reproduces its tables and
figures: its cross-validation, bootstrap intervals and samples use a fixed seed (0).

## License

MIT (see `LICENSE`). © 2026 Lavran Pagano.

