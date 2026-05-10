# Reference results

Small reference outputs from the original DISCO training runs, kept in git so
you can verify your local setup reproduces the published numbers without
needing to refit. Five subfolders, one per cell type — each contains the same
five files:

| File | Source (original repo) | Purpose |
|------|------------------------|---------|
| `config.json` | `<cellType>_test/config.json` | The exact `ImPartialConfig` saved by `02_Testing.py` (`seed=42, lr=5e-4, EPOCHS=400`, etc.). Load it via `general.utils.load_json` to reproduce the run. |
| `history.json` | `<cellType>_test/history.json` | Per-epoch losses (`loss / rec / seg_fore / seg_back`, train + val). Use this as a sanity baseline for your own training curves. |
| `pd_summary_results.csv` | `<cellType>_test/pd_summary_results.csv` | Per-image prediction summary that ImPartial writes during evaluation. |
| `files_results.csv` | `<cellType>/files_results.csv` | Test-set manifest (per-image `input_dir`, `input_file`, etc.) consumed by `02_Testing.py` and `03_Tunning_crossValidate.py`. |
| `files_2tasks1x2classes_3images_scribble_train_150_results.csv` | same | Manifest for the three scribble-annotated training images (with `gt_index_task0`, `gt_index_task1`, `prefix` columns). |

Cell types: **NeuN** (neurons), **GFAP** (astrocytes), **iba1** (microglia),
**Olig2** (oligodendrocytes), **PECAM** (endothelial cells).

Pretrained weights (`weights_best.pth` and `model_val_best_save{0,1}.pth` for
each cell type, ~17 MB each, ~247 MB total) are not in git — they are
attached to the project's GitHub Release and archived on Zenodo. See the
project README and `checkpoints/README.md` for download links.
