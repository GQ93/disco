# Pretrained checkpoints

Trained DISCO checkpoints (`.pth`) are **not** committed to git. They are attached to the project's GitHub Release and archived on Zenodo with a citable DOI:

➡ **GitHub Release:** https://github.com/GQ93/disco/releases/tag/v0.1.0
➡ **Zenodo DOI:** *added once v0.1.0 publishes*

## Bundle contents

A single tarball, `disco_checkpoints_v0.1.0.tar.gz` (~247 MB), contains all 15 checkpoint files — three per cell type:

| Cell type | Marker | Files |
|---|---|---|
| Neurons | NeuN | `NeuN_test/{weights_best,model_val_best_save0,model_val_best_save1}.pth` |
| Astrocytes | GFAP | `GFAP_test/{weights_best,model_val_best_save0,model_val_best_save1}.pth` |
| Microglia | IBA1 | `iba1_test/{weights_best,model_val_best_save0,model_val_best_save1}.pth` |
| Oligodendrocytes | OLIG2 | `Olig2_test/{weights_best,model_val_best_save0,model_val_best_save1}.pth` |
| Endothelial | PECAM | `PECAM_test/{weights_best,model_val_best_save0,model_val_best_save1}.pth` |

Each `.pth` is ~17 MB. The two `model_val_best_save{0,1}.pth` files form the 2-checkpoint ensemble used at inference time (`nsaves=2` in the training config).

## Download

```bash
# from the project root
mkdir -p checkpoints && cd checkpoints
gh release download v0.1.0 -p 'disco_checkpoints_v0.1.0.tar.gz'
tar -xzf disco_checkpoints_v0.1.0.tar.gz
```

After extraction, the per-cell-type subdirectories (`NeuN_test/`, `GFAP_test/`, …) match what `scripts/02_Testing.py` writes during a fresh training run, so they can be loaded by the same `ImPartialModel` code path.

## Provenance

Trained with the configuration captured in `results/<cellType>/config.json` and the loss curves in `results/<cellType>/history.json`. See the project README for the full hyperparameter listing.
