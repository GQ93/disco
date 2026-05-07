# Data

Raw IHC images and ROSMAP neuropathology data are **not** redistributed in this repository.

## Source datasets

- **DLPFC IHC images.** Patrick et al. (2020). 48 ROSMAP subjects, ~30 images per cell type per subject (~1,500 images per of five cell classes), 1040×1388 px, two-channel (DAPI + marker). See the original publication for access.
- **ROSMAP neuropathology.** Available through the [Rush Alzheimer's Disease Center](https://www.radc.rush.edu/) (data-use application required).
- **Bulk cortical transcriptomics.** Used as input for EnsDeconv comparisons; sourced from ROSMAP.

## Layout (planned)

```
data/
├── examples/        # small, redistributable example tiles for sanity-checking inference
├── manifests/       # CSV/TSV manifests describing splits and per-image metadata
└── (raw/processed)  # NOT tracked; populated locally
```

A small set of example tiles will be added under `data/examples/` once the upload is complete.
