# DISCO

**A Deep Learning Algorithm for Digital Cell Segmentation and Counting with Immunofluorescence Images**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![Status: v0.1.0](https://img.shields.io/badge/release-v0.1.0-blue.svg)](#release-history)

> 📌 **v0.1.0 — research-code release.** This repository hosts the official implementation used in the DISCO manuscript. Code is shared *as-is* in `scripts/` (the development workflow); a clean library API is on the roadmap. Pretrained weights are attached to the GitHub Release and archived on Zenodo. **Raw images and intermediate prediction dumps are not redistributed.**

---

## Overview

**DISCO** (**D**eep learning for d**I**gital cell **S**egmentation and **CO**unting) is a unified, weakly supervised pipeline for **cell segmentation, counting, and cell-type deconvolution** from multichannel immunohistochemistry (IHC) images. DISCO pairs a U-Net-based segmentation backbone (the [ImPartial](#external-dependency-impartial) framework of Martinez et al., 2021) with **sparse expert scribble annotations** to identify nuclei and marker-positive cell bodies, then estimates sample-level cell-type proportions through a standardized thresholding procedure.

DISCO was developed and validated on dorsolateral prefrontal cortex (DLPFC) IHC data spanning **five major brain cell types** — neurons, astrocytes, microglia, oligodendrocytes, and endothelial cells — and benchmarked against established image-based and transcriptomics-based deconvolution methods. DISCO-derived proportions recover biologically interpretable associations with Alzheimer's disease neuropathology (Braak stage, neuritic/diffuse plaques, NFTs, global pathology burden) using ROSMAP subjects with matched IHC images.

### Key features
- **Weakly supervised.** Trains from ~20 scribble-annotated images per cell type — no dense pixel-wise masks required.
- **Per-cell-type models.** Separate U-Net heads per marker (NeuN, GFAP, IBA1, OLIG2, PECAM); fixed ImPartial configuration across cell types for reproducibility.
- **Standardized deconvolution.** Threshold sweep → binary masks → cell-type proportion = (nuclei inside marker-positive regions) / (total nuclei).
- **Reproducible benchmarks.** Helper scripts to compute concordance against Patrick et al. (image-based) and EnsDeconv (transcriptomics-based), and to generate the manuscript's main figures.

## Method at a glance

```
┌─────────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│ Multichannel IHC    │ →  │ DISCO (U-Net,        │ →  │ Probability maps       │
│ (DAPI + marker)     │    │ ImPartial-based,     │    │ marker / nuc_in / nuc_ou
│                     │    │ scribble-supervised) │    │                        │
└─────────────────────┘    └──────────────────────┘    └────────────┬───────────┘
                                                                    │
                                                                    ▼
┌─────────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│ Cell-type proportion    │ ←  │ Connected-component  │ ←  │ Threshold sweep +   │
│ = nuc_in / (nuc_in +    │    │ counting per map     │    │ Gaussian smoothing  │
│   nuc_ou)               │    │                      │    │ (σ=1)               │
└─────────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

The training objective combines a self-supervised blind-spot reconstruction loss with a scribble-guided segmentation loss, following the ImPartial framework (Martinez et al., 2021). See the manuscript for the full formulation.

## Repository layout

```
disco/
├── disco/              # Python package placeholder (clean API target — see Roadmap)
├── scripts/            # Original research scripts used to generate the manuscript
│   ├── 01_TestingImages.py        # Step 1: TIFF → 2-channel .npz data prep
│   ├── 02_Testing.py              # Step 2: train ImPartial + run inference
│   ├── 03_Tunning_crossValidate.py # Step 3: threshold tuning + proportion estimation
│   └── legacy/                    # 13 helper scripts (IoU calc, concordance, figures, …)
├── results/            # Reference outputs from the published runs (per cell type)
│   ├── NeuN/  GFAP/  iba1/  Olig2/  PECAM/
│   │   ├── config.json                   # exact ImPartialConfig used in the paper
│   │   ├── history.json                  # training loss/metric curves
│   │   ├── pd_summary_results.csv        # per-image prediction summary
│   │   ├── files_results.csv             # test-set manifest
│   │   └── files_2tasks*_train_150_results.csv  # scribble-training manifest
├── checkpoints/        # not in git — see Pretrained weights below
├── data/               # not in git — see Data below
├── notebooks/  tests/  docs/   # placeholders for v0.2+
├── pyproject.toml      # package metadata
├── CITATION.cff        # citation metadata
└── README.md
```

## Installation

DISCO v0.1.0 is shipped as a **research-code release**: most of the code lives in `scripts/` and is meant to be run cell-by-cell in Spyder or Jupyter, not imported as a library. A minimal install for executing the scripts:

```bash
git clone https://github.com/GQ93/disco.git
cd disco

# 1. Create an environment (Python 3.10+; CUDA recommended for training)
conda create -n disco python=3.10 -y
conda activate disco

# 2. Runtime dependencies
pip install numpy pandas scipy scikit-image pillow matplotlib seaborn \
            imageio imagecodecs csbdeep "torch>=2.0" "torchvision>=0.15" \
            openpyxl

# 3. Install DISCO itself (placeholder package today; populated in v0.2+)
pip install -e .
```

### External dependency: ImPartial

The training/inference scripts (`02_Testing.py`, several files in `scripts/legacy/`) rely on three modules — `impartial`, `dataprocessing`, `general` — from the **ImPartial** weakly supervised cell-segmentation framework. Clone that codebase separately and ensure its root is on your `PYTHONPATH` (or set the working directory to its root) before running step 2:

> Original ImPartial paper: Martinez et al., "ImPartial: Partial Annotations for Cell Instance Segmentation," 2021.
> The DISCO experiments used a local fork named `ImPartial2`. The exact upstream URL/commit will be linked here in v0.1.1.

GPU strongly recommended for training. Inference can run on CPU for individual tiles.

## Reproducing the paper

The numbered scripts in `scripts/` correspond to the three stages described in the manuscript's *Materials and Methods*. They are **Spyder/Jupyter cell-mode** files (note the `# %%` cell separators and `%load_ext autoreload` lines) — execute the cells sequentially in an IPython environment, editing the hardcoded paths near the top of each file to match your environment.

### Step 1 — `scripts/01_TestingImages.py`
Reads two-channel TIFF files (DAPI + marker), normalizes each channel via `csbdeep.utils.normalize(p1=1, p99=99.8, clip=True)`, concatenates them, and writes 2-channel `.npz` test images. Loops over the first 49 ROSMAP subjects listed in `EnsDeconv_ROS.csv`.

### Step 2 — `scripts/02_Testing.py`
Constructs the ImPartial configuration (matches the manuscript: `seed=42, lr=5e-4, EPOCHS=400, npatches_epoch=4096, weight_objectives={seg_fore: 0.45, seg_back: 0.45, rec: 0.10}`), trains, and runs evaluation on the test set. For each test image, three probability maps are pickled to disk: `marker`, `nuc_in`, `nuc_ou`. Per-cell-type training artifacts are saved under `<basedir>/<cellType>_test/` — the `config.json` and `history.json` from each published run are mirrored under `results/<cellType>/`.

### Step 3 — `scripts/03_Tunning_crossValidate.py`
Sweeps thresholds `t ∈ {0.00, 0.05, …, 0.95}` over a leave-one-out cross-validation on three scribble-annotated training images, picks the threshold whose predicted proportion is closest to the per-cell-type **reference proportion** `True_prop`, and applies that threshold to the test set to obtain the per-image cell-type proportion `prop = nuc_in / (nuc_in + nuc_ou)`.

### Helper scripts — `scripts/legacy/`

| Script | What it does |
|---|---|
| `IoU_cal.py`, `IoU_loop.py` | IoU calculation between predicted masks and scribble labels |
| `IoU_thresholds.py`, `IoU_thresholds_imageSave.py` | Sample-level proportion using IoU-tuned thresholds (Section 3.4 of the manuscript) |
| `Tunning.py` | Earlier non-CV version of the threshold-tuning step |
| `Testing_crossValidate.py` | Earlier CV-flavored training/inference variant |
| `CrossValidationVisual.py` | CV result visualization |
| `PerformanceTrend.py` | Lin's CCC trend vs. number of training images |
| `CorBiasCCC.py`, `CorBiasCCC_normalized.py` | Concordance scatter plots vs. Patrick / EnsDeconv |
| `concateCell.py` | Concatenate per-cell-type proportions across all five cell classes |
| `StackSampProp.py` | Stacked sample-proportion plots |
| `PatrickOrganize.py` | Reformat Patrick et al. (2020) deconvolution outputs into the comparison schema |

## Two threshold conventions in this codebase

Read `True_prop` and the IoU-optimal thresholds carefully — they are *different* numbers used in *different* places:

| | Per-cell-type values (NeuN, GFAP, iba1, Olig2, PECAM) | Where used |
|---|---|---|
| `True_prop` (reference cell-type proportion) | 0.36, 0.18, 0.07, 0.19, 0.12 | `scripts/03_Tunning_crossValidate.py` — the LOO-CV target for threshold tuning |
| IoU-optimal threshold (marker / nuclei) | NeuN 0.45/0.25, IBA1 0.85/0.15, GFAP 0.80/0.20, OLIG2 0.25/0.35, PECAM 0.75/0.20 | Section 3.4 of the manuscript and `scripts/legacy/IoU_thresholds*.py` |

## Pretrained weights

Per-cell-type ImPartial checkpoints (`weights_best.pth`, `model_val_best_save0.pth`, `model_val_best_save1.pth` for each of the five cell types — 15 files, ~17 MB each, ~247 MB total) are attached to the [GitHub Release `v0.1.0`](https://github.com/GQ93/disco/releases/tag/v0.1.0) and archived on Zenodo with a citable DOI:

| Cell type | Marker | Bundle |
|---|---|---|
| Neurons | NeuN | `disco_checkpoints_v0.1.0.tar.gz` (single archive containing all five cell types) |
| Astrocytes | GFAP | ↑ |
| Microglia | IBA1 | ↑ |
| Oligodendrocytes | OLIG2 | ↑ |
| Endothelial | PECAM | ↑ |

> **Zenodo DOI:** *to be filled in once the v0.1.0 release is published.* The persistent "concept DOI" at Zenodo always resolves to the latest release.

After download, untar into the local `checkpoints/` folder; each `<cellType>_test/` subdirectory matches what `02_Testing.py` writes during training.

## Data

DISCO was developed on the DLPFC IHC dataset of **Patrick et al. (2020)** — 48 ROSMAP subjects, ~30 images per cell type per subject (~1,500 images per of five cell classes), 1040×1388 px, two-channel (DAPI + marker). The marker panel is **NeuN, GFAP, IBA1, OLIG2, PECAM**.

**Raw image data is not redistributed in this repository.** See Patrick et al. (2020) for access details. ROSMAP neuropathology variables are available through the [Rush Alzheimer's Disease Center](https://www.radc.rush.edu/) under their data-use application.

The small per-cell-type CSV/JSON manifests required by the scripts (`config.json`, `history.json`, `pd_summary_results.csv`, `files_results.csv`, `files_2tasks*_train_150_results.csv`) are committed under `results/` so you can verify the published numbers match your local re-run.

## Benchmarks

Evaluated against:
- **Patrick et al. (2020)** — image-based estimates from the same DLPFC data using EBImage.
- **EnsDeconv** (Cai et al., 2022) — transcriptomics-based deconvolution applied to matched bulk cortical RNA-seq.
- **ViT_Pat / ViT_Ens** — Vision Transformer baselines trained against Patrick / EnsDeconv proportions.

Concordance is reported via Lin's CCC, Euclidean-based CCC_E, Spearman correlation, and mean squared difference; see `scripts/legacy/CorBiasCCC.py`, `scripts/legacy/PerformanceTrend.py`, and the manuscript's *Benchmarking* section.

## Roadmap

- [x] **v0.1.0** — Repository scaffold, original research scripts (`scripts/`, `scripts/legacy/`), per-cell-type configs and training histories under `results/`, pretrained weights via GitHub Release + Zenodo.
- [ ] **v0.1.1** — Pin upstream ImPartial URL/commit; add a top-level `requirements.txt`; CI smoke test.
- [ ] **v0.2.0** — Refactor common logic (model loading, IoU calc, threshold sweep, proportion estimation) into the importable `disco/` package; add a Quickstart notebook running on a single example tile.
- [ ] **v0.2.x** — R scripts for ROSMAP AD-pathology association analysis; figure-regeneration entry points.

## Citation

If you use DISCO, please cite:

```bibtex
@article{meng2026disco,
  title   = {DISCO: A Deep Learning Algorithm for Digital Cell Segmentation
             and Counting with Immunofluorescence Images},
  author  = {Meng, Guanqun and Qu, Gang and Ma, Wenjing and Tang, Wen and
             Wang, Jiebiao and Zhao, Zhongming and Feng, Hao},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

If you use the pretrained weights specifically, please *also* cite the Zenodo archive (DOI added in v0.1.0).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- DISCO is built on the **ImPartial** weakly supervised segmentation framework (Martinez et al., 2021) and the **U-Net** architecture (Ronneberger et al., 2015).
- IHC data from **Patrick et al. (2020)**; AD neuropathology from **ROSMAP**.
- Supported by **NIH NIGMS R35GM154862** to H.F. The content is solely the responsibility of the authors.

## Authors and affiliations

- **Guanqun Meng**¹†, **Gang Qu**²† — co-first authors
- Wenjing Ma³, Wen Tang¹, Jiebiao Wang⁴, Zhongming Zhao²
- **Hao Feng**¹\* — corresponding author

¹ Department of Population and Quantitative Health Sciences, Case Western Reserve University
² McWilliams School of Biomedical Informatics and School of Public Health, The University of Texas Health Science Center at Houston
³ Department of Biostatistics, University of Michigan
⁴ Department of Biostatistics, University of Pittsburgh
