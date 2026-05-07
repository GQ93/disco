# DISCO

**A Deep Learning Algorithm for Digital Cell Segmentation and Counting with Immunofluorescence Images**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-gxm324%2Fdisco-blue)](https://huggingface.co/gxm324/disco)
[![Status: pre-release](https://img.shields.io/badge/status-pre--release-orange.svg)](#roadmap)

> 🚧 **Pre-release.** This repository hosts the official implementation of DISCO. The code, trained checkpoints, and example data are being uploaded incrementally — see [Roadmap](#roadmap).

---

## Overview

**DISCO** (**D**eep learning for d**I**gital cell **S**egmentation and **CO**unting) is a unified, weakly supervised pipeline for **cell segmentation, counting, and cell-type deconvolution** from multichannel immunohistochemistry (IHC) images. DISCO pairs a U-Net-based segmentation backbone with **sparse expert scribble annotations** to identify nuclei and marker-positive cell bodies, then estimates sample-level cell-type proportions through a standardized thresholding procedure.

DISCO was developed and validated on dorsolateral prefrontal cortex (DLPFC) IHC data spanning **five major brain cell types** — neurons, astrocytes, microglia, oligodendrocytes, and endothelial cells — and benchmarked against established image-based and transcriptomics-based deconvolution methods. DISCO-derived proportions recover biologically interpretable associations with Alzheimer's disease neuropathology (Braak stage, neuritic/diffuse plaques, NFTs, global pathology burden) using ROSMAP subjects with matched IHC images.

### Key features
- **Weakly supervised.** Trains from ~20 scribble-annotated images per cell type — no dense pixel-wise masks required.
- **Per-cell-type models.** Separate U-Net heads per marker (NeuN, GFAP, IBA1, OLIG2, PECAM); fixed configuration across cell types for reproducibility.
- **Standardized deconvolution.** IoU-optimized thresholding → binary masks → cell-type proportion = (nuclei inside marker-positive regions) / (total nuclei).
- **Reproducible benchmarks.** Scripts to reproduce comparisons against Patrick et al. (image-based) and EnsDeconv (transcriptomics-based), plus ViT baselines.
- **Downstream analysis.** R scripts for AD-pathology association analysis on ROSMAP.

## Method at a glance

```
┌─────────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│ Multichannel IHC    │ →  │ DISCO (U-Net,        │ →  │ Probability maps       │
│ (DAPI + marker)     │    │ ImPartial-based,     │    │ (marker / nuclei)      │
│                     │    │ scribble-supervised) │    │                        │
└─────────────────────┘    └──────────────────────┘    └────────────┬───────────┘
                                                                    │
                                                                    ▼
┌─────────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│ Cell-type proportion    │ ←  │ Nucleus-in-marker    │ ←  │ IoU-optimal         │
│ = N_in / N              │    │ counting             │    │ thresholding        │
└─────────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

DISCO's training objective combines a self-supervised blind-spot reconstruction loss with a scribble-guided segmentation loss, following the ImPartial framework (Martinez et al., 2021). See the manuscript for the full formulation.

## Repository layout

```
disco/
├── disco/              # Python package (model, training, inference, IO)
├── notebooks/          # Demo + tutorial Jupyter notebooks
├── scripts/            # Paper-reproduction scripts (training, eval, figures)
├── tests/              # Unit + smoke tests
├── checkpoints/        # Pretrained .pth files (mirrored on Hugging Face)
├── data/               # Sample data and dataset manifests (large data not committed)
├── docs/               # Long-form documentation
├── pyproject.toml      # Package metadata
├── CITATION.cff        # Citation metadata
└── README.md
```

## Installation

> Pre-release — pinned dependencies will be added once the package is uploaded.

```bash
git clone https://github.com/GQ93/disco.git
cd disco
pip install -e .
```

Recommended Python version: **3.10+**. GPU strongly recommended for training; inference can run on CPU for small images.

## Quickstart *(planned API — subject to change)*

```python
from disco import load_pretrained, segment, deconvolve

# Load a per-cell-type checkpoint (pulled from Hugging Face on first call)
model = load_pretrained("disco-neuron")  # or astrocyte / microglia / oligodendrocyte / endothelial

# Run segmentation on a 2-channel (DAPI + marker) image
masks = segment(model, image, threshold="auto")

# Estimate cell-type proportion (nuclei in marker-positive regions / total nuclei)
proportion = deconvolve(masks)
```

CLI:

```bash
disco segment   --model disco-neuron   --input image.tif    --out masks.tif
disco deconv    --masks  masks.tif     --out proportions.csv
disco benchmark --config configs/benchmark.yaml
```

## Pretrained models

Per-cell-type checkpoints are hosted on **Hugging Face** at [`gxm324/disco`](https://huggingface.co/gxm324/disco):

| Cell type        | Marker  | Checkpoint              | IoU-optimal threshold (marker, nuclei) |
|------------------|---------|-------------------------|----------------------------------------|
| Neurons          | NeuN    | `disco-neuron.pth`      | 0.45 / 0.25                            |
| Microglia        | IBA1    | `disco-microglia.pth`   | 0.85 / 0.15                            |
| Astrocytes       | GFAP    | `disco-astrocyte.pth`   | 0.80 / 0.20                            |
| Oligodendrocytes | OLIG2   | `disco-oligo.pth`       | 0.25 / 0.35                            |
| Endothelial      | PECAM   | `disco-endothelial.pth` | 0.75 / 0.20                            |

Thresholds are calibrated by maximizing IoU between thresholded predictions and scribble annotations on the training set; nuclei thresholds are adaptive (not fixed at 0.5).

## Data

DISCO was developed on the DLPFC IHC dataset of **Patrick et al. (2020)** — 48 ROSMAP subjects, ~30 images per cell type per subject (~1,500 images per of five cell classes), 1040×1388 px, two fluorescence channels (DAPI + marker). The marker panel comprises **NeuN, GFAP, IBA1, OLIG2, PECAM**. Raw image data is not redistributed here; see Patrick et al. for access details. ROSMAP neuropathology variables are available through the [Rush Alzheimer's Disease Center](https://www.radc.rush.edu/).

A small set of example tiles for sanity-checking inference will be added under `data/examples/` once the upload is complete.

## Benchmarks

Evaluated against:
- **Patrick et al. (2020)** — image-based estimates from the same DLPFC data using EBImage.
- **EnsDeconv** (Cai et al., 2022) — transcriptomics-based deconvolution applied to matched bulk cortical RNA-seq.
- **ViT_Pat / ViT_Ens** — Vision Transformer baselines trained against Patrick / EnsDeconv proportions.

Concordance is reported via Lin's CCC, Euclidean-based CCC_E, Spearman correlation, and mean squared difference. See `scripts/benchmark/` and Section "Benchmarking" of the manuscript for full results.

## Roadmap

- [x] Repository scaffold + README
- [ ] Upload `disco` Python package (model definitions, training loop, inference)
- [ ] Upload pretrained `.pth` checkpoints (mirror on Hugging Face)
- [ ] Upload paper-reproduction scripts (training, benchmarking)
- [ ] Upload R scripts for ROSMAP AD-pathology association analysis
- [ ] Add example tiles + Quickstart notebook
- [ ] CI: lint + smoke test on push

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
