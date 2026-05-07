# Pretrained checkpoints

Trained DISCO checkpoints (`.pth`) are **not** committed to git. They are mirrored on Hugging Face:

➡ https://huggingface.co/gxm324/disco

Expected files (one per cell type):

| File                       | Cell type          | Marker | IoU-optimal thresholds (marker, nuclei) |
|----------------------------|--------------------|--------|-----------------------------------------|
| `disco-neuron.pth`         | Neurons            | NeuN   | 0.45 / 0.25                             |
| `disco-microglia.pth`      | Microglia          | IBA1   | 0.85 / 0.15                             |
| `disco-astrocyte.pth`      | Astrocytes         | GFAP   | 0.80 / 0.20                             |
| `disco-oligo.pth`          | Oligodendrocytes   | OLIG2  | 0.25 / 0.35                             |
| `disco-endothelial.pth`    | Endothelial cells  | PECAM  | 0.75 / 0.20                             |

The package will pull these on demand via `disco.load_pretrained(...)`.
