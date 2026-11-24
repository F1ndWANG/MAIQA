# MAIQA

Core model implementation for **MAIQA: Quality-Attributable Evidence Routing for Blind Image Quality Assessment**. The model exposes source-labelled depth/support roles, a pixel-domain distortion sidecar, quality-importance estimation, adapters, role interaction, and the quality prediction head. The repository also includes the model's training objectives.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later is required. PyTorch must be installed for your CPU or CUDA environment. The standard ViT backbone uses `timm`.

## Model usage

```python
import torch
from maiqa import MAIQA, MAIQAConfig

# Paper architecture: ViT-B/16, three feature depths, seven roles.
# pretrained=False avoids downloading backbone weights for this example.
cfg = MAIQAConfig(pretrained=False, adapter_dim=128, adapter_bottleneck=32)
model = MAIQA(cfg).eval()
# Replace this tensor with ImageNet-normalized 224 x 224 RGB crops.
crops = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    outputs = model(crops, return_features=True)
    crop_scores = outputs['score']              # [2]
    role_scores = outputs['role_sub_scores']    # [2, 7]
    role_nodes = outputs['role_nodes']          # [2, 7, 128]
```

With no trained MAIQA checkpoint, scores are random and have no quality-assessment meaning. `pretrained=True` initializes the ViT backbone using timm's pretrained weights, which may require network access; it does not load a trained MAIQA model. For the paper's image-level protocol, predict each of the 25 crops independently and take their arithmetic mean. The implementation also contains experimental grouped and multiscale paths; their aggregation is not necessarily the paper's arithmetic-mean protocol.

## Contents and reproducibility

- `maiqa/model.py`: reference model, its configuration, and optional experimental variants.
- `maiqa/objectives.py`: CCC, ranking, QIE, and auxiliary role losses, among other objectives.
- `maiqa/__init__.py`: model and objective exports.

This initial release contains core model and objective code only. It does not include benchmark datasets, data splits, training/evaluation scripts, or trained MAIQA checkpoints, and is therefore not a complete reproduction package for the paper's reported numbers. The model file also contains optional experimental settings; the example above selects the main architecture rather than those optional paths.
