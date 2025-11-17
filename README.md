# Noise-Boosted Activation Functions for Visual Attention

Implementation code for noise-boosted activation functions applied to attention mechanisms in computer vision.

## Programs

**`vit_glu.py`** - Vision Transformer with GLU-style feed-forward networks and noise-boosted activation functions for CIFAR-10 image classification.

**`vit_glu_optuna.py`** - Automated hyperparameter optimization using Optuna to search for optimal sigma values across transformer layers on CIFAR-10.

**`stl10_cbam.py`** - CBAM CNN classifier with channel attention using noise-boosted activation functions on STL-10 dataset.

**`kvasir_dattnet.py`** - DATTNet medical image segmentation network with dual attention modules and sigma parameter tracking for Kvasir-SEG polyp segmentation.

## Datasets

**STL-10**: https://cs.stanford.edu/~acoates/stl10/

**Kvasir-SEG**: https://datasets.simula.no/kvasir-seg/

## Installation

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow optuna
```
