# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for AAMAS 2025 investigating **Focal Loss for Neural Collaborative Filtering (NCF)** to address class imbalance in recommendation systems with implicit feedback. The project contains:

- **LaTeX paper**: `AAMAS-2025-Formatting-Instructions-CCBY/` - Academic paper targeting AAMAS 2025
- **Experiments**: `experiments/` - Jupyter notebooks and Python utilities for NCF experiments with Focal Loss

## Key Components

### Experiments (`experiments/`)

- `focal_loss_utils.py` - Shared utilities module containing:
  - `FocalLoss` and `AlphaBalancedBCE` loss classes
  - Custom NeuMF model variants (`NeuMF_FocalLoss`, `NeuMF_AlphaBCE`)
  - Training dynamics tracking for mechanism validation
  - Experiment configuration presets for MovieLens datasets
  - Multi-seed experiment runners

- `ml100k_*.ipynb` / `ml1m_*.ipynb` - Jupyter notebooks for MovieLens 100K and 1M experiments
  - `*_original.ipynb` - Original experimental design
  - `*_improved.ipynb` - Improved methodology with Alpha-BCE controls and robustness studies

### Paper (`AAMAS-2025-Formatting-Instructions-CCBY/`)

- `main.tex` - Main LaTeX document
- `references.bib` - Bibliography
- Uses AAMAS 2025 formatting (aamas.cls)
- Comment macros defined: `\todo{}`, `\omer{}`, `\dvir{}`, `\guy{}`, `\rotem{}`

## Technical Details

### Dependencies

- **Python**: PyTorch, NumPy, Pandas
- **RecBole**: Unified recommendation framework (`recbole==1.2.0`)
- **NumPy**: Requires version 1.x (not 2.x) for RecBole compatibility

### Research Hypotheses

- **H1 (Efficacy)**: Focal Loss improves NeuMF over BCE
- **H2 (Robustness)**: Improvements hold across negative sampling ratios (1:4, 1:10, 1:50)
- **H3 (Mechanism)**: Focusing effect (gamma > 0) is necessary beyond class weighting

### Focal Loss Parameters

- `gamma`: Focusing parameter (typically 2.0) - controls down-weighting of easy examples
- `alpha`: Class balancing weight (typically 0.25) - weight for positive class

## Repository Rules

- Don't add credits to Claude on any commit message
- Always look for existing scripts/code instead of rewriting - avoid duplicates
