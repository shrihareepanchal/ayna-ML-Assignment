# Ayna ML Assignment – Report & Insights

## Problem Statement

Design a deep learning model that takes:
- A grayscale polygon image
- A color name (e.g., "red", "cyan", "purple")
and generates a colorized polygon image matching the shape and color.


## Architecture Overview

We implemented a **Conditional UNet** architecture in PyTorch:

- **UNet encoder-decoder** with skip connections for spatial consistency.
- **Color conditioning** via `nn.Embedding`:
  - A color name is converted to a dense vector.
  - The vector is spatially expanded and concatenated with the image before the first convolution.
- Output image: 3-channel RGB prediction of size 128×128.

---

## ⚙️ Hyperparameters

    | Hyperparameter     | Value          | Rationale                                 |
    |--------------------|----------------|-------------------------------------------|
    | Optimizer          | Adam           | Handles sparse gradients and works well with UNet |
    | Learning Rate      | 1e-3           | Empirically stable starting point         |
    | Loss Function      | MSELoss        | Pixel-wise reconstruction objective       |
    | Batch Size         | 8              | Balanced memory usage and convergence     |
    | Epochs             | 20             | Enough for convergence on synthetic data  |
    | Embedding Dimension| 32             | Compact but expressive for color encoding |
    | Input Resolution   | 128 × 128      | Suitable tradeoff between speed & quality |


## Training Dynamics

   ### Loss Curves
   - Loss decreased smoothly over 20 epochs.
   - Model converged to a low reconstruction error without overfitting.

   ### Output Trends
   - The model correctly fills polygons with requested color.
   - Consistently handles all shapes (circle, hexagon, triangle, etc.)
   - Colors like **red, blue, purple, cyan** are well learned.


## Failure Cases & Fixes

    | Issue                                   | Fix Applied                          |
    |----------------------------------------|--------------------------------------|
    | Initial crashes due to shape mismatch  | Refactored upsampling logic with exact concat sizes |
    | KeyError on `'input'` in data.json     | Matched key names like `'input_polygon'` and `'colour'` |
    | Color mismatch due to vocab issues     | Expanded `color_vocab` to match all JSON entries |



## Key Learnings

- How to condition UNet on auxiliary non-image inputs (e.g., text).
- Handling shape and channel mismatches when using skip connections.
- Using `wandb` for experiment tracking and reproducibility.
- Importance of consistent dataset schema and augmentation possibilities.


## Artifacts

- [wandb Run Link](https://wandb.ai/shriharee0004-ayna/ayna-polygon-coloring/runs/h5cx7mdb)
- `inference.ipynb`: Visualizes model performance
-  `README.md`: Clean documentation for code and workflow
-  `unet_model.pth`: Final trained model


### Conclusion

The model meets the goal: generating colorized polygons conditioned on shape and color input. The architecture generalizes well on both training and validation data, and the design choices are robust and reproducible.
