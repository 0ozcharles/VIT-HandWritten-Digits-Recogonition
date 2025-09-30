# VIT-HandWritten-Digits-Recogonition

An end-to-end pipeline for handwritten **digit recognition** built on **Vision Transformer (ViT)**.  
It covers **detect/split -> preprocess -> classify -> evaluate/visualize** and works on **MNIST** as well as noisy, real-world images that may contain **multiple digits per image**.

![DBNet idea](./DBNet.png) ![Prediction](./prediction.png)

---
## Architecture

###  Vanilla Vision Transformer (ViT)
![ViT Overview](./ViT_Net.png)

###  Adapted ViT for 28×28 Digits (this repo)
![Adapted ViT](./Tiny_ViT.png)

**Key specs of the adapted ViT**
- **Input**: 28×28×1 grayscale
- **Patch embedding**: `Conv2d(k=4, s=4)` -> **7×7=49** patch tokens
- **Embedding dim**: **64**
- **Class token**: prepend -> sequence length **50**
- **Positional embedding**: learnable, added to all tokens
- **Encoder depth**: **L = 6** encoder blocks
- **Each encoder block**: `LN -> MHA -> Dropout -> Residual->LN -> MLP(64 -> 128-> 64, GELU) -> Dropout -> Residual`
- **Head**: `LN -> [CLS] extract -> Pre-Logits -> Linear(num_classes=10)`

**Why this adaptation works for MNIST/noisy digits**
- Smaller **embed dim (64)** + **shallower depth (L=6)** = efficient on laptops/3070.
- **Conv2d patch embedding** is robust to local noise compared to flat linear projection.
- **Patch size 4** fits 28×28 perfectly (no padding), preserving fine stroke details.
- Lightweight **MLP expansion 2× (64-> 128-> 64)** keeps capacity while controlling overfit.
- Dropout after **MHA/MLP** stabilizes training on small datasets.


##  Environment

- **OS**: Windows 10/11
- **GPU**: NVIDIA **GeForce RTX 3070 Laptop GPU** (Ampere, compute 8.6)  
- **Driver**: 551.x or newer recommended
- **Python**: 3.9 – 3.11 (tested on 3.10)
- **PyTorch**: 2.4+ with CUDA **12.1** wheels (no standalone CUDA Toolkit needed)

Create a clean environment and install deps:

```bash
# from repo root
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# PyTorch (CUDA 12.1 wheels). If you are CPU-only, drop the --index-url line.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Common libs
pip install timm opencv-python pillow numpy matplotlib scikit-learn tqdm einops
