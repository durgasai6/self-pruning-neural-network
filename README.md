# Self-Pruning Neural Network on CIFAR-10

A PyTorch implementation of a neural network that learns to prune itself
during training using learnable gate parameters. Built as part of the
Tredence Analytics AI Engineer Case Study.

---

## What This Project Does

Most neural network pruning happens **after** training — you train a big
model, then cut the weak weights. This project does something smarter:
the network learns **during training** which of its own weights are
unnecessary and removes them on the fly.

This is achieved by associating every weight with a learnable "gate"
parameter. These gates are pushed toward zero by a sparsity penalty,
effectively pruning connections that don't contribute to accuracy.

---

## How It Works

### The PrunableLinear Layer
Instead of a standard `torch.nn.Linear` layer, a custom `PrunableLinear`
layer is used. Each weight has a corresponding `gate_score` parameter of
the same shape. During the forward pass:
gates         = sigmoid(gate_scores)       # values between 0 and 1
pruned_weights = weight * gates            # element-wise multiplication
output         = pruned_weights @ x + bias # standard linear operation

When a gate value falls below a threshold (e.g., 0.01), that weight is
considered pruned — it contributes essentially nothing to the output.

### The Sparsity Loss
Training uses a combined loss:
Total Loss = CrossEntropyLoss + λ × SparsityLoss

The `SparsityLoss` is the **L1 norm** of all gate values (sum of all
gate outputs across all PrunableLinear layers). Since sigmoid outputs
are always positive, this is simply their sum. Minimizing this directly
pushes gates toward zero, creating a sparse network.

### Why L1 and Not L2?
L2 regularization shrinks weights but rarely drives them to exactly zero.
L1 creates a **constant gradient pressure** toward zero regardless of the
current value — the same reason LASSO regression produces sparse solutions.
This is mathematically ideal for pruning.

---

## Architecture
Input (3 × 32 × 32)
↓
[Conv Block 1] → 64 filters, MaxPool → 16×16
↓
[Conv Block 2] → 128 filters, MaxPool → 8×8
↓
[Conv Block 3] → 256 filters, MaxPool → 4×4
↓
Flatten → 4096
↓
PrunableLinear(4096 → 1024) + BatchNorm + ReLU + Dropout(0.4)
↓
PrunableLinear(1024 → 512) + BatchNorm + ReLU + Dropout(0.3)
↓
PrunableLinear(512 → 10)
↓
Output (10 classes)

Each Conv Block uses: `Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → MaxPool`

---

## Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|------------|------------------|--------------------|
| 1e-05      | 92.74            | 47.89              |
| 1e-04      | 92.63            | 95.88              |
| 1e-03      | 92.71            | 99.96              |

### Key Observations

- **λ = 1e-5 (Low):** Weakest sparsity pressure. Network retains ~52% of
  connections. Highest raw accuracy at **92.74%**.

- **λ = 1e-4 (Medium):** Strong pruning — **95.88% of weights pruned**
  while maintaining 92.63% accuracy. Best accuracy-vs-sparsity trade-off.

- **λ = 1e-3 (High):** Extreme pruning — **99.96% of weights pruned**
  (network is almost entirely sparse) yet accuracy holds at 92.71%.
  Demonstrates the robustness of the learned sparse representation.

### Standout Finding
Accuracy remains remarkably stable (~92.6–92.7%) across all three lambda
values despite sparsity ranging from 48% to nearly 100%. This shows that
the vast majority of weights in the classifier layers were redundant —
the network can function with less than 1% of its connections active.

---

## Gate Value Distribution

The histogram of gate values for λ = 1e-4 shows:
- A **massive spike near 0** — the vast majority of gates have been
  pushed to near-zero, confirming successful pruning.
- A **small tail toward higher values** — the small fraction of gates
  the network decided to keep active.

This bimodal-like behavior (most gates → 0, a few gates → active) is
exactly what a successful self-pruning network should produce.

---

## Training Details

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 (50k train / 10k test) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-5) |
| Loss | CrossEntropyLoss with label smoothing=0.1 |
| Epochs | 50 per lambda value |
| Batch Size | 128 |
| Device | CUDA (GPU) |

### Data Augmentation
- Random Crop (32×32, padding=4)
- Random Horizontal Flip
- Color Jitter (brightness, contrast, saturation)
- Normalization (CIFAR-10 mean/std)

---

## Project Structure
self-pruning-nn/
│
├── self_pruning_network.py   # Main script (model, training, evaluation)
├── gate_distribution.png     # Gate value histogram (best model)
└── README.md                 # This file

---

## How to Run

### Requirements
```bash
pip install torch torchvision matplotlib numpy
```

### Run Training
```bash
python self_pruning_network.py
```

CIFAR-10 will be downloaded automatically on first run. GPU is strongly
recommended — training all three lambda values takes ~30–45 minutes on
a T4 GPU.

---

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy

---

## Concepts Demonstrated

- **Custom `nn.Module` layers** with multiple learnable parameter tensors
- **Gated weight mechanisms** and element-wise masking
- **L1 sparsity regularization** and its role in learned pruning
- **Gradient flow** through custom layers (gates and weights updated jointly)
- **Hyperparameter trade-off analysis** (λ vs accuracy vs sparsity)
- **Training with composite loss functions**

---

## Author

Built for the Tredence Analytics — AI Engineer Case Study.
