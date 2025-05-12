# Deeplearning Showcase

A collection of PyTorch experiments demonstrating (1) function approximation and digit classification with MLPs, and (2) image classification on Fashion-MNIST with a CNN.

## Repository Structure

```
.
├── MLP_Function_Approx_and_MNIST.ipynb   # 1D regression & MNIST classification with MLPs
├── FashionMNIST_CNN.ipynb               # Fashion-MNIST classification with a convolutional network
└── requirements.txt                     # Python dependencies
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/deeplearning-showcase.git
   cd deeplearning-showcase
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and explore the notebooks:
   ```bash
   jupyter notebook
   ```

## 1. MLP Experiments

### 1.1 Function Approximation (1D Regression)

- **Goal**: Approximate $f(x)=x^2\sin(8x)$ on $[-\pi,\pi]$ with mean squared error (MSE) < 1e-3 while minimizing parameter count.
- **Model**: `MyMLP`  
  - Input → ReLU-activated hidden layers → Output  
  - Tunable via `hidden_size` (# hidden layers) and `hidden_width` (# units per layer)
- **Hyperparameters**  
  - `hidden_size=6`, `hidden_width=70`  
  - Optimizer: Adam (lr = 1e-3, weight_decay = 1e-4)  
  - Epochs: 2000, batch_size=1000
- **Results**  
  - Final test MSE ≈ 0.001308  
  - Best MSE ≈ 0.000764 at epoch 1700  
  - Train vs. validation loss curves demonstrate convergence and slight overfitting after epoch 1700

### 1.2 MNIST Digit Classification

- **Goal**: ≤ 2% test error on MNIST using a 3-layer MLP with dropout.
- **Model**: `ClassifierMLP`  
  - FC(784→512) → ReLU → Dropout(p=0.666) → FC(512→256) → ReLU → FC(256→10)
- **Hyperparameters**  
  - Epochs: 10, batch_size=128, lr=1e-3  
  - Optimizer: Adam
- **Results**  
  - Achieved classification error ≈ 1.84% on test set  
  - Training loss steadily decreased each epoch  
  - Confusion matrix visualizes per-digit performance

## 2. CNN Experiment

### Fashion-MNIST Classification with CNN

- **Goal**: Classify 10 classes of Fashion-MNIST images with high accuracy and measure GPU speedup.
- **Model**: `Network_FashionMNIST`  
  1. Conv1: 1→128 filters (3×3) → ReLU  
  2. Conv2: 128→256 filters (3×3) → ReLU → MaxPool(2×2)  
  3. Conv3: 256→512 filters (3×3) → ReLU → MaxPool(2×2)  
  4. Flatten → FC(12800→2560) → ReLU → FC(2560→1280) → ReLU → FC(1280→480) → ReLU → FC(480→10)
- **Hyperparameters**  
  - Epochs: 5, batch_size=16, lr=0.075  
  - Optimizer: SGD
- **Results**  
  - Final training accuracy ≈ 95.8% after 5 epochs  
  - GPU vs. CPU: ~44× speedup on training loop  
  - Detailed layer-wise sizes printed in notebook for inspection  
  - Confusion matrix & accuracy score visualized with seaborn

## Conclusions & Next Steps

- **MLP experiments** showed that modestly deep ReLU networks can approximate complex functions and classify digits with low error when combined with dropout.
- **CNN experiment** achieved strong image-classification performance and highlighted GPU acceleration benefits.
---
