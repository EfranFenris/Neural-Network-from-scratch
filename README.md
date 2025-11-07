# Neural-Network-from-scratch

Manual implementation of a small feed-forward neural network (MLP) in PyTorch **without using autograd or `nn.Module`** for learning purposes. Forward pass, backward pass, BatchNorm, and SGD parameter updates are all written by hand using only basic tensor operations.

## Highlights
- 1–2 hidden layers with configurable width
- He or Xavier weight initialization
- ReLU or Sigmoid activations
- Optional Batch Normalization on hidden layers (manual forward + backward, running stats)
- Separate subclasses for Classification (logits + cross-entropy) and Regression (raw values + MSE)
- Mini-batch SGD training loop written manually
- Simple synthetic demo datasets for sanity checking
- Saves learning curve plots (loss + metric) to PNGs via matplotlib

## Repository Structure
```
models.py      # Manual MLP core + classification/regression subclasses
train.py       # Training/evaluation loops + demo runs and plotting
Hola.py        # Simple hello-world script (not part of the NN code)
*.png          # Generated plots from previous runs (learning curves)
README.md      # This documentation
```

## Requirements
- Python 3.9+
- PyTorch
- matplotlib (for plots)

Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```
Install dependencies:
```bash
pip install torch matplotlib
```

## Running the demos
Execute the training script to run both the classification and regression examples:
```bash
python train.py
```
This will:
1. Build synthetic datasets.
2. Train the chosen model for each task.
3. Save learning curve images:
   - `clf_curves_loss.png`, `clf_curves_acc.png`
   - `reg_curves_loss.png`, `reg_curves_mse.png`

## How it works (brief)
`models.py` defines a base class that manually performs:
- Linear layers: `h @ W + b`
- BatchNorm (optional): computes batch statistics, keeps running estimates
- Activation: ReLU or Sigmoid with custom backward
- Output handling: logits for classification, raw values for regression
- Backward pass: explicit gradient propagation and accumulation
- Update: simple SGD via `param -= lr * grad`

Loss functions:
- Classification: stable softmax + cross-entropy with manual gradient
- Regression: Mean Squared Error

## Extending the project
Ideas you can try next:
- Add numerical gradient checking (finite differences) to validate manual backprop.
- Implement additional activations (e.g., Tanh, LeakyReLU) and initializations.
- Add learning rate schedulers or momentum.
- Support more than 2 hidden layers by generalizing construction loops.
- Add dropout (manual forward/backward) for regularization.
- Port the training loop to use PyTorch DataLoader for larger datasets.
- Compare manual gradients against autograd (wrap tensors with `requires_grad=True`).

## Testing suggestion
Create a small test to ensure backward gradients match numerical gradients:
```python
# pseudo-test idea
import torch
from models import MyFFNetworkForRegression

model = MyFFNetworkForRegression(input_dim=3, hidden_dim=5, output_dim=2, num_hidden_layers=1, use_batchnorm=False)
X = torch.randn(4, 3)
y = torch.randn(4, 2)
ypred, cache = model.forward(X)
loss, dY = model.loss_and_grad(ypred, y)
model.backward(X, cache, dY)
# Now compare model.dW[0] to numerical approx...
```

## License
This repository is currently unlicensed. Consider adding an open-source license (e.g., MIT) if you plan to share or extend publicly.

## Attribution / Purpose
Built for educational clarity: understand each part of a neural network pipeline by removing abstractions. Great for learning internal mechanics before using higher-level frameworks.


