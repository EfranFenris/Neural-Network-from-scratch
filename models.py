# models.py
# From-scratch MLP using ONLY tensor ops (no autograd, no nn.Module).
# Covers: 1–2 hidden layers, He/Xavier init, BatchNorm (pre-activation),
# manual forward/backward, and SGD step.

import torch
from typing import List, Tuple, Dict

Tensor = torch.Tensor

# ---------- Init helpers ----------
# Weights are in R^{fan_in x fan_out}
def he_init(fan_in: int, fan_out: int) -> Tensor: 
    """He (Kaiming) normal init: good with ReLU."""
    std = (2.0 / fan_in) ** 0.5
    return torch.randn(fan_in, fan_out) * std

def xavier_init(fan_in: int, fan_out: int) -> Tensor:
    """Xavier (Glorot) normal init: good with sigmoid/tanh."""
    std = (2.0 / (fan_in + fan_out)) ** 0.5
    return torch.randn(fan_in, fan_out) * std

# ---------- Activations ----------

def relu_forward(z: Tensor) -> Tuple[Tensor, Tensor]:
    a = torch.maximum(z, torch.zeros_like(z))
    return a, z  # cache z for backward

def relu_backward(da: Tensor, cache_z: Tensor) -> Tensor:
    dz = da.clone()
    dz[cache_z <= 0] = 0.0
    return dz

def sigmoid_forward(z: Tensor) -> Tuple[Tensor, Tensor]:
    s = 1.0 / (1.0 + torch.exp(-z))
    return s, s  # cache s

def sigmoid_backward(da: Tensor, cache_s: Tensor) -> Tensor:
    return da * cache_s * (1.0 - cache_s)

# ---------- Base network (shared logic) ----------

class BaseNetwork:
    """
    Shared MLP core:
      - 1 or 2 hidden layers
      - He / Xavier init
      - ReLU or Sigmoid hidden activations
      - BatchNorm before activation on hidden layers
      - Manual forward + backward
      - SGD update

    Subclasses must implement:
      - output_forward(z_out)      -> predictions (e.g., logits or values)
      - loss_and_grad(y_pred, y)   -> (loss_scalar, dY at output)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 1,  # {1,2}
        init: str = "he",             # "he" or "xavier"
        activation: str = "relu",     # "relu" or "sigmoid"
        use_batchnorm: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.9,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        assert num_hidden_layers in (1, 2)
        assert init in ("he", "xavier")
        assert activation in ("relu", "sigmoid")
        self.device, self.dtype = device, dtype
        self.activation_name = activation
        self.use_bn = use_batchnorm
        self.eps = eps
        self.momentum = momentum

        # Layer sizes: [Din, H, (H), Dout]
        sizes: List[int] = [input_dim]
        for _ in range(num_hidden_layers):
            sizes.append(hidden_dim)
        sizes.append(output_dim)
        self.sizes = sizes
        self.L_hidden = num_hidden_layers
        self.L_total = len(sizes) - 1  # number of linear transforms

        # --- Parameters: W[i] in R^{sizes[i] x sizes[i+1]}, b[i] in R^{sizes[i+1]}
        self.W: List[Tensor] = []
        self.b: List[Tensor] = []
        for i in range(self.L_total):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            w = he_init(fan_in, fan_out) if init == "he" else xavier_init(fan_in, fan_out)
            self.W.append(w.to(device=device, dtype=dtype))
            self.b.append(torch.zeros(fan_out, device=device, dtype=dtype))

        # --- BatchNorm params for hidden layers only
        if self.use_bn:
            self.gamma: List[Tensor] = []
            self.beta:  List[Tensor] = []
            self.running_mean: List[Tensor] = []
            self.running_var:  List[Tensor] = []
            for i in range(self.L_hidden):
                D = sizes[i + 1]
                self.gamma.append(torch.ones(D, device=device, dtype=dtype))
                self.beta.append(torch.zeros(D, device=device, dtype=dtype))
                self.running_mean.append(torch.zeros(D, device=device, dtype=dtype))
                self.running_var.append(torch.ones(D, device=device, dtype=dtype))

        # Gradient buffers (same shapes) 
        self._alloc_grads()

    # ---- internals ----

    def _alloc_grads(self):
        self.dW = [torch.zeros_like(W) for W in self.W]
        self.db = [torch.zeros_like(b) for b in self.b]
        if self.use_bn:
            self.dgamma = [torch.zeros_like(g) for g in self.gamma]
            self.dbeta  = [torch.zeros_like(b) for b in self.beta]

    def _act_forward(self, z: Tensor):
        return relu_forward(z) if self.activation_name == "relu" else sigmoid_forward(z)

    def _act_backward(self, da: Tensor, cache):
        return relu_backward(da, cache) if self.activation_name == "relu" else sigmoid_backward(da, cache)

    # ---- BatchNorm ----

    def _bn_forward(self, z: Tensor, i: int, training: bool):
        if not training:
            mu = self.running_mean[i]
            var = self.running_var[i]
            inv_std = 1.0 / torch.sqrt(var + self.eps)
            z_hat = (z - mu) * inv_std
            y = self.gamma[i] * z_hat + self.beta[i]
            cache = {"z_hat": z_hat, "inv_std": inv_std, "i": i}
            return y, cache

        mu = z.mean(dim=0)
        var = z.var(dim=0, unbiased=False)
        inv_std = 1.0 / torch.sqrt(var + self.eps)
        z_hat = (z - mu) * inv_std
        y = self.gamma[i] * z_hat + self.beta[i]

        # running stats
        self.running_mean[i] = self.momentum * self.running_mean[i] + (1 - self.momentum) * mu
        self.running_var[i]  = self.momentum * self.running_var[i]  + (1 - self.momentum) * var

        cache = {"z_hat": z_hat, "inv_std": inv_std, "i": i}
        return y, cache
    
    #---- BatchNorm backward ----

    def _bn_backward(self, dy: Tensor, cache: Dict):
        i = cache["i"]
        z_hat = cache["z_hat"]
        inv_std = cache["inv_std"]

        dgamma = (dy * z_hat).sum(dim=0)
        dbeta  = dy.sum(dim=0)
        B = dy.shape[0]
        dz = (self.gamma[i] * inv_std / B) * (B * dy - dbeta - z_hat * dgamma)
        return dz, dgamma, dbeta

    # ---- Forward ----

    def forward(self, X: Tensor, training: bool = True) -> Tuple[Tensor, Dict]:
        """
        Returns (y_pred, cache). Cache has everything needed for backward.
        """
        caches_hidden: List[Dict] = []
        h = X

        for i in range(self.L_hidden):
            z_lin = h @ self.W[i] + self.b[i]                # Linear
            if self.use_bn:
                z_bn, bn_cache = self._bn_forward(z_lin, i, training)
            else:
                z_bn, bn_cache = z_lin, None
            a, act_cache = self._act_forward(z_bn)           # Activation
            caches_hidden.append({"x_in": h, "z_lin": z_lin, "z_bn": z_bn,
                                  "bn_cache": bn_cache, "act_cache": act_cache, "i": i})
            h = a

        # Output linear
        z_out = h @ self.W[self.L_hidden] + self.b[self.L_hidden]
        caches_out = {"h_before_out": h, "z_out": z_out}
        return self.output_forward(z_out), {"hidden": caches_hidden, "out": caches_out}

    # ---- Hooks for subclasses ----
    def output_forward(self, z_out: Tensor) -> Tensor:
        raise NotImplementedError

    def loss_and_grad(self, y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    # ---- Backward ----

    def backward(self, X: Tensor, cache: Dict, dY: Tensor) -> None:
        """
        Fill self.dW, self.db (and BN grads). dY is gradient at output (wrt z_out or y_pred
        depending on subclass; here we assume it's wrt z_out as we return logits/values).
        """
        # Output linear layer
        Lout = self.L_hidden
        h = cache["out"]["h_before_out"]
        self.dW[Lout] = h.t() @ dY
        self.db[Lout] = dY.sum(dim=0)
        dh = dY @ self.W[Lout].t()

        # Hidden layers (reverse)
        for i in reversed(range(self.L_hidden)):
            c = cache["hidden"][i]
            # Activation backward
            dz_bn = self._act_backward(dh, c["act_cache"])
            # BN backward (if used)
            if self.use_bn:
                dz, dgamma, dbeta = self._bn_backward(dz_bn, c["bn_cache"])
                self.dgamma[i] = dgamma
                self.dbeta[i]  = dbeta
            else:
                dz = dz_bn
            # Linear backward
            x_in = c["x_in"]
            self.dW[i] = x_in.t() @ dz
            self.db[i] = dz.sum(dim=0)
            dh = dz @ self.W[i].t()

    # ---- SGD update ----

    def zero_grad(self):
        for i in range(self.L_total):
            self.dW[i].zero_()
            self.db[i].zero_()
        if self.use_bn:
            for i in range(self.L_hidden):
                self.dgamma[i].zero_()
                self.dbeta[i].zero_()

    def step(self, lr: float = 1e-3):
        for i in range(self.L_total):
            self.W[i] -= lr * self.dW[i]
            self.b[i] -= lr * self.db[i]
        if self.use_bn:
            for i in range(self.L_hidden):
                self.gamma[i] -= lr * self.dgamma[i]
                self.beta[i]  -= lr * self.dbeta[i]

    # Optional: handy for later experiments with torch.optim (still manual grads).
    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = []
        for i in range(self.L_total):
            ps += [self.W[i], self.b[i]]
        if self.use_bn:
            for i in range(self.L_hidden):
                ps += [self.gamma[i], self.beta[i]]
        return ps

# ---------- Classification: logits + cross-entropy ----------

class MyFFNetworkForClassification(BaseNetwork):
    def output_forward(self, z_out: Tensor) -> Tensor:
        return z_out  # return logits; softmax used inside loss for stability

    def loss_and_grad(self, logits: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Cross-entropy for integer labels y_true in [0..C-1].
        Returns (loss scalar, d/d logits).
        """
        B, C = logits.shape
        z = logits - logits.max(dim=1, keepdim=True).values  # stab
        expz = torch.exp(z)
        probs = expz / expz.sum(dim=1, keepdim=True)
        loss = -torch.log(probs[torch.arange(B), y_true]).mean()
        dlogits = probs
        dlogits[torch.arange(B), y_true] -= 1.0
        dlogits /= B
        return loss, dlogits

    @staticmethod
    def accuracy(logits: Tensor, y_true: Tensor) -> float:
        return float((logits.argmax(dim=1) == y_true).float().mean().item())

# ---------- Regression: identity + MSE ----------

class MyFFNetworkForRegression(BaseNetwork):
    def output_forward(self, z_out: Tensor) -> Tensor:
        return z_out  # raw values

    def loss_and_grad(self, y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Mean Squared Error. Returns (loss scalar, d/d y_pred).
        """
        B, D = y_pred.shape
        diff = y_pred - y_true
        loss = (diff ** 2).mean()
        dY = (2.0 / (B * D)) * diff
        return loss, dY