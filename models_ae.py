# models_ae.py
# Manual Autoencoder reusing init + activations from models.py
import torch
from typing import List, Dict, Tuple
from models import he_init, xavier_init, relu_forward, relu_backward, sigmoid_forward, sigmoid_backward

Tensor = torch.Tensor

class Autoencoder:
    """
    Symmetric AE with arbitrary layer sizes.
    - sizes = [Din, h1, h2, ..., hk, ..., h2, h1, Din]
    - BN on hidden layers (optional), ReLU or Sigmoid hidden activations
    - Output is linear (identity) -> use MSE reconstruction loss
    """

    def __init__(
        self,
        input_dim: int,
        enc_dims: List[int],           # e.g. [256, 64]  (64 = latent)
        dec_dims: List[int],           # e.g. [256]
        init: str = "he",              # "he" | "xavier"
        activation: str = "relu",      # "relu" | "sigmoid"
        use_batchnorm: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.9,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        assert init in ("he", "xavier")
        assert activation in ("relu", "sigmoid")
        self.device, self.dtype = device, dtype
        self.activation_name = activation
        self.use_bn = use_batchnorm
        self.eps = eps
        self.momentum = momentum

        self.sizes: List[int] = [input_dim] + enc_dims + dec_dims + [input_dim]
        self.idx_enc_end = len(enc_dims)          # number of encoder hidden layers
        self.L_total = len(self.sizes) - 1        # number of linears
        self.L_hidden = self.L_total - 1          # all except final output

        # params
        # params
        self.W: List[Tensor] = []
        self.b: List[Tensor] = []

        for i in range(self.L_total):
            fan_in, fan_out = self.sizes[i], self.sizes[i + 1]  # note the underscore!
            w = he_init(fan_in, fan_out) if init == "he" else xavier_init(fan_in, fan_out)
            self.W.append(w.to(device=device, dtype=dtype))
            self.b.append(torch.zeros(fan_out, device=device, dtype=dtype)) 
        if self.use_bn:
            self.gamma: List[Tensor] = []
            self.beta:  List[Tensor] = []
            self.running_mean: List[Tensor] = []
            self.running_var:  List[Tensor] = []
            for i in range(self.L_hidden):
                D = self.sizes[i+1]
                self.gamma.append(torch.ones(D, device=device, dtype=dtype))
                self.beta.append(torch.zeros(D, device=device, dtype=dtype))
                self.running_mean.append(torch.zeros(D, device=device, dtype=dtype))
                self.running_var.append(torch.ones (D, device=device, dtype=dtype))
        self._alloc_grads()

    # ---- utils ----
    def _alloc_grads(self):
        self.dW = [torch.zeros_like(W) for W in self.W]
        self.db = [torch.zeros_like(b) for b in self.b]
        if self.use_bn:
            self.dgamma = [torch.zeros_like(g) for g in self.gamma]
            self.dbeta  = [torch.zeros_like(b) for b in self.beta]

    def _act_forward(self, z):
        if self.activation_name == "relu":   return relu_forward(z)
        else:                                return sigmoid_forward(z)

    def _act_backward(self, da, cache):
        if self.activation_name == "relu":   return relu_backward(da, cache)
        else:                                return sigmoid_backward(da, cache)

    def _bn_forward(self, z, i: int, training: bool):
        if not self.use_bn: return z, None
        if not training:
            mu, var = self.running_mean[i], self.running_var[i]
            inv_std = 1.0 / torch.sqrt(var + self.eps)
            z_hat = (z - mu) * inv_std
            y = self.gamma[i]*z_hat + self.beta[i]
            return y, {"z_hat": z_hat, "inv_std": inv_std, "i": i}
        mu = z.mean(0)
        var = z.var(0, unbiased=False)
        inv_std = 1.0 / torch.sqrt(var + self.eps)
        z_hat = (z - mu) * inv_std
        y = self.gamma[i]*z_hat + self.beta[i]
        self.running_mean[i] = self.momentum*self.running_mean[i] + (1-self.momentum)*mu
        self.running_var[i]  = self.momentum*self.running_var[i]  + (1-self.momentum)*var
        return y, {"z_hat": z_hat, "inv_std": inv_std, "i": i}

    def _bn_backward(self, dy, cache):
        if cache is None: return dy, None, None
        i = cache["i"]; z_hat = cache["z_hat"]; inv_std = cache["inv_std"]
        dgamma = (dy*z_hat).sum(0); dbeta = dy.sum(0)
        B = dy.shape[0]
        dz = (self.gamma[i] * inv_std / B) * (B*dy - dbeta - z_hat*dgamma)
        return dz, dgamma, dbeta

    # ---- forward/backward ----
    def forward(self, X: Tensor, training: bool = True) -> Tuple[Tensor, Dict]:
        caches_h: List[Dict] = []
        h = X
        for i in range(self.L_hidden):
            z = h @ self.W[i] + self.b[i]
            z_bn, bn_cache = self._bn_forward(z, i, training) if self.use_bn else (z, None)
            a, act_cache = self._act_forward(z_bn)
            caches_h.append({"x_in": h, "z": z, "z_bn": z_bn,
                             "bn_cache": bn_cache, "act_cache": act_cache, "i": i})
            h = a
        # output layer (linear, no BN/activation)
        z_out = h @ self.W[self.L_hidden] + self.b[self.L_hidden]
        return z_out, {"hidden": caches_h, "h_before_out": h}

    def loss_and_grad(self, x_hat: Tensor, x_true: Tensor) -> Tuple[Tensor, Tensor]:
        # MSE reconstruction
        B, D = x_hat.shape
        diff = x_hat - x_true
        loss = (diff**2).mean()
        dY = (2.0 / (B*D)) * diff  # grad w.r.t. x_hat
        return loss, dY

    def backward(self, X: Tensor, cache: Dict, dY: Tensor) -> None:
        # output linear
        Lout = self.L_hidden
        h = cache["h_before_out"]
        self.dW[Lout] = h.t() @ dY
        self.db[Lout] = dY.sum(0)
        dh = dY @ self.W[Lout].t()

        # hidden (reverse)
        for i in reversed(range(self.L_hidden)):
            c = cache["hidden"][i]
            dz_bn = self._act_backward(dh, c["act_cache"])
            dz, dgamma, dbeta = self._bn_backward(dz_bn, c["bn_cache"]) if self.use_bn else (dz_bn, None, None)
            x_in = c["x_in"]
            self.dW[i] = x_in.t() @ dz
            self.db[i] = dz.sum(0)
            if self.use_bn:
                self.dgamma[i] = dgamma
                self.dbeta[i]  = dbeta
            dh = dz @ self.W[i].t()

    def zero_grad(self):
        for i in range(self.L_total):
            self.dW[i].zero_(); self.db[i].zero_()
        if self.use_bn:
            for i in range(self.L_hidden):
                self.dgamma[i].zero_(); self.dbeta[i].zero_()

    def step(self, lr: float = 1e-3):
        for i in range(self.L_total):
            self.W[i] -= lr * self.dW[i]
            self.b[i] -= lr * self.db[i]
        if self.use_bn:
            for i in range(self.L_hidden):
                self.gamma[i] -= lr * self.dgamma[i]
                self.beta[i]  -= lr * self.dbeta[i]

    # ---- convenience ----
    def encode(self, X: Tensor) -> Tensor:
        """Return latent (output of last encoder hidden layer)."""
        h = X
        for i in range(self.idx_enc_end):   # only encoder part
            z = h @ self.W[i] + self.b[i]
            z_bn, _ = self._bn_forward(z, i, training=False) if self.use_bn else (z, None)
            a, _ = self._act_forward(z_bn)
            h = a
        return h  # latent z

    def parameters(self) -> List[Tensor]:
        ps = []
        for i in range(self.L_total): ps += [self.W[i], self.b[i]]
        if self.use_bn:
            for i in range(self.L_hidden): ps += [self.gamma[i], self.beta[i]]
        return ps