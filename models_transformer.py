"""
Decoder-only 2-layer Transformer for modular addition (a + b) mod p.

Input sequence: [a, b, "="] (length 3), prediction taken at the last position.
Vocabulary: p number tokens (0..p-1) plus one "=" token -> vocab = p + 1.
Output classes: p.

Architecture (per experiment spec):
  d_model = 128, 4 heads (d_k = 32), d_ff = 128, 2 blocks, no dropout.
  Parameter count ~ 223K (224,768 with biases / bias-less unembed).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

P = 97
EQ_TOKEN = P          # "=" token id
VOCAB = P + 1          # 98
SEQ_LEN = 3


class CausalSelfAttention(nn.Module):
    def __init__(self, d=128, n_heads=4, learnable_temperature=False):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.d_k = d // n_heads
        self.q = nn.Linear(d, d, bias=True)
        self.k = nn.Linear(d, d, bias=True)
        self.v = nn.Linear(d, d, bias=True)
        self.o = nn.Linear(d, d, bias=True)
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            # T_attn replaces d_k under the square root: softmax(QK^T / sqrt(T)).
            # Initialized to d_k, so t=0 is exactly standard 1/sqrt(d_k) attention.
            # Stored in log for positivity.
            self.log_temp = nn.Parameter(torch.full((n_heads,), math.log(float(self.d_k))))
        self.register_buffer("causal_mask",
                             torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).bool())

    def temperature(self):
        """T_attn in d_k units (32 at initialization)."""
        if self.learnable_temperature:
            return self.log_temp.exp()
        return torch.full((self.n_heads,), float(self.d_k),
                          device=self.q.weight.device)

    def forward(self, x, fixed_temp=None, return_attn=False):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # attention = softmax(QK^T / sqrt(T_attn)); T_attn = d_k recovers the
        # standard scaling; fixed_temp (inference sweep) is also in d_k units.
        if fixed_temp is not None:
            scale = math.sqrt(float(fixed_temp))
        elif self.learnable_temperature:
            scale = (0.5 * self.log_temp).exp().view(1, self.n_heads, 1, 1)
        else:
            scale = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(~self.causal_mask.view(1, 1, T, T), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.d)
        out = self.o(out)
        if return_attn:
            return out, attn
        return out


class Block(nn.Module):
    def __init__(self, d=128, n_heads=4, d_ff=128, learnable_temperature=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads, learnable_temperature)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d, bias=True),
        )

    def forward(self, x, fixed_temp=None, return_attn=False):
        attn_out = self.attn(self.ln1(x), fixed_temp=fixed_temp,
                             return_attn=return_attn)
        if return_attn:
            a, attn = attn_out
        else:
            a, attn = attn_out, None
        x = x + a
        x = x + self.mlp(self.ln2(x))
        if return_attn:
            return x, attn
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, d=128, n_heads=4, d_ff=128, n_layers=2,
                 learnable_temperature=False):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, d)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        self.blocks = nn.ModuleList([
            Block(d, n_heads, d_ff, learnable_temperature) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, P, bias=False)
        self.d = d
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "emb" in name:
                nn.init.normal_(p, mean=0.0, std=1.0)
            elif "weight" in name and p.dim() >= 2:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif "bias" in name:
                nn.init.zeros_(p)
        # layernorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tokens, fixed_temp=None, return_acts=False, return_attn=False):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None]
        attns = []
        for blk in self.blocks:
            x, attn = blk(x, fixed_temp=fixed_temp, return_attn=True)
            attns.append(attn)
        x_last = self.ln_f(x[:, -1, :])          # (B, d) -- input to unembed
        logits = self.unembed(x_last)
        out = {"logits": logits}
        if return_acts:
            out["acts"] = x_last
        if return_attn:
            out["attns"] = attns                 # list over layers, (B,H,T,T)
        return out


def make_modular_addition(p=P, train_frac=0.30, split_seed=42):
    """All (a,b) pairs; 30% train / 70% test, encoded [a, b, =] -> (a+b)%p."""
    import numpy as np
    a = np.arange(p)
    aa, bb = np.meshgrid(a, a, indexing="ij")
    aa, bb = aa.ravel(), bb.ravel()
    x = np.stack([aa, bb, np.full_like(aa, EQ_TOKEN)], axis=1).astype("int64")
    y = ((aa + bb) % p).astype("int64")
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(y))
    n_tr = int(round(train_frac * len(y)))
    tr, te = perm[:n_tr], perm[n_tr:]
    return (torch.from_numpy(x[tr]), torch.from_numpy(y[tr]),
            torch.from_numpy(x[te]), torch.from_numpy(y[te]))


if __name__ == "__main__":
    model = GrokkingTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GrokkingTransformer parameters: {n_params}")
    xtr, ytr, xte, yte = make_modular_addition()
    print("train", xtr.shape, "test", xte.shape)
    out = model(xtr[:4], return_acts=True, return_attn=True)
    print("logits", out["logits"].shape, "acts", out["acts"].shape,
          "attn", out["attns"][0].shape)
    mt = GrokkingTransformer(learnable_temperature=True)
    print("init T_attn =", mt.blocks[0].attn.temperature().tolist(), "d_k =", 32)
