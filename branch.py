"""The fine-tuned part of the image model: the last 2 transformer blocks of TinyCLIP's image tower,
plus its final layer norm and projection. Input: the 50 tokens after block 10 (the base model's
`tok` output). Output: a 512-number embedding, normalized. Written in plain torch (no open_clip)
so it exports to a small, plain ONNX file the browser can run."""
import torch
from torch import nn

SPLIT, WIDTH, HEADS, LAYERS = 10, 512, 8, 12

class Block(nn.Module):
    def __init__(s, d=WIDTH, h=HEADS):
        super().__init__()
        s.h = h
        s.ln_1 = nn.LayerNorm(d); s.ln_2 = nn.LayerNorm(d)
        s.attn = nn.Module()
        s.attn.in_proj_weight = nn.Parameter(torch.empty(3 * d, d)); s.attn.in_proj_bias = nn.Parameter(torch.empty(3 * d))
        s.attn.out_proj = nn.Linear(d, d)
        s.mlp = nn.Sequential(); s.mlp.add_module('c_fc', nn.Linear(d, 4 * d)); s.mlp.add_module('gelu', nn.GELU()); s.mlp.add_module('c_proj', nn.Linear(4 * d, d))
    def forward(s, x):
        n, t, d = x.shape; hd = d // s.h
        y = s.ln_1(x)
        qkv = y @ s.attn.in_proj_weight.t() + s.attn.in_proj_bias
        q, k, v = qkv.split(d, dim=-1)
        q = q.reshape(n, t, s.h, hd).transpose(1, 2); k = k.reshape(n, t, s.h, hd).transpose(1, 2); v = v.reshape(n, t, s.h, hd).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)) * (hd ** -0.5), dim=-1) @ v
        x = x + s.attn.out_proj(a.transpose(1, 2).reshape(n, t, d))
        return x + s.mlp(s.ln_2(x))

class Branch(nn.Module):
    def __init__(s):
        super().__init__()
        s.blocks = nn.ModuleList([Block() for _ in range(LAYERS - SPLIT)])
        s.ln_post = nn.LayerNorm(WIDTH)
        s.proj = nn.Parameter(torch.empty(WIDTH, 512))
    def forward(s, tok):
        x = tok
        for b in s.blocks: x = b(x)
        e = s.ln_post(x)[:, 0] @ s.proj
        return e / e.norm(dim=-1, keepdim=True)
    def load_np(s, d):
        s.load_state_dict({k: torch.from_numpy(v.astype('float32')) for k, v in d.items()}, strict=True)
