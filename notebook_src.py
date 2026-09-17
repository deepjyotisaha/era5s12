# %% [markdown]
# # Session 12 — Thirty-two virtual GPUs, and what ZeRO actually moves
#
# **The brief.** Build 32 virtual GPUs, write a demo model that runs on top of them, simulate
# ZeRO-1, ZeRO-2 and ZeRO-3, and show how memory and computation change.
#
# **The claim being tested.** ZeRO changes *where the bytes live* and never *what gets
# computed*. If that is true, then four arrangements — plain data parallelism and the three
# ZeRO stages — must train to **bit-identical weights**, while holding very different
# amounts of memory per rank and moving different amounts of data between ranks. This
# notebook builds a cluster in which both halves of that sentence are measured.
#
# ### How this is built
#
# One Python process holds 32 `Rank` objects. Each rank owns real tensors — its own bf16
# weights, bf16 gradients, fp32 master copy and two Adam moments, or only its slice of them —
# and memory is **counted by walking the tensors a rank actually holds**, not by formula.
# Ranks talk only through **ring collectives implemented hop by hop**, and every hop adds the
# bytes it sent to the sender's counter. So the 2P / 2P / 2P / 3P communication pattern from
# the lecture has to *fall out of the code*; nothing asserts it.
#
# ```
# CELL 1-2     environment, configuration, thresholds, the lecture's numbers
# CELL 3-4     corpus and the demo model (the Session 11 GPT, shrunk)
# CELL 5       the cluster primitives: shards, ranks, ring collectives
# CELL 6       ring collectives vs a naive sum ─────────────── ⛔ GATE A
# CELL 7       DP / ZeRO-1 / ZeRO-2 / ZeRO-3 as one step function
# CELL 8       data parallel == one big-batch GPU, fp64 ─────── ⛔ GATE B
# CELL 8b      why Adam is written as separate ops ──────────── ⛔ GATE B′
# CELL 9       train all four on 32 ranks: identical weights? ── ⛔ GATE C
# CELL 10      two planted bugs, the gate must catch both ───── ⛔ GATE C'
# CELL 11      memory per rank, counted vs modelled ─────────── ⛔ GATE D
# CELL 12      world-size sweep 1..32, bytes on the wire ────── ⛔ GATE E
# CELL 13      what "computation" changes, and what it does not
# CELL 14      how a shard is cut: flat vs whole-tensor (the hot rank)
# CELL 15      projection to V5: 30B parameters, 8-64 GPUs ──── ⛔ GATE F
# CELL 16      real GPU memory check (runs only when CUDA is present)
# CELL 17      figures, drawn from the results dictionary
# CELL 18      acceptance checks, summary.json, RESULTS.md, bundle
# ```
#
# ### Measured, counted, modelled, projected — four words kept apart
#
# | word | meaning here |
# |---|---|
# | **counted** | `numel × element_size` summed over the tensors a simulated rank holds |
# | **measured** | `torch.cuda.memory_allocated()` on a real GPU (Cell 16 only) |
# | **modelled** | the lecture's formula — 16, 4+12/N, 2+14/N, 16/N bytes per weight |
# | **projected** | the modelled formula evaluated at a size this notebook never ran (30B) |
#
# Wall-clock seconds from a single-process simulator measure Python, not a cluster. Where
# seconds appear they are labelled *simulator time* and no conclusion rests on them.
#
# ### Running this
#
# Runtime → Run all. Everything the write-up needs lands in `outputs/` and is zipped by the
# last cell. The simulator itself runs on the CPU in every case, so its numbers are identical
# on a laptop and on Colab; a GPU runtime adds Cell 16's real-memory cross-check.

# %% [markdown]
# ## Cell 1 — environment
#
# ### What this block does
# Imports, fixes the seed, and records what the machine is — then prints it.
#
# ### How it works
# The simulator runs on the **CPU on purpose**. Its job is exact accounting and a bit-exact
# equality test, both of which want one deterministic device. If CUDA is present it is used
# for exactly one thing: Cell 16 re-runs one configuration on the real card and compares
# `torch.cuda.memory_allocated()` against the simulator's counted bytes.
#
# ### Inputs / Outputs
# In: nothing. Out: `SEED`, `HAS_CUDA`, `SIM_DEVICE`, the results dictionary `R`.
#
# ### What you should see
# A version box, the torch version, `simulator device: cpu`, and either the GPU's name or a
# note that Cell 16 will be skipped.

# %%
import sys, subprocess

def _pip(pkg):
    """Install only if the import actually fails - Colab has most of this already."""
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_pip("tiktoken")

import os, math, time, json, zipfile, urllib.request, random, platform
from collections import defaultdict
from dataclasses import dataclass, asdict

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTEBOOK_VERSION = "v1"
_BANNER = [
    f"NOTEBOOK {NOTEBOOK_VERSION}  -  Session 12, ZeRO on 32 virtual GPUs",
    "",
    "If Cell 2 does not report world = 32, you are running an old copy.",
]
_w = max(len(x) for x in _BANNER) + 4
print("+" + "-" * _w + "+")
for _line in _BANNER:
    print("|  " + _line.ljust(_w - 2) + "|")
print("+" + "-" * _w + "+\n")

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

SIM_DEVICE = "cpu"
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    _props = torch.cuda.get_device_properties(0)
    GPU_NAME, CAPABILITY, VRAM_GB = _props.name, torch.cuda.get_device_capability(0), _props.total_memory / 1e9
else:
    GPU_NAME, CAPABILITY, VRAM_GB = None, None, 0.0

print(f"python           : {platform.python_version()}")
print(f"torch            : {torch.__version__}")
print(f"cpu threads      : {torch.get_num_threads()}")
print(f"simulator device : {SIM_DEVICE}   <- every simulated rank lives here")
if HAS_CUDA:
    print(f"gpu              : {GPU_NAME}  ({VRAM_GB:.1f} GB, capability {CAPABILITY})  <- Cell 16 only")
else:
    print("gpu              : none. Cell 16 (real-memory cross-check) will be SKIPPED and")
    print("                   reported as skipped - every other number is unaffected.")

R = {"meta": {"notebook_version": NOTEBOOK_VERSION, "torch": torch.__version__,
              "python": platform.python_version(), "seed": SEED, "sim_device": SIM_DEVICE,
              "cuda": HAS_CUDA, "gpu": GPU_NAME, "cpu_threads": torch.get_num_threads()},
     "gates": {}}

# %% [markdown]
# ## Cell 2 — configuration, thresholds, and the lecture's numbers
#
# ### What this block does
# Declares every choice in one place: the demo model, the cluster, the optimizer, the
# pass/fail thresholds, and the numbers quoted on the lesson page that Cell 15 must reproduce.
#
# ### How it works
# - **The precision stack is the lecture's 16 bytes**: bf16 weights (2) + bf16 gradients (2)
#   + fp32 master (4) + Adam m (4) + Adam v (4). The simulator stores exactly these tensors.
# - **Micro-batch is fixed per rank**, so the global batch grows with world size — the way data
#   parallelism is actually run — and activation memory per rank stays constant across N.
# - **No gradient clipping and no weight decay.** Clipping needs a global norm, which is itself
#   a collective; leaving it out keeps the equality test about ZeRO alone. V4 also ran with
#   `weight_decay: 0.0`.
# - Thresholds are declared *before* any result exists, so none can be tuned to fit.
#
# ### Inputs / Outputs
# In: nothing. Out: `MCFG`, `SIM`, `TH`, `LECTURE_GIB`, `CARD_GIB`.
#
# ### What you should see
# A 4-layer, 128-wide GPT; world 32; the sweep 1..32; and the lecture's 30B table.

# %%
@dataclass(frozen=True)
class ModelCfg:
    n_layer:     int = 4
    n_head:      int = 4
    d_model:     int = 128
    block_size:  int = 64
    bias:        bool = False
    tie_weights: bool = False    # untied: a heavy table at BOTH ends of the flat parameter vector


@dataclass(frozen=True)
class SimCfg:
    world:         int = 32           # the brief's 32 virtual GPUs
    micro_batch:   int = 1            # sequences per rank per step
    steps:         int = 20
    lr:            float = 2e-3
    warmup:        int = 5
    beta1:         float = 0.9
    beta2:         float = 0.999
    eps:           float = 1e-8
    sweep_worlds:  tuple = (1, 2, 4, 8, 16, 32)
    control_world: int = 8            # negative controls run smaller; they only need to fire
    control_steps: int = 2
    gpu_world:     int = 32


@dataclass(frozen=True)
class Thresholds:
    fp64_rel_tol:           float = 1e-12   # Gate B: DP gradient vs big-batch gradient
    min_loss_drop:          float = 1.0     # nats; the equality claim must be about a model that learned
    gpu_persistent_rel_tol: float = 0.01    # Cell 16: allocator rounding is allowed, nothing else
    projection_gib_tol:     float = 0.05    # Gate F: lecture quotes one decimal place


MCFG, SIM, TH = ModelCfg(), SimCfg(), Thresholds()
WORK_DTYPE, MASTER_DTYPE = torch.bfloat16, torch.float32
STAGES = ("dp", "zero1", "zero2", "zero3")
LABEL = {"dp": "DP", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}

# The lesson page, section 7 / report Section 16: GiB per GPU for a 30B model, training state only.
V5_PARAMS = 30_000_000_000
CARD_GIB = 80e9 / 2**30                      # an "80 GB" card, as the OS reports it
LECTURE_GIB = {
    "dp":    {8: 447.0, 16: 447.0, 32: 447.0, 64: 447.0},
    "zero1": {8: 153.7, 16: 132.7, 32: 122.2, 64: 117.0},
    "zero2": {8: 104.8, 16: 80.3,  32: 68.1,  64: 62.0},
    "zero3": {8: 55.9,  16: 27.9,  32: 14.0,  64: 7.0},
}

print(f"model      : {MCFG.n_layer}L x {MCFG.n_head}H x {MCFG.d_model}D, block {MCFG.block_size}, "
      f"tied={MCFG.tie_weights}")
print(f"precision  : weights+grads {WORK_DTYPE}, master+Adam {MASTER_DTYPE}  (2+2+4+4+4 = 16 B/weight)")
print(f"world      : {SIM.world} ranks, micro-batch {SIM.micro_batch} -> global batch {SIM.world*SIM.micro_batch}")
print(f"training   : {SIM.steps} steps, Adam lr {SIM.lr} (warmup {SIM.warmup}), betas ({SIM.beta1}, {SIM.beta2})")
print(f"sweep      : world sizes {SIM.sweep_worlds}")
print(f"card       : 80 GB = {CARD_GIB:.1f} GiB")
R["config"] = {"model": asdict(MCFG), "sim": asdict(SIM), "thresholds": asdict(TH),
               "work_dtype": str(WORK_DTYPE), "master_dtype": str(MASTER_DTYPE)}

# %% [markdown]
# ## Cell 3 — the corpus, and a vocabulary sized to it
#
# ### What this block does
# Loads Tiny Shakespeare, tokenises it with GPT-2's BPE, and **compacts the vocabulary to the
# token ids that actually occur**.
#
# ### How it works
# GPT-2 has 50,257 ids, but this corpus uses far fewer. An embedding row that is never looked
# up still costs 16 bytes per weight to train, and here that dead weight would swamp
# everything else. Remapping to the ids in use keeps the model small while leaving the
# embedding and output head still far larger than a transformer block — which is exactly
# the uneven-size property Cell 14 needs.
#
# ### Inputs / Outputs
# In: `data/tinyshakespeare.txt` (downloaded if missing). Out: `STREAM`, `V`, `get_batch`.
#
# ### What you should see
# About 338k tokens and a compact vocabulary of a little under 12k ids.

# %%
DATA_DIR, OUT_DIR = "data", "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
_CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_corpus():
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"  downloading corpus -> {path}")
        urllib.request.urlretrieve(_CORPUS_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


import tiktoken
_text = load_corpus()
_raw = tiktoken.get_encoding("gpt2").encode_ordinary(_text)
_used = sorted(set(_raw))
_remap = {tok: i for i, tok in enumerate(_used)}
STREAM = torch.tensor([_remap[t] for t in _raw], dtype=torch.long)
V = len(_used)


def get_batch(step, n_ranks, device="cpu"):
    """Rank r's micro-batch at a given step. Drawn from a generator seeded by the step alone,
    so every arrangement - and every world size - sees exactly the same sequences."""
    T, mb = MCFG.block_size, SIM.micro_batch
    g = torch.Generator().manual_seed(SEED * 100_003 + step)
    ix = torch.randint(len(STREAM) - T - 1, (n_ranks * mb,), generator=g)
    x = torch.stack([STREAM[i:i + T] for i in ix]).view(n_ranks, mb, T)
    y = torch.stack([STREAM[i + 1:i + 1 + T] for i in ix]).view(n_ranks, mb, T)
    return x.to(device), y.to(device)


print(f"corpus       : {len(_text):,} characters -> {len(STREAM):,} GPT-2 tokens")
print(f"vocabulary   : {V:,} ids in use, of 50,257  ({V/50257:.1%})")
R["data"] = {"characters": len(_text), "tokens": len(STREAM), "vocab": V}

# %% [markdown]
# ## Cell 4 — the demo model, and the flat parameter vector ZeRO slices
#
# ### What this block does
# Defines the Session 11 GPT (pre-norm blocks, causal attention, GELU MLP), shrunk, and lays
# its parameters end to end in one flat vector. That vector is the object every stage shards.
#
# ### How it works
# Two changes from Session 11, both deliberate:
#
# 1. **Weights are untied.** The token embedding sits at the front of the vector and the output
#    head at the back, so the model is heavy at both ends — the shape the lecture drew in
#    Section 15.
# 2. **The model is split into units** — embed, each block, head — and each unit can run on its
#    own. ZeRO-3 gathers one unit's weights at a time, and ZeRO-2 reduces one unit's gradients
#    at a time. The attention softmax and the loss are computed in fp32 even when everything
#    around them is bf16 (report Section 20: the ops where precision is nearly free).
#
# `GPT.forward` is literally the units run in sequence, so the big-batch reference in Cell 8
# and the simulator share one code path.
#
# ### Inputs / Outputs
# In: `MCFG`, `V`. Out: `GPT`, `LAYOUT`, `THETA0` (initial fp32 weights, flat).
#
# ### What you should see
# A unit table in which `embed` and `head` are each several times larger than a block.

# %%
def _hp(dtype):
    """The dtype an op 'kept at high precision' runs in: fp32 for half types, else unchanged."""
    return torch.promote_types(dtype, torch.float32)


class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.d_model % c.n_head == 0
        self.n_head, self.d_model = c.n_head, c.d_model
        self.c_attn = nn.Linear(c.d_model, 3 * c.d_model, bias=c.bias)
        self.c_proj = nn.Linear(c.d_model, c.d_model, bias=c.bias)
        self.register_buffer("mask", torch.tril(torch.ones(c.block_size, c.block_size, dtype=torch.bool))
                             .view(1, 1, c.block_size, c.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.to(_hp(att.dtype)).masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1).to(x.dtype)      # softmax in fp32, see markdown
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c.d_model, 4 * c.d_model, bias=c.bias)
        self.c_proj = nn.Linear(4 * c.d_model, c.d_model, bias=c.bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(c.d_model), CausalSelfAttention(c)
        self.ln_2, self.mlp = nn.LayerNorm(c.d_model), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, c, vocab_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, c.d_model)
        self.wpe = nn.Embedding(c.block_size, c.d_model)
        self.h = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, vocab_size, bias=False)
        if c.tie_weights:
            self.lm_head.weight = self.wte.weight
        self.n_units = c.n_layer + 2
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def unit(self, u, x, targets=None):
        """Run unit u alone. 0 = embed, 1..L = blocks, L+1 = head + loss."""
        if u == 0:
            return self.wte(x) + self.wpe(torch.arange(x.shape[-1], device=x.device))
        if u <= len(self.h):
            return self.h[u - 1](x)
        logits = self.lm_head(self.ln_f(x))
        logits = logits.to(_hp(logits.dtype))           # loss in fp32
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def forward(self, idx, targets):
        x = idx
        for u in range(self.n_units - 1):
            x = self.unit(u, x)
        return self.unit(self.n_units - 1, x, targets)


class Layout:
    """The flat parameter vector: every parameter end to end, in unit order.

    Each unit records [start, end) in that vector and, for every parameter in it,
    (name, shape, start, numel). A rank's shard, a ZeRO-2 bucket and a ZeRO-3 gather are all
    just ranges in this one coordinate system."""
    def __init__(self, net):
        names = {id(p): n for n, p in net.named_parameters()}
        groups = [("embed", [net.wte.weight, net.wpe.weight])]
        groups += [(f"block{i}", list(b.parameters())) for i, b in enumerate(net.h)]
        groups += [("head", [net.ln_f.weight, net.ln_f.bias, net.lm_head.weight])]
        self.units, off = [], 0
        for uname, ps in groups:
            start, entries = off, []
            for p in ps:
                entries.append((names[id(p)], tuple(p.shape), off, p.numel()))
                off += p.numel()
            self.units.append({"name": uname, "start": start, "end": off, "params": entries})
        self.numel = off
        assert off == sum(p.numel() for p in net.parameters()), "a parameter is missing from the layout"

    def tensor_cuts(self):
        return sorted({e[2] for u in self.units for e in u["params"]} | {self.numel})

    def params(self, net, u=None):
        # Cached ON the module, not by id(net): a deleted module's id is reused by the next one,
        # and an id-keyed cache then hands back a dead module's parameters.
        if "_layout_named" not in net.__dict__:
            net.__dict__["_layout_named"] = dict(net.named_parameters())
        named, units = net.__dict__["_layout_named"], (self.units if u is None else [self.units[u]])
        return [named[e[0]] for unit in units for e in unit["params"]]

    def bind(self, net, u, buf, base):
        """Point unit u's parameters at views into `buf`, whose index 0 is flat position `base`.
        No copy: the rank's own tensor IS the parameter while the unit runs."""
        ps = self.params(net, u)
        for p, (_, shape, start, n) in zip(ps, self.units[u]["params"]):
            p.data = buf[start - base:start - base + n].view(shape)
        return ps

    def flatten(self, net):
        return torch.cat([p.detach().reshape(-1) for p in self.params(net)])


torch.manual_seed(SEED)
_ref = GPT(MCFG, V)
LAYOUT = Layout(_ref)
THETA0 = LAYOUT.flatten(_ref).to(MASTER_DTYPE).clone()
PSI = LAYOUT.numel
del _ref

print(f"parameters Ψ : {PSI:,}   (P = Ψ x 2 bytes = {2*PSI/2**20:.2f} MiB in bf16)")
print(f"{'unit':<8} {'flat range':>22} {'params':>11} {'share':>7}")
for u in LAYOUT.units:
    n = u["end"] - u["start"]
    print(f"{u['name']:<8} {u['start']:>10,}-{u['end']:<11,} {n:>11,} {n/PSI:>7.1%}")
_blk = LAYOUT.units[1]["end"] - LAYOUT.units[1]["start"]
_heavy = max(LAYOUT.units[0]["end"] - LAYOUT.units[0]["start"], LAYOUT.units[-1]["end"] - LAYOUT.units[-1]["start"])
print(f"\nlargest end unit is {_heavy/_blk:.1f}x a transformer block"
      + ("  -> the vector is heavy at its ends" if _heavy > 2 * _blk else ""))
R["model"] = {"params": PSI, "P_bytes": 2 * PSI,
              "units": [{"name": u["name"], "start": u["start"], "end": u["end"]} for u in LAYOUT.units],
              "n_tensors": len(LAYOUT.tensor_cuts()) - 1}

# %% [markdown]
# ## Cell 5 — the cluster primitives: shards, ranks, and a ring
#
# ### What this block does
# Three small pieces that everything later is made of.
#
# ### How it works
# - **`partition`** cuts the flat vector into one contiguous range per rank. `flat` cuts at
#   equal element counts, the way DeepSpeed partitions a flattened buffer. `whole-tensor`
#   only cuts at tensor boundaries, so a parameter is never split — Cell 14 shows what that
#   costs.
# - **`Rank`** is a dictionary of named tensors plus a memory ledger. `held()` walks the
#   dictionary and sums `numel × element_size` by category, and every sample updates the
#   rank's peak. Tensors that autograd saves for the backward pass are counted too, through
#   `saved_tensors_hooks` in Cell 7.
# - **`Ring`** implements the two collectives that everything else is built from, one hop at
#   a time:
#   - **reduce-scatter**: chunk *c* starts at rank *c+1* and travels right, and each rank it
#     reaches adds its own values. After N−1 hops it arrives complete at rank *c*, its owner.
#   - **all-gather**: each rank's chunk is passed right N−1 times until everyone has
#     everything.
#
#   **all-reduce is not a third primitive.** It is reduce-scatter followed by all-gather,
#   and Gate A checks exactly that. Every hop adds `numel × element_size` to the sender's
#   counter, tagged with a purpose.
#
# One detail makes Gate C possible. Within a chunk, the order in which gradients are added
# depends only on the chunk's **owner**, not on how many elements are sent together. So
# reducing the whole gradient vector in one call (DP, ZeRO-1) or one unit at a time (ZeRO-2,
# ZeRO-3) adds the same numbers in the same order.
#
# ### Inputs / Outputs
# In: nothing. Out: `partition`, `Rank`, `Ring`, `CATEGORIES`.
#
# ### What you should see
# Nothing printed. These are definitions.

# %%
CATEGORIES = ("weights", "grads", "master", "adam_m", "adam_v", "activations", "gathered")
PERSISTENT = ("weights", "grads", "master", "adam_m", "adam_v")


def partition(numel, n, how="flat", layout=None):
    """One contiguous [lo, hi) range of the flat vector per rank."""
    if how == "flat":
        base, rem = divmod(numel, n)
        sizes = [base + (1 if r < rem else 0) for r in range(n)]
        bounds = [0]
        for s in sizes:
            bounds.append(bounds[-1] + s)
    elif how == "whole-tensor":
        cuts = layout.tensor_cuts()
        bounds = [0]
        for k in range(1, n):
            target = k * numel / n
            nearest = min(cuts, key=lambda c: (abs(c - target), c))
            bounds.append(max(nearest, bounds[-1]))
        bounds.append(numel)
    else:
        raise ValueError(how)
    return [(bounds[r], bounds[r + 1]) for r in range(n)]


class Rank:
    def __init__(self, idx, lo, hi):
        self.idx, self.lo, self.hi = idx, lo, hi
        self.t = {}                 # name -> tensor; the name's prefix is its category
        self.saved = 0              # bytes autograd is holding for this rank right now
        self.peak, self.peak_at, self.peak_breakdown = 0, None, None
        self.peak_activations = 0

    @property
    def shard(self):
        return self.hi - self.lo

    def put(self, name, tensor):
        self.t[name] = tensor

    def free(self, name):
        self.t.pop(name, None)

    def held(self):
        out = dict.fromkeys(CATEGORIES, 0)
        for name, x in self.t.items():
            out[name.split("/")[0]] += x.numel() * x.element_size()
        out["activations"] += self.saved
        return out

    def storage_ptrs(self):
        ptrs = set()
        for x in self.t.values():
            try:
                ptrs.add(x.untyped_storage().data_ptr())
            except Exception:
                pass
        return ptrs


class Ring:
    """Ring collectives over ownership chunks, counting bytes per hop per sender."""
    def __init__(self, bounds, elsize, bug=None):
        self.bounds, self.n, self.elsize, self.bug = bounds, len(bounds), elsize, bug
        self.sent = [defaultdict(int) for _ in bounds]
        self.calls = defaultdict(int)
        self.seconds = 0.0

    def spans(self, a, b):
        """Each rank's chunk of the range [a, b), relative to a. Empty chunks are (0, 0)."""
        out = []
        for lo, hi in self.bounds:
            x, y = max(lo, a), min(hi, b)
            out.append((x - a, y - a) if y > x else (0, 0))
        return out

    def reduce_scatter(self, bufs, a, b, purpose):
        """bufs[r] covers [a, b) on rank r. Returns acc[r]: the AVERAGE of everyone's chunk r,
        delivered to rank r. The inputs are not modified."""
        t0, n, sp = time.perf_counter(), self.n, self.spans(a, b)
        acc = [bufs[(c + 1) % n][sp[c][0]:sp[c][1]].clone() for c in range(n)]
        for s in range(n - 1):                         # hop s: every rank sends one chunk
            for c in range(n):
                lo, hi = sp[c]
                if hi == lo:
                    continue
                holder, receiver = (c + 1 + s) % n, (c + 2 + s) % n
                self.sent[holder][purpose] += (hi - lo) * self.elsize
                if self.bug == "drop_contribution" and receiver == n - 1:
                    continue                           # PLANTED BUG (Cell 10 only)
                acc[c] = acc[c] + bufs[receiver][lo:hi]
        divisor = n * n if self.bug == "double_average" else n   # PLANTED BUG (Cell 10 only)
        acc = [x / divisor for x in acc]
        self.calls["reduce-scatter"] += 1
        self.seconds += time.perf_counter() - t0
        return acc

    def all_gather(self, chunks, outs, a, b, purpose):
        """chunks[r] is rank r's chunk of [a, b); outs[r] covers [a, b) on rank r and ends up
        holding every rank's chunk."""
        t0, n, sp = time.perf_counter(), self.n, self.spans(a, b)
        for r in range(n):
            lo, hi = sp[r]
            if hi > lo and outs[r][lo:hi].data_ptr() != chunks[r].data_ptr():
                outs[r][lo:hi].copy_(chunks[r])
        for s in range(n - 1):
            for r in range(n):
                lo, hi = sp[(r - s) % n]               # the chunk rank r received last hop
                if hi == lo:
                    continue
                outs[(r + 1) % n][lo:hi].copy_(outs[r][lo:hi])
                self.sent[r][purpose] += (hi - lo) * self.elsize
        self.calls["all-gather"] += 1
        self.seconds += time.perf_counter() - t0

# %% [markdown]
# ## Cell 6 — GATE A: the ring computes what an all-reduce computes
#
# ### What this block does
# Checks the collectives before anything is built on them. The three checks are the three
# claims from report Sections 9 and 10.
#
# ### How it works
# Each rank gets a random vector of **small integers stored as floats**. Adding integers is
# exact in any order, so the ring's result can be compared **bit for bit** against a naive
# `sum / N`, with no tolerance. That comparison is only fair on integer data, which is why
# the data is integers.
#
# 1. **reduce-scatter + all-gather == all-reduce**, bit for bit, at every position on every
#    rank.
# 2. **The intermediate slice is the answer.** After reduce-scatter alone, rank r already holds
#    exactly its slice of the final average. Data parallelism throws that slice away; ZeRO-1
#    and ZeRO-2 keep it.
# 3. **Bytes per rank equal 2(N−1)/N · P** with even chunks. With uneven chunks, each rank sends
#    everything except one chunk per phase, and the counter must match that exactly.
#
# ### Inputs / Outputs
# In: `Ring`, `partition`. Out: `R["gates"]["A_collectives"]`.
#
# ### What you should see
# Three `GATE PASS` lines for N=32 with even chunks and three for N=5 with uneven chunks.

# %%
class GateError(AssertionError):
    pass


def gate(ok, msg):
    if not ok:
        raise GateError(msg)
    print(f"  GATE PASS  {msg}")


def _gate_a(n, numel):
    g = torch.Generator().manual_seed(7)
    bufs = [torch.randint(-8, 9, (numel,), generator=g).float() for _ in range(n)]
    naive = torch.stack(bufs).sum(0) / n
    ring = Ring(partition(numel, n), elsize=4)
    acc = ring.reduce_scatter(bufs, 0, numel, "rs")
    slice_ok = all(torch.equal(acc[r], naive[lo:hi]) for r, (lo, hi) in enumerate(ring.bounds))
    outs = [x.clone() for x in bufs]
    for r, (lo, hi) in enumerate(ring.bounds):
        outs[r][lo:hi].copy_(acc[r])
    ring.all_gather([outs[r][lo:hi] for r, (lo, hi) in enumerate(ring.bounds)], outs, 0, numel, "ag")
    ar_ok = all(torch.equal(o, naive) for o in outs)
    sizes = [hi - lo for lo, hi in ring.bounds]
    exp = [4 * (numel - sizes[r]) + 4 * (numel - sizes[(r + 1) % n]) for r in range(n)]
    got = [ring.sent[r]["rs"] + ring.sent[r]["ag"] for r in range(n)]
    even = len(set(sizes)) == 1
    formula = 2 * (n - 1) / n * numel * 4
    print(f"N={n:>2}  Ψ={numel:,}  chunks {'even' if even else 'UNEVEN'} {min(sizes)}..{max(sizes)}")
    gate(ar_ok, f"N={n}: reduce-scatter + all-gather == naive all-reduce, bit for bit on every rank")
    gate(slice_ok, f"N={n}: after reduce-scatter alone, rank r holds exactly its slice of the answer")
    gate(got == exp, f"N={n}: bytes sent per rank == (Ψ - own chunk) + (Ψ - next chunk), exact")
    if even:
        gate(all(x == formula for x in got), f"N={n}: every rank sent 2(N-1)/N x P = {formula/(numel*4):.4f} P")
    return {"n": n, "numel": numel, "allreduce_exact": ar_ok, "slice_exact": slice_ok,
            "bytes_exact": got == exp, "bytes_per_rank_over_P": [x / (4 * numel) for x in got]}


R["gates"]["A_collectives"] = [_gate_a(32, 32 * 4096), _gate_a(5, 1003)]

# %% [markdown]
# ## Cell 7 — four arrangements, one step function
#
# ### What this block does
# Defines `Cluster`: N ranks, one stage (`dp`, `zero1`, `zero2`, `zero3`), and a `step()` that
# runs forward, backward, gradient synchronisation and Adam, all ranks in lockstep.
#
# ### How it works — what each rank holds
#
# | | bf16 weights | bf16 grads | fp32 master + Adam m + v | per weight |
# |---|---|---|---|---|
# | **DP** | full | full | full | 16 |
# | **ZeRO-1** | full | full | **own shard** | 4 + 12/N |
# | **ZeRO-2** | full | **own shard** | own shard | 2 + 14/N |
# | **ZeRO-3** | **own shard** | own shard | own shard | 16/N |
#
# ### How it works — what each rank does in a step
#
# - **Forward, unit by unit.** Every rank runs unit *u* on its own micro-batch and keeps only
#   the unit's *output*. Under **ZeRO-3** the ranks first **all-gather** that unit's weights into
#   a temporary buffer, run it, and free the buffer.
# - **Backward, units in reverse.** Each rank recomputes unit *u* from its stored input with
#   the graph on, then back-propagates into it. That is activation checkpointing at unit
#   boundaries; every arrangement does it identically, and it is what lets 32 simulated ranks
#   move through backward in lockstep. **ZeRO-3 all-gathers the weights again** here.
#   **ZeRO-2 and ZeRO-3** put the unit's gradients into a bucket and **reduce-scatter** it
#   immediately, so each rank keeps only its own slice and frees the rest.
# - **Gradient sync after backward.** **DP**: all-reduce the full gradient vector.
#   **ZeRO-1**: reduce-scatter it, each rank keeping its slice.
# - **Adam.** Each rank updates the master weights it owns and re-makes its bf16 working copy
#   from them. **ZeRO-1 and ZeRO-2** then **all-gather** the updated weights so every rank's full
#   copy is current again. ZeRO-3 has nothing to gather, because its bf16 copy only ever was
#   a shard.
#
# DP's all-reduce uses the same chunk boundaries as the ZeRO shards. A real ring may cut its
# chunks anywhere; using the same cut is what makes the comparison bit-exact.
#
# ### Memory accounting
# `sample()` walks every rank's tensors at fixed points in the step. During each recompute,
# a `saved_tensors_hooks` pack hook counts the tensors autograd saves, skipping anything the
# rank already holds (its weights, its stored activations) so nothing is counted twice.
# Tensors in flight inside a collective are not counted — a real library sends from a fixed
# communication buffer.
#
# ### Inputs / Outputs
# In: `LAYOUT`, `THETA0`, `Ring`, `Rank`. Out: `Cluster`, `adam_`, `make_compute_net`, `lr_at`.
#
# ### What you should see
# Nothing printed. These are definitions.

# %%
def adam_(p, m, v, g16, lr, t):
    """Adam on one contiguous range, written with one arithmetic operation per kernel call.

    Not the usual `m.mul_(b1).add_(g, alpha=1-b1)`. Cell 8b measures why: the scaled add
    `add_(g, alpha=...)` gives different last bits on a shard than on the full vector, so a
    ZeRO rank's update would not match DP's. A separate multiply and a separate add each
    give the same result for an element wherever it sits in the tensor."""
    if p.numel() == 0:
        return
    b1, b2 = SIM.beta1, SIM.beta2
    g = g16.to(MASTER_DTYPE)
    m.mul_(b1)
    m.add_(g * (1 - b1))
    v.mul_(b2)
    v.add_((g * g) * (1 - b2))
    denom = (v / (1 - b2 ** t)).sqrt_().add_(SIM.eps)
    upd = m / denom
    upd.mul_(lr / (1 - b1 ** t))
    p.sub_(upd)


def lr_at(t):
    return SIM.lr * min(1.0, t / SIM.warmup)


def make_compute_net(device, dtype):
    """One module whose parameters are re-pointed at a rank's tensors before each unit runs.
    Its own initial storage is dropped immediately, so it holds no parameter memory itself."""
    net = GPT(MCFG, V).to(device=device, dtype=dtype)
    for p in net.parameters():
        p.data = torch.empty(0, dtype=dtype, device=device)
    return net


class Cluster:
    def __init__(self, stage, n, net, *, how="flat", device=SIM_DEVICE, work=WORK_DTYPE,
                 bug=None, trace_rank=0):
        assert stage in STAGES
        self.stage, self.n, self.net, self.device, self.work, self.how = stage, n, net, device, work, how
        self.bounds = partition(PSI, n, how, LAYOUT)
        self.ranks = [Rank(r, lo, hi) for r, (lo, hi) in enumerate(self.bounds)]
        self.elsize = torch.tensor([], dtype=work).element_size()
        self.ring = Ring(self.bounds, self.elsize, bug)
        self.trace_rank, self.timeline, self.t_step = trace_rank, [], 0
        self.seconds = defaultdict(float)
        self.execs = defaultdict(int)
        self.steady = {}
        full = THETA0.to(device)
        for r in self.ranks:
            sl = slice(r.lo, r.hi)
            if stage in ("dp", "zero1", "zero2"):
                r.put("weights/full", full.to(work))
            else:
                r.put("weights/shard", full[sl].to(work))
            if stage in ("dp", "zero1"):
                r.put("grads/full", torch.zeros(PSI, dtype=work, device=device))
            else:
                r.put("grads/shard", torch.zeros(r.shard, dtype=work, device=device))
            src = full if stage == "dp" else full[sl]
            r.put("master/" + ("full" if stage == "dp" else "shard"), src.clone())
            r.put("adam_m/state", torch.zeros_like(src))
            r.put("adam_v/state", torch.zeros_like(src))
        del full
        self.sample("init")

    # ---------------------------------------------------------------- ledger
    def _record(self, r, label):
        h = r.held()
        tot = sum(h.values())
        if tot > r.peak:
            r.peak, r.peak_at, r.peak_breakdown = tot, f"step {self.t_step}: {label}", h
        r.peak_activations = max(r.peak_activations, h["activations"])
        if r.idx == self.trace_rank:
            self.timeline.append({"step": self.t_step, "label": label, **h})

    def sample(self, label):
        for r in self.ranks:
            self._record(r, label)

    # ---------------------------------------------------------------- helpers
    def _weights_for(self, r, u):
        if self.stage == "zero3":
            return r.t["gathered/unit"], LAYOUT.units[u]["start"]
        return r.t["weights/full"], 0

    def _gather_unit(self, u, purpose):
        us, ue = LAYOUT.units[u]["start"], LAYOUT.units[u]["end"]
        chunks, outs = [], []
        for r in self.ranks:
            lo, hi = max(r.lo, us), min(r.hi, ue)
            w = r.t["weights/shard"]
            chunks.append(w[lo - r.lo:hi - r.lo] if hi > lo else w[:0])
            buf = torch.empty(ue - us, dtype=self.work, device=self.device)
            r.put("gathered/unit", buf)
            outs.append(buf)
        self.ring.all_gather(chunks, outs, us, ue, purpose)

    def _free_gathered(self):
        for r in self.ranks:
            r.free("gathered/unit")

    # ---------------------------------------------------------------- one step
    def step(self, X, Y, t):
        self.t_step = t
        st, n, U = self.stage, self.n, len(LAYOUT.units)
        losses = [0.0] * n
        for r in self.ranks:
            r.put("activations/tokens", X[r.idx])
            r.put("activations/targets", Y[r.idx])

        # ---- forward: keep each unit's output, nothing else
        for u, unit in enumerate(LAYOUT.units):
            if st == "zero3":
                self._gather_unit(u, "param all-gather (fwd)")
            t0 = time.perf_counter()
            for r in self.ranks:
                src, base = self._weights_for(r, u)
                LAYOUT.bind(self.net, u, src, base)
                inp = r.t["activations/tokens"] if u == 0 else r.t[f"activations/in{u}"]
                with torch.no_grad():
                    out = self.net.unit(u, inp, r.t["activations/targets"] if u == U - 1 else None)
                if u < U - 1:
                    r.put(f"activations/in{u + 1}", out)
                else:
                    losses[r.idx] = float(out)
                self.execs["unit forward"] += 1
            self.seconds["compute"] += time.perf_counter() - t0
            self.sample(f"fwd {unit['name']}")
            if st == "zero3":
                self._free_gathered()

        # ---- backward: recompute each unit with the graph on, then back-propagate into it
        for u in reversed(range(U)):
            unit = LAYOUT.units[u]
            us, ue = unit["start"], unit["end"]
            if st == "zero3":
                self._gather_unit(u, "param all-gather (bwd)")
            t0 = time.perf_counter()
            for r in self.ranks:
                src, base = self._weights_for(r, u)
                params = LAYOUT.bind(self.net, u, src, base)
                if u == 0:
                    inp, wrt = r.t["activations/tokens"], params
                else:
                    inp = r.t[f"activations/in{u}"].detach().requires_grad_(True)
                    wrt = [inp] + params
                exclude, seen = r.storage_ptrs(), {}

                def _pack(x, exclude=exclude, seen=seen):
                    try:
                        key = x.untyped_storage().data_ptr()
                        if key not in exclude and key not in seen:
                            seen[key] = x.untyped_storage().nbytes()
                    except Exception:
                        pass
                    return x

                with torch.autograd.graph.saved_tensors_hooks(_pack, lambda x: x):
                    out = self.net.unit(u, inp, r.t["activations/targets"] if u == U - 1 else None)
                r.saved = sum(seen.values())
                self._record(r, f"bwd {unit['name']}")          # graph alive: the activation peak
                grads = torch.autograd.grad(out, wrt, grad_outputs=None if u == U - 1
                                            else r.t["activations/grad_in"])
                del out
                r.saved = 0
                pgrads = grads if u == 0 else grads[1:]
                flat = torch.cat([g.reshape(-1) for g in pgrads])
                if st in ("dp", "zero1"):
                    r.t["grads/full"][us:ue].copy_(flat)
                else:
                    r.put("grads/bucket", flat)
                del flat, pgrads
                if u == U - 1:
                    r.free("activations/targets")
                if u > 0:
                    r.free(f"activations/in{u}")
                    r.put("activations/grad_in", grads[0])
                else:
                    r.free("activations/tokens")
                    r.free("activations/grad_in")
                del grads
                self.execs["unit recompute"] += 1
                self.execs["unit backward"] += 1
            self.seconds["compute"] += time.perf_counter() - t0

            if st in ("zero2", "zero3"):
                self.sample(f"bucket {unit['name']}")
                acc = self.ring.reduce_scatter([r.t["grads/bucket"] for r in self.ranks], us, ue,
                                               "grad reduce-scatter")
                for r in self.ranks:
                    lo, hi = max(r.lo, us), min(r.hi, ue)
                    if hi > lo:
                        r.t["grads/shard"][lo - r.lo:hi - r.lo].copy_(acc[r.idx])
                    r.free("grads/bucket")
                del acc
            if st == "zero3":
                self._free_gathered()

        # ---- gradient synchronisation for the stages that waited until backward finished
        if st in ("dp", "zero1"):
            self.sample("grads ready")
            bufs = [r.t["grads/full"] for r in self.ranks]
            acc = self.ring.reduce_scatter(bufs, 0, PSI, "grad reduce-scatter")
            for r in self.ranks:
                if r.hi > r.lo:
                    r.t["grads/full"][r.lo:r.hi].copy_(acc[r.idx])
            del acc
            if st == "dp":      # second half of the all-reduce: everyone gets every average
                self.ring.all_gather([b[lo:hi] for b, (lo, hi) in zip(bufs, self.bounds)], bufs, 0, PSI,
                                     "grad all-gather")

        # ---- Adam on what each rank owns
        t0, lr = time.perf_counter(), lr_at(t)
        for r in self.ranks:
            opt = [r.t[k] for k in ("master/full" if st == "dp" else "master/shard", "adam_m/state", "adam_v/state")]
            if st == "dp":
                g = r.t["grads/full"]
            elif st == "zero1":
                g = r.t["grads/full"][r.lo:r.hi]
            else:
                g = r.t["grads/shard"]
            adam_(*opt, g, lr, t)
            w16 = opt[0].to(self.work)
            if st == "dp":
                r.t["weights/full"].copy_(w16)
            elif st in ("zero1", "zero2"):
                r.t["weights/full"][r.lo:r.hi].copy_(w16)
            else:
                r.t["weights/shard"].copy_(w16)
            self.execs["adam elements"] += opt[0].numel()
        self.seconds["optimizer"] += time.perf_counter() - t0

        if st in ("zero1", "zero2"):
            outs = [r.t["weights/full"] for r in self.ranks]
            self.ring.all_gather([o[lo:hi] for o, (lo, hi) in zip(outs, self.bounds)], outs, 0, PSI,
                                 "param all-gather (sync)")
        self.sample("step done")
        self.steady[t] = [r.held() for r in self.ranks]
        return sum(losses) / n

    # ---------------------------------------------------------------- read-outs
    def master_full(self):
        if self.stage == "dp":
            return self.ranks[0].t["master/full"].clone()
        return torch.cat([r.t["master/shard"] for r in self.ranks])

    def weights_full(self):
        if self.stage == "zero3":
            return torch.cat([r.t["weights/shard"] for r in self.ranks])
        return self.ranks[0].t["weights/full"].clone()

    def stats(self, steps):
        last = self.steady[max(self.steady)]
        first = self.steady[min(self.steady)]
        ranks = []
        for r, h in zip(self.ranks, last):
            ranks.append({"rank": r.idx, "lo": r.lo, "hi": r.hi, "shard": r.shard,
                          "steady": h, "persistent": sum(h[k] for k in PERSISTENT),
                          "transient_at_step_end": h["activations"] + h["gathered"],
                          "peak": r.peak, "peak_at": r.peak_at, "peak_breakdown": r.peak_breakdown,
                          "peak_activations": r.peak_activations,
                          "sent_per_step": {k: v / steps for k, v in self.ring.sent[r.idx].items()},
                          "sent_total_per_step": sum(self.ring.sent[r.idx].values()) / steps})
        return {"stage": self.stage, "n": self.n, "partition": self.how, "steps": steps,
                "ranks": ranks, "no_leak": first == last,
                "timeline_step1": [e for e in self.timeline if e["step"] == 1],
                "calls_per_step": {k: v / steps for k, v in self.ring.calls.items()},
                "execs_per_rank_per_step": {k: v / (steps * self.n) for k, v in self.execs.items()},
                "sim_seconds_per_step": {"compute": self.seconds["compute"] / steps,
                                         "collectives": self.ring.seconds / steps,
                                         "optimizer": self.seconds["optimizer"] / steps}}

# %% [markdown]
# ## Cell 8 — GATE B: data parallelism *is* one GPU with an N-times larger batch
#
# ### What this block does
# Tests the claim from report Section 7 that lets ZeRO exist at all: averaging N ranks'
# gradients gives the gradient one GPU would compute on the whole global batch —
# **exactly, not approximately**.
#
# ### How it works
# One model in **fp64**, the same initial weights, the step-1 batch of 32 sequences. First the
# gradient of the loss over all 32 sequences at once, then the mean of 32 separate
# one-sequence gradients. Each micro-batch has the same number of tokens, so the mean of the
# per-rank losses *is* the global loss, and the gradient of a mean is the mean of the
# gradients. fp64 is used because the claim is mathematical: any difference left should be
# round-off at ~1e-16, far below the declared tolerance.
#
# ### Inputs / Outputs
# In: `THETA0`, `get_batch`. Out: `R["gates"]["B_dp_equals_big_batch"]`.
#
# ### What you should see
# A relative difference around 1e-15, and `GATE PASS`.

# %%
def _gate_b():
    net = GPT(MCFG, V).double()
    theta = THETA0.double()
    for u in range(len(LAYOUT.units)):
        LAYOUT.bind(net, u, theta.clone(), 0)
    params = LAYOUT.params(net)
    X, Y = get_batch(1, SIM.world)
    T = MCFG.block_size
    g_big = torch.autograd.grad(net(X.reshape(-1, T), Y.reshape(-1, T)), params)
    g_big = torch.cat([g.reshape(-1) for g in g_big])
    g_sum = torch.zeros_like(g_big)
    for r in range(SIM.world):
        gr = torch.autograd.grad(net(X[r], Y[r]), params)
        g_sum += torch.cat([g.reshape(-1) for g in gr])
    g_avg = g_sum / SIM.world
    rel = float((g_big - g_avg).abs().max() / g_big.abs().max())
    print(f"one GPU, batch {SIM.world * SIM.micro_batch}   vs   mean of {SIM.world} ranks x batch {SIM.micro_batch}   (fp64)")
    print(f"  max |g_big - g_avg| / max |g_big| = {rel:.3e}   (tolerance {TH.fp64_rel_tol:.0e})")
    gate(rel <= TH.fp64_rel_tol, f"DP gradient == big-batch gradient to {rel:.1e} relative")
    return {"rel_diff": rel, "tolerance": TH.fp64_rel_tol, "passed": rel <= TH.fp64_rel_tol}


R["gates"]["B_dp_equals_big_batch"] = _gate_b()

# %% [markdown]
# ## Cell 8b — GATE B′: "elementwise" is not enough for bit-exact
#
# ### What this block does
# Explains, by measurement, why `adam_` in Cell 7 is written as separate multiplies and adds.
#
# ### How it works
# The first working draft of this notebook wrote Adam the usual way,
# `m.mul_(β1).add_(g, alpha=1-β1)`, and **Gate C fired**. After 20 steps ZeRO-1's fp32 master
# weights differed from DP's by 7.451e-09, while the losses and bf16 weights still matched.
#
# Every Adam operation is elementwise, so a shard's update *should* equal the same slice of a
# full-vector update. This cell tests that directly. It applies one Adam step to identical
# random state twice — to the whole vector, and shard by shard over the 32 flat shards — and
# counts elements whose bits differ. Three forms are tested: the usual fused form, this
# notebook's `adam_`, and the scaled add `add_(g, alpha=…)` on its own.
#
# The difference is not in the mathematics but in the kernel: the scaled add can produce
# different last bits for the same element depending on where the tensor it sits in begins
# and ends. The most likely mechanism is the scaled add being fused into a single
# multiply-add instruction on some code paths and not on others. That is an inference; what
# the cell *measures* is which forms disagree.
#
# The consequence for ZeRO in general: "identical to data parallelism" is a guarantee about
# the order of arithmetic. An optimizer kernel that handles a shard differently from a full
# tensor can break it without anything being wrong with the sharding.
#
# ### Inputs / Outputs
# In: `adam_`, `partition`. Out: `R["adam_kernel_diagnostic"]`, `R["gates"]["B2_adam_shard_exact"]`.
#
# ### What you should see
# `adam_` with 0 differing elements and `GATE PASS`. The usual form's count depends on the CPU,
# and the printout says which case this machine is.

# %%
def _adam_usual(p, m, v, g, lr, t):
    b1, b2 = SIM.beta1, SIM.beta2
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    denom = (v / (1 - b2 ** t)).sqrt_().add_(SIM.eps)
    p.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))


def _scaled_add_only(p, m, v, g, lr, t):
    m.add_(g, alpha=1 - SIM.beta1)


def _shard_vs_full(fn):
    gen = torch.Generator().manual_seed(11)
    state = [torch.randn(PSI, generator=gen) * 0.02, torch.randn(PSI, generator=gen) * 1e-3,
             torch.rand(PSI, generator=gen) * 1e-6]
    grad = torch.randn(PSI, generator=gen) * 1e-3
    full = [x.clone() for x in state]
    fn(*full, grad, SIM.lr, 3)
    shard = [x.clone() for x in state]
    for lo, hi in partition(PSI, SIM.world):
        fn(shard[0][lo:hi], shard[1][lo:hi], shard[2][lo:hi], grad[lo:hi], SIM.lr, 3)
    return {k: int((a != b).sum()) for k, a, b in zip(("master", "adam_m", "adam_v"), full, shard)}


_cap = torch.backends.cpu.get_cpu_capability() if hasattr(torch.backends.cpu, "get_cpu_capability") else "unknown"
_forms = {"usual fused form": _adam_usual, "adam_ (this notebook)": adam_, "add_(g, alpha) alone": _scaled_add_only}
diag = {"cpu_capability": _cap, "threads": torch.get_num_threads(), "forms": {}}
print(f"one Adam step on {PSI:,} elements: full vector vs {SIM.world} shards   (cpu: {_cap}, {torch.get_num_threads()} threads)")
for name, fn in _forms.items():
    d = _shard_vs_full(fn)
    diag["forms"][name] = d
    print(f"  {name:<24} elements whose bits differ:  master {d['master']:>4}   m {d['adam_m']:>4}   v {d['adam_v']:>4}")
_usual_diff = sum(diag["forms"]["usual fused form"].values())
_ours_diff = sum(diag["forms"]["adam_ (this notebook)"].values())
print()
if _usual_diff:
    print(f"On this CPU the usual form disagrees with itself across a shard boundary in {_usual_diff} places;")
    print("that is the 7.5e-9 Gate C caught in the first draft. The form in adam_ disagrees in "
          f"{_ours_diff}.")
else:
    print("On this CPU the usual form happens to agree as well. It did not on the AVX-512 machine this")
    print("notebook was developed on, which is why adam_ does not rely on it.")
gate(_ours_diff == 0, "adam_ updates a shard bit-identically to the same slice of a full-vector update")
R["adam_kernel_diagnostic"] = diag | {"first_draft_gate_c_master_diff": 7.451e-09}
R["gates"]["B2_adam_shard_exact"] = {"passed": _ours_diff == 0, "usual_form_differing": _usual_diff}

# %% [markdown]
# ## Cell 9 — GATE C: train all four arrangements on 32 ranks
#
# ### What this block does
# Trains the same model, from the same weights, on the same data, under DP, ZeRO-1, ZeRO-2
# and ZeRO-3 on 32 ranks. Then it demands **bit-identical** results.
#
# ### How it works
# After the run it compares:
# - the per-step loss,
# - the final fp32 master weights, reassembled from shards where they are sharded,
# - the final bf16 working weights,
#
# using `torch.equal`, with no tolerance. It also checks that DP's 32 full replicas are still
# identical to each other — the property that makes data parallelism work — and that the loss
# actually fell, so the equality is a claim about a model that learned.
#
# If any of these are not identical, that is a bug in the sharding — a wrong slice boundary, a
# double average, a dropped contribution — and the gate refuses to continue rather than
# loosening a tolerance.
#
# ### Inputs / Outputs
# In: `Cluster`, `get_batch`. Out: `TRAIN` (per-stage stats), `R["train"]`, `R["gates"]["C_identical"]`.
#
# ### What you should see
# Four loss columns that agree to every printed digit, `max |Δ| = 0` for every ZeRO stage, and
# `GATE PASS` lines.

# %%
def train(stage, n, steps, bug=None, how="flat"):
    cl = Cluster(stage, n, make_compute_net(SIM_DEVICE, WORK_DTYPE), how=how, bug=bug)
    losses = []
    for t in range(1, steps + 1):
        X, Y = get_batch(t, n)
        losses.append(cl.step(X, Y, t))
    return cl, losses


def assert_same_training(ref, test, name):
    (_, l_ref, m_ref, w_ref), (_, l_t, m_t, w_t) = ref, test
    dm = float((m_ref - m_t).abs().max())
    dw = float((w_ref.float() - w_t.float()).abs().max())
    dl = max(abs(a - b) for a, b in zip(l_ref, l_t))
    same = (l_ref == l_t) and torch.equal(m_ref, m_t) and torch.equal(w_ref, w_t)
    gate(same, f"{name}: losses, fp32 master and bf16 weights bit-identical to DP "
               f"(max |Δmaster| = {dm:.3e}, |Δw16| = {dw:.3e}, |Δloss| = {dl:.3e})")
    return {"identical": same, "max_abs_master": dm, "max_abs_w16": dw, "max_abs_loss": dl}


TRAIN, _runs = {}, {}
for st in STAGES:
    t0 = time.perf_counter()
    cl, losses = train(st, SIM.world, SIM.steps)
    extra = {}
    if st == "dp":
        w0 = cl.ranks[0].t["weights/full"]
        m0 = cl.ranks[0].t["master/full"]
        extra["replica_max_diff"] = max(max(float((r.t["weights/full"].float() - w0.float()).abs().max()),
                                            float((r.t["master/full"] - m0).abs().max()))
                                        for r in cl.ranks)
    _runs[st] = (st, losses, cl.master_full(), cl.weights_full())
    TRAIN[st] = cl.stats(SIM.steps) | extra | {"losses": losses, "wall_seconds": time.perf_counter() - t0}
    print(f"  trained {LABEL[st]:<7} {SIM.world} ranks x {SIM.steps} steps   "
          f"simulator time {TRAIN[st]['wall_seconds']:.1f}s")
    del cl

print(f"\n{'step':>4} " + " ".join(f"{LABEL[s]:>12}" for s in STAGES))
for i in range(SIM.steps):
    print(f"{i+1:>4} " + " ".join(f"{TRAIN[s]['losses'][i]:>12.6f}" for s in STAGES))

_l = TRAIN["dp"]["losses"]
_drop = _l[0] - _l[-1]
print(f"\nloss {_l[0]:.4f} -> {_l[-1]:.4f}: "
      + (f"fell {_drop:.3f} nats, so the model learned" if _drop > 0 else f"ROSE {-_drop:.3f} nats"))
print(f"DP replicas: largest difference between any rank's weights and rank 0's = {TRAIN['dp']['replica_max_diff']:.1e}")
print()
gate(_drop >= TH.min_loss_drop, f"loss fell {_drop:.3f} nats >= {TH.min_loss_drop} over {SIM.steps} steps")
gate(TRAIN["dp"]["replica_max_diff"] == 0.0, "all 32 DP replicas bit-identical after training")
R["gates"]["C_identical"] = {st: assert_same_training(_runs["dp"], _runs[st], LABEL[st]) for st in STAGES[1:]}
R["train"] = {"losses": {s: TRAIN[s]["losses"] for s in STAGES}, "loss_drop": _drop,
              "replica_max_diff": TRAIN["dp"]["replica_max_diff"],
              "wall_seconds": {s: TRAIN[s]["wall_seconds"] for s in STAGES}}
del _runs

# %% [markdown]
# ## Cell 10 — GATE C′: two planted bugs, and the gate has to catch both
#
# ### What this block does
# Shows that Gate C can fail. A gate that has never fired proves nothing.
#
# ### How it works
# Two bugs of the kind the report names, each planted in the ring and run as ZeRO-2 on 8 ranks
# for 2 steps against a clean DP run:
#
# - **`drop_contribution`**: one hop skips adding its rank's gradient. This is the "all-gather
#   silently dropped a rank's contribution" failure.
# - **`double_average`**: the reduced gradient is divided by N twice. This one is subtle. Adam
#   divides the first moment by the square root of the second, which cancels a uniform scale
#   for most weights. The exceptions are weights whose gradient, after the extra 1/N, is small
#   enough for ε or bf16 rounding to matter; those can move a lot. So the loss can barely
#   change while individual weights change noticeably. The cell prints both numbers.
#
# ### Inputs / Outputs
# In: `train`, `assert_same_training`. Out: `R["gates"]["C_negative_controls"]`.
#
# ### What you should see
# Both bugs reported as `caught`, with how far each moved the weights and the loss.

# %%
def _control(bug):
    cl, l = train("dp", SIM.control_world, SIM.control_steps)
    ref = ("dp", l, cl.master_full(), cl.weights_full())
    cl, l = train("zero2", SIM.control_world, SIM.control_steps, bug=bug)
    test = ("zero2", l, cl.master_full(), cl.weights_full())
    dm = float((ref[2] - test[2]).abs().max())
    dl = max(abs(a - b) for a, b in zip(ref[1], test[1]))
    try:
        assert_same_training(ref, test, f"ZeRO-2 with bug '{bug}'")
        caught, msg = False, "gate did NOT fire"
    except GateError as e:
        caught, msg = True, str(e)
    print(f"  {bug:<18} caught={caught}   max |Δmaster| = {dm:.3e}   max |Δloss| = {dl:.3e}")
    return {"bug": bug, "caught": caught, "max_abs_master": dm, "max_abs_loss": dl, "message": msg}


print(f"negative controls: ZeRO-2 on {SIM.control_world} ranks, {SIM.control_steps} steps, vs clean DP\n")
_controls = [_control("drop_contribution"), _control("double_average")]
R["gates"]["C_negative_controls"] = _controls
_small = min(_controls, key=lambda c: c["max_abs_master"])
print(f"\nThe smaller bug, '{_small['bug']}', moved the loss by {_small['max_abs_loss']:.1e} "
      f"while moving some weights by up to {_small['max_abs_master']:.1e} (the learning rate is {SIM.lr:.0e}).")
print(f"A {_small['max_abs_loss']:.0e} loss difference is invisible on a loss plot; the bit-exact gate caught it anyway."
      if _small["caught"] and _small["max_abs_loss"] < 1e-3 else
      "Its effect on the loss is large enough to see without the gate." if _small["caught"] else "")
for c in _controls:
    gate(c["caught"], f"negative control '{c['bug']}' was caught by the equality gate")

# %% [markdown]
# ## Cell 11 — GATE D: memory per rank on 32 ranks, counted vs modelled
#
# ### What this block does
# Reads the memory ledgers from the Cell 9 runs and compares what each rank **actually held**
# against the lecture's formula — to the byte, on every rank.
#
# ### How it works
# **Persistent state** is weights + grads + master + Adam m + v, read at the end of a step
# when nothing transient is alive. The modelled bytes use each rank's *exact* shard size
# `s_r`:
#
# - DP: `16Ψ`
# - ZeRO-1: `4Ψ + 12·s_r`
# - ZeRO-2: `2Ψ + 14·s_r`
# - ZeRO-3: `16·s_r`
#
# **Peak** adds what exists only during the step:
# - **activations**: boundary outputs plus whatever autograd saves during a recompute;
# - **gathered**: ZeRO-3's all-gathered weights for the unit currently running.
#
# Three things are checked:
# 1. counted persistent == modelled, exactly, on every rank of every arrangement;
# 2. nothing leaks: the ledger after step 1 equals the ledger after the last step, and no
#    transient is still held when a step ends;
# 3. peak activation bytes are identical across arrangements — ZeRO does not touch the
#    compute, so it cannot touch activations.
#
# ### Inputs / Outputs
# In: `TRAIN`. Out: `R["memory_n32"]`, `R["gates"]["D_ledger"]`.
#
# ### What you should see
# A table falling roughly 16 → 4.4 → 2.4 → 0.5 bytes per weight, all gates passing, and a
# ZeRO-3 peak well above its persistent state. The last printed line says what fills that gap.

# %%
def modelled_persistent(stage, shard, numel=PSI):
    w = 2 * (numel if stage in ("dp", "zero1", "zero2") else shard)
    g = 2 * (numel if stage in ("dp", "zero1") else shard)
    o = 12 * (numel if stage == "dp" else shard)
    return w + g + o


def bytes_per_weight(stage, n):
    """The lecture's formula at even shards - the same function Cell 15 projects to 30B."""
    return modelled_persistent(stage, 1 / n, numel=1)


MiB = 2**20
mem32, ledger_ok = {}, True
print(f"{'':<8} {'persistent/rank':>16} {'B/weight':>9} {'modelled':>9} {'peak/rank':>11} "
      f"{'peak - persistent':>18} {'peak happens at':<28}")
for st in STAGES:
    rk = TRAIN[st]["ranks"]
    exact = all(r["persistent"] == modelled_persistent(st, r["shard"]) for r in rk)
    ledger_ok &= exact
    r0 = rk[0]
    mem32[st] = {
        "persistent_per_rank": [r["persistent"] for r in rk],
        "peak_per_rank": [r["peak"] for r in rk],
        "rank0_steady": r0["steady"], "rank0_peak_breakdown": r0["peak_breakdown"], "rank0_peak_at": r0["peak_at"],
        "bytes_per_weight_counted": r0["persistent"] / PSI, "bytes_per_weight_modelled": bytes_per_weight(st, SIM.world),
        "modelled_exact": exact, "no_leak": TRAIN[st]["no_leak"],
        "transient_at_step_end": max(r["transient_at_step_end"] for r in rk),
        "peak_activations_rank0": r0["peak_activations"],
    }
    m = mem32[st]
    print(f"{LABEL[st]:<8} {r0['persistent']/MiB:>12.3f} MiB {m['bytes_per_weight_counted']:>9.4f} "
          f"{m['bytes_per_weight_modelled']:>9.4f} {r0['peak']/MiB:>7.3f} MiB {(r0['peak']-r0['persistent'])/MiB:>14.3f} MiB "
          f"{r0['peak_at']:<28}")

print()
gate(ledger_ok, "counted persistent bytes == modelled bytes, exactly, on all 32 ranks of all four arrangements")
gate(all(mem32[s]["no_leak"] and mem32[s]["transient_at_step_end"] == 0 for s in STAGES),
     "no leak: ledger after step 1 == after the last step, and no transient survives a step")
_acts = {s: mem32[s]["peak_activations_rank0"] for s in STAGES}
gate(len(set(_acts.values())) == 1,
     f"peak activation bytes identical across arrangements ({_acts['dp']/MiB:.3f} MiB): ZeRO does not touch compute")
_z3 = mem32["zero3"]
_z3_ratio = _z3["peak_per_rank"][0] / _z3["persistent_per_rank"][0]
print(f"\nZeRO-3 rank 0: persistent {_z3['persistent_per_rank'][0]/MiB:.3f} MiB, peak {_z3['peak_per_rank'][0]/MiB:.3f} MiB "
      f"({_z3_ratio:.1f}x). At the peak it holds: "
      + ", ".join(f"{k} {v/MiB:.3f}" for k, v in _z3["rank0_peak_breakdown"].items() if v) + " MiB.")
R["memory_n32"] = mem32
R["gates"]["D_ledger"] = {"modelled_exact": ledger_ok, "no_leak": True, "activations_identical": True,
                          "activation_bytes": _acts["dp"]}

# %% [markdown]
# ## Cell 12 — GATE E: the world-size sweep, and bytes on the wire
#
# ### What this block does
# Runs one step of every arrangement at N = 1, 2, 4, 8, 16, 32 and records per-rank memory
# and bytes sent. This is the lecture's memory table grown from four points into a curve,
# measured on a real model, plus the communication pattern at every N.
#
# ### How it works
# From the ring's counters, per rank per step, in units of P = 2Ψ bytes:
#
# | | gradient sync | weight all-gather | total |
# |---|---|---|---|
# | DP | reduce-scatter + all-gather of grads | — | 2(N−1)/N P |
# | ZeRO-1 / ZeRO-2 | reduce-scatter of grads | once, after Adam | 2(N−1)/N P |
# | ZeRO-3 | reduce-scatter of grads | forward **and** backward | 3(N−1)/N P |
#
# Gate E checks, on every rank at every N:
# - every rank's counted bytes equal the exact ring expression;
# - ZeRO-1 and ZeRO-2 send **exactly** what DP sends;
# - ZeRO-3 sends exactly 1.5× DP when shards are even;
# - N = 1 sends nothing;
# - the memory ledger matches the model at every N.
#
# ### Inputs / Outputs
# In: `Cluster`. Out: `SWEEP`, `R["sweep"]`, `R["gates"]["E_comm"]`.
#
# ### What you should see
# ZeRO-3's memory halving at every doubling of N. ZeRO-1 flattening towards 4 bytes per
# weight. Communication rising from 0 at N=1 towards 2P or 3P, and never above.

# %%
SWEEP = {st: {} for st in STAGES}
comm_ok, sweep_ledger_ok = True, True
for n in SIM.sweep_worlds:
    X, Y = get_batch(1, n)
    for st in STAGES:
        cl = Cluster(st, n, make_compute_net(SIM_DEVICE, WORK_DTYPE))
        cl.step(X, Y, 1)
        s = cl.stats(1)
        del cl
        sizes = [r["shard"] for r in s["ranks"]]
        per_rank = []
        for r in s["ranks"]:
            i = r["rank"]
            rs = 2 * (PSI - sizes[i])
            ag = 2 * (PSI - sizes[(i + 1) % n])
            expect = rs + ag + (ag if st == "zero3" else 0)
            comm_ok &= (r["sent_total_per_step"] == expect)
            sweep_ledger_ok &= (r["persistent"] == modelled_persistent(st, r["shard"]))
            per_rank.append(r["sent_total_per_step"])
        SWEEP[st][n] = {"persistent_max": max(r["persistent"] for r in s["ranks"]),
                        "peak_max": max(r["peak"] for r in s["ranks"]),
                        "comm_bytes_rank0": s["ranks"][0]["sent_total_per_step"],
                        "comm_over_P_rank0": s["ranks"][0]["sent_total_per_step"] / (2 * PSI),
                        "comm_by_purpose_rank0": s["ranks"][0]["sent_per_step"],
                        "comm_per_rank": per_rank,
                        "calls": s["calls_per_step"]}

print(f"{'N':>3} | " + " ".join(f"{LABEL[s]+' mem':>12}" for s in STAGES) + " | " +
      " ".join(f"{LABEL[s]+' comm':>12}" for s in STAGES))
for n in SIM.sweep_worlds:
    print(f"{n:>3} | " + " ".join(f"{SWEEP[s][n]['persistent_max']/MiB:>8.3f} MiB" for s in STAGES) + " | " +
          " ".join(f"{SWEEP[s][n]['comm_over_P_rank0']:>10.4f} P" for s in STAGES))

print()
gate(comm_ok, "bytes sent == exact ring expression, every rank, every arrangement, every N")
gate(sweep_ledger_ok, "persistent bytes == modelled, every rank, every arrangement, every N")
same_12 = all(SWEEP[s][n]["comm_per_rank"] == SWEEP["dp"][n]["comm_per_rank"]
              for s in ("zero1", "zero2") for n in SIM.sweep_worlds)
gate(same_12, "ZeRO-1 and ZeRO-2 send exactly the bytes DP sends, rank for rank, at every N")
z3_ratio = SWEEP["zero3"][SIM.world]["comm_bytes_rank0"] / SWEEP["dp"][SIM.world]["comm_bytes_rank0"]
gate(z3_ratio == 1.5, f"ZeRO-3 sends {z3_ratio:.4f}x DP at N={SIM.world} (even shards)")
gate(all(SWEEP[s][1]["comm_bytes_rank0"] == 0 for s in STAGES), "N=1: no arrangement sends a byte")
_p32 = SWEEP["dp"][SIM.world]["comm_over_P_rank0"]
print(f"\nAt N={SIM.world}, DP sends {_p32:.4f} P = 2 x {SIM.world-1}/{SIM.world} P, and ZeRO-3 sends "
      f"{SWEEP['zero3'][SIM.world]['comm_over_P_rank0']:.4f} P. The lecture's '2P' and '3P' are these at N -> infinity.")
R["sweep"] = {st: {str(n): v for n, v in SWEEP[st].items()} for st in STAGES}
R["gates"]["E_comm"] = {"ring_exact": comm_ok, "ledger_exact_all_N": sweep_ledger_ok,
                        "zero12_equal_dp": same_12, "zero3_over_dp": z3_ratio}

# %% [markdown]
# ## Cell 13 — what "computation" changes, and what it does not
#
# ### What this block does
# Puts every per-step operation count for one rank at N=32 side by side.
#
# ### How it works
# There are four kinds of work a rank does in a step, and ZeRO treats them differently:
#
# - **Model arithmetic** — unit forwards, recomputes and backwards. Identical in every
#   arrangement. ZeRO never changes what a rank computes on its own data.
# - **Optimizer arithmetic** — Adam elements updated per rank. This *does* shrink by N,
#   because a rank only updates the state it owns. It is the one piece of computation ZeRO
#   genuinely removes.
# - **Collective calls** — how many times a rank has to stop and talk: DP 2, ZeRO-1 2,
#   ZeRO-2 one per unit plus 1, ZeRO-3 three per unit. More, smaller calls are what let a real
#   system overlap communication with the backward pass (report Section 19).
# - **Bytes on the wire** — from Cell 12.
#
# Simulator seconds are shown last and labelled as such. They measure Python loops in one
# process, not a cluster.
#
# ### Inputs / Outputs
# In: `TRAIN`, `SWEEP`. Out: `R["compute_n32"]`.
#
# ### What you should see
# The first three rows identical across arrangements; Adam elements falling by 32×; collective
# calls and bytes rising with the stage.

# %%
U = len(LAYOUT.units)
compute = {}
for st in STAGES:
    ex = TRAIN[st]["execs_per_rank_per_step"]
    calls = TRAIN[st]["calls_per_step"]
    compute[st] = {"unit_forward": ex["unit forward"], "unit_recompute": ex["unit recompute"],
                   "unit_backward": ex["unit backward"], "adam_elements_per_rank": ex["adam elements"],
                   "reduce_scatter_calls": calls.get("reduce-scatter", 0), "all_gather_calls": calls.get("all-gather", 0),
                   "collective_calls": calls.get("reduce-scatter", 0) + calls.get("all-gather", 0),
                   "comm_over_P": SWEEP[st][SIM.world]["comm_over_P_rank0"],
                   "comm_by_purpose_over_P": {k: v / (2 * PSI) for k, v in SWEEP[st][SIM.world]["comm_by_purpose_rank0"].items()},
                   "sim_seconds_per_step": TRAIN[st]["sim_seconds_per_step"]}

_rows = [("unit forwards / rank", "unit_forward", "{:.0f}"), ("unit recomputes / rank", "unit_recompute", "{:.0f}"),
         ("unit backwards / rank", "unit_backward", "{:.0f}"), ("Adam elements / rank", "adam_elements_per_rank", "{:,.0f}"),
         ("reduce-scatter calls", "reduce_scatter_calls", "{:.0f}"), ("all-gather calls", "all_gather_calls", "{:.0f}"),
         ("bytes sent / rank (xP)", "comm_over_P", "{:.4f}")]
print(f"{'per step, N=' + str(SIM.world):<26}" + "".join(f"{LABEL[s]:>13}" for s in STAGES))
for name, key, fmt in _rows:
    print(f"{name:<26}" + "".join(f"{fmt.format(compute[s][key]):>13}" for s in STAGES))
print(f"{'-- simulator time, s --':<26}")
for k in ("compute", "collectives", "optimizer"):
    print(f"{'  ' + k:<26}" + "".join(f"{compute[s]['sim_seconds_per_step'][k]:>13.3f}" for s in STAGES))

_model_same = all(compute[s][k] == compute["dp"][k] for s in STAGES for k in ("unit_forward", "unit_recompute", "unit_backward"))
_adam_ratio = compute["dp"]["adam_elements_per_rank"] / compute["zero1"]["adam_elements_per_rank"]
print(f"\nmodel arithmetic identical across arrangements: {_model_same}")
print(f"Adam work per rank: DP / ZeRO = {_adam_ratio:.1f}x")
print(f"ZeRO-3 makes {compute['zero3']['collective_calls']:.0f} collective calls per step against DP's "
      f"{compute['dp']['collective_calls']:.0f}, and sends {compute['zero3']['comm_over_P'] / compute['dp']['comm_over_P']:.2f}x the bytes.")
R["compute_n32"] = compute | {"model_arithmetic_identical": _model_same, "adam_ratio": _adam_ratio}

# %% [markdown]
# ## Cell 14 — what a shard is: cut the vector flat, or keep tensors whole
#
# ### What this block does
# Repeats one ZeRO-3 step on 32 ranks with a different partition rule. The comparison is the
# lecture's Section 15 point: *equal count per GPU does not mean equal memory per GPU.*
#
# ### How it works
# - **flat** (Cells 9–13) cuts at equal element counts, even if that splits a tensor through
#   the middle. Every rank holds within one element of Ψ/N.
# - **whole-tensor** only cuts between parameters, choosing the tensor boundary nearest each
#   ideal cut. That is the natural thing to write by hand — and this model's embedding table is
#   a large fraction of the whole vector, so no boundary exists near most of the ideal cuts.
#
# The imbalance is measured as max/mean of per-rank persistent bytes, together with how many
# ranks end up owning nothing at all.
#
# ### Inputs / Outputs
# In: `Cluster`. Out: `R["hot_rank"]`.
#
# ### What you should see
# A flat partition near 1.00×, and a whole-tensor partition far above it with empty ranks.

# %%
def _partition_run(how):
    X, Y = get_batch(1, SIM.world)
    cl = Cluster("zero3", SIM.world, make_compute_net(SIM_DEVICE, WORK_DTYPE), how=how)
    cl.step(X, Y, 1)
    s = cl.stats(1)
    del cl
    per = [r["persistent"] for r in s["ranks"]]
    mean = sum(per) / len(per)
    hot_ranks = [i for i, p in enumerate(per) if p >= 0.99 * max(per)]
    owners = {r["rank"]: [u["name"] for u in LAYOUT.units if min(u["end"], r["hi"]) > max(u["start"], r["lo"])]
              for r in s["ranks"]}
    return {"persistent_per_rank": per, "max_over_mean": max(per) / mean,
            "hot_rank": int(np.argmax(per)), "hot_ranks": hot_ranks,
            "hot_units": {str(i): owners[i] for i in hot_ranks},
            "empty_ranks": sum(1 for r in s["ranks"] if r["shard"] == 0),
            "shards": [r["shard"] for r in s["ranks"]]}


hot = {"flat": _partition_run("flat"), "whole-tensor": _partition_run("whole-tensor")}
for how, h in hot.items():
    print(f"{how:<13} max/mean {h['max_over_mean']:>6.2f}x   hot rank {h['hot_rank']:>2} holds "
          f"{max(h['persistent_per_rank'])/MiB:.3f} MiB   empty ranks: {h['empty_ranks']}")
_wt = hot["whole-tensor"]
_hr = _wt["hot_ranks"]
print(f"\nKeeping tensors whole puts {len(_hr)} rank{'s' if len(_hr) > 1 else ''} at {_wt['max_over_mean']:.1f}x the average - "
      + ", ".join(f"rank {k} (holding {'+'.join(v)})" for k, v in _wt["hot_units"].items())
      + f" - and leaves {_wt['empty_ranks']} of {SIM.world} ranks with no parameters.")
print("Under a memory limit, " + ("those are the ranks" if len(_hr) > 1 else "that is the rank") + " that fail first.")
R["hot_rank"] = hot

# %% [markdown]
# ## Cell 15 — GATE F: projecting to V5 — 30B parameters on 8 to 64 GPUs
#
# ### What this block does
# Evaluates `bytes_per_weight` at Ψ = 30 billion and N = 8, 16, 32, 64, and checks the result
# against the lecture's memory table.
#
# ### How it works
# This is the **same function** Gates D and E verified against counted bytes on every rank at
# every N. That is what licenses using it at a size this notebook cannot run. It is labelled
# *projected* because it is. As in the lecture, activations are **not** included — Cell 11
# shows how large they can be relative to sharded state.
#
# Two derived facts from report Section 16 are checked as well:
# - **ZeRO-1's floor**: `4Ψ` stays on every card at any N, which is 111.8 GiB at 30B.
# - **Where that floor fills a card**: 74.5 GiB ÷ 4 bytes = 20B parameters.
#
# ### Inputs / Outputs
# In: `bytes_per_weight`, `LECTURE_GIB`. Out: `R["projection"]`, `R["gates"]["F_projection"]`.
#
# ### What you should see
# Sixteen cells matching the lecture to one decimal, a fits-on-card column, and `GATE PASS`.

# %%
GiB = 2**30
proj, proj_ok = {}, True
print(f"{'GiB per GPU (projected)':<24}" + "".join(f"{'N=' + str(n):>18}" for n in (8, 16, 32, 64)))
for st in STAGES:
    proj[st] = {}
    cells = []
    for n in (8, 16, 32, 64):
        gib = bytes_per_weight(st, n) * V5_PARAMS / GiB
        lec = LECTURE_GIB[st][n]
        ok = abs(gib - lec) <= TH.projection_gib_tol
        proj_ok &= ok
        fits = gib <= CARD_GIB
        proj[st][str(n)] = {"gib": gib, "lecture": lec, "match": ok, "fits_80gb": fits}
        cells.append(f"{gib:>7.1f} ({lec:>5.1f}) {'fits' if fits else '  - '}")
    print(f"{LABEL[st]:<24}" + "".join(f"{c:>18}" for c in cells))
print("(value in brackets = lesson page;  'fits' = under the 74.5 GiB card, activations NOT included)")

floor_gib = 4 * V5_PARAMS / GiB
boundary_params = CARD_GIB * GiB / 4
print(f"\nZeRO-1 floor: 4 B x 30B = {floor_gib:.1f} GiB on every card at any N  "
      f"-> {'never fits' if floor_gib > CARD_GIB else 'fits'}")
print(f"4 B/weight fills a {CARD_GIB:.1f} GiB card at {boundary_params/1e9:.1f}B parameters")
_fit = {st: [n for n in (8, 16, 32, 64) if proj[st][str(n)]["fits_80gb"]] for st in STAGES}
print("smallest world size that fits: " + ", ".join(f"{LABEL[s]} {min(v) if v else 'none'}" for s, v in _fit.items()))
print(f"P = {2*V5_PARAMS/1e9:.0f} GB; per step DP/ZeRO-1/ZeRO-2 move 2P = {4*V5_PARAMS/1e9:.0f} GB per GPU, "
      f"ZeRO-3 moves 3P = {6*V5_PARAMS/1e9:.0f} GB")
print()
gate(proj_ok, "all 16 projected cells match the lecture's table to 0.05 GiB")
gate(floor_gib > CARD_GIB and abs(boundary_params / 1e9 - 20.0) < 0.05,
     f"ZeRO-1 floor {floor_gib:.1f} GiB exceeds the card; the 4-byte floor fills it at {boundary_params/1e9:.1f}B")
R["projection"] = {"params": V5_PARAMS, "card_gib": CARD_GIB, "table": proj, "zero1_floor_gib": floor_gib,
                   "zero1_boundary_params": boundary_params, "smallest_fitting_world": _fit}
R["gates"]["F_projection"] = {"table_matches": proj_ok, "floor_and_boundary": True}

# %% [markdown]
# ## Cell 16 — the real GPU check: counted bytes against `torch.cuda.memory_allocated()`
#
# ### What this block does
# Only when CUDA is present: re-runs the simulator **on the GPU** for one step of each
# arrangement at N=32 and compares the allocator's numbers with the ledger's. This is the
# check that the counted bytes are not fiction.
#
# ### How it works
# All 32 ranks live on one card, so the comparison is cluster-wide:
#
# - **Persistent**: allocated bytes after building the cluster, minus the baseline, against
#   the sum of every rank's counted persistent bytes. The caching allocator rounds each
#   allocation up to a 512-byte block, so a small positive gap is expected; anything beyond
#   1% fails.
# - **Peak**: `max_memory_allocated()` during one step, against the ledger. Reported, not
#   gated. The allocator also sees scratch tensors the ledger deliberately ignores — kernel
#   workspaces, gradients in the moment before they are copied, ring buffers in flight — and
#   the gap is shown rather than hidden.
#
# A T4 has no native bf16, so this cell uses **fp16** there. It is also 2 bytes, so the
# accounting is unchanged, and one step needs no loss scaling to measure memory.
#
# ### Inputs / Outputs
# In: `Cluster` on `cuda`. Out: `R["gpu_check"]`.
#
# ### What you should see
# On a GPU runtime: measured/counted persistent ratios of 1.00x–1.01x for all four
# arrangements and `GATE PASS`. On a CPU runtime: `SKIPPED`.

# %%
def _gpu_check():
    gpu_work = torch.bfloat16 if CAPABILITY >= (8, 0) else torch.float16
    net_gpu = make_compute_net("cuda", gpu_work)
    Xg, Yg = get_batch(1, SIM.gpu_world, "cuda")
    gpu = {"device": GPU_NAME, "work_dtype": str(gpu_work), "n": SIM.gpu_world, "stages": {}}
    print(f"{GPU_NAME}, {gpu_work}, N={SIM.gpu_world}\n")
    print(f"{'':<8} {'counted persist':>16} {'measured':>12} {'ratio':>7}   {'sum of rank peaks':>18} {'measured peak':>14} {'ratio':>7}")
    for st in STAGES:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        cl = Cluster(st, SIM.gpu_world, net_gpu, device="cuda", work=gpu_work)
        torch.cuda.synchronize()
        meas_init = torch.cuda.memory_allocated() - base
        counted_init = sum(sum(r.held()[k] for k in PERSISTENT) for r in cl.ranks)
        torch.cuda.reset_peak_memory_stats()
        cl.step(Xg, Yg, 1)
        torch.cuda.synchronize()
        meas_peak = torch.cuda.max_memory_allocated() - base
        counted_peak = sum(r.peak for r in cl.ranks)
        del cl
        gpu["stages"][st] = {"counted_persistent": counted_init, "measured_persistent": meas_init,
                             "persistent_ratio": meas_init / counted_init,
                             "counted_sum_of_rank_peaks": counted_peak, "measured_peak": meas_peak,
                             "peak_ratio": meas_peak / counted_peak}
        g = gpu["stages"][st]
        print(f"{LABEL[st]:<8} {counted_init/MiB:>12.2f} MiB {meas_init/MiB:>8.2f} MiB {g['persistent_ratio']:>6.3f}x"
              f"   {counted_peak/MiB:>14.2f} MiB {meas_peak/MiB:>10.2f} MiB {g['peak_ratio']:>6.3f}x")
    print()
    ok = all(abs(g["persistent_ratio"] - 1) <= TH.gpu_persistent_rel_tol for g in gpu["stages"].values())
    print("The ledger's persistent bytes are what the GPU allocator actually holds, to within rounding."
          if ok else "The ledger and the allocator DISAGREE on persistent state - see the ratios above.")
    print(f"Peak: the allocator sits {min(g['peak_ratio'] for g in gpu['stages'].values()):.2f}x-"
          f"{max(g['peak_ratio'] for g in gpu['stages'].values()):.2f}x the ledger's sum of rank peaks - "
          "the ledger does not count kernel scratch or in-flight buffers, and ranks here peak one at a time.")
    # Reported, not raised: this is a cross-check on an optional device, and a failure here must not
    # stop Run-all before the figures and the bundle are written. Cell 18 counts it as FAIL.
    print(f"  {'CHECK PASS' if ok else 'CHECK FAIL'}  GPU-measured persistent memory within "
          f"{TH.gpu_persistent_rel_tol:.0%} of counted, all arrangements")
    gpu["passed"] = ok
    del net_gpu, Xg, Yg
    torch.cuda.empty_cache()
    return gpu


if HAS_CUDA:
    try:
        gpu = _gpu_check()
    except Exception as e:
        gpu = {"skipped": False, "passed": False, "error": repr(e), "device": GPU_NAME}
        print(f"GPU check FAILED TO RUN: {e!r}\nRecorded as a failed check; every other result is CPU-side and unaffected.")
else:
    gpu = {"skipped": True, "reason": "no CUDA device in this runtime"}
    print("SKIPPED - no CUDA. Run on a Colab GPU runtime to cross-check counted bytes against the allocator.")
R["gpu_check"] = gpu

# %% [markdown]
# ## Cell 17 — figures
#
# ### What this block does
# Draws seven figures from `R` and saves them to `outputs/`. Every number on a figure comes
# from the results dictionary, never from a literal.
#
# ### How it works
# One colour per arrangement (DP, ZeRO-1, ZeRO-2, ZeRO-3) and one per memory category, each
# held fixed across every figure. Values are labelled directly on the marks, because three of
# the colours are too light to identify from a legend swatch alone.
#
# ### Inputs / Outputs
# In: `R`. Out: `outputs/f1..f7*.png`.
#
# ### What you should see
# 1. bytes per rank at N=32
# 2. memory through one step
# 3. the memory ladder
# 4. communication
# 5. flat vs whole-tensor shards
# 6. the 30B projection
# 7. four coincident loss curves

# %%
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
S_COL = dict(zip(STAGES, PAL))
C_COL = {"weights": PAL[0], "grads": PAL[1], "optimizer state": PAL[2], "activations": PAL[3], "gathered weights": PAL[4]}
PURPOSES = ["grad reduce-scatter", "grad all-gather", "param all-gather (sync)", "param all-gather (fwd)", "param all-gather (bwd)"]
P_COL = dict(zip(PURPOSES, PAL))

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                     "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "legend.frameon": False})


def _style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1)
        ax.set_axisbelow(True)


def _log_y(ax):
    """Log scale with plain tick labels (2, 5, 10, 20) instead of 2x10^0."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _cats(h):
    return {"weights": h["weights"], "grads": h["grads"],
            "optimizer state": h["master"] + h["adam_m"] + h["adam_v"],
            "activations": h["activations"], "gathered weights": h["gathered"]}


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# F1 - bytes per rank at N=32: persistent (left) and at the peak (right)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, key, title in ((axes[0], "rank0_steady", "Persistent state per rank"),
                       (axes[1], "rank0_peak_breakdown", "At the rank's peak moment in a step")):
    for i, st in enumerate(STAGES):
        left = 0.0
        for cat, val in _cats(R["memory_n32"][st][key]).items():
            if val:
                ax.barh(i, val / MiB, left=left, height=0.5, color=C_COL[cat], edgecolor=SURF, linewidth=2)
                left += val / MiB
        ax.text(left, i, f"  {left:.2f} MiB", va="center", ha="left", fontsize=8, color=INK2)
    ax.set_yticks(range(len(STAGES)), [LABEL[s] for s in STAGES])
    ax.invert_yaxis()
    ax.set_xlabel("MiB per rank (counted)")
    ax.set_title(title, loc="left")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.18)
    _style(ax, "x")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in C_COL.values()], labels=list(C_COL),
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
fig.suptitle(f"What one rank holds, {SIM.world} ranks, {PSI/1e6:.2f}M-parameter GPT", x=0.01, y=1.05, ha="left", fontsize=12)
_save(fig, "f1_memory_per_rank.png")

# F2 - memory through one step, rank 0, small multiples
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, st in zip(axes, STAGES):
    tl = TRAIN[st]["timeline_step1"]
    xs = np.arange(len(tl))
    stack = {c: np.array([_cats(e)[c] for e in tl]) / MiB for c in C_COL}
    ax.stackplot(xs, *stack.values(), colors=list(C_COL.values()), edgecolor=SURF, linewidth=0.5)
    labels = [e["label"] for e in tl]
    marks = [0, labels.index(next(l for l in labels if l.startswith("bwd"))),
             next(i for i, l in enumerate(labels) if l in ("step done",)) ]
    ax.set_xticks(marks, ["fwd", "bwd", "end"])
    ax.set_title(LABEL[st], loc="left")
    peak = max(sum(_cats(e).values()) for e in tl) / MiB
    ax.text(1.0, 1.02, f"peak {peak:.2f} MiB", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    _style(ax)
axes[0].set_ylabel("MiB held by rank 0 (counted)")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in C_COL.values()], labels=list(C_COL),
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Memory through one training step, rank 0 of 32", x=0.01, y=1.05, ha="left", fontsize=12)
_save(fig, "f2_step_timeline.png")

# F3 - the memory ladder: persistent and peak per rank vs N
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
ns = list(SIM.sweep_worlds)
for ax, key, title in ((axes[0], "persistent_max", "Persistent state"), (axes[1], "peak_max", "Peak within a step")):
    for st in STAGES:
        ys = [R["sweep"][st][str(n)][key] / MiB for n in ns]
        ax.plot(ns, ys, color=S_COL[st], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURF, markeredgewidth=1.5, label=LABEL[st])
        ax.text(ns[-1] * 1.12, ys[-1], LABEL[st], va="center", fontsize=8, color=INK2)
    if key == "persistent_max":
        mod = [bytes_per_weight("zero3", n) * PSI / MiB for n in ns]
        ax.plot(ns, mod, color=MUTED, linewidth=1, zorder=0, label="ZeRO-3 modelled 16Ψ/N")
    ax.set_xscale("log", base=2)
    _log_y(ax)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlim(0.8, ns[-1] * 2.2)
    ax.set_xlabel("world size N")
    ax.set_title(title, loc="left")
    _style(ax, "both")
axes[0].set_ylabel("MiB per rank (largest rank)")
axes[0].legend(loc="lower left")
fig.suptitle("The memory ladder, counted on the demo model", x=0.01, y=1.05, ha="left", fontsize=12)
_save(fig, "f3_memory_ladder.png")

# F4 - communication: by purpose at N=32, and total vs N
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
for i, st in enumerate(STAGES):
    left = 0.0
    for p in PURPOSES:
        val = R["compute_n32"][st]["comm_by_purpose_over_P"].get(p, 0.0)
        if val:
            ax.barh(i, val, left=left, height=0.5, color=P_COL[p], edgecolor=SURF, linewidth=2)
            left += val
    ax.text(left, i, f"  {left:.3f} P", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(STAGES)), [LABEL[s] for s in STAGES])
ax.invert_yaxis()
ax.set_xlim(0, 3.6)
ax.set_xlabel(f"bytes sent per rank per step, in P = 2Ψ  (N={SIM.world})")
ax.set_title("What crosses the wire", loc="left")
_style(ax, "x")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=P_COL[p]) for p in PURPOSES], labels=PURPOSES,
          loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
ax = axes[1]
for st in ("dp", "zero3"):
    ys = [R["sweep"][st][str(n)]["comm_over_P_rank0"] for n in ns]
    ax.plot(ns, ys, color=S_COL[st], linewidth=2, marker="o", markersize=5, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.text(ns[-1] * 1.12, ys[-1], "ZeRO-3" if st == "zero3" else "DP = ZeRO-1 = ZeRO-2", va="center", fontsize=8, color=INK2)
ax.set_xscale("log", base=2)
ax.set_xticks(ns, [str(n) for n in ns])
ax.set_xlim(0.8, ns[-1] * 4)
ax.set_ylim(0, 3.2)
ax.set_xlabel("world size N")
ax.set_ylabel("xP per rank per step")
ax.set_title("Volume per rank approaches 2P and 3P, never more", loc="left")
_style(ax)
fig.suptitle("Communication, counted hop by hop in the ring", x=0.01, y=1.05, ha="left", fontsize=12)
_save(fig, "f4_communication.png")

# F5 - flat vs whole-tensor shards
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, how in zip(axes, ("flat", "whole-tensor")):
    h = R["hot_rank"][how]
    ys = np.array(h["persistent_per_rank"]) / MiB
    ax.bar(range(SIM.world), ys, width=0.6, color=S_COL["zero3"])
    for hr in (h["hot_ranks"] if len(h["hot_ranks"]) <= 4 else h["hot_ranks"][:1]):
        ax.text(hr, ys[hr], f"{ys[hr]:.2f} MiB", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_title(f"{how}: max/mean {h['max_over_mean']:.2f}x, {h['empty_ranks']} empty ranks", loc="left")
    ax.set_xlabel("rank")
    _style(ax)
axes[0].set_ylabel("ZeRO-3 persistent MiB")
fig.suptitle("Where the cut falls decides which rank runs hot", x=0.01, y=1.05, ha="left", fontsize=12)
_save(fig, "f5_hot_rank.png")

# F6 - projection to 30B
fig, ax = plt.subplots(figsize=(7.5, 4))
pn = [8, 16, 32, 64]
for st in STAGES:
    ys = [R["projection"]["table"][st][str(n)]["gib"] for n in pn]
    ax.plot(pn, ys, color=S_COL[st], linewidth=2, marker="o", markersize=5, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.scatter(pn, [LECTURE_GIB[st][n] for n in pn], s=90, facecolors="none", edgecolors=MUTED, linewidths=1, zorder=3)
    ax.text(64 * 1.1, ys[-1], LABEL[st], va="center", fontsize=8, color=INK2)
ax.axhline(CARD_GIB, color=INK2, linewidth=1)
ax.text(60, CARD_GIB * 1.07, f"one 80 GB card = {CARD_GIB:.1f} GiB", ha="right", fontsize=8, color=INK2)
ax.set_xscale("log", base=2)
_log_y(ax)
ax.set_xticks(pn, [str(n) for n in pn])
ax.set_xlim(6.5, 110)
ax.set_xlabel("GPUs")
ax.set_ylabel("GiB per GPU, training state only")
ax.set_title("Projected to V5 (30B). Open circles: the lesson page's values", loc="left")
_style(ax, "both")
_save(fig, "f6_v5_projection.png")

# F7 - loss curves
fig, ax = plt.subplots(figsize=(7.5, 3.6))
steps_x = np.arange(1, SIM.steps + 1)
for st, lw in zip(STAGES, (6, 4.5, 3, 1.5)):
    ax.plot(steps_x, R["train"]["losses"][st], color=S_COL[st], linewidth=lw, label=LABEL[st], solid_capstyle="round")
_dmax = max(R["gates"]["C_identical"][s]["max_abs_loss"] for s in STAGES[1:])
ax.set_title(("Four arrangements, one curve: max |Δloss| vs DP = " + f"{_dmax:.1e}") if _dmax == 0
             else f"Loss curves differ: max |Δloss| = {_dmax:.2e}", loc="left")
ax.set_xlabel("step")
ax.set_xticks(range(0, SIM.steps + 1, 5))
ax.set_ylabel(f"mean loss over {SIM.world} ranks")
ax.legend(loc="upper right")
_style(ax)
_save(fig, "f7_loss_identical.png")

# %% [markdown]
# ## Cell 18 — acceptance checks, summary.json, RESULTS.md, and the bundle
#
# ### What this block does
# Collects every check into one list, prints `N/N`, writes `summary.json` and `RESULTS.md`, and
# zips `outputs/` into `era5s12_outputs.zip`.
#
# ### How it works
# Each gate above already raised on failure, so reaching this cell means they passed. They are
# collected here so the tally lives in one place and in the JSON the README is generated from.
# A skipped GPU check is counted as **skipped**, not as passed.
#
# ### Inputs / Outputs
# In: `R`. Out: `outputs/summary.json`, `outputs/RESULTS.md`, `outputs/era5s12_outputs.zip`.
#
# ### What you should see
# `ACCEPTANCE: N/N passed`, followed by the skip count and the bundle path.

# %%
G = R["gates"]
CHECKS = [
    ("A. ring reduce-scatter + all-gather == all-reduce, bit-exact", all(g["allreduce_exact"] for g in G["A_collectives"])),
    ("A. reduce-scatter alone leaves each rank its exact slice", all(g["slice_exact"] for g in G["A_collectives"])),
    ("A. ring bytes == exact expression (even and uneven chunks)", all(g["bytes_exact"] for g in G["A_collectives"])),
    ("B. DP gradient == big-batch gradient (fp64)", G["B_dp_equals_big_batch"]["passed"]),
    ("B'. adam_ on a shard == same slice of full-vector update, bit-exact", G["B2_adam_shard_exact"]["passed"]),
    ("C. loss fell by at least the declared threshold", R["train"]["loss_drop"] >= TH.min_loss_drop),
    ("C. 32 DP replicas bit-identical", R["train"]["replica_max_diff"] == 0.0),
    ("C. ZeRO-1 bit-identical to DP", G["C_identical"]["zero1"]["identical"]),
    ("C. ZeRO-2 bit-identical to DP", G["C_identical"]["zero2"]["identical"]),
    ("C. ZeRO-3 bit-identical to DP", G["C_identical"]["zero3"]["identical"]),
    ("C'. planted bug 'drop_contribution' caught", G["C_negative_controls"][0]["caught"]),
    ("C'. planted bug 'double_average' caught", G["C_negative_controls"][1]["caught"]),
    ("D. counted persistent == modelled, every rank (N=32)", G["D_ledger"]["modelled_exact"]),
    ("D. no leak across steps, no transient survives a step", G["D_ledger"]["no_leak"]),
    ("D. activation bytes identical across arrangements", G["D_ledger"]["activations_identical"]),
    ("E. ring bytes == exact expression, every rank, every N", G["E_comm"]["ring_exact"]),
    ("E. ledger == modelled, every rank, every N", G["E_comm"]["ledger_exact_all_N"]),
    ("E. ZeRO-1 and ZeRO-2 send exactly DP's bytes", G["E_comm"]["zero12_equal_dp"]),
    ("E. ZeRO-3 sends exactly 1.5x DP (even shards)", G["E_comm"]["zero3_over_dp"] == 1.5),
    ("13. model arithmetic identical across arrangements", R["compute_n32"]["model_arithmetic_identical"]),
    ("14. whole-tensor partition more imbalanced than flat",
     R["hot_rank"]["whole-tensor"]["max_over_mean"] > R["hot_rank"]["flat"]["max_over_mean"]),
    ("F. 30B projection matches the lesson page (16 cells)", G["F_projection"]["table_matches"]),
    ("F. ZeRO-1 floor exceeds the card; 20B boundary", G["F_projection"]["floor_and_boundary"]),
    ("16. GPU-measured persistent within 1% of counted", None if R["gpu_check"].get("skipped") else R["gpu_check"]["passed"]),
]
print("ACCEPTANCE CHECKS")
for name, ok in CHECKS:
    print(f"  [{'PASS' if ok else ('SKIP' if ok is None else 'FAIL')}] {name}")
_ran = [ok for _, ok in CHECKS if ok is not None]
_npass, _nskip = sum(1 for ok in _ran if ok), sum(1 for _, ok in CHECKS if ok is None)
print(f"\nACCEPTANCE: {_npass}/{len(_ran)} passed" + (f", {_nskip} skipped" if _nskip else ""))
R["acceptance"] = {"passed": _npass, "total": len(_ran), "skipped": _nskip,
                   "checks": [{"name": n, "result": None if ok is None else bool(ok)} for n, ok in CHECKS]}
R["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, default=float)

L = []
A = L.append
A(f"# Session 12 results — notebook {NOTEBOOK_VERSION}\n")
A(f"Model: {PSI:,} parameters (P = {2*PSI/MiB:.2f} MiB bf16). World size {SIM.world}, "
  f"{SIM.steps} training steps. Simulator on {SIM_DEVICE}; GPU check: "
  f"{'skipped' if R['gpu_check'].get('skipped') else R['gpu_check']['device']}.\n")
A(f"Acceptance: **{_npass}/{len(_ran)}**" + (f" ({_nskip} skipped)" if _nskip else "") + "\n")
A("## Memory per rank, N=32 (counted)\n")
A("| arrangement | persistent MiB | bytes/weight | modelled | peak MiB | peak at |")
A("|---|---|---|---|---|---|")
for st in STAGES:
    m = R["memory_n32"][st]
    A(f"| {LABEL[st]} | {m['persistent_per_rank'][0]/MiB:.3f} | {m['bytes_per_weight_counted']:.4f} | "
      f"{m['bytes_per_weight_modelled']:.4f} | {m['peak_per_rank'][0]/MiB:.3f} | {m['rank0_peak_at']} |")
A("\n## Sweep: persistent MiB per rank / bytes sent per rank (×P)\n")
A("| N | " + " | ".join(LABEL[s] for s in STAGES) + " |")
A("|---|" + "---|" * len(STAGES))
for n in SIM.sweep_worlds:
    A(f"| {n} | " + " | ".join(f"{R['sweep'][s][str(n)]['persistent_max']/MiB:.3f} / {R['sweep'][s][str(n)]['comm_over_P_rank0']:.4f}"
                               for s in STAGES) + " |")
A("\n## Computation per rank per step, N=32\n")
A("| | " + " | ".join(LABEL[s] for s in STAGES) + " |")
A("|---|" + "---|" * len(STAGES))
for name, key, fmt in _rows:
    A(f"| {name} | " + " | ".join(fmt.format(R["compute_n32"][s][key]) for s in STAGES) + " |")
A("\n## Projection to 30B (GiB per GPU; lesson page in brackets)\n")
A("| | " + " | ".join(f"N={n}" for n in (8, 16, 32, 64)) + " |")
A("|---|---|---|---|---|")
for st in STAGES:
    A(f"| {LABEL[st]} | " + " | ".join(f"{R['projection']['table'][st][str(n)]['gib']:.1f} ({LECTURE_GIB[st][n]})"
                                     for n in (8, 16, 32, 64)) + " |")
A("\n## Acceptance\n")
for c in R["acceptance"]["checks"]:
    A(f"- [{'PASS' if c['result'] else ('SKIP' if c['result'] is None else 'FAIL')}] {c['name']}")
with open(os.path.join(OUT_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

_zip = os.path.join(OUT_DIR, "era5s12_outputs.zip")
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(OUT_DIR)):
        if name.endswith((".png", ".json", ".md")):
            z.write(os.path.join(OUT_DIR, name), arcname=name)
print(f"\nbundle: {_zip}  ({os.path.getsize(_zip)/1024:.0f} KB)")
print("Download it with the executed .ipynb.")
