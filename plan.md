# KDA-MoE 1B — Implementation Plan

A sparse KDA + MoE language model: **~1B total params, ~100M active per token, top-4 routing**.
Built to train for free on **Google Colab (T4, 15GB)** and smoke-test on the **local RTX 4050 (6GB)**.

This plan fixes the contradictions in the prior draft and makes one honest scoping decision up front:
**the 1B model does not fit on the 4050** (steady-state optimizer+weights ≈ 9.2GB > 5.9GB VRAM),
so the local GPU is a fast-iteration/smoke box running a ~450M config, and Colab runs the real training.

---

## 0. Decisions (locked)

| Decision | Value | Why |
|---|---|---|
| Total params | **~990M** | user target: 1B |
| Active params / token | **~95M** | user target: ~100M; keeps compute cheap so free GPUs buy more tokens |
| Routing | **top-4** of 96 experts | user target: 4 active; high sparsity (4.2%) is the design |
| Context length | **2048** (curriculum 512→1024→2048) | the most that fits VRAM + an overnight budget; not a paper 1M |
| Primary training machine | **Colab free T4 (15GB VRAM, 12GB RAM, 12h sessions)** | the 1B fits there with 5.8GB headroom; it does not fit locally |
| Smoke / correctness machine | **local RTX 4050 (6GB), ~450M config** | uv is already installed; no Nix detour |
| Checkpoint store | **Google Drive** (mounted) | Colab sessions are ephemeral; local disk otherwise |
| Global attention | **GQA, strict NoPE** | MLA→GQA substitution is deliberate; 2k ctx KV cache is cheap, MLA adds complexity we can't validate at this scale |
| KDA backend | **FLA `fla.ops.kda` if it imports; PyTorch reference otherwise** | reference is the correctness oracle and is written first |
| Depth mixing | **Block AttnRes = OFF in v1, phase-2 flag** | least-validated component at this scale; ship the core first |
| Dense baseline | **yes — one control run** | without it we can't attribute any result to KDA/MoE |
| Pretraining data | DCLM + FineWeb-Edu + code + math, capped | reproducible, ungated |
| Post-train data | FABLE.5 traces (56,700 verified rows) → OpenAI Responses-style JSON | coding-agent behavior |
| Tool set | `read, write, edit, bash, grep, glob` | coding core |

Open questions that block nothing (defaults chosen): exact SFT oversample ratio, tool-span loss weight, AttnRes on/off ablation.

---

## 1. Hardware reality (measured)

| | Local RTX 4050 | Colab free T4 |
|---|---|---|
| VRAM | 5.9 GiB usable | 15 GB |
| System RAM | 15 GB total (~7 GB free) | 12 GB |
| Achievable bf16 | ~20 TFLOPS | ~65 TFLOPS |
| Session limit | none | **12 h**, ~90 min idle disconnect, GPU not guaranteed |
| torch/CUDA | install via uv | **preinstalled** — pin to it, do not fight it |
| Persistence | local disk | **none — must checkpoint to Drive** |

### VRAM budget, 990M model (8-bit Adam + fp32 master, bf16 weights/grads)
```
bf16 weights   1.84 GB
bf16 grads     1.84 GB
fp32 master    3.69 GB
8-bit m+v      1.84 GB
------------------------
steady state   9.22 GB
T4 headroom    5.78 GB  → activations + CUDA ctx: COMFORTABLE
4050 headroom  -3.3 GB  → DOES NOT FIT
```
**Local optimizer-offload fallback is rejected**: offloading 5.5GB of state over PCIe per step
collapses throughput to unusable, and full fp32 AdamW (11GB) exceeds free RAM. Do not attempt.

### Throughput (fwd+bwd ≈ 6·N_active, +30% recompute from grad checkpointing)
| Machine | MFU 40% | Tokens/hour | 2.5B-token shard |
|---|---|---|---|
| Colab T4 | ~35,000 tok/s | ~126M | **~20 h ≈ 1.6 × 12h sessions** |
| RTX 4050 | ~10,800 tok/s | ~39M | ~64 h (only the 450M smoke config) |

**Set expectations:** an overnight Colab run trains the 1B on ~0.5–1.5B tokens. That is a real
signal (loss curve, router behavior, tool-call emergence) but it is a **pilot**, not a converged
frontier model. The full shard is ~2 Colab sessions.

---

## 2. Model

### 2.1 Primary: `kda-moe-1b` (Colab)
| Component | Value |
|---|---|
| Vocabulary | 32,000 BPE (trained fresh) |
| Layers | 16 |
| `d_model` | 640 |
| Attention heads | 10 (head_dim 64) |
| KDA : global ratio | 3:1 (12 KDA, 4 global GQA) |
| Global attention | GQA, **NoPE** |
| KDA q/k/v prep | depthwise ShortConv1D (k=4) + Swish |
| KDA q/k norm | L2Norm |
| KDA decay | K3 bounded: `g = g_min·sigmoid(exp(A_h)·z)`, `g_min=-5`, `alpha=exp(g)` |
| Attention out gate | `sigmoid(W_g x) * RMSNorm(o)` |
| FFN | LatentMoE: down-project to latent `l`, route, shared expert + up-project |
| Dense warmup | layer 0 dense FFN (no MoE) |
| **Routed experts E** | **96** |
| **Active experts** | **top-4** |
| Shared experts | 1 (hidden 640) |
| Latent width `l` | 320 |
| Routed expert hidden | 660 |
| MoE activation | SiTU-GLU: `softcap(g,4)·sigmoid(g)·softcap(u,25)` |
| Router | sigmoid scores + top-4 + auxiliary-loss-free bias; z-loss only if saturating |
| Block AttnRes | **off (v1)** |
| Context | 512→1024→2048 curriculum |
| Precision | bf16 |

**Param budget (computed):** emb 20.5M + attn ~33M + shared 36.9M→(hidden 640) 18.4M + latent proj 6.1M + routed experts 3·320·660·96·15 ≈ 912M ≈ **990M total**. Active (top-4) ≈ **95M**.

> Note: shared-expert hidden dropped 1280→640 vs. the old draft. At 640 hidden a 1280-wide shared
> expert plus 96 routed experts over-weights the dense path; hidden 640 = model width is cleaner.

### 2.2 Smoke: `kda-moe-450m` (local 4050)
Same code, one config flag set. Only the MoE dims shrink:
| Component | Value |
|---|---|
| Routed experts E | 32 |
| Active experts | top-2 |
| Shared expert hidden | 1280 |
| Latent width `l` | 320 |
| Routed expert hidden | 768 |
| → total / active | **~450M / ~98M** |

Steady-state VRAM ≈ 4.2GB → fits the 4050. Used for correctness tests, KDA equivalence, and
loss-decrease smoke runs. **Not** the deliverable.

### 2.3 Dense control: `dense-100m` (Colab, one run)
A vanilla 16-layer ~100M dense transformer (no KDA, no MoE, GQA+NoPE) trained on the same shard
with the same tokenizer. **Purpose:** the control. If `kda-moe-1b` does not beat `dense-100m` at
equal active-compute, the architecture is not pulling its weight and we find out cheap.

### 2.4 Layer schedule (both MoE models)
```
0: KDA + dense FFN          4: KDA + MoE    8: KDA + MoE    12: KDA + MoE
1: KDA + MoE                5: KDA + MoE    9: KDA + MoE    13: KDA + MoE
2: KDA + MoE                6: KDA + MoE   10: KDA + MoE    14: KDA + MoE
3: global GQA + MoE         7: global GQA  11: global GQA   15: global GQA
```

---

## 3. KDA implementation (reference first, FLA optional)

Order matters — write the oracle before the kernel:
1. **Slow recurrent KDA** (pure PyTorch) — correctness oracle.
2. **Equivalence + causality tests** (below).
3. **Chunked PyTorch KDA** (chunk 64, Kimi Linear's value) — local speed path.
4. **FLA `fla.ops.kda`** — only after tests pass; drop-in behind a config flag.

**Correction to prior draft:** the "16-token tile" rationale was a mis-citation. K3's bounded decay
*removes* tile-size coupling; Kimi Linear's chunk size is 64. Use **chunk 64**, `head_dim 64`.

Reference recurrence:
```
q = l2norm(silu(shortconv(Wq x)));  k = l2norm(silu(shortconv(Wk x)));  v = silu(shortconv(Wv x))
beta = sigmoid(W_beta x)
z = W_au(W_ad(x)) + b_a
log_decay = g_min * sigmoid(exp(A_h) * z);  alpha = exp(log_decay)     # g_min = -5
S_t = (I - beta_t k_t k_tᵀ) Diag(alpha_t) S_{t-1} + beta_t k_t v_tᵀ
o_t = S_tᵀ q_t
out = W_o( sigmoid(W_g x_t) * rmsnorm(o_t) )
```

**Numerics:** accumulate the KDA state `S` in fp32 even when the rest is bf16 (cheap at this size,
kills the NaN mode). Required tests:
1. recurrent == chunked within tolerance on random tensors.
2. causality: output at `t` unchanged when future tokens change.
3. `log_decay ∈ (-5, 0)`, no NaN/Inf in cumulative decay over 2048 tokens.
4. router: over **real-data** batches (not random), no expert starves for many consecutive steps.

---

## 4. Environment

**Local (4050) — no Nix detour; `uv` is already installed.**
```bash
uv init --package kda-poc && uv python pin 3.11
uv add torch --index-url https://download.pytorch.org/whl/cu121
uv add datasets tokenizers transformers safetensors tqdm numpy einops tomli-w tensorboard
uv add --dev pytest ruff
# optional: uv pip install git+https://github.com/fla-org/flash-linear-attention --no-deps
```
FLA needs Python ≥3.10, torch ≥2.4, Triton. FlashKDA (the fast CUTLASS path) needs SM90+ — the 4050
(sm_89) and T4 (sm_75) both fall back to the Triton kernel. **If FLA doesn't import cleanly, skip it;
the chunked PyTorch path is the supported fallback.** Do not burn a day on Triton pinning.

**Colab (T4) — torch/CUDA preinstalled; pin to it.**
```python
import torch; print(torch.__version__, torch.cuda.is_available())   # use THIS torch
!pip install datasets tokenizers transformers safetensors einops tomli-w
# clone repo or %cd into Drive copy
from google.colab import drive; drive.mount('/content/drive')
```
Do **not** `pip install torch` on Colab — match the preinstalled build. Put the repo + checkpoints on
Drive so a disconnect costs at most one checkpoint interval.

---

## 5. Data

### 5.1 Pretraining (token mix / GB cap — percentages are tokens, caps are disk)
```
55% DCLM-baseline        data/shards/dclm/          ≤ 3.0 GB
10% FineWeb-Edu          data/shards/fineweb_edu/   ≤ 1.0 GB
25% code (CSN-Python + flytech/python-codes-25k)    ≤ 3.5 GB
10% math (Proof-Pile-2 OWM/AlgStack)                ≤ 1.0 GB
```
Total ≤ **10GB ≈ 2.5B tokens**. Note: the token% and GB caps only agree if bytes/token are similar
across sources — **measure bytes/token per shard during prep and rebalance the caps**, don't assume.

Stream + materialize bounded shards (verify streaming works per-dataset; DCLM Parquet streaming can
degrade to shard downloads — if so, download a fixed file subset instead of `streaming=True`).

**Carve a held-out split** (every-Nth document → `data/val/`): validation loss means nothing without it.

### 5.2 Tokenizer (32k BPE)
Train on a bounded sample of the pretraining shards. Add chat/special tokens:
`<|system|> <|user|> <|assistant|> <|tool_call|> <|tool_result|> <|reasoning|> <|final|> <|end|>`
and Responses-style sentinels `<|tools|> <|function_call|> <|tool_json|> <|tool_name|> <|tool_arguments|>`.
**Acceptance gate (new):** measure fertility (tokens/word) on held-out web, code, and a
`{"type":"function_call",...}` JSON sample; if JSON tool-call fertility is pathological, adjust before training.

### 5.3 Post-train (FABLE.5 → Responses-style JSON)
`Crownelius/Complete-FABLE.5-traces-2M`: 2,006,487 original rows → **56,700 verified**, `row_json`
payloads across 9 source schemas. ≤0.8GB.
1. **Normalize** each source's `row_json` into one internal trace schema (keep raw row for audit).
2. **Convert** tool spans to OpenAI Responses-style calls: `{"type":"function_call","name":"read","arguments":{"path":"..."}}`.
3. **Clean** reasoning into `<|reasoning|>` blocks (strip envelope noise / UI metadata).
4. **Filter** to rows with real user intent + parseable tool spans + substantive response.
5. Decide per-example **turn structure**: keep multi-turn call→result→call trajectories (a coding
   agent loops) — do not flatten to single turns.
6. **Loss masking:** assistant-only; **tool-result tokens masked out** (model emits calls, not results);
   weight JSON tool-call spans by `w_tool` (start 2.0, tune).

---

## 6. Training

### Optimizer (boring first; Muon is a phase-2 upgrade)
```
AdamW, lr 3e-4 (smoke) → 1.5e-4 (1B run), betas (0.9,0.95), wd 0.1
warmup 1–2%, cosine decay, clip_grad_norm 1.0
8-bit Adam states (bitsandbytes) + fp32 master; bf16 autocast
```
### Memory tactics
Gradient checkpointing every block · microbatch 1 · grad accumulation · tied embeddings ·
sequence curriculum · no activation logging (scalars only).

### Batch schedule
| Phase | Seq | Micro | Accum | Tokens/update |
|---|---|---|---|---|
| smoke (4050, 450M) | 128 | 1 | 4 | 512 |
| pilot-1 | 512 | 1 | 16 | 8K |
| pilot-2 | 1024 | 1 | 16–32 | 16–32K |
| pilot-3 (Colab, 1B) | 2048 | 1 | 32 | 65K |

### Checkpointing (Drive on Colab; local disk otherwise)
Frequent **model-only** checkpoints + less frequent **full-resume** (optimizer+step). Keep latest +
best-validation, prune the rest, **≤20GB total**. On Colab, save every N steps *to Drive* — a 12h
disconnect must never lose more than one interval.

---

## 7. Phases & acceptance

- **P0 env smoke** — local: `uv run python -c "import torch;print(torch.cuda.is_available())"` → True; tiny CUDA matmul. Colab: preinstalled torch sees the T4; Drive mounts; repo on Drive.
- **P1 data+tokenizer** — stream 1k examples from each source; parse 100 FABLE rows; write shards ≤10GB; **held-out split exists**; **fertility gate passes**.
- **P2 core model** — forward (batch1, seq128) + loss + backward on GPU, both configs (450M local, 1B shape on CPU/Colab).
- **P3 KDA equivalence** — recurrent==chunked, no causal leak, no NaN at seq2048; FLA optional behind flag.
- **P4 smoke train (4050, 450M)** — 200–1000 updates, loss decreases, checkpoint save/reload, non-empty generation, no NaN. *Exit: plumbing works.*
- **P5 pilot pretrain (Colab, 1B)** — stable run at seq512→2048; router not collapsed; **val loss (held-out) trends down**. *Exit: clean loss curve + throughput ≥ 25K tok/s on T4.*
- **P6 dense control (Colab)** — train `dense-100m` on same shard; record val loss as the bar.
- **P7 FABLE SFT (Colab)** — traces render to Responses-style JSON; assistant-only masked loss with tool-span weight; SFT doesn't wreck base generation; **JSON validity + tiny coding-task eval improve**.

**Quantitative gates (new):** abort/iterate if T4 throughput < 25K tok/s; if router entropy collapses
before pilot-2; if `kda-moe-1b` pilot val loss ≥ `dense-100m` at equal active-compute (architecture not paying off).

---

## 8. Evaluation

Always-on scalars: train/val loss, tokens/sec, VRAM, grad norm, KDA decay min/max, router entropy,
expert-load histogram, dead-expert count, JSON tool-call validity, shard disk usage.

Fixed smoke-prompt file (factual QA, arithmetic, short Python fn, Responses-style JSON call,
`<|reasoning|>`+final, JSON-validity gate, 1024/2048 needle).

**Coding-agent eval (the actual product metric):** build a small held-out set of ~30 terminal-task
prompts, each with a pass criterion (emits valid JSON call for the right tool with the right args, or
completes a 2-step call→result→call loop). Report pass rate. This set does not exist yet — create it in P7.

Formal benchmarks (HellaSwag/ARC/WinoGrande/HumanEval/needle) only after meaningful pretraining.

---

## 9. Risk register
| Risk | Symptom | Fix |
|---|---|---|
| 1B OOMs on T4 | crash in backward | accum up, seq down to 1024, then E 96→64 |
| 1B won't fit 4050 | (known) | expected — use 450M smoke config locally, Colab for 1B |
| Colab disconnect | lost progress | checkpoint to Drive every interval; resume from latest |
| Colab GPU unavailable | CPU-only runtime | retry later / run 450M locally meanwhile |
| KDA NaN | NaN in decay/state | fp32 state, lower LR, verify g_min bound |
| Router collapse | few experts get all tokens | bias update up, z-loss, then Quantile Balancing |
| FLA won't build | import/build error | use chunked PyTorch path; don't fight Triton |
| DCLM streaming degrades | huge downloads | fixed file subset instead of streaming |
| Data-mix skew | token% ≠ GB% | measure bytes/token, rebalance caps |
| SFT verbosity collapse | model rambles traces | low LR, mix pretraining PTX, cap `<|reasoning|>` length |
| Architecture underperforms | val ≥ dense-100m | the control run tells us; revisit KDA/MoE value |

---

## 10. Repo layout
```
kda-poc/
  plan.md  pyproject.toml  uv.lock
  src/kda_moe/
    __init__.py config.py tokenizer.py data.py
    kda.py attention.py moe.py model.py train.py eval.py
    fable.py            # FABLE row_json → normalized traces → Responses-style JSON
  tests/
    test_kda_equivalence.py test_causal.py test_moe_router.py
  configs/
    kda_moe_1b.toml       # Colab primary
    kda_moe_450m.toml     # local smoke
    dense_100m.toml       # control
    smoke.toml
  scripts/
    prepare_tokenizer.py sample_dataset.py make_eval_prompts.py
```

## 11. References
- Kimi Linear / KDA: `kimi-linear-report.pdf` (= `kimi delta attention.pdf`), chunk 64, 3:1 KDA:MLA, NoPE via KDA diagonal decay, sigmoid out-gate, ShortConv+Swish, L2-norm q/k.
- Kimi K3: `k3_tech_report.pdf`, bounded decay `g=g_min·sigmoid(exp(A_h)·z)` g_min=-5, SiTU-GLU β1=4/β2=25, sigmoid router+top-k, AttnRes (phase-2 here), Quantile Balancing (phase-2).
- DCLM: hf.co/datasets/mlfoundations/dclm-baseline-1.0 · FineWeb-Edu: hf.co/datasets/HuggingFaceFW/fineweb-edu
- Proof-Pile-2: hf.co/datasets/EleutherAI/proof-pile-2 · CodeSearchNet: hf.co/datasets/code-search-net/code_search_net
- flytech 25k: hf.co/datasets/flytech/python-codes-25k · FABLE.5: hf.co/datasets/Crownelius/Complete-FABLE.5-traces-2M
- FLA: github.com/fla-org/flash-linear-attention (`fla.ops.kda`; FlashKDA needs SM90+, Triton fallback otherwise)
