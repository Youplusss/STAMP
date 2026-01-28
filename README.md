# LLM-TSAD

LLM-TSAD is a **time-series anomaly detection** project built by combining:

- **STAMP**: *Compatible Unsupervised Anomaly Detection with Multi-Perspective Spatio-Temporal Learning* (prediction + reconstruction + adversarial coupling + top-k aggregation).
- **Time-LLM**: *Time Series Forecasting by Reprogramming Large Language Models* (freeze LLM, train a small reprogramming module to map patches into the LLM embedding space).

In this repo, the LLM is used **as the prediction branch** (forecasting under normal behavior), and the anomaly detector is obtained by comparing **forecast errors + reconstruction errors** (same evaluation protocol as STAMP).

---

## 1) What is implemented

### 1.1 LLM prediction branch (new)

`pred_model=llm` uses a Time-LLM-style forecaster:

- Patchify the context window.
- Use a trainable **ReprogrammingLayer** (cross-attention) to align patch embeddings into a frozen LLM embedding space.
- Feed **prompt + reprogrammed patch tokens** into the frozen LLM.
- Predict the next `n_pred` steps using a small output head.

Code:

- `model/llm_time_llm.py` (core: PatchEmbedding + Reprogramming + frozen HF LLM)
- `model/llm_wrappers.py` (STAMP adapter: [B,T,N,C] <-> [B,T,features])
- `dataset/prompt_bank/*.txt` (dataset descriptions)

### 1.2 STAMP reconstruction branch (existing)

Unchanged. You can use:

- `recon_model=mamba` (recommended in this codebase)
- `recon_model=ae` (classic AE)

### 1.3 Anomaly scoring & evaluation (existing)

Unchanged. `test.py` calls `lib/evaluate.get_final_result`:

- Per-feature error normalization.
- Top-K feature aggregation.
- Threshold search.
- Segment-aware point-adjustment (STAMP-style).

---

## 2) Quick start

### 2.1 Install

```bash
pip install -r requirements.txt
```

This repo uses HuggingFace `transformers` for the LLM. If you run on a machine without internet, make sure the model weights are available locally.

### 2.2 Dataset

This repo keeps STAMP's data conventions and loaders:

- SWaT/WADI: `dataset/<DATA>/*_train.csv`, `*_test.csv`
- SMD/MSL/SMAP: CSV mode supported (see `lib/paths.py` + `scripts/process_*.py`)

### 2.3 Train

Example: **SWaT** with LLM predictor (default uses GPT2, first 6 layers):

```bash
python run.py \
  --data SWaT --pred_model llm --recon_model mamba \
  --nnodes 45 --window_size 15 --n_pred 3 \
  --down_len 100 --epochs 10 --batch_size 8 \
  --llm_model gpt2 --llm_backend gpt2 --llm_layers 6 \
  --llm_patch_len 4 --llm_stride 2 --llm_prompt_mode stats_short
```

### 2.4 Test

```bash
python test.py \
  --data SWaT --pred_model llm --recon_model mamba \
  --nnodes 45 --window_size 15 --n_pred 3 --down_len 100 \
  --test_alpha 0.8 --test_beta 0.1 --test_gamma 0.1
```

---

## 3) LLM-specific ablations (to prove the LLM matters)

You can do controlled ablations without touching code:

### 3.1 Pretrained vs random-init LLM

```bash
--llm_pretrained True   # pretrained HF checkpoint
--llm_pretrained False  # random init (ablation baseline)
```

### 3.2 Prompt ablations

```bash
--llm_prompt_mode none        # no prompt
--llm_prompt_mode dataset     # dataset description only
--llm_prompt_mode stats_short # dataset + min/max/median/trend
--llm_prompt_mode stats       # stats_short + top-k FFT lags
```

### 3.3 MAS ablation

If you train with `--is_mas True`, you can decide whether the LLM sees MAS channels:

```bash
--llm_use_mas True/False
```

---

## 4) Notes on compute

- Default: `gpt2` + 6 layers is the safest starting point.
- Large LLMs (LLaMA) may require 4-bit/8-bit loading:

```bash
--llm_backend llama --llm_model <hf-llama-name> --llm_load_in_4bit True
```

---

## 5) Where things are saved

Same as STAMP:

- checkpoints: `expe/pth/best_model_<DATA>_<MODEL>.pth`
- logs: `expe/log/...`
