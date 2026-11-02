# tmem-repro

A re-implementation of the inference-time part of **TMEM**:
*Scaling Self-Evolving Agents via Parametric Memory* (Ren et al., arXiv:2606.04536, 2026).
No official code was public when this repo was written.

This is the first step of a larger study on **persistent, revocable parametric memory for agents**:
can fast LoRA memory be kept across episodes without breaking old memories or general ability,
and can one memory be deleted exactly on request?

## What is implemented

| Paper part | File | Status |
|---|---|---|
| SVD-initialized LoRA, A frozen, only B trained (Sec. 3.2, Eq. 7) | `tmem/svd_lora.py` | done, unit-tested |
| LoRA on FFN `gate/up/down_proj` of the last 4 layers, rank 6 (Sec. 5.1) | `tmem/svd_lora.py` | done |
| Online SFT operator T: SGD, lr 5e-4, 5 epochs, batch 16, cumulative within episode (Sec. 5.1) | `tmem/update.py` | done, unit-tested |
| Fast-weight rollout: trigger at `len(h)+len(m) > L_max`, extract QA, update Delta, clear h, carry no text (Sec. 3.1, Eq. 2-6) | `tmem/agent.py` | done |
| Extraction prompt (Fig. 1 + App. D Fig. 6) | `tmem/prompts.py` | done |
| Summary baseline (MemAgent-style overwrite) | `tmem/agent.py` | approximation, prompt not given in paper |
| No-memory baseline | `tmem/agent.py` | done |
| Token F1 / EM (Sec. 5.1) | `tmem/metrics.py` | done |
| LongMemEval-S runner | `scripts/run_longmemeval.py` | done, resumable |
| A-MEM retrieval baseline | | not yet |
| LoCoMo, multi-objective search, CL-Bench | | not yet |
| RL on theta_0 with stop-gradient through Delta (Sec. 4) | | out of scope for now |

Every memory write is logged (`WriteRecord`: which sessions, which QA pairs, losses, ||Delta|| after),
so later experiments can replay or revert individual writes.

## Paper targets (Qwen3-4B, before RL, Table 1)

| Method | LongMemEval-S F1 | EM |
|---|---|---|
| No memory | 5.30 | 0.00 |
| MemAgent | 36.45 | 23.20 |
| A-MEM | 29.22 | 17.76 |
| TMEM | 41.24 | 25.54 |

Paper efficiency for TMEM on LongMemEval-S: 6.55 s/episode, 22.9 GB peak (Table 7).

## Usage

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
pytest -q

# full LongMemEval-S, paper setting (one GPU)
python scripts/run_longmemeval.py --model Qwen/Qwen3-4B --strategy tmem    --l-max 12288 --out runs/lme_tmem_4b.jsonl
python scripts/run_longmemeval.py --model Qwen/Qwen3-4B --strategy summary --l-max 12288 --out runs/lme_summary_4b.jsonl
python scripts/run_longmemeval.py --model Qwen/Qwen3-4B --strategy none                  --out runs/lme_none_4b.jsonl

# smoke test on a laptop
python scripts/run_longmemeval.py --model Qwen/Qwen3-0.6B --strategy tmem --limit 1 --max-sessions 4 \
    --l-max 4096 --device mps --dtype float32 --out runs/smoke.jsonl
```

The dataset (`longmemeval_s_cleaned.json`, ~277 MB) is downloaded on first run into `data/`.

## Choices the paper does not pin down

- **Chat template / thinking**: Qwen3 is run with `enable_thinking=False` for extraction and answering.
- **QA history in the extraction prompt**: App. D shows previous QA pairs inside the prompt. Kept on by default
  (last 50 pairs), `--no-qa-history` turns it off.
- **SFT loss**: loss on answer tokens only, question as a user turn in the chat template.
- **Sessions longer than L_max** are split into token windows before triggering.
- **No-memory baseline**: the paper keeps "only the current working context"; here it keeps the last
  `--max-context` tokens (default 32k).
- **Decoding**: greedy. The paper reports mean ± std over 3 runs but does not state the sampling setup.
- **B dtype**: float32 even when the base model is bf16.

## Status

- 7 unit tests pass on CPU with a tiny random Qwen3. They check:
  - the SVD init
  - the model starts exactly at theta_0
  - only B moves and the loss goes down
  - state save/load/reset
  - determinism under a fixed seed
  - JSON parsing
  - metrics
- End-to-end smoke run: Qwen3-0.6B on Apple MPS, 1 episode, last 4 sessions, L_max 4096. The full loop runs: trigger → extract → online SFT → clear → answer.
  - Write 0: 4 QA pairs, loss 2.79 → 2.19.
  - Write 1: the model only copied the QA history back. Those 4 duplicates are now dropped and counted in `dropped_dup`.
  - The answer was wrong (F1 0). A 0.6B model is not expected to extract well; this run checks the plumbing, not quality.
- Not yet run at paper scale. No number here should be read as a reproduction yet.
