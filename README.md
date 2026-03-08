# NanoGPTDolly (Emeldar)

A customized NanoGPT codebase for training and running a real-time conversational character model.

This project extends classic NanoGPT with:
- a compact character tokenizer ("Caseifer")
- instruction-style masked-loss training
- streaming and instructed generation modes
- chat-history aware inference tooling for live interactions

---

## What is different from vanilla NanoGPT?

### 1) Caseifer character encoding
The project uses a character-level vocabulary with a case marker:
- `↨` marks uppercase letters (e.g. `A` becomes `↨a`)
- `§` is used as an end-of-text marker

This keeps vocabulary small while preserving case.

### 2) Multiple generation modes in the model
[model.py](model.py) includes:
- `generate()`
- `generate_streaming()`
- `generate_instructional()`
- `generate_instructed_streaming()`

### 3) Instruction tuning workflow
[train_instruct.py](train_instruct.py) supports prompt/response training with response-focused masking and mixed dataset training.

### 4) Live-chat style sampling stack
[sample-Eml2.py](sample-Eml2.py) adds:
- chat/follow/donation input file polling
- conversation history management
- bad-word filtering
- sample logging to SQLite (`samples.db`)
- optional local socket output

---

## Repository map

Core model/training:
- [model.py](model.py): main GPT implementation + extra generation functions
- [model2.py](model2.py): alternate GPT model variant
- [train_instruct.py](train_instruct.py): instruction-style training loop
- [train_ramble_tqdm.py](train_ramble_tqdm.py): standard/continuous training loop with tqdm
- [configurator.py](configurator.py): config-file and CLI override mechanism

Inference/runtime:
- [sample.py](sample.py): basic sampling from checkpoints / GPT-2
- [sample-Eml2.py](sample-Eml2.py): interactive streaming/chat-oriented sampler
- [history.py](history.py): rolling context helpers (`History`, `DonationHistory`, `FollowHistory`)

Utilities:
- [bench.py](bench.py): training-step benchmark/profiling helper
- [scaling_laws.ipynb](scaling_laws.ipynb): scaling-law calculations
- [transformer_sizing.ipynb](transformer_sizing.ipynb): parameter/FLOP sizing notes

Data/config:
- [data/](data/): datasets and preparation scripts
- [config/](config/): example training/eval config overrides

Checkpoints:
- `Eml2b_*` folders hold existing training outputs/checkpoints.

---

## Requirements

Recommended:
- Python 3.10+
- PyTorch 2.x
- CUDA GPU (training/inference is configured for CUDA by default)

Python packages used across scripts:
- `torch`
- `numpy`
- `tqdm`
- `tiktoken`
- `transformers` (for GPT-2 weight loading)
- `datasets` (OpenWebText prep path)
- `wandb` (optional)
- `pyttsx3` (optional, voice output in [sample-Eml2.py](sample-Eml2.py))
- `requests` (used in some prep scripts)

Example install:

```bash
pip install torch numpy tqdm tiktoken transformers datasets wandb pyttsx3 requests
```

---

## Quick start

### Run basic sampling from a local checkpoint
Edit defaults in [sample.py](sample.py) or override with CLI via [configurator.py](configurator.py):

```bash
python sample.py --out_dir=Eml2b_hl --init_from=resume
```

### Run interactive chat-style sampler

```bash
python sample-Eml2.py --out_dir=Eml2b_hl --init_from=resume
```

This script reads/writes files under `files/` and maintains history in `files/past_*.txt`.

---

## Training workflows

## 1) Instruction tuning
Use [train_instruct.py](train_instruct.py).

Default behavior:
- reads supervised data from `data.json`
- optionally mixes additional JSONL-style data (e.g. `gpt4all_stripped.json` if present)
- applies response-focused masking for loss
- can mix in original token stream dataset from `data/<dataset>/train.bin`

Run:

```bash
python train_instruct.py
```

Override settings:

```bash
python train_instruct.py --out_dir=Eml2b_FT1 --batch_size=8 --learning_rate=1e-5
```

## 2) Continuous/pretraining loop
Use [train_ramble_tqdm.py](train_ramble_tqdm.py) for standard next-token training on memmapped bins.

Run:

```bash
python train_ramble_tqdm.py
```

Distributed launch example:

```bash
torchrun --standalone --nproc_per_node=4 train_ramble_tqdm.py
```

---

## Data preparation

Included scripts:
- [data/openwebtext/prepare.py](data/openwebtext/prepare.py)
- [data/shakespeare/prepare.py](data/shakespeare/prepare.py)
- [data/shakespeare_char/prepare.py](data/shakespeare_char/prepare.py)
- [data/whisper_2/prepare_chars.py](data/whisper_2/prepare_chars.py)

Note: [data/whisper_2/prepare_chars.py](data/whisper_2/prepare_chars.py) uses project-specific local paths by default and may need path edits before use.

---

## Configuration system

[configurator.py](configurator.py) is executed by major scripts and supports:

1. config file injection
```bash
python train_instruct.py config/train_gpt2.py
```

2. direct key overrides
```bash
python train_instruct.py --batch_size=4 --compile=False
```

Overrides are type-checked against script globals.

---

## Runtime files used by interactive sampler

[sample-Eml2.py](sample-Eml2.py) integrates with files under `files/`, including:
- `input.txt`
- `new_follower.txt`
- `new_donation.txt`
- `direction.txt`
- `autoplay.txt`
- `badwords.txt`

It also persists generated-history metadata in `samples.db`.

---

## Notes and caveats

- Defaults are tuned for CUDA; for CPU/Windows fallback use `--device=cpu --compile=False` where needed.
- Some scripts assume optional local files exist (for example extra JSON data sources).
- Several folders contain historical checkpoints and experimental artifacts; they are intentionally preserved.

---

## Acknowledgement

This project is built on top of NanoGPT concepts and code style, then extended for character-level conversational and streaming use cases.
