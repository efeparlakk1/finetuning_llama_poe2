# Path of Exile 2 crafting fine-tune

Supervised fine-tune of Llama Instruct models so they answer Path of Exile 2 crafting questions like a step-by-step guide: which orb/omen/essence/rune to use, what it does, and how to apply it to a given base item.

This is **SFT, not RAG**. Training updates adapter weights so the model replies in the style of `poe2_data.jsonl`. It does not look up a wiki at inference time.

## What you get

Given a scenario (base item, current mods, inventory omen, goal), the model explains the currency or omen and walks through the craft.

**Example prompt**

> How do I use Orb of Augmentation in PoE 2 to solve this crafting situation?
>
> Base Item: Magic Titanium Spirit Shield with 1 explicit modifier (+80 to Maximum Life - Prefix)
> Goal: Add a Suffix without altering the Life roll.

**Expected style** — name the tool, say what it does, then give the concrete action and outcome.

## Two training paths

| Path | Model | How | Best for |
| --- | --- | --- | --- |
| `ft_notebook.ipynb` | Llama 3.1 8B Instruct (QLoRA via Unsloth) | Lesson-style notebook | Learning the pipeline end to end |
| `finetune_sample.py` | Llama 3.2 3B Instruct (QLoRA via PEFT/TRL) | Single script | Faster / lower VRAM run |

`ft_notebook_turkish.md` is a Turkish walkthrough of the same notebook.

Both use the same Alpaca JSONL dataset.

## Dataset

`poe2_data.jsonl` — 260 rows:

| Field | Role |
| --- | --- |
| `instruction` | Task wording (“explain this omen”, “give a step-by-step craft”) |
| `input` | Scenario: base item, goal, inventory state |
| `output` | Gold crafting guide |

Coverage includes currency orbs (Transmute, Augment, Alteration, Regal, Chaos, Exalt, Annul, Divine, Vaal, Scouring), crafting omens, essences, runes, and catalysts.

This is a **small domain set**. It teaches format and typical PoE 2 craft reasoning. It will not encode the whole affix wiki. If answers stay generic after training, add better / more rows before raising LoRA rank.

## Setup

Needs a CUDA GPU (the notebook was developed on a 16 GB card). Python 3.13+.

```bash
uv sync
```

Or with pip after creating a venv:

```bash
pip install -e .
```

Dependencies: Unsloth, Transformers, TRL, PEFT, bitsandbytes, datasets, JupyterLab.

## Train

**Notebook (8B, Unsloth)**

```bash
uv run jupyter lab ft_notebook.ipynb
```

Pipeline: load 4-bit Instruct weights → attach LoRA (`r=16`) → convert Alpaca rows to Llama chat text → `SFTTrainer` → save adapter.

**Script (3B, QLoRA)**

```bash
uv run python finetune_sample.py
```

Defaults: 4-bit NF4, LoRA `r=16` on all linear layers, batch 2 × grad accum 2, 250 steps, cosine LR `2e-4`. Writes checkpoints to `qlora-output/` and the final adapter to `qlora-final-adapter/`.

Update the dataset path in the script if you move the repo (it currently points at a local absolute path).

## Merge and export

`sample_to_gguf.py` merges the 3B LoRA into the base model (load base in bf16/fp16, **not** 4-bit) and writes `llama-3.2-3b-merged/`.

That merged folder is a Hugging Face checkpoint. Convert to GGUF afterward with [llama.cpp](https://github.com/ggml-org/llama.cpp) `convert_hf_to_gguf.py` if you want Ollama / llama.cpp.

The notebook can also save the LoRA only, or merge + export GGUF from Unsloth.

## Repo layout

```
poe2_data.jsonl          training data
finetune_sample.py       3B QLoRA train
sample_to_gguf.py        merge LoRA into base weights
ft_notebook.ipynb        8B Unsloth lesson notebook
ft_notebook_turkish.md   Turkish notes for the notebook
pyproject.toml           uv / pip deps
```

Large outputs (`qlora-output/`, `qlora-final-adapter/`, `llama-3.2-3b-merged/`, `ft_lora/`, …) are gitignored.

## License / game data

Path of Exile 2 names and mechanics belong to Grinding Gear Games. This repo is a personal fine-tune experiment, not an official tool.
