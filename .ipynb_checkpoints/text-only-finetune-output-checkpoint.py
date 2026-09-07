"""
Strip a merged Llama-3.2-Vision (MllamaForConditionalGeneration) checkpoint
down to a plain LlamaForCausalLM checkpoint, by:

  1. Dropping the vision encoder (`vision_model.*`) entirely.
  2. Dropping the multimodal projector (`multi_modal_projector.*`) entirely.
  3. Dropping the cross-attention decoder layers inside the language model
     (these are architecturally distinct layers at specific indices, not a
     wrapper around the self-attention layers).
  4. Re-indexing the remaining self-attention decoder layers into a
     contiguous 0..N-1 range so the result matches a standard Llama layout.
  5. Writing a matching plain LlamaConfig.

ASSUMPTION THIS RELIES ON: your fine-tuning only modified self-attention /
MLP weights in the text backbone (not the vision encoder or cross-attention
layers, and no image data was used in training). If that's not the case,
this will silently discard part of what you trained.

Usage:
    python strip_vision_to_text_llama.py \
        --src /home/eplinux/unsloth-ft/finetuned_model \
        --dst /home/eplinux/unsloth-ft/finetuned_model_text_only
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def strip_model(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(src, "config.json")) as f:
        config = json.load(f)

    if "text_config" not in config:
        raise ValueError(
            "config.json has no 'text_config' block — this doesn't look like "
            "a MllamaForConditionalGeneration checkpoint. Nothing to strip."
        )

    text_config = config["text_config"]
    cross_attention_layers = set(text_config["cross_attention_layers"])
    num_hidden_layers = text_config["num_hidden_layers"]

    self_attn_old_indices = sorted(
        i for i in range(num_hidden_layers) if i not in cross_attention_layers
    )
    old_to_new = {old: new for new, old in enumerate(self_attn_old_indices)}

    print(f"Total language-model layers in source: {num_hidden_layers}")
    print(f"Cross-attention layers (dropped): {sorted(cross_attention_layers)}")
    print(
        f"Self-attention layers kept: {len(self_attn_old_indices)} "
        f"-> reindexed 0..{len(self_attn_old_indices) - 1}"
    )

    shard_files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {src}")

    new_state_dict = {}
    kept, dropped = 0, 0

    for shard in shard_files:
        path = os.path.join(src, shard)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("vision_model.") or key.startswith(
                    "multi_modal_projector."
                ):
                    dropped += 1
                    continue
                if not key.startswith("language_model."):
                    dropped += 1
                    continue

                new_key = key[len("language_model.") :]

                if new_key.startswith("model.layers."):
                    parts = new_key.split(".")
                    old_idx = int(parts[2])
                    if old_idx in cross_attention_layers:
                        dropped += 1
                        continue
                    parts[2] = str(old_to_new[old_idx])
                    new_key = ".".join(parts)

                new_state_dict[new_key] = f.get_tensor(key)
                kept += 1

    print(f"Kept {kept} tensors, dropped {dropped} tensors")

    leaked = [k for k in new_state_dict if "cross_attn" in k]
    assert not leaked, f"Cross-attention keys leaked into output: {leaked}"

    save_file(
        new_state_dict,
        os.path.join(dst, "model.safetensors"),
        metadata={"format": "pt"},
    )

    llama_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": text_config["hidden_size"],
        "intermediate_size": text_config["intermediate_size"],
        "num_attention_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config.get(
            "num_key_value_heads", text_config["num_attention_heads"]
        ),
        "num_hidden_layers": len(self_attn_old_indices),
        "vocab_size": text_config["vocab_size"],
        "max_position_embeddings": text_config.get(
            "max_position_embeddings", 131072
        ),
        "rms_norm_eps": text_config.get("rms_norm_eps", 1e-5),
        "rope_theta": text_config.get("rope_theta", 500000.0),
        "rope_scaling": text_config.get("rope_scaling"),
        "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
        "torch_dtype": config.get("torch_dtype", "bfloat16"),
        "hidden_act": text_config.get("hidden_act", "silu"),
        "attention_bias": text_config.get("attention_bias", False),
        "attention_dropout": text_config.get("attention_dropout", 0.0),
        "pad_token_id": text_config.get("pad_token_id"),
        "bos_token_id": text_config.get("bos_token_id"),
        "eos_token_id": text_config.get("eos_token_id"),
    }

    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(llama_config, f, indent=2)

    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "added_tokens.json",
    ]:
        src_path = os.path.join(src, fname)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dst, fname))

    print(f"\nText-only checkpoint written to: {dst}")
    print("Next: verify, then run GGUF conversion (see bottom of this file).")


def verify_equivalence(src: str, dst: str, prompt: str = "The capital of France is") -> None:
    """
    Optional but recommended: confirms the stripped model produces identical
    output to the original on text-only input. Loads BOTH models, so this
    needs enough RAM/VRAM for the full 11B vision model plus the ~8B
    text-only model. Run this once before deleting the original checkpoint.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, MllamaForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(src)
    inputs = tok(prompt, return_tensors="pt")

    print("Loading original vision model...")
    original = MllamaForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.bfloat16, device_map="auto"
    )
    with torch.no_grad():
        out_orig = original.language_model.generate(
            **inputs, max_new_tokens=20, do_sample=False
        )
    del original
    torch.cuda.empty_cache()

    print("Loading stripped text-only model...")
    stripped = AutoModelForCausalLM.from_pretrained(
        dst, torch_dtype=torch.bfloat16, device_map="auto"
    )
    with torch.no_grad():
        out_stripped = stripped.generate(**inputs, max_new_tokens=20, do_sample=False)

    text_orig = tok.decode(out_orig[0], skip_special_tokens=True)
    text_stripped = tok.decode(out_stripped[0], skip_special_tokens=True)

    print(f"\nOriginal:  {text_orig}")
    print(f"Stripped:  {text_stripped}")
    print(f"\nMATCH: {text_orig == text_stripped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to merged Mllama checkpoint")
    parser.add_argument("--dst", required=True, help="Output path for text-only checkpoint")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also load both models and compare generation output (needs more RAM/VRAM)",
    )
    args = parser.parse_args()

    strip_model(args.src, args.dst)

    if args.verify:
        verify_equivalence(args.src, args.dst)

# ---------------------------------------------------------------------------
# After this completes, convert the text-only checkpoint to GGUF as normal —
# it now reports architecture "LlamaForCausalLM", which llama.cpp supports:
#
#   python convert_hf_to_gguf.py finetuned_model_text_only \
#       --outfile model-bf16.gguf --outtype bf16
#   ./llama-quantize model-bf16.gguf model-q4_k_m.gguf q4_k_m
#
# Or reload it with Unsloth's FastLanguageModel (not FastVisionModel) and
# call save_pretrained_gguf() again — same effect, since it's now a
# standard architecture.
# ---------------------------------------------------------------------------