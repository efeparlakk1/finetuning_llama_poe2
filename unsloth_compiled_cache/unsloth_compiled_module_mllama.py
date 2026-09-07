"""
2026.9.1
2026.9.2
5.5.0
0.24.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import sys
import torch
import importlib.util
import math
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)
UNSLOTH_COMPILE_LOCATION = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
if UNSLOTH_COMPILE_LOCATION not in sys.path:
    sys.path.insert(0, UNSLOTH_COMPILE_LOCATION)

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass

from unsloth_zoo import DEVICE_TYPE_TORCH, DEVICE_COUNT


from unsloth_zoo.loss_utils import (
    fused_linear_cross_entropy,
    unsloth_fused_ce_loss,
)

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 12, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True}

from torch.nn import CrossEntropyLoss
from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
    # Stateless pickling so accelerate gather_object works on the sentinel
    def __reduce__(self): return (type(self), ())
    # Gathered copies must compare equal in accelerate debug mode
    def __eq__(self, other): return type(other).__name__ == "EmptyLogits"
    __hash__ = object.__hash__
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass
# The loop above stomps pickle hooks with stubs returning None; restore them.
for function in ("__reduce__", "__reduce_ex__", "__getstate__", "__setstate__"):
    try: delattr(EMPTY_LOGITS, function)
    except Exception: pass
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    if labels is not None and attention_mask is not None:
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
from unsloth_zoo.temporary_patches.common import torch_compile
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.mllama.modeling_mllama import (F, Callable, Optional, torch, nn, init, ACT2FN, Cache, GenerationMixin, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, ALL_ATTENTION_FUNCTIONS, PreTrainedModel, Unpack, TransformersKwargs, can_return_tuple, maybe_autocast, MllamaConfig, MllamaTextConfig, MllamaVisionConfig, logger, __name__, MllamaPreTrainedModel, MllamaModel, MllamaForConditionalGeneration)

@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def _prepare_cross_attention_mask(cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,) -> tuple[torch.Tensor, torch.Tensor]:
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min)

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = ((cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None])
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask!=torch.finfo(cross_attention_mask.dtype).min, full_text_row_masked_out_mask

@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def _prepare_aspect_ratio_attention_mask(aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(-1, -2) * torch.finfo(dtype).min
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask!=torch.finfo(attention_mask.dtype).min

@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaPrecomputedAspectRatioEmbedding_forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
    embeddings = self.embedding(aspect_ratio_ids)
    embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

    if self.is_gated:
        embeddings = embeddings * self.gate.tanh()

    hidden_state = hidden_state + embeddings
    return hidden_state

class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        return MllamaPrecomputedAspectRatioEmbedding_forward(self, hidden_state=hidden_state, aspect_ratio_ids=aspect_ratio_ids)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaPrecomputedPositionEmbedding_forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
    # position embeddings
    gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
    hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

    # precomputed tile position embeddings
    tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
    batch_size = hidden_state.shape[0]
    tile_position_embedding = tile_position_embedding.reshape(
        batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
    )
    gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
    hidden_state = hidden_state + gated_tile_position_embedding

    return hidden_state

class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size
        )

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        return MllamaPrecomputedPositionEmbedding_forward(self, hidden_state=hidden_state, aspect_ratio_ids=aspect_ratio_ids)


@torch_compile_with_fallback(fullgraph = False, dynamic = True, options = torch_compile_options)
def MllamaVisionMLP_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = self.fc2(hidden_states)
    return hidden_states

class MllamaVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return MllamaVisionMLP_forward(self, hidden_states=hidden_states)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:

        if isinstance(attention_mask, dict):

            attention_mask = attention_mask.get(getattr(module, 'layer_type', None), None)

        if attention_mask is not None:

            attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(attn_weights.dtype).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@torch.compiler.disable(recursive = False)
def MllamaVisionAttention_forward(
    self,
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    query = self.q_proj(hidden_state)
    key = self.k_proj(hidden_state)
    value = self.v_proj(hidden_state)

    batch_size, q_seq_len, _ = query.shape
    _, kv_seq_len, _ = key.shape

    query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, q_seq_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = 1

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return MllamaVisionAttention_forward(self, hidden_state=hidden_state, attention_mask=attention_mask, **kwargs)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaTextRMSNorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

class MllamaTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        MllamaTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return MllamaTextRMSNorm_forward(self, hidden_states=hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@torch.compiler.disable(recursive = False)
def MllamaTextCrossAttention_forward(
    self,
    hidden_states: torch.Tensor,
    cross_attention_states: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    attention_mask: torch.Tensor | None = None,
    use_cache: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    """Input shape: Batch x Time x Channel"""
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    query_states = self.q_norm(query_states)

    if cross_attention_states is not None:
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_norm(key_states)
        if past_key_values is not None:
            # if we have a new image + new tokens, we only computed key_states on that new image
            # we still update the cross key states, past_image, new_image. And use it!
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    elif past_key_values is not None and past_key_values.get_seq_length() > 0:
        key_states, value_states = (
            past_key_values.layers[self.layer_idx].keys,
            past_key_values.layers[self.layer_idx].values,
        )
    else:
        raise ValueError(
            "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
        )

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: MllamaTextConfig | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        return MllamaTextCrossAttention_forward(self, hidden_states=hidden_states, cross_attention_states=cross_attention_states, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=use_cache, **kwargs)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compiler.disable(recursive = False)
def MllamaTextSelfAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: torch.Tensor,
    past_key_values=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # Unsloth: align V dtype with Q after RoPE (fixes 4-bit dtype mismatch)
    if value_states.dtype != query_states.dtype:
        value_states = value_states.to(query_states.dtype)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

class MllamaTextSelfAttention(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.layer_idx = layer_idx
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        past_key_values=None,
        **kwargs,
    ):
        return MllamaTextSelfAttention_forward(self, hidden_states=hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings, past_key_values=past_key_values, **kwargs)


@torch_compile_with_fallback(fullgraph = False, dynamic = True, options = torch_compile_options)
def MllamaTextMLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class MllamaTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Ignore copy
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return MllamaTextMLP_forward(self, x=x)


@torch_compile_with_fallback(fullgraph = False, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def MllamaRotaryEmbedding_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class MllamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: MllamaTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: MllamaTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    # Ignore copy

    def forward(self, x, position_ids):
        return MllamaRotaryEmbedding_forward(self, x=x, position_ids=position_ids)


@torch.compiler.disable(recursive = False)
@can_return_tuple
def MllamaForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    aspect_ratio_mask: torch.Tensor | None = None,
    aspect_ratio_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    cross_attention_mask: torch.Tensor | None = None,
    cross_attention_states: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | CausalLMOutputWithPast:
    r"""
    aspect_ratio_mask (`torch.Tensor` of shape `(batch_size, max_num_images, max_num_tiles)`, *optional*):
        Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:

        - 1 for tiles that are **not masked**,
        - 0 for tiles that are **masked**.
    aspect_ratio_ids (`torch.Tensor` of shape `(batch_size, max_num_images)`, *optional*):
        Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
        These ids correspond to indices in the model's list of supported aspect ratios, offset by 1.

        For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:
        - An image with aspect ratio [1, 1] would have ID 1
        - An image with aspect ratio [1, 2] would have ID 2
        - An image with aspect ratio [2, 1] would have ID 3

        The id 0 is reserved for padding (i.e., no image).

        If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its `aspect_ratio_id` would be 2.
    cross_attention_mask (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*):
        Cross-attention mask to control the interaction between text tokens and image tiles.
        This 4D tensor defines which image tiles each text token should attend to.

        For each text token (in seq_length):
        - 1 indicates the token **should attend** to the corresponding image tile
        - 0 indicates the token **should not attend** to the corresponding image tile
    cross_attention_states (`torch.FloatTensor`, *optional*):
        Output of the vision model, used for cross-attention. This tensor contains the processed image features that
        the language model will attend to.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import httpx
    >>> from io import BytesIO
    >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

    >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
    >>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
    >>> processor = AutoProcessor.from_pretrained(checkpoint)

    >>> prompt = "<|image|>If I had to write a haiku for this one"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> with httpx.stream("GET", url) as response:
    ...     image = Image.open(BytesIO(response.read()))

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> output = model.generate(**inputs, max_new_tokens=15)

    >>> prompt_len = inputs.input_ids.shape[-1]
    >>> generated_ids = output[:, prompt_len:]
    >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    >>> print(generated_text)
    [', it would be:.\\nA stop sign in Chinatown.\\n']
    ```
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        aspect_ratio_mask=aspect_ratio_mask,
        aspect_ratio_ids=aspect_ratio_ids,
        cross_attention_mask=cross_attention_mask,
        cross_attention_states=cross_attention_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None)
        if n_items is None: n_items = (kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None)
                    if n_items is None: n_items = __kwargs.get("n_items", None)
                    break
    pass
    
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        if torch_dynamo_eval_frame is not None:
            old_stance = torch_dynamo_eval_frame._stance.stance
        else:
            old_stance = None
        if old_stance is not None and INFERENCE_RUNS == 1:
            # Skip guards and return to eager -> we still need guards!
            torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Removing compiler guards after 1 inference run. "\
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
        elif old_stance == "eager_on_recompile":
            pass
        elif old_stance == "default" and INFERENCE_RUNS > 1:
            # Reset compiler stance
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Reseting guards. "\
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
            INFERENCE_RUNS = 0
        INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # Inference returns these logits to the caller, so they must carry the same
        # scale and softcap transforms the training branches below apply. Leaving
        # them off does not change greedy decoding, since tanh is monotonic, but it
        # does change the distribution: sampling, returned logprobs and any scoring
        # that reads the logits all see uncapped values.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
    elif (() == () and () == ()) and (UNSLOTH_ENABLE_CCE) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if () == () else (),
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and NOT_RETURN_LOGITS:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = labels,
            mask                 = None,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () != () else 0,
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        # UNSLOTH_RETURN_LOGITS=1 path. Prepended `logits = self.lm_head(...)`
        # already materialised the full lm_head matmul; apply the captured logit
        # scale/softcap transforms and route loss through self.loss_function on
        # those logits instead of letting unsloth_fused_ce_loss redo the matmul.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.config.text_config.vocab_size, **kwargs)
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.config.text_config.vocab_size, **kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class MllamaForConditionalGeneration(MllamaPreTrainedModel, GenerationMixin):
    # _tied_weights_keys = {"lm_head.weight": "model.language_moddel.embed_tokens.weight"}

    def __init__(self, config: MllamaConfig):
        super().__init__(config)
        self.model = MllamaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        aspect_ratio_mask: torch.Tensor | None = None,
        aspect_ratio_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        cross_attention_states: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        return MllamaForConditionalGeneration_forward(self, input_ids=input_ids, pixel_values=pixel_values, aspect_ratio_mask=aspect_ratio_mask, aspect_ratio_ids=aspect_ratio_ids, attention_mask=attention_mask, cross_attention_mask=cross_attention_mask, cross_attention_states=cross_attention_states, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["aspect_ratio_ids"] = None
            model_inputs["aspect_ratio_mask"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        return model_kwargs

MllamaForConditionalGeneration.__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

