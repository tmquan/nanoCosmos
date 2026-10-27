"""Tokenizer cache and packed-attention freeze split for Cosmos 3."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from nanocosmos.models.cosmos_3_common.wrapper import Cosmos3OmniWrapper


class _FakeTokenizer:
    def __init__(self):
        self.calls = 0
        self.model_max_length = 32

    def __call__(self, prompt, **kwargs):
        self.calls += 1
        # Distinct length > 1 so und_len would exceed the null-text path.
        return {"input_ids": [11, 22, 33, 44, 55]}


def test_tokenize_prompt_caches_cpu_ids_by_string():
    tok = _FakeTokenizer()
    obj = SimpleNamespace(_text_tokenizer=tok, _prompt_ids_cache={})
    prompt = "ssTEM · z40 y8 x8 nm · ssl"

    ids1 = Cosmos3OmniWrapper._tokenize_prompt(obj, prompt, torch.device("cpu"))
    ids2 = Cosmos3OmniWrapper._tokenize_prompt(obj, prompt, torch.device("cpu"))

    assert tok.calls == 1
    assert torch.equal(ids1, ids2)
    assert ids1.dtype == torch.long
    assert ids1.device.type == "cpu"
    assert ids1.tolist() == [11, 22, 33, 44, 55]
    assert prompt in obj._prompt_ids_cache
    assert obj._prompt_ids_cache[prompt].device.type == "cpu"


def test_tokenize_prompt_without_tokenizer_is_null_token():
    obj = SimpleNamespace(_text_tokenizer=None, _prompt_ids_cache={})
    ids = Cosmos3OmniWrapper._tokenize_prompt(
        obj, "FIB-SEM · z8 y8 x8 nm · gapped", torch.device("cpu"),
    )
    assert ids.tolist() == [0]
    assert len(obj._prompt_ids_cache) == 1


def test_packed_attention_roles_keep_dit_half_out_of_ar_freeze():
    """AR projections freeze with the reasoner; DiT projections stay trainable."""
    role = Cosmos3OmniWrapper._packed_attn_role
    assert role("layers.0.self_attn.to_q.weight") == "ar"
    assert role("layers.0.self_attn.to_out.weight") == "ar"
    assert role("layers.0.self_attn.to_add_out.weight") == "gen"
    assert role("layers.0.self_attn.add_q_proj.weight") == "gen"
    assert role("layers.0.self_attn.norm_added_q.weight") == "gen"
    assert role("layers.0.mlp_moe_gen.up_proj.weight") is None

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.to_q = nn.Linear(4, 4, bias=False)
            self.self_attn.to_add_out = nn.Linear(4, 4, bias=False)
            self.self_attn.add_k_proj = nn.Linear(4, 4, bias=False)
            self.mlp = nn.Linear(4, 4, bias=False)
            self.mlp_moe_gen = nn.Linear(4, 4, bias=False)

    dit = nn.Module()
    dit.layers = nn.ModuleList([_Block()])
    holder = SimpleNamespace(
        dit=dit,
        _UNDERSTANDING_STREAM_SEGMENTS=Cosmos3OmniWrapper._UNDERSTANDING_STREAM_SEGMENTS,
    )
    selected = {id(p) for p in Cosmos3OmniWrapper._understanding_stream_params(holder)}
    frozen = {n for n, p in dit.named_parameters() if id(p) in selected}
    assert "layers.0.self_attn.to_q.weight" in frozen
    assert "layers.0.mlp.weight" in frozen
    assert "layers.0.self_attn.to_add_out.weight" not in frozen
    assert "layers.0.self_attn.add_k_proj.weight" not in frozen
    assert "layers.0.mlp_moe_gen.weight" not in frozen
