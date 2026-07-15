"""Tokenizer input_ids cache for Cosmos3 metadata prompts."""

from types import SimpleNamespace

import torch

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
