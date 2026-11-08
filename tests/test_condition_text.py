"""Tests for fixed metadata text-condition templates."""

from nanocosmos.datamodules.condition_text import (
    LABEL_CONVENTIONS,
    build_condition_text,
    prompt_from_volume_spec,
    validate_label_convention,
)
import pytest


def test_build_condition_text_examples():
    assert (
        build_condition_text("ssTEM", task="sft", label_convention="filled",
        )
        == "ssTEM · filled"
    )
    assert (
        build_condition_text("FIB-SEM", task="sft", label_convention="gapped",
        )
        == "FIB-SEM · gapped"
    )
    assert (
        build_condition_text("ssTEM", task="ssl")
        == "ssTEM · ssl"
    )


def test_ssl_ignores_label_convention():
    # SSL always uses the ssl tail even if a convention sneaks in.
    assert (
        build_condition_text("ssTEM", task="ssl", label_convention="gapped",
        )
        == "ssTEM · ssl"
    )


def test_sft_defaults_missing_convention_to_filled():
    assert (
        build_condition_text("FIB-SEM", task="sft")
        == "FIB-SEM · filled"
    )


def test_validate_label_convention():
    assert validate_label_convention(None) is None
    assert validate_label_convention("gapped") == "gapped"
    assert LABEL_CONVENTIONS == {"gapped", "filled"}
    with pytest.raises(ValueError, match="label_convention"):
        validate_label_convention("abutting")


def test_prompt_from_volume_spec():
    vol = {
        "vol": "flywire_x",
        "imaging": "ssTEM",
        "native_resolution": [40, 8, 8],
        "label_convention": "filled",
    }
    assert prompt_from_volume_spec(vol, task="sft") == "ssTEM · filled"
    assert prompt_from_volume_spec({**vol, "native_resolution": [4, 4, 4]}, task="sft") == (
        "ssTEM · filled"
    )
    with pytest.raises(ValueError, match="imaging"):
        prompt_from_volume_spec(
            {"vol": "x", "native_resolution": [8, 8, 8]}, task="ssl",
        )
