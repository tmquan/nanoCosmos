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
        build_condition_text(
            "ssTEM", [40, 8, 8], task="sft", label_convention="no_gap",
        )
        == "ssTEM · z40 y8 x8 nm · no_gap"
    )
    assert (
        build_condition_text(
            "FIB-SEM", [8, 8, 8], task="sft", label_convention="bg_gap",
        )
        == "FIB-SEM · z8 y8 x8 nm · bg_gap"
    )
    assert (
        build_condition_text("ssTEM", [40, 8, 8], task="ssl")
        == "ssTEM · z40 y8 x8 nm · ssl"
    )


def test_ssl_ignores_label_convention():
    # SSL always uses the ssl tail even if a convention sneaks in.
    assert (
        build_condition_text(
            "ssTEM", [33, 4, 4], task="ssl", label_convention="bg_gap",
        )
        == "ssTEM · z33 y4 x4 nm · ssl"
    )


def test_sft_defaults_missing_convention_to_no_gap():
    assert (
        build_condition_text("FIB-SEM", [8.0, 8.0, 8.0], task="sft")
        == "FIB-SEM · z8 y8 x8 nm · no_gap"
    )


def test_validate_label_convention():
    assert validate_label_convention(None) is None
    assert validate_label_convention("bg_gap") == "bg_gap"
    assert LABEL_CONVENTIONS == {"bg_gap", "no_gap"}
    with pytest.raises(ValueError, match="label_convention"):
        validate_label_convention("abutting")


def test_prompt_from_volume_spec():
    vol = {
        "vol": "flywire_x",
        "imaging": "ssTEM",
        "native_resolution": [40, 8, 8],
        "label_convention": "no_gap",
    }
    assert prompt_from_volume_spec(vol, task="sft") == (
        "ssTEM · z40 y8 x8 nm · no_gap"
    )
    with pytest.raises(ValueError, match="imaging"):
        prompt_from_volume_spec(
            {"vol": "x", "native_resolution": [8, 8, 8]}, task="ssl",
        )
