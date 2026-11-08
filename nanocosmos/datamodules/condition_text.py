"""Fixed short batch prompts for joint Cosmos3 training.

Template::

    {imaging} · {tail}

where ``tail`` is ``ssl`` on the SSL branch and ``gapped`` / ``filled``
(``label_convention``) on SFT.  The prompt is logged to TensorBoard for
debugging; the network is not conditioned on it (null text).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

LABEL_CONVENTIONS = frozenset({"gapped", "filled"})


def validate_label_convention(raw: Any, *, vol: Optional[str] = None) -> Optional[str]:
    """Return validated convention or ``None`` if unset."""
    if raw is None or raw == "":
        return None
    conv = str(raw)
    if conv not in LABEL_CONVENTIONS:
        where = f" on vol={vol!r}" if vol else ""
        raise ValueError(
            f"label_convention must be one of {sorted(LABEL_CONVENTIONS)}; "
            f"got {raw!r}{where}."
        )
    return conv


def build_condition_text(
    imaging: str,
    *,
    task: str,
    label_convention: Optional[str] = None,
) -> str:
    """Build the fixed metadata prompt string.

    Examples
    --------
    >>> build_condition_text("ssTEM", task="sft", label_convention="filled")
    'ssTEM · filled'
    >>> build_condition_text("FIB-SEM", task="ssl")
    'FIB-SEM · ssl'
    """
    if not imaging or not str(imaging).strip():
        raise ValueError("imaging must be a non-empty string (e.g. 'ssTEM', 'FIB-SEM').")
    task_l = str(task).lower()
    if task_l == "ssl":
        tail = "ssl"
    else:
        conv = validate_label_convention(label_convention)
        tail = conv if conv is not None else "filled"
    return f"{str(imaging).strip()} · {tail}"


def prompt_from_volume_spec(vol: Mapping[str, Any], *, task: str) -> str:
    """Build the condition text from a joint volume YAML dict."""
    imaging = vol.get("imaging")
    if imaging is None or str(imaging).strip() == "":
        raise ValueError(
            f"volume {vol.get('vol')!r} is missing required 'imaging' "
            f"(e.g. ssTEM / FIB-SEM) for text conditioning."
        )
    conv = validate_label_convention(
        vol.get("label_convention"), vol=str(vol.get("vol")),
    )
    return build_condition_text(str(imaging), task=task, label_convention=conv)


__all__ = [
    "LABEL_CONVENTIONS",
    "build_condition_text",
    "prompt_from_volume_spec",
    "validate_label_convention",
]
