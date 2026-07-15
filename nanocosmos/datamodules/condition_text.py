"""Fixed short text-condition templates for joint Cosmos3 training.

Template::

    {imaging} · z{Z} y{Y} x{X} nm · {tail}

where ``tail`` is ``ssl`` on the SSL branch and ``gapped`` / ``filled``
(``label_convention``) on SFT.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

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
    native_resolution: Sequence[float],
    *,
    task: str,
    label_convention: Optional[str] = None,
) -> str:
    """Build the fixed metadata prompt string.

    Examples
    --------
    >>> build_condition_text("ssTEM", [40, 8, 8], task="sft", label_convention="filled")
    'ssTEM · z40 y8 x8 nm · filled'
    >>> build_condition_text("FIB-SEM", [8, 8, 8], task="ssl")
    'FIB-SEM · z8 y8 x8 nm · ssl'
    """
    if not imaging or not str(imaging).strip():
        raise ValueError("imaging must be a non-empty string (e.g. 'ssTEM', 'FIB-SEM').")
    if len(native_resolution) < 3:
        raise ValueError(
            f"native_resolution must be [z, y, x]; got {list(native_resolution)!r}."
        )
    z, y, x = (int(r) for r in native_resolution[:3])
    task_l = str(task).lower()
    if task_l == "ssl":
        tail = "ssl"
    else:
        conv = validate_label_convention(label_convention)
        tail = conv if conv is not None else "filled"
    return f"{str(imaging).strip()} · z{z} y{y} x{x} nm · {tail}"


def prompt_from_volume_spec(vol: Mapping[str, Any], *, task: str) -> str:
    """Build the condition text from a joint volume YAML dict."""
    imaging = vol.get("imaging")
    if imaging is None or str(imaging).strip() == "":
        raise ValueError(
            f"volume {vol.get('vol')!r} is missing required 'imaging' "
            f"(e.g. ssTEM / FIB-SEM) for text conditioning."
        )
    res = vol.get("native_resolution")
    if res is None:
        raise ValueError(
            f"volume {vol.get('vol')!r} is missing 'native_resolution'."
        )
    conv = validate_label_convention(
        vol.get("label_convention"), vol=str(vol.get("vol")),
    )
    return build_condition_text(
        str(imaging), res, task=task, label_convention=conv,
    )


__all__ = [
    "LABEL_CONVENTIONS",
    "build_condition_text",
    "prompt_from_volume_spec",
    "validate_label_convention",
]
