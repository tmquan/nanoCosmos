"""
Cross-cutting utilities used throughout nanocosmos.

A small bag of helpers that don't fit cleanly into ``datasets`` /
``transforms`` / ``inference``:

I/O (:mod:`nanocosmos.utils.io`)
    * :func:`find_folder` -- locate a volume file in a directory by
      base name (any of the supported extensions).
    * :func:`load_volume` / :func:`save_volume` -- format-agnostic
      reader / writer dispatching to :mod:`nanocosmos.preprocessors`.

Extending this module: anything reused by **two or more** subpackages
and with no better home belongs here.  Anything used by exactly one
subpackage should live next to its consumer.
"""

from nanocosmos.utils.io import find_folder, load_volume, save_volume

__all__ = [
    "find_folder",
    "load_volume",
    "save_volume",
]
