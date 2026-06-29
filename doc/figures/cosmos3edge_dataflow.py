#!/usr/bin/env python
"""Data-flow / method figure: Cosmos3-Edge (4B) connectomics pipeline.

Mirrors the requested flowchart style:

  * TOP-LEFT  (SSL): isotropic source tensors 4^3 / 8^3 / 16^3  --Degrade-->
    anisotropic target volumes (6x6x30, 8x8x30, 4x4x40, 8x8x40).
  * TOP-RIGHT (SFT): the same native volumes WITHOUT the degrade op (real
    labeled data).
  * BOTTOM (model): resample -> Wan2.2 VAE encoder -> Cosmos3-Edge backbone ->
    decoder -> resample, predicting three 4 nm heads (aff / sem / raw).

Tensor glyphs: isotropic = square sized by voxel; anisotropic = wide thin
rectangle (z >> xy).  Ops = circles.  Encoder/decoder = trapezoids; backbone =
rounded rectangle.

Outputs: cosmos3edge_dataflow.{pdf,svg,png}.  Run: python doc/figures/cosmos3edge_dataflow.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle, FancyArrowPatch, FancyBboxPatch, Polygon,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

INK = "#1b2430"
BLU_F, BLU_E = "#dce8f7", "#6f9fd8"
YEL_F, YEL_E = "#fde7c4", "#e0a93f"
TEAL_F, TEAL_E = "#cfeae6", "#2e9b8a"
PINK_F, PINK_E = "#f6d6d9", "#d06b76"
CYA_F, CYA_E = "#cfe6f3", "#3a8ec0"
ORA_F, ORA_E = "#fde6c6", "#e0982f"
GRN_F, GRN_E = "#d8ebca", "#6fae53"
PUR_F, PUR_E = "#e2d7f3", "#7b5bb6"
SSL_C, SFT_C = "#2e7d55", "#2c5c9e"
AR = "#2e9b8a"

fig, ax = plt.subplots(figsize=(13.6, 8.4))
ax.set_xlim(0, 204)
ax.set_ylim(0, 128)
ax.axis("off")


def sq(cx, cy, side, label, fc=BLU_F, ec=BLU_E, fs=8.5, lw=1.4):
    ax.add_patch(FancyBboxPatch((cx - side / 2, cy - side / 2), side, side,
                 boxstyle="round,pad=0.2,rounding_size=1.2", facecolor=fc,
                 edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def rect(x, y, w, h, label, fc=BLU_F, ec=BLU_E, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                 boxstyle="round,pad=0.2,rounding_size=1.4", facecolor=fc,
                 edgecolor=ec, linewidth=1.3, zorder=3))
    ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def circ(cx, cy, r, label, fc=TEAL_F, ec=TEAL_E, fs=7.0):
    ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(p1, p2, color=AR, lw=1.5, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                 lw=lw, color=color, shrinkA=2, shrinkB=2,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def trap(x, y, w, h, taper, label, fc, ec, fs=8.0, flip=False):
    if not flip:  # narrowing to the right (encoder)
        pts = [(x, y - h / 2), (x, y + h / 2), (x + w, y + h / 2 - taper), (x + w, y - h / 2 + taper)]
    else:         # widening to the right (decoder)
        pts = [(x, y - h / 2 + taper), (x, y + h / 2 - taper), (x + w, y + h / 2), (x + w, y - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3))
    ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


# ============================ data panel =====================================
ANISO = ["6×6×30", "8×8×30", "4×4×40", "8×8×40"]
ANISO_DS = ["SNEMI", "MitoEM", "CREMI", "MICrONS"]
RY = [104, 92.5, 81, 69.5]  # rect y-centers


def data_panel(ox, with_degrade):
    xl = ox + 4
    sq(xl + 4.5, 104, 9, "4³", fc=YEL_F, ec=YEL_E, fs=8.5)   # finest
    ax.text(xl + 4.5, 110.5, "★ 4 nm", ha="center", va="center", fontsize=5.6,
            color=YEL_E, fontweight="bold")
    sq(xl + 6, 86, 12, "8³")
    sq(xl + 8.5, 66, 17, "16³", fs=9)
    rx = ox + 40
    for yy, lab, ds in zip(RY, ANISO, ANISO_DS):
        rect(rx, yy, 44, 9.4, lab)
        ax.text(rx + 44 + 1.5, yy, ds, ha="left", va="center", fontsize=5.2, color="#7a8794")
    if with_degrade:
        dcx, dcy = ox + 26, 86
        circ(dcx, dcy, 7.5, "Degrade", fc=TEAL_F, ec=TEAL_E, fs=6.0)
        for cy_i in (104, 86, 66):
            arr((xl + 18, cy_i), (dcx - 7.5, dcy), rad=0.0 if cy_i == 86 else (0.12 if cy_i < 86 else -0.12))
        for yy in RY:
            arr((dcx + 7.5, dcy), (rx, yy), rad=(yy - dcy) * 0.004)
    else:
        # native: a light bracket from iso/aniso block down (no degrade)
        ax.text(ox + 26, 86, "native\n(no degrade)", ha="center", va="center",
                fontsize=6.0, color="#9aa4b0", style="italic")


data_panel(0, with_degrade=True)
data_panel(110, with_degrade=False)
ax.text(40, 120, "SSL · synthetic degrade pairs (label-free)", ha="center",
        va="center", fontsize=9.5, fontweight="bold", color=SSL_C)
ax.text(152, 120, "SFT · native labeled volumes", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=SFT_C)
ax.text(102, 113, "isotropic 4³/8³/16³  vs  anisotropic  y×x×z (nm)", ha="center",
        va="center", fontsize=6.6, color="#9aa4b0", style="italic")

# ============================ model row ======================================
ymid = 30
sq(16, ymid, 9, "4³", fc=PINK_F, ec=PINK_E, fs=8.0)
circ(36, ymid, 8, "Resample", fc=PINK_F, ec=PINK_E, fs=5.8)
trap(52, ymid, 26, 30, 6, "Wan2.2\nVAE enc.", ORA_F, ORA_E, fs=7.6)
ax.add_patch(FancyBboxPatch((84, ymid - 14), 44, 28, boxstyle="round,pad=0.3,rounding_size=2.5",
             facecolor=GRN_F, edgecolor=GRN_E, linewidth=1.8, zorder=3))
ax.text(106, ymid + 3.5, "Cosmos3-Edge", ha="center", va="center", fontsize=9.5,
        fontweight="bold", color=GRN_E, zorder=4)
ax.text(106, ymid - 4.5, "4B · 28L / 2048d", ha="center", va="center", fontsize=6.8,
        color=INK, zorder=4)
trap(134, ymid, 26, 30, 6, "decoder\n+ head", TEAL_F, TEAL_E, fs=7.6, flip=True)
circ(178, ymid, 8, "Resample", fc=CYA_F, ec=CYA_E, fs=5.8)
for yy, lab, fc, ec in [(ymid + 11, "aff", PINK_F, PINK_E),
                        (ymid, "sem", PUR_F, PUR_E),
                        (ymid - 11, "raw", "#ffffff", "#9aa4b0")]:
    sq(196, yy, 8, "4³", fc=fc, ec=ec, fs=6.6)
    ax.text(196 + 6.5, yy, lab, ha="left", va="center", fontsize=6.0, color="#54606e")

# model-row arrows
arr((20.5, ymid), (28, ymid))
arr((44, ymid), (52, ymid))
arr((78, ymid - 0.0), (84, ymid))
arr((128, ymid), (134, ymid))
arr((160, ymid), (170, ymid))
for yy in (ymid + 11, ymid, ymid - 11):
    arr((186, ymid), (192, yy), rad=(yy - ymid) * 0.02)

# data -> model connectors (resample to fine grid)
arr((40, 60), (34, 40), color="#9aa4b0", lw=1.3, rad=-0.18, style="-|>")
arr((150, 60), (40, 38), color="#9aa4b0", lw=1.3, rad=0.20, style="-|>")
ax.text(64, 49, "resample → 4 nm fine grid", ha="center", va="center",
        fontsize=6.4, color="#7a8794", style="italic")
ax.text(150, 14, "predict at 4 nm · pool to native res. for loss", ha="center",
        va="center", fontsize=6.4, color="#7a8794", style="italic")

# ============================ title ==========================================
ax.text(102, 126.5, "Cosmos3-Edge (4B) for Connectomics Segmentation",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color=INK)

here = os.path.dirname(os.path.abspath(__file__))
stem = os.path.join(here, "cosmos3edge_dataflow")
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300, facecolor="white", pad_inches=0.16)
    print("wrote", f"{stem}.{ext}")
