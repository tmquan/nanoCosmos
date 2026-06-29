#!/usr/bin/env python
"""Graphical-abstract figure: Cosmos3-Edge (4B) for connectomics segmentation.

Single-row flow for a journal graphical abstract.  The EM input is a 2x2
DATA MATRIX -- columns = task role (SSL reconstruct / SFT segment), rows =
geometry (isotropic / anisotropic) -- covering ALL training datasets across
the resolution ladder.  Anisotropy is drawn LITERALLY: isotropic volumes are
near-cubic voxel glyphs; anisotropic volumes are squashed flat slabs (thin z /
single thick section, fine in-plane xy).  Dual-role datasets (FIB 8 nm, CREMI
40x4x4) appear in both their SSL and SFT cells.  The finest 4 nm (COSEM) rung
is marked as the reconstruction anchor.  Output: a coloured-instance cube.

Outputs (next to this file): cosmos3edge_graphical_abstract.{pdf,svg,png}
Run:  python doc/figures/cosmos3edge_graphical_abstract.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

INK = "#1b2430"
ditE, ditF = "#5b3e9b", "#e8e1f4"
vaeE = "#1f7a72"
headE = "#b5791f"
poolE = "#6a52a3"
mwsE = "#c2541b"
sslF, sslE = "#dceee4", "#2e7d55"
sftF, sftE = "#dbe6f6", "#2c5c9e"

fig, ax = plt.subplots(figsize=(17.0, 6.3))
ax.set_xlim(0, 342)
ax.set_ylim(0, 122)
ax.axis("off")


def cube(ox, oy, w, h, d, front, top, right, edge,
         gx=4, gy=4, gz=4, voxels=None, lw=1.0):
    """Oblique 3D voxel cube with per-axis grid (gx,gy in-plane; gz depth)."""
    dx, dy = d * 0.95, d * 0.85
    f = [(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h)]
    t = [(ox, oy + h), (ox + w, oy + h), (ox + w + dx, oy + h + dy), (ox + dx, oy + h + dy)]
    r = [(ox + w, oy), (ox + w + dx, oy + dy), (ox + w + dx, oy + h + dy), (ox + w, oy + h)]
    ax.add_patch(Polygon(r, closed=True, facecolor=right, edgecolor=edge, lw=lw, zorder=4))
    ax.add_patch(Polygon(t, closed=True, facecolor=top, edgecolor=edge, lw=lw, zorder=4))
    ax.add_patch(Polygon(f, closed=True, facecolor=front, edgecolor=edge, lw=lw, zorder=5))
    for i in range(1, gx):
        ax.plot([ox + i * w / gx] * 2, [oy, oy + h], color=edge, lw=0.4, alpha=0.5, zorder=5)
    for j in range(1, gy):
        ax.plot([ox, ox + w], [oy + j * h / gy] * 2, color=edge, lw=0.4, alpha=0.5, zorder=5)
    for i in range(1, gx):
        ax.plot([ox + i * w / gx, ox + i * w / gx + dx], [oy + h, oy + h + dy],
                color=edge, lw=0.35, alpha=0.35, zorder=5)
    for k in range(1, gz):
        fx, fy = k / gz * dx, k / gz * dy
        ax.plot([ox + fx, ox + w + fx], [oy + h + fy, oy + h + fy], color=edge, lw=0.5, alpha=0.5, zorder=5)
        ax.plot([ox + w + fx, ox + w + fx], [oy + fy, oy + h + fy], color=edge, lw=0.5, alpha=0.5, zorder=5)
    for j in range(1, gy):
        ax.plot([ox + w, ox + w + dx], [oy + j * h / gy, oy + j * h / gy + dy],
                color=edge, lw=0.35, alpha=0.35, zorder=5)
    if voxels is not None:
        rng = np.random.default_rng(7)
        cols = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                "#46c2c2", "#bcbd22", "#e377c2", "#2e7d55"]
        for a in range(gx):
            for b in range(gy):
                if rng.random() < 0.62:
                    ax.add_patch(Rectangle((ox + a * w / gx, oy + b * h / gy), w / gx, h / gy,
                                 facecolor=cols[(a * gx + b) % len(cols)],
                                 edgecolor="white", lw=0.5, alpha=0.92, zorder=5))


def caption(cx, text, color=INK, size=10.5, y=20, weight="bold"):
    ax.text(cx, y, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color)


def arrow(x1, x2, y=63, color=INK, lw=2.3):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=18,
                 lw=lw, color=color, shrinkA=1, shrinkB=1, zorder=1))


# ============ 1. EM input: 2x2 data matrix (SSL/SFT x ISO/ANISO) =============
BX, BY, BW, BH = 4, 24, 96, 72
ax.add_patch(FancyBboxPatch((BX, BY), BW, BH, boxstyle="round,pad=0.4,rounding_size=2.5",
             facecolor="#f7f8fa", edgecolor="#3b4a5a", linewidth=1.3, zorder=0))
DIVX, DIVY = 58, 61
ax.add_patch(Rectangle((13, 30), DIVX - 13, 60, facecolor=sslF, alpha=0.55, ec="none", zorder=0))
ax.add_patch(Rectangle((DIVX, 30), 38, 60, facecolor=sftF, alpha=0.55, ec="none", zorder=0))
ax.plot([DIVX, DIVX], [30, 90], color="#c2ccd6", lw=0.8, zorder=1)
ax.plot([13, 96], [DIVY, DIVY], color="#c2ccd6", lw=0.8, zorder=1)
ax.text(35, 92.5, "SSL · reconstruct", ha="center", va="center", fontsize=7.6,
        fontweight="bold", color=sslE)
ax.text(77, 92.5, "SFT · segment", ha="center", va="center", fontsize=7.6,
        fontweight="bold", color=sftE)
ax.text(8.6, 71, "isotropic", rotation=90, ha="center", va="center", fontsize=7.0, color="#5b6675")
ax.text(8.6, 44, "anisotropic", rotation=90, ha="center", va="center", fontsize=7.0, color="#5b6675")

ISO_T, ISO_B, ISO_R = "#eef2f6", "#f4f7fa", "#cdd5df"


def em_iso(ox, label, gx, gy, gz, finest=False):
    edge = headE if finest else "#46566a"
    cube(ox, 68, 10, 10, 5, "#e7edf3", ISO_T, ISO_R, edge, gx=gx, gy=gy, gz=gz,
         lw=1.6 if finest else 1.0)
    ax.text(ox + 6.4, 64.2, label, ha="center", va="center", fontsize=4.8, color="#46566a")
    if finest:
        ax.text(ox + 5, 82.5, "★ finest", ha="center", va="center", fontsize=5.2,
                color=headE, fontweight="bold")


def em_aniso(ox, label, gx, gy, ly=35.2):
    # squashed flat slab: thin z (d) + single thick section (gz=1), fine xy
    cube(ox, 39, 10, 6, 2.2, "#dde6ec", "#e8eef4", "#c6d0da", "#46566a",
         gx=gx, gy=gy, gz=1, lw=1.0)
    ax.text(ox + 5.4, ly, label, ha="center", va="center", fontsize=4.1, color="#46566a")


# ISO row (near-cubic voxel glyphs)
em_iso(15, "COSEM 4", 8, 8, 5, finest=True)   # SSL-ISO
em_iso(31, "FIB 8", 6, 6, 5)                   # SSL-ISO (dual-role)
em_iso(46, "MitoEM 16", 4, 4, 4)               # SSL-ISO
em_iso(70, "FIB 8", 6, 6, 5)                   # SFT-ISO (dual-role)
# ANISO row (flat slabs); stagger SFT labels to avoid horizontal collisions
em_aniso(16, "MitoEM 30·8·8", 6, 6)            # SSL-ANISO
em_aniso(38, "CREMI 40·4·4", 10, 10)           # SSL-ANISO (dual-role)
em_aniso(59, "SNEMI 30·6·6", 8, 8, ly=35.6)    # SFT-ANISO
em_aniso(72, "MICrONS 40·8·8", 6, 6, ly=32.4)  # SFT-ANISO
em_aniso(85, "CREMI 40·4·4", 10, 10, ly=35.6)  # SFT-ANISO (dual-role)
caption(50, "EM volumes", size=10.5)

# ============ 2. Wan2.2 VAE encoder (frozen) =================================
vx = 114
ax.add_patch(Polygon([(vx, 49), (vx + 28, 58), (vx + 28, 78), (vx, 87)],
             closed=True, facecolor="#d6edea", edgecolor=vaeE, lw=1.7, zorder=2))
ax.text(vx + 14, 72, "16×/4×", ha="center", va="center", fontsize=8.4, color=vaeE, fontweight="bold")
ax.text(vx + 14, 64, "→ z48", ha="center", va="center", fontsize=7.8, color=INK)
caption(vx + 14, "Wan2.2 VAE", color=vaeE, size=10.5)

# ============ 3. Cosmos3-Edge transformer ====================================
tx = 166
for i in range(9):
    hl = i in (2, 5, 8)
    ax.add_patch(Rectangle((tx, 52 + i * 3.7), 34, 2.8,
                 facecolor=(ditE if hl else "#c9bce6"), edgecolor="none", zorder=3))
ax.add_patch(Rectangle((tx - 1, 50.5), 36, 36, facecolor="none", edgecolor=ditE,
             lw=1.7, zorder=2, joinstyle="round"))
ax.text(tx + 17, 81, "4B", ha="center", va="center", fontsize=9.6, color="white",
        fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=ditE, ec="none"))
caption(tx + 17, "Cosmos3-Edge", color=ditE, size=10.5)
caption(tx + 17, "Nano→reduced · 28L / 2048d", size=7.4, y=15, weight="normal", color="#5b6675")

# ============ 4. Unified head ================================================
hx = 218
for i, (lab, col) in enumerate([("raw", "#e7c98a"), ("sem", "#d9a44a"), ("aff", headE)]):
    ax.add_patch(Rectangle((hx, 55 + i * 7.7), 24, 6.4, facecolor=col,
                 edgecolor="white", lw=1.0, zorder=3))
    ax.text(hx + 12, 58.2 + i * 7.7, lab, ha="center", va="center", fontsize=7.4,
            color="white", fontweight="bold", zorder=4)
caption(hx + 12, "unified head", color=headE, size=10.5)

# ============ 5. Multi-resolution pooling ====================================
pcx, pcy = 272, 69
for i, s in enumerate([20, 15, 10, 5.5]):
    ax.add_patch(Rectangle((pcx - s / 2, pcy - s / 2), s, s,
                 facecolor=("white" if i % 2 else ditF), edgecolor=poolE, lw=1.1, zorder=3 + i))
ax.text(pcx, pcy - 13.5, "8–40 nm", ha="center", va="center", fontsize=6.6, color=poolE)
caption(pcx, "pool → res.", color=poolE, size=10.0)

# ============ 6. Mutex Watershed → instance segmentation =====================
cube(296, 52, 24, 30, 10, "#dfe5ec", "#eef1f5", "#cdd5df", "#3b4a5a",
     gx=5, gy=5, gz=4, voxels=True)
caption(311, "instance seg.", size=10.0)
caption(311, "Mutex Watershed", size=7.4, y=15, weight="normal", color=mwsE)

# ============ arrows =========================================================
arrow(100, 114)
arrow(142, 166)
arrow(202, 218)
arrow(244, 260)
arrow(284, 296, y=62)

# ============ title ==========================================================
ax.text(176, 114, "Cosmos3-Edge (4B) for Connectomics Segmentation",
        ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)
ax.text(176, 107, "SSL/SFT × isotropic/anisotropic data ladder · frozen Wan2.2 "
        "tokenizer · Nano→Edge generator · one fine-grid head pooled to native res.",
        ha="center", va="center", fontsize=7.8, color="#5b6675", style="italic")

here = os.path.dirname(os.path.abspath(__file__))
stem = os.path.join(here, "cosmos3edge_graphical_abstract")
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300, facecolor="white", pad_inches=0.18)
    print("wrote", f"{stem}.{ext}")
