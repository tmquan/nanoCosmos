#!/usr/bin/env python
"""Publication-quality teaser figure: Cosmos3-Edge (4B) for connectomics seg.

Renders a vector (PDF + SVG) + raster (PNG) architecture teaser in the style of
SIGGRAPH / IEEE TVCG / IEEE TMI / MedIA method figures:

  * snake (U-shaped) two-row layout, left->right then right->left;
  * flat panels with a restrained, colour-blind-safe palette;
  * the multi-resolution "resolution ladder" + single-fine-grid prediction that
    fans out (pools) to every native GT resolution;
  * frozen Wan2.2-TI2V VAE tokenizer, Cosmos3-Edge omni generator tower
    (reduced from Nano), unified affinity/semantic/raw head, Mutex Watershed.

Outputs (next to this file): cosmos3edge_teaser.{pdf,svg,png}
Run:  python doc/figures/cosmos3edge_teaser.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

# ---- camera-ready rcParams (text stays text in SVG; TrueType embed in PDF) ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.0,
})

# ---- palette (muted, colour-blind-safe families) ----------------------------
INK = "#1b2430"
C = {
    "in":    ("#eef1f5", "#3b4a5a"),   # inputs / slate
    "ssl":   ("#dceee4", "#2e7d55"),   # green
    "sft":   ("#dbe6f6", "#2c5c9e"),   # blue
    "vae":   ("#d6edea", "#1f7a72"),   # teal
    "dit":   ("#e8e1f4", "#5b3e9b"),   # violet
    "head":  ("#fbeed9", "#b5791f"),   # amber
    "pool":  ("#eae6f3", "#6a52a3"),   # light violet
    "loss":  ("#f3e7ef", "#9c3f7a"),   # mauve
    "mws":   ("#fae3d4", "#c2541b"),   # orange
    "seg":   ("#eef1f5", "#3b4a5a"),
}

fig, ax = plt.subplots(figsize=(18.0, 9.4))
ax.set_xlim(0, 180)
ax.set_ylim(0, 98)
ax.axis("off")


def box(x, y, w, h, key, title, subs=None, ts=10.5, ss=8.0, lw=1.5):
    fc, ec = C[key]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.2",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    ))
    cx = x + w / 2
    ax.text(cx, y + h - 2.6, title, ha="center", va="top",
            fontsize=ts, fontweight="bold", color=ec, zorder=3)
    if subs:
        ax.text(cx, y + h - 2.6 - ts * 0.42, "\n".join(subs), ha="center",
                va="top", fontsize=ss, color=INK, linespacing=1.45, zorder=3)
    return cx, y + h / 2


def arrow(p1, p2, color=INK, lw=1.7, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=13, lw=lw, color=color,
        shrinkA=3, shrinkB=3, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
    ))


# ===================== TOP ROW (left -> right) ===============================
# 1) Inputs + resolution ladder ------------------------------------------------
ix, iy, iw, ih = 3, 50, 27, 40
box(ix, iy, iw, ih, "in", "EM volumes", ["resolution ladder"], ts=11)
ladder = [("4 nm", "COSEM", 7), ("8 nm", "FlyEM / FIB", 9.5),
          ("16 nm", "MitoEM", 12), ("30 nm", "SNEMI", 15), ("40 nm", "CREMI / MICrONS", 18)]
ly = iy + 4
for i, (nm, ds, bw) in enumerate(ladder):
    shade = 0.74 - i * 0.10
    g = f"{shade:.2f}"
    ax.add_patch(Rectangle((ix + 3, ly), bw, 3.0, facecolor=g, edgecolor="#33414f",
                           linewidth=0.7, zorder=3))
    ax.text(ix + 3 + bw + 1.2, ly + 1.5, f"{nm}  {ds}", ha="left", va="center",
            fontsize=6.6, color=INK, zorder=3)
    ly += 5.3

# 2) Two task branches ---------------------------------------------------------
ssl_c = box(35, 71, 27, 16, "ssl", "SSL  ·  label-free",
            ["degrade small-voxel EM", "→ reconstruct clean EM"], ts=9.5, ss=7.6)
sft_c = box(35, 51, 27, 16, "sft", "SFT  ·  labeled",
            ["pool head → native grid", "→ affinity + semantic"], ts=9.5, ss=7.6)

# 3) Frozen VAE tokenizer ------------------------------------------------------
vae_c = box(67, 56, 26, 26, "vae", "Wan2.2-TI2V VAE",
            ["frozen tokenizer (❄)", "16× spatial / 4× temporal", "→ 48-ch latent",
             "EM depth ↔ video time", "repatch latent-patch 2→1"], ts=10, ss=7.4)

# 4) Cosmos3-Edge transformer (text panel + right-side layer-stack strip) ------
tx, ty, tw, th = 96, 48, 36, 42
_fc, _ec = C["dit"]
ax.add_patch(FancyBboxPatch(
    (tx, ty), tw, th, boxstyle="round,pad=0.4,rounding_size=2.2",
    facecolor=_fc, edgecolor=_ec, linewidth=1.5, zorder=2))
_tcx = tx + (tw - 7) / 2 + 1.0
ax.text(_tcx, ty + th - 2.6, "Cosmos3-Edge omni transformer", ha="center",
        va="top", fontsize=9.4, fontweight="bold", color=_ec, zorder=3)
ax.text(_tcx, ty + th - 8.0, "\n".join([
    "generator tower · null cond.",
    "2048 d · 28 L · 16 h · 8 kv",
    "FFN 6144 · head_dim 128",
    "MoT · 3D mRoPE",
    "Edge ≡ Qwen3-1.7B shape",
    "reduced from Nano:",
    "36→28 L · 4096→2048 d",
]), ha="center", va="top", fontsize=7.0, color=INK, linespacing=1.55, zorder=3)
# right-side layer-stack strip with 4 highlighted feature taps
n_layers, taps = 28, {7, 14, 21, 27}
sx0, sw_ = tx + tw - 5.4, 3.6
sy0, gap = ty + 2.4, (th - 4.8) / n_layers
tap_pts = []
for i in range(n_layers):
    yy = sy0 + i * gap
    is_tap = i in taps
    ax.add_patch(Rectangle((sx0, yy), sw_, gap * 0.72,
                 facecolor=(_ec if is_tap else "#c9bce6"), edgecolor="none", zorder=3))
    if is_tap:
        tap_pts.append((sx0 + sw_, yy + gap * 0.36))

# 5) Feature projector + 3D decoder head --------------------------------------
proj_c = box(135, 74, 24, 12, "dit", "feature projector",
             ["4 layer taps → feature_size"], ts=9, ss=7.2)
head_c = box(135, 56, 24, 14, "head", "3D decoder head",
             ["ONE prediction", "on FINE 4 nm grid"], ts=9.5, ss=7.6)

# top-row arrows
arrow((ix + iw, 76), (35, 79))                       # input -> SSL
arrow((ix + iw, 64), (35, 59))                       # input -> SFT
arrow((35 + 27, 79), (67, 73))                       # SSL -> VAE
arrow((35 + 27, 59), (67, 67))                       # SFT -> VAE
arrow((67 + 26, 69), (tx, 69))                       # VAE -> transformer
for p in tap_pts:                                    # taps -> projector
    arrow(p, (135, 80), color=C["dit"][1], lw=1.0, rad=-0.12)
arrow((147, 74), (147, 70))                          # projector -> head

# ===================== TURN: head -> bottom row ==============================
arrow((159, 63), (170, 63), rad=0.0)
arrow((170, 63), (170, 44), rad=0.0)                 # go down on the right

# ===================== BOTTOM ROW (right -> left) ============================
# unified head outputs (right) ------------------------------------------------
aff_c = box(150, 38, 24, 8.5, "head", "Affinities · 30 offsets", ts=8.6)
sem_c = box(150, 28, 24, 8.5, "head", "Semantic / foreground", ts=8.6)
raw_c = box(150, 18, 24, 8.5, "head", "Raw EM reconstruction", ts=8.6)
arrow((170, 44), (162, 46.5))                        # turn into outputs
ax.add_patch(FancyArrowPatch((174, 42.25), (174, 36.5), arrowstyle="-",
             lw=1.2, color="#999", zorder=1))
ax.add_patch(FancyArrowPatch((174, 32.25), (174, 26.5), arrowstyle="-",
             lw=1.2, color="#999", zorder=1))

# multi-resolution pooling fan-out --------------------------------------------
pool_c = box(96, 30, 40, 14, "pool", "multi-resolution pooling",
             ["fan-out  4 → 8 → 16 → 30 → 40 nm",
              "pool factor = native / fine"], ts=9.5, ss=7.8)
arrow((150, 42), (136, 39), rad=0.10)                # aff -> pool
arrow((150, 32), (136, 34))                          # sem -> pool

# training losses (below pool) ------------------------------------------------
loss_c = box(96, 12, 40, 12, "loss", "Joint supervision",
             ["AffinityFGLoss @ each native grid (SFT)",
              "L1 reconstruction vs clean EM (SSL)"], ts=9, ss=7.4)
arrow((116, 30), (116, 24))                          # pool -> loss
arrow((150, 20.5), (136, 16.5), rad=0.08, color=C["loss"][1])  # raw -> L1

# Mutex Watershed + segmentation (left) ---------------------------------------
mws_c = box(54, 30, 34, 14, "mws", "Mutex Watershed",
            ["semantic-gated agglomeration", "(inference / eval)"], ts=9.5, ss=7.6)
arrow((96, 37), (88, 37))                            # pool -> MWS

# segmentation result with coloured neuron fragments
sx, sy, sw, sh = 8, 22, 38, 24
box(sx, sy, sw, sh, "seg", "3D instance segmentation", [""], ts=9.5)
arrow((54, 37), (sx + sw, 34))                       # MWS -> seg
frag_colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#46c2c2", "#bcbd22", "#e377c2"]
fx = [13, 22, 31, 39, 17, 27, 36, 21]
fy = [27, 30, 26, 31, 34, 36, 33, 39]
fr = [3.2, 2.6, 3.0, 2.4, 2.2, 2.8, 2.3, 2.0]
for cx, cy, r, col in zip(fx, fy, fr, frag_colors):
    ax.add_patch(plt.Circle((cx, cy), r, facecolor=col, edgecolor="white",
                            linewidth=0.8, alpha=0.92, zorder=3))

# ===================== titles / framing ======================================
ax.text(90, 95.5, "Cosmos3-Edge (4B) for Connectomics Segmentation",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color=INK)
ax.text(90, 92.0,
        "frozen Wan2.2 tokenizer  ·  Nano→Edge reduced omni generator  ·  "
        "single fine-grid head pooled across the resolution ladder",
        ha="center", va="center", fontsize=9.2, color="#54606e", style="italic")

here = os.path.dirname(os.path.abspath(__file__))
stem = os.path.join(here, "cosmos3edge_teaser")
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300,
                facecolor="white", pad_inches=0.12)
    print("wrote", f"{stem}.{ext}")
