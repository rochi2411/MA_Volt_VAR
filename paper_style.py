"""
Shared matplotlib style for publication-ready (IEEE Transactions) figures.

Import and call ``apply_paper_style()`` once, right after importing pyplot, in
every script that produces paper figures. This guarantees a single, consistent
look across all figures:

  * One sans-serif family (Arial/Helvetica, DejaVu Sans fallback).
  * Font sizes that stay readable when a figure is scaled to single-column
    (3.5 in) width: 9 pt minimum for ticks/legends.
  * 300 DPI raster export with tight bounding boxes.
  * TrueType (Type 42) font embedding in PDF/PS output. IEEE rejects the
    default Type-3 fonts, and Type-3 is a common cause of garbled text on
    export -- Type 42 embeds the actual glyphs.
"""

import matplotlib as mpl

PAPER_RCPARAMS = {
    # --- Consistent sans-serif family across every figure ---
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    # --- Sizes: readable at single-column (3.5 in) scaling ---
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,

    # --- Export: 300 DPI, tight bbox, embedded TrueType fonts ---
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style():
    """Apply the shared publication style to the global matplotlib rcParams."""
    mpl.rcParams.update(PAPER_RCPARAMS)
