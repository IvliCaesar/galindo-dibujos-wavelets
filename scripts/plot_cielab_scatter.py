"""
Renders the five palettes as points in the CIELAB a*-b* (hue-chroma) plane,
color-coded by register, each point drawn in its own actual sRGB color.
This is the direct visual counterpart of Table 2 (mean L*, mean C*, mean h,
hue spread): the table gives the summary numbers, this figure shows the
raw scatter they were computed from, including the two facts the numbers
alone must be read carefully to see -- Cronenberg's points cluster tightly
along a narrow ray from the origin at high radius (narrow hue, high
chroma), while Kafka's cluster just as tightly in angle but sit close to
the origin (narrow hue, low chroma).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_theory_analysis import hex_to_lab
from generative_field import PALETTES

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 180,
})

MARKERS = {"totoro": "o", "shishigami": "^", "cronenberg": "s",
           "kafka": "D", "university": "P"}
LABELS = {"totoro": "Totoro", "shishigami": "Shishigami",
          "cronenberg": "Cronenberg", "kafka": "Kafka",
          "university": "Institutional"}

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6, 6))

    max_r = 0
    for name, hexes in PALETTES.items():
        for hexcolor in hexes:
            lab = hex_to_lab(hexcolor)
            a, b = lab[1], lab[2]
            max_r = max(max_r, np.hypot(a, b))
            ax.scatter(a, b, s=140, marker=MARKERS[name], color=hexcolor,
                       edgecolors="black", linewidths=0.7, zorder=3)
        # one extra invisible point per palette, just to build the legend
        ax.scatter([], [], marker=MARKERS[name], color="none",
                   edgecolors="black", label=LABELS[name])

    ax.axhline(0, color="grey", linewidth=0.6, zorder=1)
    ax.axvline(0, color="grey", linewidth=0.6, zorder=1)
    lim = max_r * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$a^*$ (green $\leftrightarrow$ red)")
    ax.set_ylabel(r"$b^*$ (blue $\leftrightarrow$ yellow)")
    ax.set_title("Five palettes in the CIELAB $a^*$-$b^*$ plane", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig("../figures/cielab_scatter.pdf")
    fig.savefig("../figures/cielab_scatter.png")
    print("Saved ../figures/cielab_scatter.pdf / .png")
