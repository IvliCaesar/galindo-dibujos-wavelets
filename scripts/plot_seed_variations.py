"""
Renders three independent random-texture realizations (different SEED
values into wavelet_texture's deterministic hash) of each of the five
registers, to give Sec. 3.1's "stochastic remark" -- that replacing the
deterministic wavelet coefficients with a random field turns Phi into a
random field, and the visible creature at a threshold into a random
compact set K_u(omega) -- an actual figure. Until this script, that whole
paragraph had no visual support at all.

This also directly demonstrates the claim underlying every register
description in Sec. 4: that a "Cronenberg register" or "Kafka register" is
a *reading* stable across many realizations of the same field and
parameters, not a single cherry-picked rendering that happens to look
right. Three seeds per register are shown; if the reading changed
qualitatively across seeds, that claim would be false, and this script is
what would have caught it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generative_field import generative_field, REGIMES, PALETTES
from render_creatures import make_cmap, STYLE, apply_vignette, TITLES
from generative_field import EXTENT

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "savefig.dpi": 200,
})

SEEDS = [3, 7, 15]
ORDER = ["totoro", "shishigami", "cronenberg", "kafka", "university"]


def render_variation(name, seed, ax):
    X, Y, Phi = generative_field(REGIMES[name], seed=seed)
    cmap = make_cmap(name)
    style = STYLE[name]
    ax.imshow(Phi, extent=[X.min(), X.max(), Y.min(), Y.max()],
              origin="lower", cmap=cmap, interpolation="bilinear", zorder=1)
    for lev in style["contour_levels"]:
        ax.contour(X, Y, Phi, levels=[Phi.max() * lev], colors="black",
                   linewidths=0.5, alpha=0.45, zorder=3)
    apply_vignette(ax, style["vignette"], style["vignette_strength"], EXTENT)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    fig, axes = plt.subplots(len(ORDER), len(SEEDS),
                              figsize=(3.0 * len(SEEDS), 3.0 * len(ORDER)))
    for row, name in enumerate(ORDER):
        for col, seed in enumerate(SEEDS):
            ax = axes[row][col]
            render_variation(name, seed, ax)
            if col == 0:
                ax.set_ylabel(TITLES[name].replace(" register", "")
                              .replace(" (same field as Totoro)", ""),
                              fontsize=11, rotation=90)
            if row == 0:
                ax.set_title(f"seed = {seed}", fontsize=10)
    fig.suptitle(r"Three independent texture realizations $T_\omega$ per register "
                 r"(Sec.\ 3.1's stochastic remark, made concrete)", fontsize=12)
    fig.tight_layout()
    fig.savefig("../figures/seed_variations.pdf")
    fig.savefig("../figures/seed_variations.png")
    print("Saved ../figures/seed_variations.pdf / .png")
