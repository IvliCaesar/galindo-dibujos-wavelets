"""
Renders the five creatures (Totoro, Shishigami, Cronenberg, Kafka,
University) from the generative field Phi(x,y) of generative_field.py,
each with its own palette, plus a combined comparison figure for the
article.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from generative_field import generative_field, REGIMES, PALETTES

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "savefig.dpi": 180,
})

SEED = 7
ORDER = ["totoro", "shishigami", "cronenberg", "kafka", "university"]
TITLES = {
    "totoro": "Totoro register",
    "shishigami": "Shishigami register",
    "cronenberg": "Cronenberg register",
    "kafka": "Kafka register",
    "university": "Institutional register (same field as Totoro)",
}


def make_cmap(name):
    return LinearSegmentedColormap.from_list(name, PALETTES[name], N=256)


def render_one(name, ax=None):
    X, Y, Phi = generative_field(REGIMES[name], seed=SEED)
    cmap = make_cmap(name)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(Phi, extent=[X.min(), X.max(), Y.min(), Y.max()],
              origin="lower", cmap=cmap, interpolation="bilinear")
    ax.contour(X, Y, Phi, levels=[Phi.max() * 0.32], colors="black",
               linewidths=0.6, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(TITLES[name], fontsize=10)
    if standalone:
        fig.tight_layout()
        fig.savefig(f"../figures/creature_{name}.pdf")
        fig.savefig(f"../figures/creature_{name}.png")
        plt.close(fig)


if __name__ == "__main__":
    for name in ORDER:
        render_one(name)
        print(f"Saved ../figures/creature_{name}.png")

    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.4))
    for ax, name in zip(axes, ORDER):
        render_one(name, ax=ax)
    fig.suptitle(r"Same construction, five color maps (four fields + one recolored field)", fontsize=11)
    fig.tight_layout()
    fig.savefig("../figures/creatures_comparison.pdf")
    fig.savefig("../figures/creatures_comparison.png")
    print("Saved ../figures/creatures_comparison.pdf / .png")
