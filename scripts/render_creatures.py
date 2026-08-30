"""
Renders the five creatures (Totoro, Shishigami, Cronenberg, Kafka,
University) from the generative field Phi(x,y) of generative_field.py,
each with its own palette AND its own register-appropriate rendering
treatment (contour count, vignette, frame), plus a combined comparison
figure for the article. Only the underlying field Phi and the color map
differ per the article's own thesis (Sec. 4); the rendering treatments
added here (vignette, extra contour levels, a shield frame) are entirely
cosmetic presentation choices layered on top for the standalone figures,
disclosed as such -- they change how the same field is *displayed*, not
the field or the palette.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from generative_field import generative_field, REGIMES, PALETTES, EXTENT

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "savefig.dpi": 220,
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

# Register-specific presentation treatments (cosmetic only; see docstring).
STYLE = {
    "totoro": dict(contour_levels=[0.32], vignette="warm", vignette_strength=0.35),
    "shishigami": dict(contour_levels=[0.22, 0.42], vignette="cool", vignette_strength=0.30),
    "cronenberg": dict(contour_levels=[0.20, 0.32, 0.48], vignette="none", vignette_strength=0.0),
    "kafka": dict(contour_levels=[0.32], vignette="none", vignette_strength=0.0),
    "university": dict(contour_levels=[0.32], vignette="none", vignette_strength=0.0, shield=True),
}


def make_cmap(name):
    return LinearSegmentedColormap.from_list(name, PALETTES[name], N=256)


def apply_vignette(ax, kind, strength, extent):
    """Cosmetic radial fade toward the panel edge. 'warm' darkens toward
    warm-neutral at the corners (Totoro: a held, contained gentleness);
    'cool' darkens toward cool-neutral (Shishigami: the night form
    encroaching at the edges of an otherwise placid day form); 'none'
    leaves the field untouched (Cronenberg, Kafka, Institutional -- an
    unvignetted, flatly lit field reads as more clinical/institutional
    for the latter two, and Cronenberg's own roughest texture needs no
    added atmosphere to read as raw)."""
    if kind == "none" or strength <= 0:
        return
    n = 300
    lin = np.linspace(-1, 1, n)
    gx, gy = np.meshgrid(lin, lin)
    r = np.sqrt(gx ** 2 + gy ** 2)
    alpha = np.clip((r - 0.55) / 0.6, 0, 1) * strength
    color = (0.25, 0.15, 0.05) if kind == "warm" else (0.05, 0.10, 0.20)
    rgba = np.zeros((n, n, 4))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = alpha
    ax.imshow(rgba, extent=[-extent, extent, -extent, extent], origin="lower", zorder=2)


def render_one(name, ax=None, size=4.6):
    X, Y, Phi = generative_field(REGIMES[name], seed=SEED)
    cmap = make_cmap(name)
    style = STYLE[name]
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(Phi, extent=[X.min(), X.max(), Y.min(), Y.max()],
              origin="lower", cmap=cmap, interpolation="bilinear", zorder=1)
    for lev in style["contour_levels"]:
        ax.contour(X, Y, Phi, levels=[Phi.max() * lev], colors="black",
                   linewidths=0.6, alpha=0.45, zorder=3)
    apply_vignette(ax, style["vignette"], style["vignette_strength"], EXTENT)
    if style.get("shield") and standalone:
        # A simple shield-shaped frame, since the institutional register's
        # whole point is to look at home on a coat of arms, not just to be
        # colored like one.
        frame = FancyBboxPatch((-EXTENT * 0.97, -EXTENT * 0.97),
                                EXTENT * 1.94, EXTENT * 1.94,
                                boxstyle="round,pad=0,rounding_size=0.35",
                                fill=False, edgecolor="black", linewidth=1.4,
                                zorder=4)
        ax.add_patch(frame)
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
