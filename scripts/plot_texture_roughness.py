"""
Renders two complementary views of the roughness exponent H's effect on
the wavelet texture field T(x,y) of eq:wavelettexture, isolated from the
contour and metaball layers, across the four H values actually used by
the article's registers (Totoro 0.80, Shishigami 0.60, Kafka 0.55,
Cronenberg 0.32), holding f0 and J fixed.

Top row: the texture field itself. A first version of this figure used
only this row and the H-driven difference was visually too subtle to
read at a glance (the per-octave contributions all remain present at
every H; only their relative weight 2^{-jH} changes, and images alone do
not make a multiplicative weighting easy to see). Bottom row: the actual
per-octave weight 2^{-jH} as a function of octave j for each H, which
makes the claim in Sec. 3 ("H near 1 favors smooth, coherent patches, H
near 0 favors sharp, high-frequency detail") precise and legible rather
than left to be inferred from the texture images alone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generative_field import wavelet_texture, N, EXTENT

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 180,
})

SEED = 7
F0 = 3.0
J = 6
H_VALUES = [0.80, 0.60, 0.55, 0.32]
LABELS = ["H = 0.80\n(Totoro)", "H = 0.60\n(Shishigami)",
          "H = 0.55\n(Kafka)", "H = 0.32\n(Cronenberg)"]
COLORS = ["tab:green", "tab:orange", "tab:gray", "tab:red"]

if __name__ == "__main__":
    lin = np.linspace(-EXTENT, EXTENT, N)
    X, Y = np.meshgrid(lin, lin)

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2),
                              gridspec_kw={"height_ratios": [2.2, 1]})

    for ax, H, label in zip(axes[0], H_VALUES, LABELS):
        T = wavelet_texture(X, Y, F0, J, H, SEED)
        ax.imshow(T, extent=[X.min(), X.max(), Y.min(), Y.max()],
                   origin="lower", cmap="Greys_r", vmin=-np.abs(T).max(),
                   vmax=np.abs(T).max())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=10)

    js = np.arange(J + 1)
    for ax, H, color in zip(axes[1], H_VALUES, COLORS):
        weights = 2.0 ** (-js * H)
        weights /= weights[0]
        ax.bar(js, weights, color=color, width=0.6)
        ax.set_xlabel("octave $j$", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(js)
        ax.tick_params(labelsize=7)
    axes[1][0].set_ylabel(r"weight $2^{-jH}$" + "\n(relative to $j=0$)", fontsize=8)

    fig.suptitle(r"Wavelet texture $T(x,y)$ (top) and its per-octave weight $2^{-jH}$ (bottom), $f_0,J$ fixed, $H$ varying", fontsize=11)
    fig.tight_layout()
    fig.savefig("../figures/texture_roughness.pdf")
    fig.savefig("../figures/texture_roughness.png")
    print("Saved ../figures/texture_roughness.pdf / .png")
