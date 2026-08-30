"""
Renders the wavelet texture field T(x,y) of eq:wavelettexture in
isolation (no contour, no metaballs), across the four roughness exponents
H actually used by the article's registers (Totoro 0.80, Shishigami 0.60,
Kafka 0.55, Cronenberg 0.32), holding f0 and J fixed.

This is a raster (matplotlib) image because it is a genuine continuous
scalar field on a fine grid (N x N, N=720) -- not practical to reproduce
as a native TikZ/pgfplots figure. The companion per-octave weight bar
chart (the quantitative half of the article's Figure 4) is instead built
natively in pgfplots directly inside the .tex file (search for
"pgfplots" / "groupplot" near \\label{fig:textureweights}), since that
data (7 values per H, 4 H values) is small and exact -- native TikZ is
both more appropriate and produces cleaner vector output than exporting
another matplotlib panel for it.
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

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4))

    for ax, H, label in zip(axes, H_VALUES, LABELS):
        T = wavelet_texture(X, Y, F0, J, H, SEED)
        ax.imshow(T, extent=[X.min(), X.max(), Y.min(), Y.max()],
                   origin="lower", cmap="Greys_r", vmin=-np.abs(T).max(),
                   vmax=np.abs(T).max())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=10)

    fig.suptitle(r"Wavelet texture $T(x,y)$, $f_0,J$ fixed, $H$ varying", fontsize=11)
    fig.tight_layout()
    fig.savefig("../figures/texture_field.pdf")
    fig.savefig("../figures/texture_field.png")
    print("Saved ../figures/texture_field.pdf / .png")

    # Print the exact per-octave weights too, for cross-checking against
    # the native pgfplots bar chart built directly in the .tex file.
    js = np.arange(J + 1)
    for H in H_VALUES:
        weights = 2.0 ** (-js * H)
        weights /= weights[0]
        print(f"H={H}: " + " ".join(f"({j},{w:.4f})" for j, w in zip(js, weights)))
