"""
Renders the Cronenberg register at higher grid resolution than the rest
of the article's figures (N=2160 vs the N=720 used everywhere else), to
better sample the finest wavelet-texture octaves and reduce the
grid-scale moire visible in Figure 6 (creature_cronenberg.png).
Cronenberg has both the largest J (7 octaves) and the lowest H (0.32,
so the finest octaves keep the most weight), making it the one register
where N=720 undersamples the texture most visibly.

This is the same field, same parameters, same palette, same presentation
treatment (three contour levels, no vignette) as Figure 6; only the
sampling grid is finer. N is overridden on the generative_field module
before calling it, since generative_field/wavelet_texture read N as a
module-level global at call time.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generative_field as gf
from render_creatures import make_cmap, STYLE, apply_vignette

DETAIL_N = 2160
SEED = 7

if __name__ == "__main__":
    gf.N = DETAIL_N
    X, Y, Phi = gf.generative_field(gf.REGIMES["cronenberg"], seed=SEED)

    cmap = make_cmap("cronenberg")
    style = STYLE["cronenberg"]

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(Phi, extent=[X.min(), X.max(), Y.min(), Y.max()],
              origin="lower", cmap=cmap, interpolation="bilinear", zorder=1)
    for lev in style["contour_levels"]:
        ax.contour(X, Y, Phi, levels=[Phi.max() * lev], colors="black",
                   linewidths=0.5, alpha=0.45, zorder=3)
    apply_vignette(ax, style["vignette"], style["vignette_strength"], gf.EXTENT)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Cronenberg register, $N={DETAIL_N}$ (detail)", fontsize=10)
    fig.tight_layout()
    fig.savefig("../figures/creature_cronenberg_detail.pdf", dpi=300)
    fig.savefig("../figures/creature_cronenberg_detail.png", dpi=300)
    print(f"Saved ../figures/creature_cronenberg_detail.{{pdf,png}} at N={DETAIL_N}")
