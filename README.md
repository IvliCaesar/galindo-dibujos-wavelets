# galindo-dibujos-wavelets

Reproducible code for César Galindo, *"One Field, Five Readings: Metaballs,
Haar-Wavelet Texture, and the Color Grammar of a Generative Creature,"*
submitted to *Journal of Mathematics and the Arts*.

## What is here

- `scripts/generative_field.py` — the generative field
  $\Phi(x,y) = c_B B + c_I I + c_M \tanh M + \eta T$ (symmetry-broken
  contour + sigmoid interior + metaballs + Haar-wavelet texture) and the
  five parameter regimes (Totoro, Shishigami, Cronenberg, Kafka,
  Institutional) discussed in the article.
- `scripts/render_creatures.py` — renders all five creatures and the
  comparison figure.
- `scripts/plot_texture_roughness.py` — renders the wavelet-texture /
  per-octave-weight figure isolating the roughness exponent $H$.
- `scripts/color_theory_analysis.py` — converts each palette to CIELAB
  (D65 white point, via the `colour-science` package) and computes mean
  lightness, mean chroma, circular mean hue, and hue spread — the exact
  numbers reported in the article's CIELAB table.
- `scripts/plot_cielab_scatter.py` — plots all five palettes as points in
  the CIELAB $a^*$-$b^*$ plane, each marker in its own true color.

## Reproduce

```
pip install -r requirements.txt
cd scripts
python generative_field.py        # sanity import; regimes/palettes only
python render_creatures.py        # -> ../figures/*.png, *.pdf
python plot_texture_roughness.py  # -> ../figures/texture_roughness.{png,pdf}
python color_theory_analysis.py   # prints Table 2's numbers; writes color_theory_summary.json
python plot_cielab_scatter.py     # -> ../figures/cielab_scatter.{png,pdf}
```

All randomness is seed-fixed (`SEED = 7` in the render scripts,
deterministic hashing in the wavelet-texture coefficients), so re-running
reproduces the article's figures and numbers exactly.

## Citation

If you use this code, please cite the article (details to be added on
acceptance) or this repository.
