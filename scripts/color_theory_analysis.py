"""
Quantifies the four palettes (Totoro, Shishigami, Cronenberg, University)
in CIELAB space using the colour-science library (sRGB -> CIE XYZ -> CIELAB,
D65 white point, the same standard pipeline behind CIEDE2000). This turns
"Cronenberg reads as visceral, Totoro reads as gentle" from an assertion
into a measured claim: mean lightness L*, mean chroma C* = sqrt(a*^2+b*^2),
and hue angle h = atan2(b*,a*) per palette, plus each palette's total hue
spread (max pairwise hue-angle difference) as a measure of how visually
"loud"/heterogeneous a palette is.
"""
import numpy as np
import colour

from generative_field import PALETTES


def hex_to_lab(hex_color: str) -> np.ndarray:
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    xyz = colour.sRGB_to_XYZ(rgb)
    lab = colour.XYZ_to_Lab(xyz)
    return lab  # [L*, a*, b*]


def circular_spread(hues_deg: np.ndarray) -> float:
    """Max pairwise circular distance between hue angles, in degrees."""
    diffs = np.abs(hues_deg[:, None] - hues_deg[None, :])
    diffs = np.minimum(diffs, 360 - diffs)
    return float(diffs.max())


def circular_mean(hues_deg: np.ndarray) -> float:
    """Mean of angles via the resultant (vector-sum) method -- a plain
    arithmetic mean of hue angles is invalid whenever a palette's hues
    straddle the 0/360 wraparound (as the university palette's do here:
    individual hues 82.0, 282.4, 12.5, 157.1 -- 282.4 and 12.5 are only
    90.1 degrees apart going through 360/0, not the ~270 degrees a naive
    arithmetic mean would imply)."""
    rad = np.radians(hues_deg)
    mean_angle = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
    return float(mean_angle % 360)


if __name__ == "__main__":
    print(f"{'Palette':<14}{'mean L*':>9}{'mean C*':>9}{'mean h (deg)':>14}{'hue spread':>12}")
    summary = {}
    for name, hexes in PALETTES.items():
        labs = np.array([hex_to_lab(h) for h in hexes])
        L, a, b = labs[:, 0], labs[:, 1], labs[:, 2]
        C = np.sqrt(a ** 2 + b ** 2)
        h = (np.degrees(np.arctan2(b, a)) + 360) % 360
        mean_h = circular_mean(h)
        spread = circular_spread(h)
        summary[name] = dict(mean_L=float(L.mean()), mean_C=float(C.mean()),
                              mean_h=mean_h, hue_spread=spread)
        print(f"{name:<14}{L.mean():>9.1f}{C.mean():>9.1f}{mean_h:>14.1f}{spread:>12.1f}")

    import json
    with open("color_theory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
