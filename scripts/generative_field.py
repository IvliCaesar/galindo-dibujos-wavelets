"""
Implements the generative field Phi(x,y) from Galindo's "El bosque que si
sentimos" / "A Totoro Spirit" notebook (SS/Sentimientos/
cronenberg_creature_final_revised.tex, Sec. "Fusion mediante campos
escalares" and "La textura como campo wavelet"), transcribed here from the
notebook's own equations (eq:metaballs, eq:totalfield, eq:wavelettexture):

    r(theta)  = 1 + a*sin(k*theta) + b*sin(m*theta + phi)
    B(x,y)    = exp(-lambda*|rho - r(theta)|)
    I(x,y)    = 1 / (1 + exp(beta*(rho - r(theta))))
    M(x,y)    = sum_i alpha_i / ((x-x_i)^2 + (y-y_i)^2 + eps)
    T(x,y)    = (1/Z) sum_j 2^(-jH) c_j(x,y) psi(f0*2^j*x) psi(f0*2^j*y)
    Phi(x,y)  = c_B*B + c_I*I + c_M*tanh(M) + eta*T

psi is the periodized Haar mother wavelet (+1 on the first half-period, -1
on the second). c_j(x,y) is a per-dyadic-cell pseudorandom coefficient in
[-1,1], made reproducible via a deterministic hash of the cell indices
(same discipline as the notebook's own C listing).

Four parameter regimes are transcribed from the notebook's Sec.
"Tres registros de un mismo campo": Totoro, Shishigami, Cronenberg, and a
fourth (institutional) palette applied to the Totoro-parameter field, per
the notebook's own point that the same field, differently colored, produces
a different reading. Only k (contour harmonic) and m (secondary harmonic)
are given exact integer values in the notebook; the remaining continuous
parameters (a, b, phi, lambda, beta, metaball centers/weights, c_B, c_I,
c_M, eta, H, J, f0) are this script's own concrete, disclosed choices
consistent with the notebook's qualitative description of each regime
(e.g. Cronenberg: "high-frequency high-amplitude contour harmonics",
"three strongly negative metaball wells", "rougher wavelet texture, lower
H, more octaves J") -- not a literal reproduction of an original
implementation, which was not located.
"""
import numpy as np

N = 720  # grid resolution
EXTENT = 2.2  # plot from -EXTENT to +EXTENT on both axes


def haar_psi(t: np.ndarray) -> np.ndarray:
    frac = np.mod(t, 1.0)
    return np.where(frac < 0.5, 1.0, -1.0)


def cell_hash_coeff(j: int, kx: np.ndarray, ky: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic pseudorandom coefficient in [-1, 1] per dyadic cell,
    via a fixed-seed hash of (j, kx, ky) -- reproducible across runs."""
    h = (kx.astype(np.int64) * 73856093) ^ (ky.astype(np.int64) * 19349663) ^ (j * 83492791) ^ seed
    rng_vals = (np.abs(h) % 1_000_003) / 1_000_003.0
    return 2.0 * rng_vals - 1.0


def wavelet_texture(X, Y, f0, J, H, seed):
    T = np.zeros_like(X)
    for j in range(J + 1):
        scale = f0 * (2 ** j)
        kx = np.floor(scale * X)
        ky = np.floor(scale * Y)
        cj = cell_hash_coeff(j, kx, ky, seed)
        T += (2.0 ** (-j * H)) * cj * haar_psi(scale * X) * haar_psi(scale * Y)
    T /= np.sqrt(J + 1)  # crude variance normalization (Z)
    return T


def contour_field(theta, k, m, a, b, phi):
    return 1.0 + a * np.sin(k * theta) + b * np.sin(m * theta + phi)


def metaball_field(X, Y, centers, weights, eps=0.05):
    M = np.zeros_like(X)
    for (cx, cy), w in zip(centers, weights):
        M += w / ((X - cx) ** 2 + (Y - cy) ** 2 + eps)
    return M


def generative_field(params: dict, seed: int = 0) -> np.ndarray:
    lin = np.linspace(-EXTENT, EXTENT, N)
    X, Y = np.meshgrid(lin, lin)
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    r_theta = contour_field(theta, params["k"], params["m"], params["a"],
                             params["b"], params["phi"])
    B = np.exp(-params["lam"] * np.abs(rho - r_theta))
    I = 1.0 / (1.0 + np.exp(params["beta"] * (rho - r_theta)))
    M = metaball_field(X, Y, params["centers"], params["weights"], params["eps"])
    T = wavelet_texture(X, Y, params["f0"], params["J"], params["H"], seed)

    Phi = (params["cB"] * B + params["cI"] * I
           + params["cM"] * np.tanh(M) + params["eta"] * T)
    return X, Y, Phi


REGIMES = {
    "totoro": dict(
        k=3, m=2, a=0.22, b=0.12, phi=0.35, lam=6.0, beta=14.0,
        centers=[(0.0, -0.35), (-0.55, 0.15), (0.55, 0.15),
                  (-0.35, 0.55), (0.35, 0.55)],
        weights=[0.55, 0.30, 0.30, -0.22, -0.22],
        eps=0.05, f0=2.6, J=5, H=0.80,
        cB=1.0, cI=0.55, cM=0.85, eta=0.10,
    ),
    "shishigami": dict(
        k=5, m=9, a=0.30, b=0.16, phi=0.55, lam=7.5, beta=16.0,
        centers=[(0.0, -0.30), (-0.30, 0.20), (0.30, 0.20)],
        weights=[0.35, -0.55, -0.55],
        eps=0.04, f0=3.2, J=6, H=0.60,
        cB=1.05, cI=0.5, cM=0.9, eta=0.13,
    ),
    "cronenberg": dict(
        k=7, m=5, a=0.42, b=0.30, phi=0.85, lam=9.0, beta=18.0,
        centers=[(-0.15, -0.25), (0.35, 0.10), (-0.45, 0.35)],
        weights=[-0.60, -0.65, -0.50],
        eps=0.035, f0=4.2, J=7, H=0.32,
        cB=1.15, cI=0.45, cM=1.05, eta=0.18,
    ),
    "kafka": dict(
        # k even: by Proposition 2.1 of the article, an even k admits no
        # value of phi at all under which the contour is exactly
        # bilaterally symmetric -- a deliberate choice for a register
        # about a transformation that is not off-center but categorically
        # wrong. One metaball well is strongly negative (a hollow,
        # unprotected underside) against an otherwise domed, segmented
        # mass; texture roughness is intermediate between Totoro's soft
        # fur and Cronenberg's raw tissue, read as a chitinous carapace.
        k=4, m=7, a=0.18, b=0.20, phi=0.65, lam=8.0, beta=17.0,
        centers=[(0.10, -0.25), (-0.40, 0.35), (0.50, 0.40), (0.0, 0.55)],
        weights=[0.50, 0.25, 0.20, -0.35],
        eps=0.045, f0=3.6, J=6, H=0.55,
        cB=1.08, cI=0.5, cM=0.95, eta=0.14,
    ),
}
REGIMES["university"] = REGIMES["totoro"]  # same field, different palette only

PALETTES = {
    # Hex colors taken directly from the notebook's own \definecolor values.
    "totoro": ["#F6F3EC", "#E3A438", "#6B8F71", "#2F5233", "#3A2B22"],
    "shishigami": ["#E3A438", "#8C2F2E", "#3A2B22", "#12241A"],
    "cronenberg": ["#F2D9D9", "#C94F4F", "#8C1C1C", "#4A0E0E", "#1A0505"],
    "university": ["#B9975B", "#1B3A6B", "#6E1E31", "#1F5C3E"],
    # Low-chroma grey-brown-beige family: bureaucratic paper, carapace,
    # shadow -- narrow in hue like Cronenberg's, but desaturated rather
    # than saturated (see the article's CIELAB analysis, Sec. 5).
    "kafka": ["#D9D4C7", "#A69C88", "#6B6459", "#3D3830", "#211E19"],
}
