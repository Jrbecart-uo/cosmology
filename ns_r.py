"""Slow-roll predictions (n_s, r) for standard inflation models vs current CMB limits.

Units: reduced Planck mass M_pl = 1. For each potential V(phi) we find phi_end
(epsilon_V = 1), go back N e-folds with dN = V / V' dphi, and evaluate the
first-order slow-roll formulas n_s = 1 - 6 eps + 2 eta, r = 16 eps at phi_*.
First order is accurate to ~1e-3 in n_s here, below the current 1-sigma (~3e-3).

Run: uv run --no-project --with numpy --with scipy --with matplotlib python ns_r.py
     -> prints a table and writes ns_r.png
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# --- Observational constraints (see before_big_bang.md, section 8, for sources) ---
DATA = {
    "Planck 2018":     (0.9649, 0.0042),  # Planck 2018 VI (Aghanim et al. 2020), TT,TE,EE+lowE+lensing
    "CMB (SPA+BK)":    (0.9682, 0.0032),  # Balkenhol et al. 2026, CMB only
    "CMB + DESI BAO":  (0.9728, 0.0029),  # Balkenhol et al. 2026, + DESI DR2
}
R_MAX = 0.034  # 95% upper limit, Balkenhol et al. 2026 (BK18 alone: 0.036)


def slow_roll(V, dV, d2V, N, phi_hi=40.0, phi_lo=1e-6):
    """(n_s, r) N e-folds before the end of inflation, for a potential whose
    plateau / large-field region is at large phi and inflation ends rolling to smaller phi."""
    eps = lambda p: 0.5 * (dV(p) / V(p)) ** 2
    eta = lambda p: d2V(p) / V(p)
    # end of inflation: largest phi below the plateau where eps = 1
    grid = np.geomspace(phi_lo, phi_hi, 4000)
    e = np.array([eps(p) for p in grid]) - 1.0
    idx = np.where(np.sign(e[:-1]) != np.sign(e[1:]))[0]
    phi_end = brentq(lambda p: eps(p) - 1.0, grid[idx[-1]], grid[idx[-1] + 1])
    efolds = lambda p: quad(lambda x: V(x) / dV(x), phi_end, p, limit=200)[0]
    phi_star = brentq(lambda p: efolds(p) - N, phi_end * (1 + 1e-9), phi_hi)
    return 1 - 6 * eps(phi_star) + 2 * eta(phi_star), 16 * eps(phi_star)


def monomial(p):
    return (lambda f: f**p, lambda f: p * f**(p - 1), lambda f: p * (p - 1) * f**(p - 2))


def starobinsky(alpha=1.0):
    """E-model alpha-attractor; alpha = 1 is Starobinsky R^2 (and Higgs inflation at large N)."""
    a = np.sqrt(2 / (3 * alpha))
    return (lambda f: (1 - np.exp(-a * f)) ** 2,
            lambda f: 2 * a * np.exp(-a * f) * (1 - np.exp(-a * f)),
            lambda f: 2 * a**2 * np.exp(-a * f) * (2 * np.exp(-a * f) - 1))


def t_model(alpha):
    """T-model alpha-attractor V = tanh^2(phi / sqrt(6 alpha))."""
    s = np.sqrt(6 * alpha)
    t = lambda f: np.tanh(f / s)
    sech2 = lambda f: 1 - t(f) ** 2
    return (lambda f: t(f) ** 2,
            lambda f: 2 * t(f) * sech2(f) / s,
            lambda f: 2 * sech2(f) * (1 - 3 * t(f) ** 2) / s**2)


def polynomial_attractor(k, mu):
    """Kallosh-Linde 2022 polynomial alpha-attractor V = phi^k / (phi^k + mu^k)."""
    return (lambda f: f**k / (f**k + mu**k),
            lambda f: k * mu**k * f**(k - 1) / (f**k + mu**k) ** 2,
            lambda f: k * mu**k * f**(k - 2) * ((k - 1) * mu**k - (k + 1) * f**k) / (f**k + mu**k) ** 3)


def natural(fdecay):
    """Natural inflation (Freese-Frieman-Olinto 1990) V = 1 - cos(phi/f), rolling to phi = 0."""
    return (lambda p: 1 - np.cos(p / fdecay),
            lambda p: np.sin(p / fdecay) / fdecay,
            lambda p: np.cos(p / fdecay) / fdecay**2)


MODELS = [
    # label, (V, V', V''), phi_hi (upper search bound for phi_*)
    ("phi^4", monomial(4), 60),
    ("phi^2", monomial(2), 60),
    ("phi^1", monomial(1), 60),
    ("phi^(2/3)", monomial(2 / 3), 60),
    ("phi^(1/3)", monomial(1 / 3), 60),
    ("Natural, f=7", natural(7.0), 7 * np.pi * 0.999),
    ("Starobinsky R^2 / Higgs", starobinsky(1.0), 40),
    ("E-model alpha=10", starobinsky(10.0), 80),
    ("T-model alpha=1/3", t_model(1 / 3), 15),
    ("Poly. attractor k=2, mu=1", polynomial_attractor(2, 1.0), 400),
    ("Poly. attractor k=4, mu=1", polynomial_attractor(4, 1.0), 400),
]
NS = (50, 55, 60)


def tension(ns, r):
    """Rough distance from each constraint: |n_s offset| in sigma; flag r above the limit.
    (Treats n_s and r as independent; the real contours are only weakly correlated.)"""
    out = {k: (ns - m) / s for k, (m, s) in DATA.items()}
    return out, r > R_MAX


if __name__ == "__main__":
    results = {}
    print(f"{'model':28s} {'N':>3s} {'n_s':>7s} {'r':>9s}   " +
          "  ".join(f"{k:>14s}" for k in DATA) + "   r<0.034?")
    for label, (V, dV, d2V), hi in MODELS:
        results[label] = []
        for N in NS:
            ns, r = slow_roll(V, dV, d2V, N, phi_hi=hi)
            results[label].append((N, ns, r))
            sig, r_bad = tension(ns, r)
            print(f"{label:28s} {N:3d} {ns:7.4f} {r:9.2e}   " +
                  "  ".join(f"{v:+13.1f}σ" for v in sig.values()) +
                  ("   EXCLUDED" if r_bad else "   ok"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (name, (m, s)), c in zip(DATA.items(), ["0.4", "tab:blue", "tab:orange"]):
        if name == "Planck 2018":  # superseded; show as dashed 1-sigma lines only
            for x in (m - s, m + s):
                ax.axvline(x, color=c, ls="--", lw=0.8)
            ax.plot([], [], color=c, ls="--", lw=0.8, label=f"{name}: {m} ± {s} (1σ)")
            continue
        ax.axvspan(m - 2 * s, m + 2 * s, color=c, alpha=0.10, lw=0)
        ax.axvspan(m - s, m + s, color=c, alpha=0.25, lw=0, label=f"{name}: {m} ± {s} (1σ, 2σ)")
    ax.axhspan(R_MAX, 1, color="red", alpha=0.10, lw=0)
    ax.axhline(R_MAX, color="red", lw=1)
    ax.text(0.9405, R_MAX * 1.15, "r < 0.034 (95%) — excluded above", color="red", fontsize=8)
    for i, (label, rows) in enumerate(results.items()):
        ns = [x[1] for x in rows]; r = [x[2] for x in rows]
        ax.plot(ns, r, "o-", ms=4, label=label, color=f"C{i}" if i < 10 else "k")  # 11 models, 10 default colours
        ax.annotate("N=60", (ns[-1], r[-1]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    # Bounce / CPT-mirror: no detectable primordial tensors; n_s is fit, not predicted (see note)
    ax.axhline(2e-4, color="k", ls=":", lw=1)
    ax.text(0.9405, 2.4e-4, "bounce (ekpyrotic) and CPT-mirror: r ≈ 0, off the bottom", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1.5e-4, 0.5); ax.set_xlim(0.94, 0.985)
    ax.set_xlabel("spectral index $n_s$"); ax.set_ylabel("tensor-to-scalar ratio $r$")
    ax.set_title("Inflation models (N = 50, 55, 60 e-folds) vs current CMB limits")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(); fig.savefig("ns_r.png", dpi=150)
    print("wrote ns_r.png")
