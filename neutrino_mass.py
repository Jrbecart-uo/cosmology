"""Neutrino masses vs the CPT-mirror prediction (lightest neutrino exactly massless).

Given the measured mass splittings, each value of the lightest mass fixes all three
masses. We compute, for both orderings:
  - sum of masses           (what cosmology measures)
  - m_beta                  (what tritium beta decay, KATRIN, measures)
  - m_betabeta band         (what neutrinoless double beta decay measures, if Majorana;
                             the band comes from the two unknown Majorana phases)
and compare with current limits. Lightest mass = 0 is the Boyle-Finn-Turok 2018 point.

Run: uv run --no-project --with numpy --with matplotlib python neutrino_mass.py
     -> prints a table and writes neutrino_mass.png
"""
import numpy as np

# --- NuFIT 6.0 (Esteban et al. 2024, JHEP 12 (2024) 216), Table 1, "IC24 with SK-atm" ---
DM21 = 7.49e-5                      # eV^2, both orderings
DM3L = {"NO": 2.513e-3,             # Delta m^2_31 > 0
        "IO": -2.484e-3}            # Delta m^2_32 < 0
S12 = {"NO": 0.308, "IO": 0.308}    # sin^2 theta_12
S13 = {"NO": 0.02215, "IO": 0.02231}  # sin^2 theta_13

# --- Current limits (see before_big_bang.md, section 10, for sources) ---
SUM_LIMITS = [  # 95% upper limits on the sum, LCDM unless stated
    ("DESI DR2 + CMB (DESI 2025)", 0.0642),
    ("+ Lyα full shape (DESI DR2 IV, 2026)", 0.0592),
    ("CMB-SPA + DESI DR2 + DES SN, oscillation prior", 0.092),  # Hou et al. 2026, hierarchy-informed prior
    ("DESI DR2 + CMB, w0wa dark energy", 0.16),
]
KATRIN_MBETA = 0.45            # eV, 90% CL (KATRIN Collaboration, Science 2025)
KLZ_MBB = (0.028, 0.122)       # eV, 90% CL, range from nuclear matrix elements (KamLAND-Zen 2024)
NEXT_GEN_MBB = (0.009, 0.021)  # eV, LEGEND-1000 design discovery sensitivity (pre-CDR 2021)
SIGMA_SUM_NOW = 0.020          # eV, marginalised error on the sum, DESI DR2 + CMB


def masses(m0, order):
    """(m1, m2, m3) in eV for lightest mass m0."""
    m0 = np.asarray(m0, dtype=float)
    if order == "NO":
        m1 = m0
        m2 = np.sqrt(m0**2 + DM21)
        m3 = np.sqrt(m0**2 + DM3L["NO"])
    else:
        m3 = m0
        m2 = np.sqrt(m0**2 - DM3L["IO"])      # m2^2 = m3^2 + |Dm32^2|
        m1 = np.sqrt(m2**2 - DM21)
    return m1, m2, m3


def mixing(order):
    """|U_e1|^2, |U_e2|^2, |U_e3|^2."""
    s12, s13 = S12[order], S13[order]
    return (1 - s12) * (1 - s13), s12 * (1 - s13), s13


def observables(m0, order):
    m = masses(m0, order)
    u = mixing(order)
    total = sum(m)
    mbeta = np.sqrt(sum(ui * mi**2 for ui, mi in zip(u, m)))
    terms = [ui * mi for ui, mi in zip(u, m)]
    # |sum_i U_ei^2 m_i e^{i phase_i}|: max with all phases aligned, min by triangle inequality
    hi = sum(terms)
    lo = np.maximum(0.0, 2 * np.max(terms, axis=0) - hi)
    return total, mbeta, lo, hi


def lightest_for_sum(target, order):
    """Lightest mass giving a sum of masses equal to target (eV), by bisection."""
    lo, hi = 0.0, 1.0
    if observables(0.0, order)[0] >= target:
        return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if observables(mid, order)[0] < target else (lo, mid)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("CPT-mirror point (lightest mass = 0):")
    for o in ("NO", "IO"):
        m = masses(0.0, o)
        s, mb, lo, hi = observables(0.0, o)
        print(f"  {o}: masses = " + ", ".join(f"{x * 1e3:.1f}" for x in m) +
              f" meV; sum = {s * 1e3:.1f} meV; m_beta = {mb * 1e3:.1f} meV; "
              f"m_bb = {lo * 1e3:.1f}-{hi * 1e3:.1f} meV")

    print("\nLightest mass allowed by each 95% limit on the sum:")
    for name, lim in SUM_LIMITS:
        row = []
        for o in ("NO", "IO"):
            smin = observables(0.0, o)[0]
            row.append(f"{o}: " + (f"m_light < {lightest_for_sum(lim, o) * 1e3:.1f} meV"
                                   if lim > smin else f"excluded (needs sum >= {smin * 1e3:.0f} meV)"))
        print(f"  {name:45s} < {lim * 1e3:5.1f} meV   " + "   ".join(row))

    print("\nHow precisely must the sum be measured to see a nonzero lightest mass (2 sigma, NO)?")
    s0 = observables(0.0, "NO")[0]
    for m0 in (0.005, 0.010, 0.020, 0.030, 0.050):
        ds = observables(m0, "NO")[0] - s0
        print(f"  m_light = {m0 * 1e3:4.0f} meV -> sum rises by {ds * 1e3:5.1f} meV -> needs sigma(sum) <= {ds / 2 * 1e3:4.1f} meV")
    m_now = lightest_for_sum(s0 + 2 * SIGMA_SUM_NOW, "NO")
    print(f"  with today's sigma = {SIGMA_SUM_NOW * 1e3:.0f} meV: only m_light >~ {m_now * 1e3:.0f} meV would be detectable")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {"NO": "#1f77b4", "IO": "#d95f02"}
    NAME = {"NO": "normal ordering", "IO": "inverted ordering"}
    m0 = np.geomspace(1e-4, 0.3, 600)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2))

    for o in ("NO", "IO"):
        s, mb, lo, hi = observables(m0, o)
        a1.plot(m0, s, color=COL[o], lw=2, label=NAME[o])
        a2.fill_between(m0, lo, hi, color=COL[o], alpha=0.30, lw=0, label=NAME[o])
        a2.plot(m0, lo, color=COL[o], lw=1); a2.plot(m0, hi, color=COL[o], lw=1)
        s0_, _, lo0, hi0 = observables(np.array([1e-4]), o)
        a1.plot(1e-4, s0_, "o", ms=9, color=COL[o], mec="k", zorder=5, clip_on=False)
        a2.plot([1e-4, 1e-4], [max(lo0[0], 1e-4), hi0[0]], color=COL[o], lw=6,
                solid_capstyle="butt", zorder=5, clip_on=False)

    for name, lim in SUM_LIMITS:
        ls = ":" if "w0wa" in name else "-"
        a1.axhline(lim, color="0.25", lw=1, ls=ls)
        below = "Lyα" in name  # two lines 5 meV apart: put this label under its line
        a1.text(1.4e-3, lim * (0.955 if below else 1.04), f"{name}: < {lim * 1e3:.0f} meV",
                va="top" if below else "bottom", fontsize=7.5, color="0.2",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
    a1.axhspan(SUM_LIMITS[0][1], 1, color="red", alpha=0.08, lw=0)
    a1.annotate("CPT-mirror prediction:\nlightest mass = 0", (1e-4, 0.0588), xytext=(2.5e-4, 0.037),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlim(1e-4, 0.3); a1.set_ylim(0.03, 1.0)
    a1.set_xlabel("lightest neutrino mass (eV)"); a1.set_ylabel("sum of the three masses (eV)")
    a1.set_title("What cosmology measures: the sum")
    a1.legend(fontsize=8, loc="lower right")

    a2.axhspan(KLZ_MBB[1], 1, color="red", alpha=0.10, lw=0)
    a2.axhspan(*KLZ_MBB, color="red", alpha=0.05, lw=0)
    a2.axhline(KLZ_MBB[0], color="0.25", lw=1, ls="--"); a2.axhline(KLZ_MBB[1], color="0.25", lw=1)
    a2.text(1.4e-3, KLZ_MBB[1] * 1.08, "KamLAND-Zen 2024: excluded above 28–122 meV\n(range = nuclear-physics uncertainty)",
            fontsize=7.5, color="0.2")
    a2.axhspan(*NEXT_GEN_MBB, color="0.5", alpha=0.18, lw=0)
    a2.text(1.4e-3, NEXT_GEN_MBB[0] * 0.92, "next generation reach (LEGEND-1000 goal: 9–21 meV)", va="top", fontsize=7.5, color="0.2")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlim(1e-4, 0.3); a2.set_ylim(1e-4, 0.5)
    a2.set_xlabel("lightest neutrino mass (eV)"); a2.set_ylabel(r"effective Majorana mass $m_{\beta\beta}$ (eV)")
    a2.set_title("Neutrinoless double beta decay (if Majorana)")
    a2.legend(fontsize=8, loc="lower right")
    fig.text(0.5, 0.005, "Markers at the left edge: lightest mass = 0 (the CPT-mirror point). "
             "Oscillation data: NuFIT 6.0. Bands: unknown Majorana phases.", ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("neutrino_mass.png", dpi=150)
    print("wrote neutrino_mass.png")
