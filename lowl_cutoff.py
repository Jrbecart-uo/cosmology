"""Is the low CMB power at the largest angles a trace of what came before inflation?

Tests the "testable version" of before_big_bang.md (section 5) against Planck 2018 data:

1. Fit a large-scale cutoff in the primordial spectrum,
       P(k) = P_LCDM(k) * [1 - exp(-(k/k_c)^lambda)]   (Contaldi et al. 2003),
   to the Planck 2018 Commander TT spectrum at l = 2..29, with all other parameters at
   the Planck 2018 mean values. A cutoff is what you expect if inflation lasted only just
   long enough and the largest scales we see were made at its very start.
2. Measure how rare the low quadrupole and the weak large-angle correlation (S_1/2) are
   in base LCDM, and in the best-fit cutoff model.
3. Forecast what cosmic-variance-limited E-mode polarisation (LiteBIRD-like) would add
   if the cutoff were real.

Likelihood: Planck's low-l TT likelihood is a Gibbs/Blackwell-Rao likelihood, which we
do not reproduce. We use the standard Wishart form per multipole,
    -2 ln L = nu_l [ Chat/C + ln(C/Chat) - 1 ],
with nu_l = (2l+1) f_l, and f_l calibrated so that the 68% interval of this likelihood
matches the asymmetric error bars Planck publishes for each l. Approximate, stated as such.

Run: uv run --no-project --with camb --with matplotlib python lowl_cutoff.py
"""
import os
import urllib.request

import numpy as np
from scipy.optimize import brentq
from scipy.special import eval_legendre

import camb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "planck")
URL = "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/"
TT_FILE = "COM_PowerSpect_CMB-TT-full_R3.01.txt"
LMIN, LMAX = 2, 29          # Commander range
RNG = np.random.default_rng(1)
NSIM = 200_000

# Planck 2018 mean values, TT,TE,EE+lowE+lensing (Planck 2018 VI, Table 2)
P18 = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544, mnu=0.06,
           As=np.exp(3.044) * 1e-10, ns=0.9649)


def load_tt():
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, TT_FILE)
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL + TT_FILE, path)
    d = np.loadtxt(path)
    d = d[(d[:, 0] >= LMIN) & (d[:, 0] <= LMAX)]
    return d[:, 0].astype(int), d[:, 1], d[:, 2], d[:, 3]


def wishart(chat, c, nu):
    """-2 ln L (zero at C = Chat)."""
    x = chat / c
    return nu * (x - np.log(x) - 1)


def calibrate_nu(ell, dl, lo, hi):
    """nu_l such that the Wishart 68% interval (Delta = 1) matches Planck's (lo, hi) width."""
    nus = []
    for l, d, m, p in zip(ell, dl, lo, hi):
        target = np.log((d + p) / (d - m))           # width of the interval in ln C

        def width(nu):
            f = lambda c: wishart(d, c, nu) - 1.0
            return np.log(brentq(f, d * 1.000001, d * 1e3) / brentq(f, d * 1e-3, d * 0.999999))
        nus.append(brentq(lambda nu: width(nu) - target, 0.5, 5000))
    return np.array(nus)


# ---------- theory ----------
_pars = camb.set_params(lmax=200, **{k: v for k, v in P18.items() if k not in ("As", "ns")},
                        As=P18["As"], ns=P18["ns"])
_pars.WantTensors = False
_transfer = camb.get_transfer_functions(_pars)


def spectra(kc=None, lam=None):
    """Lensed-free scalar D_l^{TT,EE,TE} [muK^2] for l = 0..LMAX+, with optional cutoff."""
    k = np.logspace(-7, 1, 3000)
    pk = P18["As"] * (k / 0.05) ** (P18["ns"] - 1)
    if kc is not None:
        pk = pk * (1 - np.exp(-(k / kc) ** lam))
    ip = camb.initialpower.SplinedInitialPower()
    ip.set_scalar_table(k, pk)
    _transfer.power_spectra_from_transfer(ip)
    cl = _transfer.get_unlensed_scalar_cls(CMB_unit="muK", lmax=60)
    return cl[:, 0], cl[:, 1], cl[:, 3]     # TT, EE, TE (as D_l)


# ---------- large-angle correlation S_1/2 ----------
def s_half_matrix(lmax):
    """S_1/2 = sum_ll' C_l C_l' I_ll' with C(theta) = sum (2l+1)/(4 pi) C_l P_l(cos theta)."""
    x = np.linspace(-1, 0.5, 4001)
    ls = np.arange(LMIN, lmax + 1)
    P = np.array([(2 * l + 1) / (4 * np.pi) * eval_legendre(l, x) for l in ls])
    w = np.full_like(x, x[1] - x[0]); w[0] = w[-1] = w[0] / 2
    return (P * w) @ P.T


def dl_to_cl(ell, dl):
    return dl * 2 * np.pi / (ell * (ell + 1))


def main():
    ell, dl, lo, hi = load_tt()
    nu = calibrate_nu(ell, dl, lo, hi)
    fsky = nu / (2 * ell + 1)
    print(f"Calibrated effective sky fraction f_l: median {np.median(fsky):.2f}, "
          f"range {fsky.min():.2f}-{fsky.max():.2f}")

    tt0, ee0, te0 = spectra()
    th0 = tt0[ell]
    chi0 = wishart(dl, th0, nu).sum()
    print(f"LCDM (Planck 2018 mean values), l=2-29: chi2 = {chi0:.1f} for {len(ell)} multipoles")
    print(f"  observed/LCDM:  l=2 {dl[0]/th0[0]:.2f},  l=3 {dl[1]/th0[1]:.2f},  "
          f"mean l=2-29 {np.mean(dl/th0):.2f}")

    # ---- 1. cutoff fit ----
    kcs = np.logspace(-4.6, -2.6, 61)
    lams = [0.5, 1.0, 2.0, 3.35, 6.0]
    grid = np.empty((len(lams), len(kcs)))
    for i, lam in enumerate(lams):
        for j, kc in enumerate(kcs):
            grid[i, j] = wishart(dl, spectra(kc, lam)[0][ell], nu).sum() - chi0
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    best = dict(kc=kcs[j], lam=lams[i], dchi2=grid[i, j])
    print("\nCutoff model: Delta chi2 vs LCDM (negative = better fit)")
    for lam, row in zip(lams, grid):
        jj = np.argmin(row)
        print(f"  lambda={lam:<5} best k_c = {kcs[jj]:.2e} /Mpc   Delta chi2 = {row[jj]:+.2f}")
    # likelihood-ratio significance, 2 extra parameters (approximate: boundary + look-elsewhere)
    from scipy.stats import chi2 as chi2d, norm
    p_lr = chi2d.sf(-best["dchi2"], 2)
    print(f"Best: k_c = {best['kc']:.2e} /Mpc, lambda = {best['lam']}, Delta chi2 = {best['dchi2']:+.2f}"
          f"  -> p = {p_lr:.2f} ({norm.isf(p_lr / 2):.1f} sigma, 2 extra parameters)")
    # robustness: full-sky weighting (nu = 2l+1) gives the quadrupole its largest weight -> upper bound
    nuf = 2 * ell + 1.0
    chi0f = wishart(dl, th0, nuf).sum()
    best_f = min((wishart(dl, spectra(kc, lam)[0][ell], nuf).sum() - chi0f, kc, lam)
                 for lam in lams for kc in kcs[::2])
    print(f"  full-sky weighting (upper bound): Delta chi2 = {best_f[0]:+.2f} "
          f"at k_c = {best_f[1]:.2e}, lambda = {best_f[2]}  -> {norm.isf(chi2d.sf(-best_f[0], 2) / 2):.1f} sigma")
    d_ls = 13_870.0   # comoving distance to last scattering [Mpc], Planck 2018 best fit (approx.)
    print(f"  cutoff scale ~ multipole l ~ k_c * D_LS = {best['kc'] * d_ls:.1f}")

    ttc, eec, tec = spectra(best["kc"], best["lam"])

    # ---- 2. anomaly statistics ----
    M = s_half_matrix(LMAX)
    cl_obs = dl_to_cl(ell, dl)
    s_obs = cl_obs @ M @ cl_obs

    def sims(th_dl, nus):
        """C_l realisations: Chat ~ C * chi2_nu / nu (Wishart, 1 field)."""
        cl = dl_to_cl(ell, th_dl)
        return cl * RNG.chisquare(nus, size=(NSIM, len(ell))) / nus

    for name, th in (("LCDM", th0), ("best cutoff", ttc[ell])):
        for label, nus in (("full sky", 2 * ell + 1.0), ("calibrated f_l", nu)):
            sm = sims(th, nus)
            p_q = np.mean(sm[:, 0] <= cl_obs[0])
            s = np.einsum("ij,jk,ik->i", sm, M, sm)
            p_s = np.mean(s <= s_obs)
            print(f"{name:12s} [{label:14s}]  P(C_2 <= observed) = {p_q:.3f}   "
                  f"P(S_1/2 <= observed) = {p_s:.4f}")
    print(f"Observed S_1/2 (l=2-29, from the Commander C_l) = {s_obs:.0f} muK^4")

    # ---- 3. polarisation forecast ----
    # Expected -2 Delta lnL between LCDM and the cutoff model when the cutoff is true:
    # sum_l nu [tr(C0^-1 Cc) - ln det(C0^-1 Cc) - n], cosmic-variance limited, f_sky = 0.7.
    def kl(n_fields, lmax, f=0.7):
        tot = 0.0
        for l in range(2, lmax + 1):
            if n_fields == 1:
                C0 = np.array([[tt0[l]]]); Cc = np.array([[ttc[l]]])
            else:
                C0 = np.array([[tt0[l], te0[l]], [te0[l], ee0[l]]])
                Cc = np.array([[ttc[l], tec[l]], [tec[l], eec[l]]])
            A = np.linalg.solve(C0, Cc)
            tot += (2 * l + 1) * f * (np.trace(A) - np.log(np.linalg.det(A)) - n_fields)
        return tot
    kl_t, kl_te = kl(1, LMAX), kl(2, LMAX)
    gain = kl_te - kl_t
    print(f"\nIf the best-fit cutoff is real (f_sky = 0.7, noise-free, l <= {LMAX}):")
    print(f"  expected Delta chi2 from T alone       = {kl_t:.1f}")
    print(f"  expected Delta chi2 from T + E         = {kl_te:.1f}  (E adds {gain:.1f})")
    print(f"  observed T Delta chi2 + expected E gain = {-best['dchi2'] + gain:.1f}")
    print(f"  EE at l=2-10, cutoff/LCDM: " + " ".join(f"{eec[l]/ee0[l]:.2f}" for l in range(2, 11)))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.errorbar(ell, dl, yerr=[lo, hi], fmt="o", ms=4, color="k", label="Planck 2018 (Commander)")
    lfine = np.arange(2, 31)
    a1.plot(lfine, tt0[lfine], color="tab:blue", label="ΛCDM (Planck 2018)")
    a1.plot(lfine, ttc[lfine], color="tab:orange",
            label=f"with cutoff, k_c = {best['kc']:.1e}/Mpc, λ = {best['lam']}")
    a1.set_xscale("log"); a1.set_xticks([2, 3, 5, 10, 20, 30]); a1.set_xticklabels(["2", "3", "5", "10", "20", "30"])
    a1.minorticks_off(); a1.set_xlabel("multipole ℓ (≈ 180°/angle)"); a1.set_ylabel("D_ℓ^TT [μK²]")
    a1.set_title("Largest angles: data vs models"); a1.legend(fontsize=8)
    for lam, row in zip(lams, grid):
        a2.plot(kcs, row, label=f"λ = {lam}")
    a2.axhline(0, color="k", lw=0.8)
    a2.axhspan(4, 12, color="tab:red", alpha=0.08, lw=0)
    a2.text(kcs[0] * 1.1, 9.5, "stronger cutoffs: disfavoured by the data", fontsize=8, color="tab:red")
    a2.text(kcs[0] * 1.1, -1.6, "below 0: cutoff fits better than ΛCDM", fontsize=8)
    a2.set_ylim(-2.5, 12)
    a2.set_xscale("log"); a2.set_xlabel("cutoff scale k_c [1/Mpc]"); a2.set_ylabel("Δχ² vs ΛCDM")
    a2.set_title("How much a cutoff improves the fit"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "lowl_cutoff.png"), dpi=150)
    print("wrote lowl_cutoff.png")


if __name__ == "__main__":
    main()
