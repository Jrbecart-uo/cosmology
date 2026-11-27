# Before the Big Bang

Personal notes on the question "what happened before the Big Bang, and what caused
it?". The goal is plausible, clearly labelled hypotheses, tested against real data
where possible. It is not a claim to have solved anything. Established physics,
well-supported ideas and speculation are kept separate, and credences are personal
guesses.

**Read it online** (popular-science version for non-experts):
[English](https://jrbecart-uo.github.io/before-the-bang/) ·
[Français](https://jrbecart-uo.github.io/before-the-bang/before_the_bang.fr.html)

## Contents

| File | What it is |
|---|---|
| `before_big_bang.md` | The main note, in 10 sections: what is known, inflation, seven hypotheses for what came before, a personal bet, how to think about "no time", and three data tests (below). |
| `before_the_bang.html`, `before_the_bang.fr.html` | The popular-science page in English and French: timeline, sketches of the seven stories, animations, an interactive "which stories survive?" lab. Standalone HTML; open in any browser. |
| `ns_r.py` → `ns_r.png` | Slow-roll predictions (n_s, r) for standard inflation models vs current CMB limits (note section 8). |
| `lowl_cutoff.py` → `lowl_cutoff.png` | Fits a "just enough inflation" large-scale cutoff to the Planck 2018 low-ℓ temperature spectrum with CAMB, with anomaly p-values and an E-mode forecast (section 9). |
| `neutrino_mass.py` → `neutrino_mass.png` | Neutrino-mass observables vs the CPT-mirror universe's massless-neutrino prediction, with DESI, KATRIN and KamLAND-Zen limits (section 10). |

## The three data tests, in one line each

- **n_s and r (section 8):** the simplest inflation models (φ², φ⁴) are dead because
  their gravitational waves were not seen; Starobinsky inflation is about 4σ off
  once DESI galaxy data are included; plateau and very shallow models survive. LiteBIRD
  (results in the mid-to-late 2030s) decides between "inflation with r ≈ 0.003" and "r ≈ 0".
- **The faint largest-scale CMB patterns (section 9):** a pre-inflation cutoff is
  allowed but not needed (under 1σ in this analysis, ~1.3σ in Planck's); cosmic
  variance means the CMB alone probably can never prove it.
- **Neutrino masses (section 10):** cosmology sits right at the CPT-mirror prediction
  (lightest neutrino massless), but partly because of an unexplained tension; a
  next-generation detection of neutrinoless double beta decay in the normal ordering
  would rule it out.

## Running the scripts

Python with [uv](https://docs.astral.sh/uv/); no project setup needed.

```bash
uv run --no-project --with numpy --with scipy --with matplotlib python ns_r.py
uv run --no-project --with numpy --with matplotlib python neutrino_mass.py
uv run --no-project --with camb --with matplotlib python lowl_cutoff.py   # downloads Planck files into data/ (git-ignored)
```

Each prints a table and writes its figure next to the script.

## Status

All numbers and citations were checked against the original papers, and in October
2026 two independent reviews re-checked them against the sources. Data limits move
quickly (DESI, ACT, SPT, JUNO, LiteBIRD), so check the dates in the note before
quoting them.

By Jean-Roch Bécart, using Claude Opus 5.5.
