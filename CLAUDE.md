# CLAUDE.md

Instructions for an agent working in this folder.

## What this is

Personal notes on the question "what happened before the Big Bang, and what
caused it?". The goal is *plausible hypotheses*, clearly labelled, not claims
of having solved anything. It started on 2026-10-22 as a side conversation
during a nanoprover session and was moved out of that repo because it is a
different subject. Keep it separate: never copy it back into
`~/tmp/mathTest/nanoprover`.

- `before_big_bang.md`: the main note. It covers what is known, inflation as the
  probable "first step back", a table of hypotheses for what came before (no-boundary,
  tunnelling, eternal inflation, bounce, conformal cyclic cosmology, black-hole
  origin, CPT-mirror universe), a personal bet with rough credences, a testable
  version (large-scale CMB anomalies as traces of the pre-inflation era), how to
  imagine "no time", whether space survives without time, and (section 8) which
  models the measured n_s and r already rule out.
- `ns_r.py` / `ns_r.png`: slow-roll (n_s, r) predictions for standard inflation
  models vs current CMB limits. Run with
  `uv run --no-project --with numpy --with scipy --with matplotlib python ns_r.py`
  (system python3 has numpy/scipy but no matplotlib).
- `before_the_bang.html`: a popular-science version of the note for non-experts
  (timeline, spacetime sketches of the seven stories, an interactive "which stories
  survive?" scenario section, button-driven animations: rewind the timeline, play
  inflation, play each sketch). Standalone file (opens offline except Google Fonts). Keep
  it in sync with the note when the science changes.

Git repository, remote `origin` = github.com/Jrbecart-uo/cosmology (private),
branch `main`.

## How to work here

- **Separate established physics from speculation, every time.** The hot Big Bang
  is solid; inflation is well-supported but not proven; everything before it is
  hypothesis. Label credences as personal guesses.
- **Cite real work by author and year** (e.g. Hartle–Hawking 1983, Vilenkin 1982,
  Borde–Guth–Vilenkin 2003, Boyle–Finn–Turok 2018). Check any number or citation
  you are not sure of before writing it down. No invented papers or values.
- **Prefer testable versions of ideas**: say what observation would support or kill
  a hypothesis (primordial gravitational waves, CMB polarisation at the largest
  scales, neutrino masses, spectral index).
- **Plain language first.** The user is curious and mathematically strong but not a
  cosmologist; explain with analogies, then give the precise statement.
- **Update `before_big_bang.md` in place** when the user asks follow-up questions
  (add a section) instead of creating scattered new files.

## Possible next steps (offered, not started)

- Update the constraint table in section 8 when new results appear (SPT-3G full
  survey, Simons Observatory, LiteBIRD, newer DESI). Look them up and cite them.
- Add the second-order slow-roll terms or a full numerical mode calculation only if
  data precision gets close to 1e-3 in n_s.

## User preferences that apply here too

- Ask before publishing anything (GitHub, forums, shared links). The user posts
  public claims personally. Byline, if ever needed: "Jean-Roch Bécart, using Claude
  Opus 5.5".
- The user prefers substantive work over polish.
