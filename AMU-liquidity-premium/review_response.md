# Response to review — verification results and action plan

Status of each point after checking claims against the code (`tardis/process_pcp.py`,
`panel_regressions.py`, `amu_panel.py`, built artifacts). Verdicts: **Correct — must fix**,
**Partially correct**, **Rebuttable**.

---

## 1. "The statistic is not identified as a risk premium" — Partially correct; reframe

The reviewer is right that the nested/straddling payoffs are conditional on queue position,
fill, leg availability, and hedge latency, so the positive part of the quote-parity gap is an
*upper bound* on demanded compensation, contaminated by quote noise and latency.

Response plan:
- Keep the measured object primary and descriptive. Headline name: **quote-level parity gap**
  (or "parity-implied maker wedge"); present ARP as the *interpretation* of its systematic
  component, not as the definition. This is close to the reviewer's own suggested reframing
  and preserves all data work.
- Add the execution-facing caveats explicitly where the measure is defined (queue, fill,
  legging, adverse selection).
- Feasible validation with data we already have: (a) wedge *persistence* (time-in-force of
  executable wedges at tick resolution — transient noise dies in seconds, demanded
  compensation persists); (b) trade-conditioned checks using the trade columns already in the
  panel (`*_trade_amount`); (c) the frequency/size decomposition already separates the
  incidence channel the reviewer worries about.
- Do not claim a structural premium without one of these; soften "This settles RQ_i".

## 2. Algebra vs implementation — **Correct — must fix (verified)**

Derived leg-by-leg from `process_pcp.py` (X_a=(F_ask−K)e^{−rT}, X_b=(F_bid−K)e^{−rT}):

| path (code) | implemented payoff |
|---|---|
| fwd_joincall | C_ask − P_ask − X_ask − c |
| fwd_joinput  | C_bid − P_bid − X_ask − c |
| bck_joincall | P_bid − C_bid + X_bid − c |
| bck_joinput  | P_ask − C_ask + X_bid − c |

The implementation is economically correct (forward crosses the futures ask; backward the
bid). The paper's Table 1 swaps **both** the futures sides and the join-call/join-put labels.
Downstream corrections required:
- **Table 1**: replace with the implemented formulas above (linear + inverse variants).
- **Band (Eq. 6)**: correct C_ask band is [P_ask+X_b−c, P_ask+X_a+c]: half-width
  cost + ½(F_ask−F_bid)e^{−rT}. The futures spread **widens** the band; it is non-empty at
  cost = 0. Delete the "interval inverts / no fair ask exists" paragraph.
- **Conjugate-sum claim**: payoffs sum to −2c − (X_a−X_b), i.e. mirror images about
  −c − half-futures-spread. The mode-at-−cost argument survives (futures spread ≪ cost in the
  data) but must be restated with the spread term.
- **Figures 1–3** (hand-built schematics): re-label paths and band edges to match.
- **Unit tests**: add a leg-by-leg cash-flow test asserting each Table-1 formula equals the
  implemented column on synthetic quotes (`tardis/tests/test_pcp_metrics.py` is the place).

## 3. c20 vs 30 bp dependent — **Correct in the built artifacts; two causes**

- `AMU_DEPENDENT` in the current source is `mean_amu_uncond_bp_c30` (= the 30 bp baseline).
  The reviewer evidently saw a `c20` state; the built `artifacts/text_numbers.tex`
  (\AmuPreBp=14.9, \AmuPostBp=6.6) is inconsistent with Figure 10's annotations (12.6, 4.5),
  so at minimum the macros must be regenerated from the c30 run.
- Second cause the review missed: even at the same cost the two numbers are different
  **estimands** — the macros are n_obs-weighted means of *filtered panel cells*; the figure
  annotates the *pooled tickpath* mean. Fix by computing the prose macros from the same
  pooled construction as the figure (or annotating the figure with the panel estimand), and
  by stating the estimand (tick-weighted vs cell-weighted) once, explicitly.
- Guard: emit the dependent-variable name and cost into `text_numbers.tex` as a macro and
  print it in the table notes, so prose/figures/regressions cannot silently diverge again.

## 4. 2024 not causally identified — Correct; reframe + add diagnostics

Accept: single break date, four co-treated markets, no control. Actions:
- Language: "descriptive regime break around 2024", drop causal phrasing ("settles",
  "signature of entry"). The regulation section states a hypothesis, not identification.
- Add: pre-trend plot with confidence bands; placebo break dates (e.g., each quarter
  2021–2025) with the 2024-01-10 estimate located in that distribution; unknown-break
  (sup-Wald/Bai–Perron) test; market-specific linear trends as robustness; vol controls
  (realized/implied) in spec (4).

## 5. HC1 SEs untenable — Correct; must fix

With ~few-thousand distinct days and 4 markets, HC1 is indefensible. Actions:
- Cluster by **day** (primary), two-way day × cell as robustness; block bootstrap by day.
- Report a collapsed daily-aggregate regression (one obs per day per market) as the honest
  small-sample version.
- Shift emphasis from stars to magnitudes, as suggested.

## 6. Staleness contamination — Partially correct; we have direct evidence

- We have already run the pipeline with `filter_stale` on and off: the headline shapes and
  the pre/post compression are unchanged (this experiment motivated adding the flag).
  Report both configurations in a robustness table rather than asserting it.
- Accept the harder version: add max-quote-age tolerance variants (drop legs older than
  {30s, 60s, 300s} within the grid) and event-time synchronization at the tick level for a
  subsample; report the by-cost figure under each.
- Accept: executable-notional refinement (common tradable notional across legs, not
  per-leg $5k minima) — the quantity columns needed are already in the aligned data.

## 7. Cost and r assumptions — Correct on r; partially on cost

- r = 5% constant over 2020–2026 is not defensible; rates moved ~0→5%. Because r enters via
  (F−K)e^{−rT}, the error scales with basis × maturity; at long maturities it is bp-order
  and time-correlated with the break — exactly the reviewer's concern. Action: time-varying
  r (SOFR/OIS curve or 3M bill) as the default; constant-r as robustness. Add r to the
  data-processing-assumptions table either way.
- Cost heterogeneity: we cannot know every VIP tier; the response is the existing cost-grid
  (5–50 bp) plus venue-specific cost scenarios (e.g., Deribit vs OKX published taker fee
  schedules) as columns of the robustness table. The grid answers level-sensitivity, not
  fee-structure *changes*; acknowledge that residual explicitly.

## 8. Mechanism partly algebraic — Half correct; rebut the half that isn't

- Rebuttal: mirror-pairing makes the pooled distribution *symmetric* about −c − s_F/2
  mechanically, but not *peaked*. The uniform benchmark is equally symmetric; the observed
  concentration at the center is empirical content (quotes tracking the synthetic top of
  book). State this distinction explicitly in the text.
- Accept: the section must be rewritten anyway (see #2); add tick-size controls, delta
  buckets, unclipped/winsorized variants, and quote-age-conditioned surface plots for the
  V/U shapes.

## 9. Controls jointly determined — Correct; soften

Spread/depth/staleness are equilibrium outcomes from the same quotes, not clean confounders.
Reframe spec (4) as an *accounting* decomposition ("how much of the break co-moves with
observable book quality") rather than causal control. Drop "participants with higher risk
tolerance entered" as a conclusion; keep it as the hypothesis the pattern is consistent with,
absent participant-level data.

## Smaller quibbles — accept nearly all

- Title vs "ARP is not the spread": revert title to gap/parity language (e.g. *"The
  Compression of Quote-Level Parity Gaps in Cryptocurrency Option Markets"*) or keep spreads
  title and define "effective spread vs the synthetic book" carefully. Decision needed.
- "This settles RQ_i" → "consistent with"/"answers descriptively". Done wholesale.
- Fig 9: relabel ("levels around event dates"), or normalize to t=−1 and de-overlap windows.
- Fig 14: switch y to unconditional ARP (one-line change in `panel_figures.py`) or drop.
- Table 4: label the tick-level count vs path-level φ distinction (partially done; add φ
  column explicitly).
- Add r row to Table 3 (with macro, once r is time-varying: "OIS curve").
- State the estimand (tick- vs cell-weighted) once, use consistently (ties to #3).
- Two-part model: agreed and natural — frequency (fractional logit / LPM) and conditional
  size regressions alongside the unconditional one; infrastructure already exists.
- "Executable profit" phrasing: align with #1 caveats.
- okex → OKX display names in figures (cosmetic, one mapping).
- References/URLs, annual sample counts, coverage timelines, missing-data diagnostics,
  data-quality-around-2024 note: add appendix table (counts per market-year already
  computable from the panel).

## Sequencing (proposed)

1. **Correctness first** (blocks everything): Table 1 + band section + schematics rewrite to
   match implementation; unit tests; regenerate macros at c30 with a single declared
   estimand. (#2, #3)
2. **Inference**: day/two-way clustering + daily-aggregate regression; placebo/unknown-break
   tests; pre-trends. (#4, #5)
3. **Measurement robustness**: stale-filter table, quote-age tolerances, time-varying r,
   executable-notional, tick-size/clipping variants. (#6, #7, #8)
4. **Reframing pass**: gap-first naming, ARP as interpretation, softened institutional
   claims, title decision, small quibbles. (#1, #9, quibbles)
5. Target Journal of Futures Markets per the reviewer's ladder.
