# Citation verification (ghost-citation check) — liquidity-premium paper

Scope: the **34 keys actually `\cite`d** in `liquidity-premium.tex`. Method: OpenAlex
by-DOI lookup (title/abstract) + Crossref bibliographic query (title+author match ⇒
real) + abstract pull where deposited (OpenAlex `abstract_inverted_index` or Crossref
JATS `abstract`). Abstract text files are named by bib key, one per cited reference
(non-abstractable/logistics entries excluded — see `_NO_ABSTRACT.md`).

## Fixed (prior pass)
- **`klemkosky1979put`** — DOI corrected to `10.2307/2327240` (Crossref-confirmed:
  Klemkosky & Resnick, "Put-Call Parity and Market Efficiency", J. Finance 1979).
- **`pcpb_frl1`** — removed from `bibl.bib` (orphan placeholder "Title of the Paper",
  cited nowhere; the user's PCP paper is already present and cited as `felfoldi2024put`).

## DOI / metadata errors found — ALL NOW FIXED in bibl.bib
Every case below resolves to a **real, matching paper** — none are ghost citations,
just wrong DOI digits/suffixes plus a couple of author slips. All corrections listed
here have now been applied to `bibl.bib`:
- six DOIs: `klemkosky1979put`, `gagnon2010multimarket`, `felfoldi2024put`, `el1998put`,
  `nisbet1992put`, `kamara1995daily`, `alexander2024arbitrage`;
- `nisbet1992put` author first name Mark → **Mary**;
- `alexander2024arbitrage` authors → **Alexander, Chen, Deng, Wang**, article number
  → 100930, and "…Crypto Derivatives ~~Markets~~" trimmed;
- `felfoldi2024put` → full author list (Felföldi-Szűcs, Králik, Váradi), correct title
  ("Put–call parity in a crypto option market — Evidence from Binance"), article no. 104874;
- `pcpb_frl1` orphan stub removed.

- **`gagnon2010multimarket`** — bib DOI `10.1016/j.jfineco.2010.03.008` resolves to an
  unrelated paper ("Comovement, information production, and the business cycle").
  Correct DOI (verified via Crossref: vol. 97, pp. 53–80, authors Gagnon & Karolyi,
  exact match to the bib entry): **`10.1016/j.jfineco.2010.03.005`**.
- **`felfoldi2024put`** — bib DOI `10.1016/j.frl.2023.104963` resolves to an unrelated
  paper ("Uncertainties and oil price volatility: Can lasso help?", Li et al.).
  Correct DOI (Felföldi-Szűcs, Králik, Váradi, "Put–call parity in a crypto option
  market — Evidence from Binance", Finance Research Letters vol. 61, 2024 — this is
  the user's own paper, coauthored with Balázs Králik): **`10.1016/j.frl.2023.104874`**.
- **`el1998put`** — bib DOI `10.1016/S1042-4431(98)00027-9` does not resolve to any
  indexed work. Correct DOI (El-Mekkaoui & Flood, "Put-call parity revisited:
  intradaily tests in the foreign currency options market", J. Int. Financial Markets,
  Institutions and Money, vol. 8, 1998 — same authors/journal/year as the bib entry,
  whose title is a paraphrase of the real one): **`10.1016/S1042-4431(98)00046-8`**.
- **`nisbet1992put`** — bib DOI `10.1016/0378-4266(92)90044-I` does not resolve to
  this paper. Correct DOI (cross-checked against `felfoldi2024put`'s own reference
  list): **`10.1016/0378-4266(92)90021-Q`**. Also, the bib author first name "Mark"
  is wrong — Crossref lists the sole author as **Mary** Nisbet.
- **`kamara1995daily`** — bib DOI `10.2307/2331271` does not resolve to any indexed
  work. Correct DOI (Kamara & Miller, "Daily and Intradaily Tests of European
  Put-Call Parity", JFQA vol. 30, 1995 — exact title/author/journal/year match):
  **`10.2307/2331275`**.
- **`alexander2024arbitrage`** — bib DOI `10.1016/j.finmar.2024.100938` does not
  resolve to any indexed work. The real paper (Journal of Financial Markets vol. 71,
  art. 100930, 2024, "Arbitrage opportunities and efficiency tests in crypto
  derivatives") has DOI **`10.1016/j.finmar.2024.100930`**. Additionally, the bib
  author list ("Alexander, Deng, Imeraj") does not match Crossref, which lists
  **Alexander, Chen, Deng, Wang** for this DOI. (A distinct related paper by
  Alexander, Chen, Imeraj does exist — "Crypto Quanto and Inverse Options",
  `mafi.12410` — but that is the separate `alexander2023crypto` entry, already
  correctly cited.)

## Resolved
- **`gagnon2010multimarket`** DOI question — see above: correct DOI is
  `10.1016/j.jfineco.2010.03.005` (bib's `...008` points to a different, unrelated
  JFE article).
- **`todorov2021bond`** — real (BIS Quarterly Review, Mar 2021) but not in
  Crossref/OpenAlex, so no machine-readable abstract (see `_NO_ABSTRACT.md`).

## Verified real (title + author match) — all 34 cited keys accounted for
Journal papers: shleifer1997limits, amihud1986asset, garleanu2011margin,
du2018deviations, cremers2010deviations, alexander2023crypto, alexander2024arbitrage,
felfoldi2024put, glosten1985bid, demsetz1968cost, ho1981optimal, gagnon2010multimarket,
krishnamurthy2002bond, gromb2010limits, pontiff1996costly, pontiff2006costly,
cameron2011robust, klemkosky1979put, lucic2024valuation, el1998put, nisbet1992put,
kamara1995daily, ofek2004limited, bollen2004modeling, chen2021implied.
Working papers (real): pan2017etf, todorov2021bond.
Non-abstractable but real: esma_mica_2024, sec_spot_bitcoin_etp_2024,
sec_spot_ether_etp_2024, sec_reg_nms_2005, tardis_main, kralik_tardisproject_repo,
kralik_amu_publication_repo.

**No ghost citations** found among the cited set — every entry resolves to a real,
matching work. The issues found are six DOI errors (listed above, all now identified
with corrected DOIs) plus two author-list/name slips (`nisbet1992put`'s "Mark" vs.
actual "Mary"; `alexander2024arbitrage`'s "Deng, Imeraj" vs. actual "Chen, Deng,
Wang"). None of these affect the paper's citation *text* (author-year in the .tex),
only the DOI/metadata fields in `bibl.bib`, which have now all been corrected.

## Abstract files written (all 27 abstractable cited keys)
shleifer1997limits, amihud1986asset, garleanu2011margin, du2018deviations,
duffie2010presidential, cremers2010deviations, pontiff2006costly,
krishnamurthy2002bond, alexander2023crypto, alexander2024arbitrage, felfoldi2024put,
glosten1985bid, demsetz1968cost, ho1981optimal, gagnon2010multimarket, gromb2010limits,
pontiff1996costly, cameron2011robust, klemkosky1979put, lucic2024valuation, el1998put,
nisbet1992put, kamara1995daily, ofek2004limited, bollen2004modeling, chen2021implied,
pan2017etf.

Of these, the following have a `.txt` file that explicitly states **no abstract is
available** from Crossref/OpenAlex (publisher page only), after existence/authorship
was verified: amihud1986asset, glosten1985bid, demsetz1968cost, ho1981optimal,
gagnon2010multimarket, klemkosky1979put, el1998put, nisbet1992put, ofek2004limited,
chen2021implied, pan2017etf, felfoldi2024put. (See also `_NO_ABSTRACT.md`.)

Working papers/non-abstractable entries with no `.txt` file (by design — see
`_NO_ABSTRACT.md`): todorov2021bond, esma_mica_2024, sec_spot_bitcoin_etp_2024,
sec_spot_ether_etp_2024, sec_reg_nms_2005, tardis_main, kralik_tardisproject_repo,
kralik_amu_publication_repo.
