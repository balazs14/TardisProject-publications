# References with no accessible abstract

Files in this folder are named by their bib key (e.g. `shleifer1997limits.txt`), so
each maps to a `\cite{key}` in the paper. This note lists entries for which no
abstract was retrieved, grouped by reason.

## Not an abstractable work (books, regulations, web/data, code)
These have no journal-style abstract by nature.

- **Books / textbooks:** `foucault2013market`, `persaud2008liquidity`, `goossens93`
  (LaTeX Companion), `jorion2001` (Value at Risk), `Hull` (Options, Futures, and
  Other Derivatives).
- **Regulations / official statements:** `sec_spot_bitcoin_etp_2024`,
  `sec_spot_ether_etp_2024`, `esma_mica_2024` (MiCA), `sec_reg_nms_2005`,
  `RTS` (EU 153/2013), `EMIR` (EU 648/2012).
- **Web pages / data sources:** `okx_options`, `okx_futures`, `usd_vs_usdt`,
  `tardis_main`, `turn3search3` (also both point to tardis.dev), .
- **Code repositories:** `kralik_tardisproject_repo`, `kralik_amu_publication_repo`.

## Flagged issues (worth fixing in bibl.bib)
- **`pcpb_frl1`** — the entry's title is literally "Title of the Paper" with no
  author/venue. This looks like a leftover placeholder/stub. Recommend replacing or
  removing it.
- **`klemkosky1979put`** — the DOI in the bib (`10.1111/j.1540-6261.1979.tb00060.x`)
  resolves in OpenAlex to a *different* paper ("Currency Option Bonds, Puts and Calls
  on Spot Exchange…") with no abstract. The DOI appears to be wrong for
  "Put-Call Parity and Market Efficiency" (Klemkosky & Resnick 1979). Recommend
  verifying/fixing the DOI.

## Abstract not deposited with the provider (older / SSRN-only / short-DOI papers)
Existence and authorship verified for all of these via Crossref/OpenAlex title+author
match; no `abstract` or `abstract_inverted_index` field is deposited, so the `.txt`
file for each states "No abstract available from Crossref/OpenAlex (publisher page
only)" instead of reproducing an abstract.

- **`amihud1986asset`** — 1986 JFE paper, no abstract deposited.
- **`glosten1985bid`** — 1985 JFE paper, no abstract deposited.
- **`demsetz1968cost`** — 1968 QJE paper; OpenAlex carries only a table-of-contents
  listing, not a genuine abstract.
- **`ho1981optimal`** — 1981 JFE paper, no abstract deposited.
- **`gagnon2010multimarket`** — 2010 JFE paper, no abstract deposited (see DOI fix
  below).
- **`klemkosky1979put`** — 1979 J. Finance paper, no abstract deposited.
- **`el1998put`** — 1998 JIFMIM paper, no abstract deposited (see DOI fix below).
- **`nisbet1992put`** — 1992 J. Banking & Finance paper, no abstract deposited (see
  DOI fix below).
- **`ofek2004limited`** — 2004 JFE paper, no abstract deposited.
- **`chen2021implied`** — 2021 J. Econometrics paper, no abstract deposited.
- **`pan2017etf`** — 2017 SSRN/ESRB working paper, abstract only on the SSRN page,
  not machine-readable via Crossref/OpenAlex.
- **`felfoldi2024put`** — 2024 FRL paper (the user's own, with Balázs Králik and Kata
  Váradi), no abstract deposited (see DOI fix below).

## DOI corrections found — NOW APPLIED to bibl.bib (see `_VERIFICATION.md` for detail)
Six bib DOIs were wrong (pointing to unrelated or non-existent Crossref records):
`gagnon2010multimarket`, `felfoldi2024put`, `el1998put`, `nisbet1992put`,
`kamara1995daily`, `alexander2024arbitrage` (plus `klemkosky1979put` earlier). All have
now been corrected in `bibl.bib`, along with author fixes for `nisbet1992put` (Mary),
`alexander2024arbitrage` (Chen, Deng, Wang), and `felfoldi2024put` (full author list +
title), and removal of the `pcpb_frl1` stub.

_(This list is now complete for all 34 cited keys; see `_VERIFICATION.md` for the
full verification summary.)_
