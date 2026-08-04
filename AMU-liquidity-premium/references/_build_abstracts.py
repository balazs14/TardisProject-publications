"""Helper: write one plain-text abstract file per reference into references/.

Abstracts are gathered from OpenAlex (reconstructed from its inverted index) or
Crossref, keyed by the bib key so each file maps to a \\cite{key}. Re-runnable;
appends/overwrites as batches are gathered.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

# key -> (title, abstract_plaintext)
ABSTRACTS: dict[str, tuple[str, str]] = {
    "shleifer1997limits": (
        "The Limits of Arbitrage",
        "Textbook arbitrage in financial markets requires no capital and entails no "
        "risk. In reality, almost all arbitrage requires capital, and is typically risky. "
        "Moreover, professional arbitrage is conducted by a relatively small number of "
        "highly specialized investors using other people's capital. Such professional "
        "arbitrage has a number of interesting implications for security pricing, including "
        "the possibility that arbitrage becomes ineffective in extreme circumstances, when "
        "prices diverge far from fundamental values. The model also suggests where anomalies "
        "in financial markets are likely to appear, and why arbitrage fails to eliminate them.",
    ),
    "garleanu2011margin": (
        "Margin-Based Asset Pricing and Deviations from the Law of One Price",
        "In a model with heterogeneous-risk-aversion agents facing margin constraints, we "
        "show how securities' required returns increase in both their betas and their margin "
        "requirements. Negative shocks to fundamentals make margin constraints bind, lowering "
        "risk-free rates and raising Sharpe ratios of risky securities, especially for "
        "high-margin securities. Such a funding-liquidity crisis gives rise to “bases,” "
        "that is, price gaps between securities with identical cash-flows but different margins. "
        "In the time series, bases depend on the shadow cost of capital, which can be captured "
        "through the interest-rate spread between collateralized and uncollateralized loans "
        "and, in the cross-section, they depend on the relative margins. We test the model "
        "empirically using the credit default swap–bond bases and other deviations from "
        "the Law of One Price, and use it to evaluate central banks' lending facilities.",
    ),
    "du2018deviations": (
        "Deviations from Covered Interest Rate Parity",
        "We find that deviations from the covered interest rate parity (CIP) condition imply "
        "large, persistent, and systematic arbitrage opportunities in one of the largest asset "
        "markets in the world. Contrary to the common view, these deviations for major "
        "currencies are not explained away by credit risk or transaction costs. They are "
        "particularly strong for forward contracts that appear on banks' balance sheets at the "
        "end of the quarter, pointing to a causal effect of banking regulation on asset prices. "
        "The CIP deviations also appear significantly correlated with other fixed income spreads "
        "and with nominal interest rates.",
    ),
    "duffie2010presidential": (
        "Presidential Address: Asset Price Dynamics with Slow-Moving Capital",
        "I describe asset price dynamics caused by the slow movement of investment capital to "
        "trading opportunities. The pattern of price responses to a supply or demand shock "
        "typically involves a sharp reaction to the shock and a subsequent and more extended "
        "reversal. The amplitude of the immediate price impact and of the subsequent recovery "
        "can reflect the institutional impediments to immediate trade, such as search costs for "
        "trading counterparties or time to raise capital by intermediaries. I discuss special "
        "impediments to capital formation during the recent financial crisis that caused asset "
        "price distortions, which subsided afterward. After presenting examples of price "
        "reactions to supply shocks in normal market settings, I offer a simple illustrative "
        "model of price dynamics associated with slow-moving capital due to the presence of "
        "inattentive investors.",
    ),
    "cremers2010deviations": (
        "Deviations from Put-Call Parity and Stock Return Predictability",
        "Deviations from put-call parity contain information about future stock returns. Using "
        "the difference in implied volatility between pairs of call and put options to measure "
        "these deviations, we find that stocks with relatively expensive calls outperform stocks "
        "with relatively expensive puts by 50 basis points per week. We find both positive "
        "abnormal performance in stocks with relatively expensive calls and negative abnormal "
        "performance in stocks with relatively expensive puts, which cannot be explained by short "
        "sale constraints. Rebate rates from the stock lending market directly confirm that our "
        "findings are not driven by stocks that are hard to borrow. The degree of predictability "
        "is larger when option liquidity is high and stock liquidity low, while there is little "
        "predictability when the opposite is true. Controlling for size, option prices are more "
        "likely to deviate from strict put-call parity when underlying stocks face more "
        "information risk. The degree of predictability decreases over the sample period. Our "
        "results are consistent with mispricing during the earlier years of the study, with a "
        "gradual reduction of the mispricing over time.",
    ),
    "pontiff2006costly": (
        "Costly arbitrage and the myth of idiosyncratic risk",
        "Transaction and holding costs make arbitrage costly. If some traders are rational, "
        "mispricing will only exist to the extent that arbitrage costs prevent rational traders "
        "from fully eliminating inefficiencies. Although the relation between mispricing and "
        "transaction costs is well-known, the relation between mispricing and holding costs is "
        "misunderstood. One holding cost, idiosyncratic risk, is particularly misunderstood. "
        "Various myths are debunked, including the common myth that arbitrageurs care about "
        "idiosyncratic risk because they are undiversified [Shleifer and Vishny (1997)]. The "
        "literature demonstrates that idiosyncratic risk is the single largest cost faced by "
        "arbitrageurs.",
    ),
    "krishnamurthy2002bond": (
        "The Bond/Old-Bond Spread",
        "This paper studies the spread between the newly issued 30 year Treasury bond and the "
        "old 30 year bond. The spread follows a systematic pattern over the auction cycle: it "
        "begins high at an auction date, and converges toward zero by the next auction date. I "
        "document the profits on establishing a long old bond/short new bond convergence trade "
        "and rolling this over every auction cycle, over a period from 6/95 to 11/99. Despite the "
        "systematic convergence, the average profits are close to zero, and profits covary with "
        "the stock market in a manner resembling a short put option position. The difference in "
        "repo market financing rates between the two bonds is an important component of the costs "
        "in carrying the position. I then ask what economic factors account for the level and time "
        "variation in the spread. Using the spread between commercial paper and T-Bills to identify "
        "changes in investor preference for liquid assets, I establish that aggregate factors "
        "affecting liquidity preference play an important role in the variation of the bond/old-bond "
        "spread. The evidence also establishes that new bonds are imperfect substitutes for other "
        "bonds, and therefore changes in the supply of new bonds, such as in recent Treasury "
        "buybacks, will have important effects on bond market spreads.",
    ),
}


def main() -> None:
    for key, (title, abstract) in ABSTRACTS.items():
        path = HERE / f"{key}.txt"
        path.write_text(f"{key}\nTitle: {title}\n\n{abstract.strip()}\n", encoding="utf-8")
    print(f"wrote {len(ABSTRACTS)} abstract files to {HERE}")


if __name__ == "__main__":
    main()
