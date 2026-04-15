# Kalshi Pricing via News Sequences

## Problem with Original Proposal

Our original proposal asked whether a transformer could distinguish "genuine" price reactions from "emotional overreactions" in Kalshi prediction markets. This framing is fundamentally broken:

- **Overreaction is undefined.** All market movements are reactions to sentiment and information. There is no ground truth for what the "correct" price should have been at the moment of a spike.
- **Labeling is circular.** Using price reversion as a proxy for overreaction just defines overreaction as mean reversion, which conflates noise with emotion.
- **Any comparison requires a benchmark.** If you compare Kalshi to futures or another market, the transformer becomes irrelevant — the question becomes about the benchmark, not the model.

---

## Refined Research Question

> **Can a transformer trained on sequences of news headlines replicate Kalshi contract pricing over the lifetime of a contract?**

The model learns to update a probability estimate day-by-day from headlines alone, mirroring what Kalshi traders do from all available information. Kalshi's price at each timestep is the ground truth label — no interpretation, no benchmark comparison needed.

This answers two things simultaneously:

1. **Is our model predictive?** Does text alone contain enough signal to track market-implied probabilities?
2. **Is Kalshi a good baseline?** If the model converges to Kalshi, it validates Kalshi as a text-predictable aggregator of public information.

**Either result is a real finding:**

- If yes → human belief updating about macro events is largely text-driven and a transformer captures that process
- If no → Kalshi incorporates something fundamentally beyond text: crowd dynamics, trading behavior, or information that never makes headlines

---

## Data Pipeline — Person 1

### Kalshi Price Histories

Pull daily prices for macro contracts via the Kalshi API.

Target contract types:

- CPI threshold
- Fed rate decision
- Unemployment
- PCE

Goal: 50–100 contracts with full lifetime price histories. Each contract has a start date, resolution date, and daily closing price.

### Headline Collection

For each contract, query GDELT daily with contract-specific keywords (e.g. `"CPI inflation"` for CPI contracts) which all must be >= 5 chars long for GDELT. Collect all headlines for each day from contract open to resolution. Match headlines to Kalshi prices by date.

Note:

- Using GDELT gives us a 3 month before the current day window to collect headlines so we will have to either choose all contract that have a significant amount of data during that time or figure out a different way of getting headlines
- NewsAPI has only a 1 month window, thats why I've opted to go with GDELT instead for now. There is a NewsAPI payed version that would allow us a much larger window but I don't think that should be necessary.

### Input Representation

One vector per day per contract, constructed by concatenating three components:

| Component      | Description                                                        | Dimension |
| -------------- | ------------------------------------------------------------------ | --------- |
| BERT embedding | Average of 384-dim BERT embeddings across all headlines that day   | 384       |
| FinBERT score  | Scalar sentiment scores averaged across headlines (pos, neg, nuet) | 3         |
| Price delta    | Kalshi price change from previous day                              | 1         |

**Final input per day:** ~388-dim vector  
**Full contract input:** sequence of `T` such vectors, where `T` = contract lifetime in days

---

## Model — Person 2

### Architecture

Transformer encoder with:

- **Causal masking** — at timestep `t` the model attends only over days `1...t`, never future headlines
- **Learned temporal positional embeddings**
- **Linear regression head** on top of encoder output, producing a scalar predicted Kalshi price at each timestep

**Loss:** MSE between predicted price and actual Kalshi price at each timestep, averaged across all timesteps and all contracts in the batch.

### Why Causal Masking is Load-Bearing

Without causal masking the model can cheat by attending to future headlines. With it, the model is forced to do the same sequential belief updating a trader would do — it must produce a valid probability estimate at every point using only what it would have known at that moment. This is the core justification for using a transformer over a simpler model.

### Ablations — Model Components

| Model | Input                                       | Question Answered                                                 |
| ----- | ------------------------------------------- | ----------------------------------------------------------------- |
| A     | FinBERT score only, no sequence             | Does sentiment alone predict Kalshi?                              |
| B     | BERT embeddings, single day, no sequence    | Does semantic content alone predict Kalshi?                       |
| C     | BERT embeddings in transformer sequence     | Does sequential context add anything over single-day semantics?   |
| D     | BERT + FinBERT + price delta in transformer | Does adding price history improve replication?                    |
| E     | Same as C but LSTM                          | Does attention specifically matter or just sequential processing? |
| F     | Same as C but mean pooling                  | Does order matter at all?                                         |

> **The gap between B and C is the most important result in the project.** It directly justifies whether the transformer is load-bearing. The gap between C, E, and F tells you whether the attention mechanism specifically is doing useful work.

### Ablations — Sequence Length

| Window        | Question Answered            |
| ------------- | ---------------------------- |
| 3 days        | Only recent headlines matter |
| 7 days        | One week of context          |
| 14 days       | Two weeks                    |
| Full contract | Entire lifetime              |

If 3 days matches full contract performance, long-range attention is not contributing. This is an important null result worth reporting.

---

## Analysis — Person 3

### Convergence Analysis

For each test contract, plot model estimate vs Kalshi price over the full contract lifetime. Compute MSE at each timestep `t` and average across all test contracts.

**Primary question:** does error decrease as `t` approaches resolution?

If yes, the model is converging to Kalshi using text alone as information accumulates. The shape of this error curve is a core result.

### Cross-Contract Analysis

Evaluate separately on CPI, Fed rate, unemployment, and PCE contracts. Compare convergence speed and final MSE across types.

**Hypothesis:** Fed rate contracts are more text-predictable because Fed communication is deliberate and heavily textual. CPI contracts may be less predictable because they depend on hard economic data releases that may not be anticipated in headlines.

Also compare attention weight distributions across contract types — does the model attend to different positions in the sequence for different contract types?

### Failure Analysis

For each test contract at each timestep compute:

```
divergence(t) = |model_estimate(t) − kalshi_price(t)|
```

Identify the top 10% highest divergence days across all test contracts. For each divergence day, characterize:

- **Sentiment:** is the FinBERT score higher on divergence days? Tests whether emotionally charged headlines cause the model to deviate from Kalshi
- **Volume:** are divergence days low-headline days where the model had little signal to work with?
- **Manual inspection:** read 10–20 divergence day headlines and characterize what type of news caused the gap
- **Next-day correction:** did the model catch up to Kalshi the next day, or did Kalshi move toward the model?

This answers the most interesting question in the project: **when does text fail to capture what the market knows?**

### Attention Visualization

On high-divergence days vs low-divergence days, extract and compare attention weight distributions. Does the model attend uniformly across the sequence on divergence days (no clear signal) vs concentrated attention on specific earlier days on convergence days? Visualize as heatmaps over contract lifetime for a handful of representative contracts.

---

## Work Split

| Person   | Role          | Responsibilities                                                                                                |
| -------- | ------------- | --------------------------------------------------------------------------------------------------------------- |
| Person 1 | Data pipeline | Kalshi API, NewsAPI, BERT embeddings, FinBERT scores, price delta computation, train/val/test split by contract |
| Person 2 | Modeling      | Transformer architecture, causal masking, training loop, all ablation models A–F, sequence length sweep         |
| Person 3 | Analysis      | Convergence curves, cross-contract comparison, failure analysis, attention heatmaps, report writing             |

---

## Report Structure

1. **Introduction** — research question, motivation from Kalshi efficiency literature, why a transformer is the right architecture
2. **Data** — contract selection, headline collection, input representation, dataset statistics
3. **Model** — architecture, causal masking, training procedure, ablation design
4. **Results** — ablation table, convergence curves, sequence length sweep
5. **Analysis** — cross-contract comparison, failure analysis, attention visualization
6. **Conclusion** — what text alone captures vs what it misses, limitations, future work

---

## Known Limitations

- **Small dataset.** ~50–100 contracts is small. The model is fine-tuned on top of pretrained BERT so less data is required than training from scratch, but results should be interpreted cautiously.
- **Headline coverage.** NewsAPI coverage may be incomplete or uneven across contract types, particularly for less prominent macro indicators.
- **Price delta information leak.** Including price delta as an input feature (Model D) gives the model partial access to the thing it is predicting. Ablation D vs C quantifies this leakage directly.

These are limitations to state explicitly in the report — not problems that sink the project.
