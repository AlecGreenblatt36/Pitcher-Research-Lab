# Pitch-count Markov Gate Receipt

**Decision:** REJECT as a replacement for the frozen seven-outcome PA model  
**Secondary decision:** RETAIN as a candidate pitch-sequence / pitch-count state layer only  
**Workflow run:** `37299217678`  
**Commit:** `1a8ad10e48fe13124c196122ff254e22a09f1613`  
**Generated:** 2026-10-05

## What actually ran

- Downloaded pitch-level Statcast for 2023-03-30 through 2026-09-27.
- 426 three-day source chunks.
- 2,861,903 raw pitch rows.
- 2,861,812 mapped pitch-event rows; 91 unknown rows.
- Pitch transition states: balls, strikes, platoon, batter, pitcher.
- Transition outcomes: ball, called strike, swinging strike, foul, in play, hit by pitch.
- Batted-ball outcome layer: BIP out, single, double/triple, home run, other reach.
- Empirical-Bayes batter/pitcher shrinkage.
- Train 2023-2024, tune 2025, refit 2023-2025, score 2026.
- Frozen comparison cohort: 183,849 PAs in 2,429 games.
- Game-clustered bootstrap: 2,000 replicates.

## Frozen comparison

| Model | Multiclass Brier ↓ | Log loss ↓ |
|---|---:|---:|
| Frozen locked PA ensemble | 0.702370507 | 1.459908409 |
| Pitch-count Markov + EB | 0.706742698 | 1.470841472 |

Candidate minus frozen:

- Brier: **+0.004372191**
- game-clustered 95% CI: **[+0.004065637, +0.004694743]**
- relative Brier change: **+0.6225%**
- log loss: **+0.010933063**
- game-clustered 95% CI: **[+0.010230194, +0.011653058]**
- relative log-loss change: **+0.7489%**

Positive differences are worse. Both proper scores clearly reject promotion.

## Robustness checks

### Source-ID corrections

The fresh Statcast pull and the frozen locked file disagreed on batter and/or pitcher identity for 68 of 183,849 PAs, while game, at-bat number, and outcome matched. Excluding those 68 PAs changed the result negligibly:

- Brier difference: +0.004368872
- log-loss difference: +0.010929990

The rejection is not caused by those source corrections.

### Month-by-month

The Markov candidate was worse in every 2026 month. The Brier disadvantage increased from +0.003115 in March to +0.005289 in September. This is not a one-month anomaly.

### Outcome diagnostics

The largest conditional log-loss losses were:

- HR: +0.0480 nats on actual HR PAs
- BB/HBP: +0.0371
- K: +0.0129
- BIP out: +0.0081

It improved actual single PAs (-0.0058) and other-reach PAs (-0.0251), but not enough to offset the losses.

### Blend diagnostic

This is post-hoc diagnosis, not a promotion test.

- Brier-optimal convex blend assigned **0.0%** weight to Markov.
- Log-loss-optimal full-sample blend assigned **0.32%** weight to Markov and changed log loss by only about 0.00000012.
- Forward month cross-fitting also failed to beat the locked model in aggregate.

The Markov probabilities do not add useful independent PA-outcome signal at this specification.

### Support diagnostic

Low batter support made the loss larger, but high-support players still lost to the frozen model. In the highest batter pitch-support decile, Markov minus locked remained:

- Brier: +0.002337
- log loss: +0.005663

Sparse samples are only part of the problem.

## Interpretation

The pitch-count model is useful for simulating realistic counts and pitch totals, but it should not replace the stronger locked PA probability vector. The locked model contains richer context and better discrimination than this count/platoon/player-count table.

## Accepted next implementation

Use an **outcome-preserving conditional pitch bridge**:

1. Sample the seven-outcome PA result from the frozen locked PA model.
2. Use the fitted Markov transition artifact to sample a pitch sequence conditional on ending in the sampled terminal group (K, BB/HBP, or ball in play).
3. Preserve the locked PA marginal exactly by construction.
4. Add the generated pitch count and count path to fatigue and manager-decision state.
5. Judge the bridge first on pitch-count distribution, then on full-game Brier/log loss/run CRPS after manager/fatigue integration.

The standalone Markov candidate is not promoted.
