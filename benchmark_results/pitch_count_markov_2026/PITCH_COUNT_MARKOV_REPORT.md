# Pitch-count Markov frozen replay — negative result

Generated: `2026-10-05T11:06:33.355695+00:00`

## Decision

**Do not promote the pitch-count Markov empirical-Bayes model over the frozen PA baseline.**

The candidate was worse on both proper scoring rules, and the game-clustered confidence intervals were entirely above zero. This result is retained intentionally rather than hidden or tuned away.

## Evaluation boundary

- Test season: 2026 regular season
- Dates: 2026-03-25 through 2026-09-27
- Plate appearances: 183,849
- Games / bootstrap clusters: 2,429
- Historical pitch rows downloaded: 2,861,903 across 426 files
- Historical coverage: 2023-03-30 through 2026-09-27
- Bootstrap replicates: 2,000, clustered by game

The 2026 period had already been inspected elsewhere in the project, so this is a frozen replay diagnostic rather than a pristine final holdout.

## Proper-score comparison

| Metric | Frozen PA baseline | Pitch-count Markov | Candidate minus frozen | 95% game-clustered CI |
|---|---:|---:|---:|---:|
| Multiclass Brier | 0.702370507 | 0.706742698 | +0.004372191 | [+0.004065637, +0.004694743] |
| Log loss | 1.459908409 | 1.470841472 | +0.010933063 | [+0.010230194, +0.011653058] |

Relative degradation:

- Brier: +0.6225%
- Log loss: +0.7489%

Lower is better for both metrics, so the positive differences are losses.

## Model protocol

- Transition model tuning seasons: 2023–2024 training, 2025 tuning
- Final refit seasons: 2023–2025
- Test season: 2026
- Pitch transition states included count, platoon, batter and pitcher empirical-Bayes components
- Ball-in-play outcomes were modeled separately and composed with the pitch-state absorbing probabilities
- Only 91 of 2,861,903 downloaded pitch rows were unmapped

Selected transition parameters:

- batter prior: 150
- pitcher prior: 150
- batter weight: 1.00
- pitcher weight: 0.75
- temperature: 1.00

Selected ball-in-play parameters:

- batter prior: 150
- pitcher prior: 150
- batter weight: 0.75
- pitcher weight: 0.25
- temperature: 1.00

## Reproducibility receipt

- GitHub Actions run: `37299217678`
- Source commit: `1a8ad10e48fe13124c196122ff254e22a09f1613`
- Actions artifact ID: `11341657363`
- Artifact ZIP SHA-256: `9b0b0bb8c20662c2242f9c7b56bed6a9bc9de07ab84f3dc645ed9138cb5d89f0`
- Input data SHA-256: `f1451acbbc2bd9e6817b3bb54524570cf3da3e93cc1d6966666c434ebfb68a35`
- Frozen baseline-score SHA-256: `62a1453f2b261a59258c4b24113c1a5fdc926819813fbc3a41bd3f1995edb3cc`
- Candidate predictions SHA-256: `b30b6b26de2c4290757258eb27208fff4612fb11f29e9ce3e1e1d48add31c579`
- Candidate game-score SHA-256: `18e684994a300692be29741b9c7f7eaad67306c2ed4e96777a040aa0ec8a2682`
- Serialized model SHA-256: `633c3db1d7e54dbb5429afc19b75a38a3b30a3908f5aba324d7e78b1534caa98`

## Engineering conclusion

Count-state dynamics are real baseball structure, but this implementation did not convert that structure into better terminal PA probabilities than the existing frozen model. The downstream full-game engine should therefore continue using the locked PA model. Any future pitch-sequence work must be treated as a new research branch and evaluated against a new development period; the failed candidate must not be blended into production simply because it is more detailed.
