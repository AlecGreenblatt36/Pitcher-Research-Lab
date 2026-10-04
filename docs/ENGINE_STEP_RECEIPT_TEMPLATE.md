# Engine Step Receipt

**Step:** `<short name>`  
**Date:** `<UTC timestamp>`  
**Base engine commit:** `<sha>`  
**Candidate commit:** `<sha>`  
**Data/artifact hashes:** `<sha256 list>`

## What changed

`<one short paragraph; no unrelated work>`

## Frozen comparison

- Games: `2,430`
- Paths/game/variant: `<N>`
- Same games and random-stream protocol: `<yes/no + receipt>`
- Pregame boundary: `<sources and cutoff>`

| Metric | Current engine | Candidate | Candidate − current | 95% CI |
|---|---:|---:|---:|---:|
| Corrected Brier |  |  |  |  |
| Corrected log loss |  |  |  |  |
| Calibration slope |  |  |  |  |
| Calibration intercept |  |  |  |  |
| Run MAE |  |  |  |  |
| Run CRPS |  |  |  |  |
| 50% interval coverage |  |  |  |  |
| 80% interval coverage |  |  |  |  |

## Fair baseline check

`<candidate vs negative-binomial team baseline on corrected Brier/log loss>`

## Decision

**`ACCEPT | REJECT | INCONCLUSIVE`**

`<one-sentence reason tied to frozen metrics; winner accuracy may be reported but is not a decision criterion>`
