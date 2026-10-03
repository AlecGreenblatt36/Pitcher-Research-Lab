# Baseball Research Lab: chronological PA model

This package is the predictive foundation of the Baseball Research Lab. It estimates a seven-outcome probability distribution before a plate appearance:

- ball-in-play out
- strikeout
- walk or hit by pitch
- single
- double or triple
- home run
- other reach (error, fielder's choice, catcher interference)

## Scientific boundary

The model uses only information that can exist before the target PA. Same-PA velocity, location, launch speed, launch angle, xwOBA, wOBA value, run value, and terminal result-derived fields are not predictors. All player, platoon, park, and recent-form counters are **date blocked**: every PA on date D is scored before outcomes from D are revealed.

The validation period is divided chronologically into separate model-tuning, calibration, and ensemble-selection blocks. The final candidate is a validation-selected blend of a calibrated multinomial model and an empirical-Bayes matchup baseline.

## Development evaluation

The reproducible development benchmark uses:

- train: 2023
- tune/calibrate/blend: disjoint date blocks in 2024
- development holdout: 2025

The benchmark compares prior-date league rates, a shrunk empirical-Bayes batter/pitcher/platoon/park/recent baseline, a talent-only multinomial model, a pre-PA context model, and the calibrated candidate ensemble.

Primary metrics are multiclass log loss, multiclass Brier score, calibration, and game-clustered uncertainty. The candidate must beat the empirical-Bayes baseline with a clustered confidence interval that excludes zero, show incremental context signal, and keep maximum absolute class calibration error below 0.015.

Because the first 2025 result informed the final calibration and ensemble architecture, 2025 is now labeled a **development holdout**, not an untouched production test.

## Locked final evaluation

A locked final run uses:

- train: 2023–2024
- tune/calibrate/blend: disjoint date blocks in 2025
- locked test: 2026
- `--evaluation-mode locked_final`

No architecture, threshold, or hyperparameter may be changed after seeing the locked result and then re-scored as though the same 2026 data remained untouched.

## Reproducible development run

Install the project requirements plus `requirements-model.txt`, then run:

```text
python -m research_lab.pa_model.cli --verbose run-all \
  --start 2023-03-20 \
  --end 2025-11-05 \
  --work-dir model_runs/pa_v1 \
  --train-years 2023 \
  --validation-years 2024 \
  --test-years 2025 \
  --evaluation-mode development
```

The run produces:

- `plate_appearances.csv.gz` and a provenance receipt
- `PA_BENCHMARK_RESULT.json`
- `TUNING_AUDIT.json`
- `test_predictions.csv.gz`
- `pa_model.joblib`
- source, model, prediction, tuning, feature, and candidate fingerprints

## What a passing PA result does not prove

A passing PA model does not validate the outcome-conditioned runner kernel, between-pitch events, batting-order logic, pitcher removal, bullpen selection, team-run distributions, winner probabilities, or exact scores. Those layers require their own frozen prospective tests after the PA layer survives the locked final holdout.
