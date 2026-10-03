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

The model uses only information that can exist before the target PA. Same-PA velocity, location, launch speed, launch angle, xwOBA, wOBA value, run value and terminal result-derived fields are not predictors. All player, platoon, park and recent-form counters are **date blocked**: every PA on date D is scored before outcomes from D are revealed.

## Evaluation design

The default freeze is:

- train: 2023
- tune/calibrate: 2024
- untouched test: 2025

The benchmark compares:

1. prior-date league outcome rates
2. a shrunk empirical-Bayes batter/pitcher/platoon/park/recent baseline
3. a regularized multinomial model using talent features
4. the same model plus pre-PA context

Primary metrics are multiclass log loss, multiclass Brier score, calibration and game-clustered uncertainty. There is no arbitrary two-percent promotion gate. The candidate must beat the strong empirical-Bayes baseline with a clustered confidence interval that excludes zero, add context signal beyond talent alone, and remain acceptably calibrated.

## Reproducible run

Install the project requirements plus `requirements-model.txt`, then run:

```text
python -m research_lab.pa_model.cli --verbose run-all --work-dir model_runs/pa_v1
```

Raw public Statcast chunks are cached outside Git. The run produces:

- `plate_appearances.csv.gz` and a provenance receipt
- `PA_BENCHMARK_RESULT.json`
- `TUNING_AUDIT.json`
- `test_predictions.csv.gz`
- `pa_model.joblib`

## What this does not yet prove

A passing PA model does not validate the existing outcome-conditioned runner kernel, between-pitch events, batting-order logic, bullpen selection, team-run distributions, winner probabilities or exact scores. Those layers are promoted only after a frozen PA model survives the untouched test.
