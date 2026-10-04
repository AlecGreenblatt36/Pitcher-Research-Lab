from __future__ import annotations
import argparse,json
from pathlib import Path


def f(value, digits=6):
    return f"{value:.{digits}f}"


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=json.loads(Path(a.result).read_text())
    v=r['variants']; c=v['candidate_locked_pa']; flat=v['ablated_flat_league_pa']; team=v['team_nb_hfa_baseline']; oracle=v.get('oracle_actual_reliever_order')
    lines=[
        '# 2025 Full-Season 1,000-Path PA Ablation and Bullpen Oracle', '',
        f"Generated: `{r['generated_at_utc']}`",'',
        '## Protocol','',
        f"- {r['protocol']['games']:,} 2025 regular-season games",
        f"- {r['protocol']['simulations_per_game_per_variant']:,} paths per game and variant",
        '- identical per-game seeds across simulator variants',
        '- candidate: locked seven-outcome PA model',
        '- ablation: prior-date flat league PA probabilities',
        '- oracle: actual postgame reliever identities/order; diagnostic only, never a forecast',
        '- 2026 was not rerun',
        '- this 1,000-path protocol was specified after reviewing the 100-path result', '',
        '## Metrics','',
        '| Variant | Brier (unbiased) | Log loss (bias-corrected) | Win-p SD | Raw slope | De-noised slope | Run CRPS | Run MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name,label in [('candidate_locked_pa','Locked PA'),('ablated_flat_league_pa','Flat PA'),('team_nb_hfa_baseline','Team NB + HFA'),('oracle_actual_reliever_order','Bullpen oracle')]:
        if name not in v: continue
        x=v[name]; raw=x['calibration']['raw']; de=x['calibration'].get('de_noised') or raw
        lines.append(f"| {label} | {f(x['winner_brier_unbiased'])} | {f(x['winner_log_loss_bias_corrected'])} | {f(x['win_probability_sd'])} | {f(raw['slope'],3)} | {f(de['slope'],3)} | {f(x['team_run_crps'])} | {f(x['team_run_mae'])} |")
    lines += ['', '## Decision', '', f"Reliever modeling worthwhile under the frozen oracle gate: **{r['decision']['reliever_model_worthwhile_only_if_oracle_clear_gain']}**",'', 'This result diagnoses the maximum value of perfect reliever identity/order under the current downstream engine. It does not make the oracle deployable and does not validate the full game model prospectively.']
    Path(a.output).write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
