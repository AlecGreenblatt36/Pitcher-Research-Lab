from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_lab.game_sim.ablation_v2 import (
    build_nb_team_baseline_map,
    denoised_calibration,
    finite_path_log_loss_correction,
    fit_nb2_dispersion,
)
from research_lab.game_sim.replay import (
    bootstrap_metric_difference,
    calibration_parameters,
    extract_historical_games,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def prefix_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    shared = [
        'game_pk','game_date','away_team','home_team','actual_away_runs',
        'actual_home_runs','actual_home_win'
    ]
    ren = {c:f'{prefix}{c}' for c in frame.columns if c not in shared}
    return frame.rename(columns=ren)


def score_variant(frame: pd.DataFrame, prefix: str, simulations: int | None) -> tuple[dict,dict]:
    y = frame.actual_home_win.to_numpy(float)
    p = np.clip(frame[f'{prefix}home_win_probability'].to_numpy(float), 1e-8, 1-1e-8)
    brier_raw=(p-y)**2
    if simulations:
        brier_corr=p*(1-p)/(simulations-1)
        ll_raw,ll_bias,ll_corr=finite_path_log_loss_correction(y,p,simulations)
        calibration=denoised_calibration(y,p,simulations)
    else:
        brier_corr=np.zeros_like(p)
        ll_raw=-(y*np.log(p)+(1-y)*np.log(1-p));ll_bias=np.zeros_like(p);ll_corr=ll_raw.copy()
        calibration={'raw':calibration_parameters(y,p),'de_noised':None}
    ae=frame[f'{prefix}away_mean_runs'].to_numpy(float)-frame.actual_away_runs.to_numpy(float)
    he=frame[f'{prefix}home_mean_runs'].to_numpy(float)-frame.actual_home_runs.to_numpy(float)
    game_crps=(frame[f'{prefix}away_crps'].to_numpy(float)+frame[f'{prefix}home_crps'].to_numpy(float))/2
    metrics={
        'winner_brier_raw':float(brier_raw.mean()),
        'winner_brier_mc_correction':float(brier_corr.mean()),
        'winner_brier_unbiased':float((brier_raw-brier_corr).mean()),
        'winner_log_loss_raw':float(ll_raw.mean()),
        'winner_log_loss_mc_bias_estimate':float(ll_bias.mean()),
        'winner_log_loss_bias_corrected':float(ll_corr.mean()),
        'winner_accuracy':float(np.mean((p>=.5)==(y>=.5))),
        'win_probability_mean':float(p.mean()),
        'win_probability_sd':float(p.std(ddof=1)),
        'calibration':calibration,
        'team_run_mae':float(np.mean(np.abs(np.r_[ae,he]))),
        'team_run_rmse':float(np.sqrt(np.mean(np.r_[ae,he]**2))),
        'team_run_crps':float(game_crps.mean()),
        'away_run_bias':float(ae.mean()),
        'home_run_bias':float(he.mean()),
    }
    per_game={'brier':brier_raw-brier_corr,'log_loss':ll_corr,'crps':game_crps}
    return metrics,per_game


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',required=True)
    ap.add_argument('--flat',required=True)
    ap.add_argument('--oracle')
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--simulations',type=int,default=1000)
    ap.add_argument('--bootstrap-replicates',type=int,default=5000)
    args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)

    candidate=prefix_frame(pd.read_csv(args.candidate,low_memory=False),'candidate_')
    flat=prefix_frame(pd.read_csv(args.flat,low_memory=False),'flat_')
    frame=candidate.merge(flat,on=['game_pk','game_date','away_team','home_team','actual_away_runs','actual_home_runs','actual_home_win'],validate='one_to_one')
    if args.oracle:
        oracle=prefix_frame(pd.read_csv(args.oracle,low_memory=False),'oracle_')
        frame=frame.merge(oracle,on=['game_pk','game_date','away_team','home_team','actual_away_runs','actual_home_runs','actual_home_win'],validate='one_to_one')
    if frame.game_pk.duplicated().any():raise RuntimeError('duplicate games')
    if not np.array_equal(frame.game_pk.to_numpy(int),candidate.game_pk.to_numpy(int)):raise RuntimeError('merge order changed')

    history_path=ROOT/'pa_model_reference/model_runs/pa_locked_2026/plate_appearances.csv.gz'
    cols=['game_pk','season','game_type','date_key','away_team','home_team','home_score','away_score','terminal_event','inning_topbot','runner_1b','runner_2b','runner_3b','at_bat_number','batter','pitcher','stand','p_throws','park']
    history=pd.read_csv(history_path,usecols=cols,low_memory=False)
    overrides={int(k):v for k,v in json.loads((ROOT/'research_lab/game_sim/official_score_overrides_2025.json').read_text()).items()}
    g23,_=extract_historical_games(history,2023);g24,_=extract_historical_games(history,2024);g25,_=extract_historical_games(history,2025,score_overrides=overrides)
    nbfit=fit_nb2_dispersion([*g23,*g24]);base=build_nb_team_baseline_map(g25,[*g23,*g24],nbfit)
    for col in ('home_win_probability','away_mean_runs','home_mean_runs','away_crps','home_crps','prior_home_win_rate','home_logit_offset','neutral_home_win_probability'):
        frame[f'team_nb_hfa_{col}']=frame.game_pk.map(lambda x:base[int(x)][col])

    variants={};per={}
    for name,prefix,sims in [('candidate_locked_pa','candidate_',args.simulations),('ablated_flat_league_pa','flat_',args.simulations),('team_nb_hfa_baseline','team_nb_hfa_',None)]:
        variants[name],per[name]=score_variant(frame,prefix,sims)
    if args.oracle:
        variants['oracle_actual_reliever_order'],per['oracle_actual_reliever_order']=score_variant(frame,'oracle_',args.simulations)

    comparisons={}
    for i,other in enumerate([k for k in variants if k!='candidate_locked_pa']):
        comparisons[f'candidate_minus_{other}']={
            'winner_brier_unbiased':bootstrap_metric_difference(per['candidate_locked_pa']['brier'],per[other]['brier'],reps=args.bootstrap_replicates,seed=700+i*10),
            'winner_log_loss_bias_corrected':bootstrap_metric_difference(per['candidate_locked_pa']['log_loss'],per[other]['log_loss'],reps=args.bootstrap_replicates,seed=701+i*10),
            'team_run_crps':bootstrap_metric_difference(per['candidate_locked_pa']['crps'],per[other]['crps'],reps=args.bootstrap_replicates,seed=702+i*10),
        }
    if args.oracle:
        comparisons['oracle_minus_team_nb_hfa_baseline']={
            'winner_brier_unbiased':bootstrap_metric_difference(per['oracle_actual_reliever_order']['brier'],per['team_nb_hfa_baseline']['brier'],reps=args.bootstrap_replicates,seed=900),
            'winner_log_loss_bias_corrected':bootstrap_metric_difference(per['oracle_actual_reliever_order']['log_loss'],per['team_nb_hfa_baseline']['log_loss'],reps=args.bootstrap_replicates,seed=901),
            'team_run_crps':bootstrap_metric_difference(per['oracle_actual_reliever_order']['crps'],per['team_nb_hfa_baseline']['crps'],reps=args.bootstrap_replicates,seed=902),
        }
        comparisons['oracle_minus_candidate']={
            'winner_brier_unbiased':bootstrap_metric_difference(per['oracle_actual_reliever_order']['brier'],per['candidate_locked_pa']['brier'],reps=args.bootstrap_replicates,seed=910),
            'winner_log_loss_bias_corrected':bootstrap_metric_difference(per['oracle_actual_reliever_order']['log_loss'],per['candidate_locked_pa']['log_loss'],reps=args.bootstrap_replicates,seed=911),
            'team_run_crps':bootstrap_metric_difference(per['oracle_actual_reliever_order']['crps'],per['candidate_locked_pa']['crps'],reps=args.bootstrap_replicates,seed=912),
        }

    hazard=json.loads((ROOT/'model_runs/starter_hazard_2025_dev/STARTER_HAZARD_2025_DEV_RESULT.json').read_text())
    result={
        'schema':'baseball_research_lab.2025_full_season_1000_path_benchmark.v2',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'claim_status':'2025 development benchmark; post-hoc protocol upgrade after 100-path review',
        'protocol':{
            'season':2025,'games':int(len(frame)),'simulations_per_game_per_variant':int(args.simulations),
            'same_per_game_seed':True,'seed':'game_pk','2026_rerun':False,
            'brier_correction':'p_hat*(1-p_hat)/(N-1), per game',
            'log_loss_correction':'second-order delta-method finite-path bias, per game',
            'crps':'unbiased empirical U-statistic for simulated variants; exact NB CDF for team baseline',
            'calibration':'raw logistic slope/intercept plus first-order errors-in-variables de-noised estimate',
            'protocol_change_disclosure':'1000 paths and baseline/log-loss/calibration corrections were specified after reviewing the 100-path result',
        },
        'negative_binomial_fit_2023_2024':nbfit.to_dict(),
        'starter_hazard':{
            'artifact':'starter_hazard_2025_dev.joblib','train_seasons':hazard['protocol']['refit_seasons'],
            'regularization_selection_season':hazard['protocol']['regularization_selection_season'],
            'development_score_season':hazard['protocol']['development_score_season'],
            '2026_accessed':hazard['protocol']['2026_accessed'],
        },
        'variants':variants,'paired_game_bootstrap':comparisons,
        'oracle_boundary':'postgame actual reliever identities/order; never a forecast' if args.oracle else None,
        'decision':{
            'reliever_model_worthwhile_only_if_oracle_clear_gain':None if not args.oracle else bool(
                comparisons['oracle_minus_team_nb_hfa_baseline']['winner_brier_unbiased']['ci_95_high']<0 or
                comparisons['oracle_minus_team_nb_hfa_baseline']['team_run_crps']['ci_95_high']<0
            )
        },
        'artifacts':{
            'candidate':{'path':str(Path(args.candidate)),'sha256':sha256(Path(args.candidate))},
            'flat':{'path':str(Path(args.flat)),'sha256':sha256(Path(args.flat))},
            'oracle':({'path':str(Path(args.oracle)),'sha256':sha256(Path(args.oracle))} if args.oracle else None),
        }
    }
    result_path=out/'2025_PA_ABLATION_1000_RESULT.json';pred_path=out/'2025_PA_ABLATION_1000_PREDICTIONS.csv.gz'
    result_path.write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8');frame.to_csv(pred_path,index=False,compression='gzip')
    print(json.dumps({'result':str(result_path),'predictions':str(pred_path),'games':len(frame),'decision':result['decision'],'metrics':variants},indent=2)[:20000])

if __name__=='__main__':main()
