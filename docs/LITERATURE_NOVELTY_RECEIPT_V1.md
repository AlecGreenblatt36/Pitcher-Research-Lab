# Literature and Novelty Receipt V1

**Purpose:** prevent the Baseball Research Lab from presenting established mathematics or prior baseball applications as original work.  
**Status:** initial targeted review; not an exhaustive systematic review.

## Conditional pitch bridge

- **Established method:** Doob `h`-transform / Markov-chain conditioning.
- **Credible BRL claim:** applying the method to generate pitch-count paths conditional on the locked seven-outcome PA result, then validating its downstream effect on workload and manager decisions.
- **Do not claim:** invention of the transform.

## Constrained PA residual tilt

- **Established method:** KL-regularized exponential tilting / offset multinomial residual modeling.
- **Credible BRL claim:** using a locked baseball PA forecast as the prior and restricting each contextual module to baseball-causal outcome families.
- **Do not claim:** invention of exponential tilting.

## Joint score-distribution correction

- **Established method:** minimum-KL `I`-projection / maximum-entropy calibration.
- **Credible BRL claim:** applying a chronologically cross-fitted joint-score calibration to preserve the simulator while correcting mean, variance, covariance, and home-win calibration.
- **Do not claim:** invention of maximum-entropy calibration.

## Runner send and execution model

- **Prior work found:** Kato and Yanai (2025), “When should a runner on third base advance?”, Journal of Sports Analytics, DOI `10.1177/22150218251313931`. It modeled stay/tag-out/sacrifice-fly outcomes using batted-ball kinematics, Sprint Speed, fielder Arm Strength, and ballpark.
- **Novelty implication:** a Sprint-Speed × Arm-Strength runner decision model is not first-ever. A broader all-base, all-PA, send-versus-execution decomposition integrated into a full pregame simulator may still be a meaningful extension.

## Reliever fatigue and bullpen capacity

- **Prior work found:** Burris and Coleman (2018), “Out of gas: quantifying fatigue in MLB relievers,” Journal of Quantitative Analysis in Sports, DOI `10.1515/jqas-2018-0007`. It used a Bayesian hierarchical dose-response/recovery framework for recent pitch workload and future velocity.
- **Novelty implication:** workload decay and reliever fatigue are established. A Bullpen Capacity Index should be presented as a fitted operational summary, not an entirely new fatigue theory.

## Hook–bullpen coupling

- **Targeted-search result:** prior work exists separately on reliever fatigue and pitcher substitution/third-time-through decisions. This initial review did not identify a public paper fitting starter-removal hazard directly as a function of the quality and availability of that night’s remaining bullpen.
- **Current label:** potentially novel integrated application; novelty unproven.
- **Required before claim:** systematic search across sports analytics journals, conference proceedings, public team/R&D talks, Baseball Prospectus, FanGraphs, SABR, and dissertations.

## Travel load

- **Prior work found:** Song, Severini, and Allada (2017), “How jet lag impairs Major League Baseball performance,” PNAS, DOI `10.1073/pnas.1608847114`.
- **Novelty implication:** jet-lag effects in MLB are established. BRL’s contribution would be a pregame-valid fitted travel-load decay, interaction with day/night scheduling, and incremental proper-score validation.

## Shadow index

- **Related work found:** Gray and Wilkins (2015), “Lost in the Lights: The Effects of Glare on Catching Performance,” Journal of Vision, DOI `10.1167/15.12.597`; non-peer-reviewed baseball analyses of sun/shade pitch conditions also exist.
- **Targeted-search result:** no strong peer-reviewed MLB pitch-level solar-geometry model was identified in this initial search.
- **Current label:** plausible research hypothesis; novelty unproven.
- **Required before claim:** broader literature search covering vision science, stadium architecture, broadcast-tracked shadows, dissertations, and team/public analytics presentations.

## Pitch tunnels

- **Prior public baseball work exists:** Baseball Prospectus popularized and quantified pitch-tunnel concepts before this project.
- **Credible BRL claim:** a reproducible, forecast-valid implementation that adds held-out residual signal, not invention of tunneling.

## Naming rule

Every report must separate:

1. established mathematical method;
2. prior baseball application;
3. BRL implementation detail;
4. held-out evidence of added value;
5. any remaining novelty uncertainty.
