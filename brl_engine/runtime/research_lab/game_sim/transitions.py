from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import BaseRunner, PlayerProfile


@dataclass
class TransitionResult:
    bases: list[BaseRunner | None]
    outs_added: int
    scored_runners: list[BaseRunner]
    retired_runner_ids: list[str]
    description: str

    @property
    def runs_scored(self) -> int:
        return len(self.scored_runners)


def _bounded_probability(value: float) -> float:
    return float(min(0.995, max(0.005, value)))


def _batter_runner(
    batter: PlayerProfile,
    responsible_pitcher_id: str | None,
) -> BaseRunner:
    return BaseRunner.from_batter(batter, responsible_pitcher_id)


def apply_outcome(
    outcome: str,
    bases: list[BaseRunner | None],
    batter: PlayerProfile,
    responsible_pitcher_id: str | None,
    outs_before: int,
    batting_team_baserunning: float,
    fielding_team_defense: float,
    rng: np.random.Generator,
) -> TransitionResult:
    """Resolve one seven-class PA outcome into a legal base/out transition.

    The transition kernel is deliberately isolated from PA probabilities. It can
    be replaced by an empirical kernel trained on historical play-by-play while
    preserving the game engine and evaluation interface.
    """

    if len(bases) != 3:
        raise ValueError("bases must contain first, second, and third")
    first, second, third = list(bases)
    scored: list[BaseRunner] = []
    retired: list[str] = []
    batter_runner = _batter_runner(batter, responsible_pitcher_id)
    baserun = float(min(2.5, max(-2.5, batting_team_baserunning)))
    defense = float(min(2.5, max(-2.5, fielding_team_defense)))

    if outcome == "strikeout":
        return TransitionResult(
            [first, second, third],
            1,
            scored,
            retired,
            f"{batter.name} struck out.",
        )

    if outcome == "bb_hbp":
        new_first, new_second, new_third = first, second, third
        if first is not None:
            if second is not None:
                if third is not None:
                    scored.append(third)
                new_third = second
            new_second = first
        new_first = batter_runner
        return TransitionResult(
            [new_first, new_second, new_third],
            0,
            scored,
            retired,
            f"{batter.name} reached on a walk or hit by pitch.",
        )

    if outcome == "home_run":
        scored.extend(runner for runner in (third, second, first) if runner is not None)
        scored.append(batter_runner)
        return TransitionResult(
            [None, None, None],
            0,
            scored,
            retired,
            f"{batter.name} homered.",
        )

    if outcome == "single":
        new_bases: list[BaseRunner | None] = [None, None, None]
        if third is not None:
            score_probability = _bounded_probability(
                0.965 + 0.025 * third.speed + 0.010 * (outs_before == 2)
            )
            if rng.random() < score_probability:
                scored.append(third)
            else:
                new_bases[2] = third

        if second is not None:
            score_probability = _bounded_probability(
                0.54
                + 0.28 * second.speed
                + 0.10 * (outs_before == 2)
                + 0.035 * baserun
                - 0.035 * defense
            )
            if rng.random() < score_probability:
                scored.append(second)
            elif new_bases[2] is None:
                new_bases[2] = second
            else:
                new_bases[1] = second

        if first is not None:
            take_third_probability = _bounded_probability(
                0.20
                + 0.26 * first.speed
                + 0.09 * (outs_before == 2)
                + 0.030 * baserun
                - 0.035 * defense
            )
            if new_bases[2] is None and rng.random() < take_third_probability:
                new_bases[2] = first
            elif new_bases[1] is None:
                new_bases[1] = first
            else:
                scored.append(new_bases[2])
                new_bases[2] = new_bases[1]
                new_bases[1] = first

        new_bases[0] = batter_runner
        return TransitionResult(
            new_bases,
            0,
            scored,
            retired,
            f"{batter.name} singled.",
        )

    if outcome == "double_triple":
        triple_probability = _bounded_probability(
            0.035 + 0.085 * batter.speed + 0.010 * baserun - 0.010 * defense
        )
        if rng.random() < triple_probability:
            scored.extend(runner for runner in (third, second, first) if runner is not None)
            return TransitionResult(
                [None, None, batter_runner],
                0,
                scored,
                retired,
                f"{batter.name} tripled.",
            )

        if third is not None:
            scored.append(third)
        if second is not None:
            scored.append(second)
        new_third: BaseRunner | None = None
        if first is not None:
            score_probability = _bounded_probability(
                0.42
                + 0.31 * first.speed
                + 0.08 * (outs_before == 2)
                + 0.035 * baserun
                - 0.040 * defense
            )
            if rng.random() < score_probability:
                scored.append(first)
            else:
                new_third = first
        return TransitionResult(
            [None, batter_runner, new_third],
            0,
            scored,
            retired,
            f"{batter.name} doubled.",
        )

    if outcome == "other_reach":
        fielder_choice_probability = 0.23 if first is not None and outs_before < 2 else 0.0
        if rng.random() < fielder_choice_probability:
            assert first is not None
            retired.append(first.player_id)
            return TransitionResult(
                [batter_runner, second, third],
                1,
                scored,
                retired,
                f"{batter.name} reached on a fielder's choice.",
            )
        if third is not None:
            scored.append(third)
        return TransitionResult(
            [batter_runner, first, second],
            0,
            scored,
            retired,
            f"{batter.name} reached on an error or other play.",
        )

    if outcome == "bip_out":
        double_play_probability = 0.0
        if first is not None and outs_before < 2:
            double_play_probability = _bounded_probability(
                0.145
                + 0.045 * defense
                - 0.055 * first.speed
                - 0.035 * batter.speed
            )
        if double_play_probability and rng.random() < double_play_probability:
            retired.extend([first.player_id, batter.player_id])
            return TransitionResult(
                [None, second, third],
                2,
                scored,
                retired,
                f"{batter.name} grounded into a double play.",
            )

        new_bases = [first, second, third]
        if third is not None and outs_before < 2:
            sacrifice_probability = _bounded_probability(
                0.135 + 0.070 * batter.power - 0.025 * defense
            )
            if rng.random() < sacrifice_probability:
                scored.append(third)
                new_bases[2] = None
                return TransitionResult(
                    new_bases,
                    1,
                    scored,
                    retired,
                    f"{batter.name} drove in a run on a sacrifice fly.",
                )

        if second is not None and new_bases[2] is None:
            advance_probability = _bounded_probability(
                0.13 + 0.12 * second.speed + 0.020 * baserun - 0.025 * defense
            )
            if rng.random() < advance_probability:
                new_bases[2] = second
                new_bases[1] = None
        if first is not None and new_bases[1] is None:
            advance_probability = _bounded_probability(
                0.05 + 0.07 * first.speed + 0.015 * baserun - 0.020 * defense
            )
            if rng.random() < advance_probability:
                new_bases[1] = first
                new_bases[0] = None

        return TransitionResult(
            new_bases,
            1,
            scored,
            retired,
            f"{batter.name} put the ball in play for an out.",
        )

    raise ValueError(f"unsupported outcome: {outcome}")


# ----- empirical kernel ----------------------------------------------------------------------------------------------
# Destinations in a pattern: '-' no runner on that base, '1' '2' '3' a base, 'H' scored, 'X' out. A pattern lists the
# runner on first, second and third and then the batter.
_SLOT_DECISIONS = {
    # outcome: {slot: (decision, letters that count as the aggressive result)}; slot 0-2 the runner on that base, 3 the batter
    "single": {1: ("single_from_2nd", "H"), 0: ("single_from_1st", "3H")},
    "double_triple": {0: ("double_from_1st", "H"), 3: ("triple", "3H")},
    "bip_out": {2: ("out_from_3rd", "H"), 1: ("out_from_2nd", "3H")},
}


class EmpiricalKernel:
    """Base running after each outcome as it happens in MLB: for each outcome, bases and outs at contact, the observed
    joint destinations of every runner and the batter (from the official play-by-play, tools/brl_transitions.py),
    tilted by the runners' speeds on the main advance decisions and on the double play.

    doc: {'cells': {'<sim outcome>|<bases mask>|<outs>': [[pattern, probability], ...]},
          'tilts': {decision: {'beta': log-odds per unit of speed, 'center': mean speed in that decision}}}.
    Each tilt multiplies the patterns where that runner takes the aggressive result (or, for the double play, where two
    or more are out) by exp(beta * (speed - center)). Cells missing from the doc (and home runs) use the hand-set kernel."""

    def __init__(self, doc: dict):
        self.name = str(doc.get("name") or "empirical base running")
        words = doc.get("descriptions") or {}
        # How the box score words a play the kernel cannot tell apart: a run from third on an out (sacrifice fly or a
        # ground ball) and a batter reaching without a hit (fielder's choice or error), by the real shares.
        self.sac_fly_share = float(words.get("sac_fly_share", 0.59))
        self.fielders_choice_share = {int(k): float(v) for k, v in (words.get("fielders_choice_share") or {}).items()}
        self.tilts = {k: (float(v["beta"]), float(v.get("center", 0.5))) for k, v in (doc.get("tilts") or {}).items()}
        self.cells = {}
        for key, rows in (doc.get("cells") or {}).items():
            outcome, mask, outs = key.split("|")
            mask, outs = int(mask), int(outs)
            pats = [str(r[0]) for r in rows]
            p = np.asarray([float(r[1]) for r in rows], float)
            if p.sum() <= 0:
                continue
            cols, beta, center, source = [], [], [], []
            for slot, (decision, letters) in _SLOT_DECISIONS.get(outcome, {}).items():
                if decision in self.tilts and (slot == 3 or (mask >> slot) & 1):
                    cols.append([1.0 if pat[slot] in letters else 0.0 for pat in pats])
                    beta.append(self.tilts[decision][0]); center.append(self.tilts[decision][1]); source.append(slot)
            if outcome == "bip_out" and mask & 1 and outs < 2:
                dp = [1.0 if pat.count("X") >= 2 else 0.0 for pat in pats]
                for slot, decision in ((0, "double_play_runner"), (3, "double_play_batter")):
                    if decision in self.tilts:
                        cols.append(dp); beta.append(self.tilts[decision][0]); center.append(self.tilts[decision][1]); source.append(slot)
            ind = np.asarray(cols, float).T if cols else np.zeros((len(pats), 0))
            self.cells[(outcome, mask, outs)] = (pats, p / p.sum(), ind, np.asarray(beta, float), np.asarray(center, float), tuple(source))

    def apply(self, outcome, bases, batter, responsible_pitcher_id, outs_before, batting_team_baserunning, fielding_team_defense, rng):
        first, second, third = list(bases)
        mask = int(first is not None) | (int(second is not None) << 1) | (int(third is not None) << 2)
        cell = self.cells.get((outcome, mask, int(outs_before)))
        if cell is None or outcome == "home_run":
            return apply_outcome(outcome, bases, batter, responsible_pitcher_id, outs_before, batting_team_baserunning, fielding_team_defense, rng)
        pats, p, ind, beta, center, source = cell
        if len(beta):
            who = (first, second, third, batter)
            speeds = np.asarray([float(getattr(who[s], "speed", 0.5)) for s in source], float)
            w = p * np.exp(ind @ (beta * (speeds - center)))
            w = w / w.sum()
        else:
            w = p
        pat = pats[int(rng.choice(len(pats), p=w))]
        batter_runner = _batter_runner(batter, responsible_pitcher_id)
        runners = [first, second, third, batter_runner]
        new_bases: list[BaseRunner | None] = [None, None, None]
        scored: list[BaseRunner] = []
        retired: list[str] = []
        outs = 0
        for slot in (2, 1, 0, 3):            # scoring order: third, second, first, batter
            runner, d = runners[slot], pat[slot]
            if runner is None or d == "-":
                continue
            if d == "H":
                scored.append(runner)
            elif d == "X":
                retired.append(runner.player_id); outs += 1
            else:
                new_bases[int(d) - 1] = runner
        if outcome == "bip_out":
            if outs >= 2:
                verb = "grounded into a double play" if outs == 2 else "hit into a triple play"
            elif pat[2] == "H" and pat[3] == "X" and int(outs_before) < 2:
                verb = "drove in a run on a sacrifice fly" if rng.random() < self.sac_fly_share else "grounded out, and a run scored"
            else:
                verb = "made an out"
        elif outcome == "other_reach":
            verb = "reached on a fielder's choice" if rng.random() < self.fielders_choice_share.get(mask, 0.0) else "reached on an error"
        elif outcome == "double_triple":
            verb = "tripled" if pat[3] == "3" else "doubled"
        else:
            verb = {"strikeout": "struck out", "bb_hbp": "reached on a walk or hit by pitch", "single": "singled"}.get(outcome, outcome)
        return TransitionResult(new_bases, outs, scored, retired, f"{batter.name} {verb}.")
