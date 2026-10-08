"""Running plays between plate appearances other than steals: wild pitches, passed balls, balks, pickoffs, defensive
indifference and the rest, as they happen in MLB.

The engine's steal step covers stolen bases and caught stealing. Everything else that moves runners before a batted
ball was missing: about half an event per team-game, worth about 0.08 runs per team-game (TRANS-02 in LEDGER.md).

doc (brl_live/running_events.json, built by tools/brl_running_events.py from the transitions lane):
  {'cells': {'<bases mask>|<outs>': {'p': chance per plate appearance, 'events': [[kind, pattern, outs, share], ...]}}}
A pattern lists where the runner on first, second and third ends up: '1' '2' '3' a base, 'H' scored, 'X' out, '-' no
runner. Runners never pass each other, so every pattern is legal for its cell.
"""
from __future__ import annotations

import numpy as np

KINDS = {
    "wild_pitch": "wild pitch",
    "passed_ball": "passed ball",
    "balk": "balk",
    "defensive_indifference": "defensive indifference",
    "pickoff": "pickoff",
    "pickoff_error": "pickoff throw error",
    "error": "error",
    "other_advance": "advance on the play",
    "runner_out": "out on the bases",
}
BASE_WORD = {"1": "first", "2": "second", "3": "third"}


class RunningEvents:
    def __init__(self, doc: dict):
        self.name = str(doc.get("name") or "running plays")
        self.cells = {}
        for key, cell in (doc.get("cells") or {}).items():
            mask, outs = (int(x) for x in key.split("|"))
            rows = [r for r in cell.get("events") or [] if float(r[3]) > 0]
            p = float(cell.get("p") or 0.0)
            if not rows or p <= 0.0:
                continue
            share = np.asarray([float(r[3]) for r in rows], float)
            self.cells[(mask, outs)] = (min(0.5, p), [(str(r[0]), str(r[1]), int(r[2])) for r in rows], share / share.sum())

    def rate(self, mask: int, outs: int) -> float:
        cell = self.cells.get((int(mask), int(outs)))
        return 0.0 if cell is None else cell[0]

    def draw(self, bases, outs: int, rng: np.random.Generator):
        """None (nothing happens) or (kind, pattern, outs made) for the runners on `bases` with `outs` out."""
        mask = sum(1 << i for i, r in enumerate(bases) if r is not None)
        cell = self.cells.get((mask, int(outs)))
        if cell is None:
            return None
        p, rows, share = cell
        if rng.random() >= p:
            return None
        return rows[int(rng.choice(len(rows), p=share))]

    @staticmethod
    def apply(bases, pattern: str):
        """New bases, the runners who scored (lead runner first) and the runners put out."""
        new = [None, None, None]
        scored, out = [], []
        for slot in (2, 1, 0):
            runner, d = bases[slot], pattern[slot]
            if runner is None or d == "-":
                continue
            if d == "H":
                scored.append(runner)
            elif d == "X":
                out.append(runner)
            else:
                new[int(d) - 1] = runner
        return new, scored, out

    @staticmethod
    def describe(kind: str, bases, pattern: str) -> tuple[str, object]:
        """Box-score words for the play and the runner it is about (the lead runner who moved or was put out)."""
        moves, lead = [], None
        for slot in (2, 1, 0):
            runner, d = bases[slot], pattern[slot]
            if runner is None or d == "-" or d == str(slot + 1):
                continue
            lead = lead or runner
            if d == "H":
                moves.append(f"{runner.name} scores")
            elif d == "X":
                moves.append(f"{runner.name} out")
            else:
                moves.append(f"{runner.name} to {BASE_WORD[d]}")
        lead = lead or next((r for r in (bases[2], bases[1], bases[0]) if r is not None), None)
        text = "; ".join(moves)
        if kind == "pickoff" and lead is not None:
            base = next((BASE_WORD[str(s + 1)] for s in (2, 1, 0) if bases[s] is lead), "base")
            return f"{lead.name} picked off {base}.", lead
        if kind == "defensive_indifference" and lead is not None and moves:
            return f"{moves[0].replace(' to ', ' takes ')} on defensive indifference.", lead
        if kind == "runner_out" and lead is not None:
            return f"{lead.name} out on the bases.", lead
        label = KINDS.get(kind, kind.replace("_", " "))
        return (label[0].upper() + label[1:] + (f": {text}." if text else ".")), lead
