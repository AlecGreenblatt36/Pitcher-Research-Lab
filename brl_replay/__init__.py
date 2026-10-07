"""Frozen season replay of the simulator on the sealed plate-appearance history (public code).

Every input for a game dated D uses only plate appearances dated before D: player history and
counters, pitch physics, bullpen candidates, handedness, starter tendencies. Lineups and
starters are the ones that actually played (the standard replay convention). Runs inside
Actions (tools/brl_replay.py) where the sealed data and the key live; only per-game model
outputs and scores leave the runner.
"""
