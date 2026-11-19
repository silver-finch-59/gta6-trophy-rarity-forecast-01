[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/silver-finch-59/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/silver-finch-59/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# Trophy Rarity Forecast

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who
will have earned it a week after launch — as five quantiles per trophy
(0.05 / 0.25 / 0.5 / 0.75 / 0.95). Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Model

`main.py` (Python 3.13, standard library only, deterministic, no network)
reads one case file per run and writes one forecast file. A trophy's earn
rate is a ratio — earners over players tracked — so the model extrapolates
the two counts separately instead of trending the rate itself:

- Both counts are modelled as power laws in (1 + hours since release), i.e.
  linear in log space: players tracked is fit by least squares over the
  case's distinct reads, shrunk toward a prior exponent of 1.0; each
  trophy's earners are fit the same way, shrunk toward a category prior
  (story chapters earn fast, grinds and the platinum earn slow).
  Predictions are anchored at the last read, and the median rate is capped
  per category.
- When the history is flat (every read identical, no refresh seen yet — the
  practice cases), the trend fit carries no information and a refresh prior
  takes over: Exophase refreshes a game's counts every few hours, and a page
  that has sat unchanged for hours is due one. The prior puts a probability
  on a refresh landing before the target and applies one refresh's jump —
  players up ~6x, earners up much less — calibrated on the house's own
  recording of the first refresh (638 → 3,868 players tracked, every rate
  but the 100% one lower).
- Quantiles are symmetric in logit space around the median, with a spread
  combining the house baseline's random-walk term, the per-trophy fit
  residual scaled by extrapolation distance, and a prior-uncertainty term
  that is larger when the history is flat.

## Local check

```sh
KIT=<kit dir>
MAKEFUN_INPUT_DIR=$KIT/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes `out/p00.out` and `out/p01.out`: 55 rows each, quantiles
non-decreasing in 0–100, deterministic across runs.

## License

MIT — see [LICENSE](LICENSE).
