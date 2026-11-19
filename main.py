"""GTA VI trophy rarity forecast: earners-over-players power-law model.

Forecasts, for each of Grand Theft Auto VI's PS5 trophies, five quantiles of
its Exophase earn rate at the case's target, as percentages.

Model
-----
A trophy's earn rate is a ratio: earners / players_tracked.  Both sides grow
after launch, but at different speeds, so the rate is modelled as the ratio
of two separately extrapolated counts, not as a trend in the rate itself:

* x(t) = ln(1 + hours since the game's release).  Counts are assumed to grow
  as a power law in (1 + hours), i.e. linear in x, which makes early growth
  fast and later growth slow -- the shape launch adoption actually takes.
* players_tracked: ordinary least squares of ln(N) on x over the case's
  distinct reads, shrunk toward a prior exponent of 1.0.
* earners of trophy i: E = rate * N / 100 at each read; OLS of ln(E) on x,
  shrunk toward a category prior exponent (story chapters earn fast, grinds
  and the platinum earn slow).  Predictions are anchored at the last read,
  so a level error in the fit cannot shift the forecast, only its slope can.
* rate = 100 * E_pred / N_pred, capped per category (a chapter trophy cannot
  plausibly pass 45% of tracked players in week one; the platinum cannot
  plausibly pass 2%) and with the total log move from the last rate limited
  to 3 logit units either way.

Flat history (the practice cases: every read identical, no refresh seen yet)
carries no trend information, so the trend fit is blended out and a refresh
prior takes over: Exophase refreshes a game's counts every few hours, and a
page that has sat unchanged for hours is due one, so the prior puts a
probability on a refresh landing before the target (higher the staler the
page).  One refresh multiplies players by a jump factor that decays with
time since release (the launch wave is biggest at the start) and earners by
a smaller factor, so the prior expects rates to fall at a refresh -- as the
house's own recording shows: the first refresh took players tracked from
638 to 3,868 while the trophies earned by tracked players went from 943 to
4,276, i.e. non-100% earners grew only ~1.3x against a ~6x player jump, and
every rate but the 100% one fell.  The jump factors are calibrated on those
magnitudes.  The blend weight on the data fit grows with the number of
distinct reads, so the final case (two days of reads, many refreshes) is
driven by its fitted trends, and by then the jumps have decayed to ~1x
anyway.

Spread
------
Quantiles are symmetric in logit space around the median: q = inv_logit(
logit(median) + z_tau * sigma).  sigma combines the house baseline's
random-walk term (0.5 * sqrt(days)), the per-trophy fit residual scaled by
how far the target extrapolates beyond the history, and a prior-uncertainty
term that is larger when the history is flat.  The near-saturated trophy
(the one at 100% in the last read) is handled separately: its median stays
just under 100% with a narrow spread.

Standard library only, deterministic: no randomness, no network, no time
dependence beyond the case file itself.  Reads every <id>.in in the input
directory and writes <id>.out for each.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Prior exponent for players_tracked growth in ln(1 + hours since release).
PRIOR_N_EXP = 1.0
# Assumed hours between Exophase refreshes of a game's counts.
REFRESH_HOURS = 4.0
# Largest log move allowed from a trophy's last rate, in either direction.
MAX_LOG_MOVE = 3.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols_slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of ys on xs, or None when xs do not vary."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def ols_resid_std(xs: list[float], ys: list[float], slope: float) -> float:
    """Residual standard deviation of the OLS line through (xs, ys)."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    intercept = my - slope * mx
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(sse / (n - 2))


# ---------------------------------------------------------------------------
# Trophy categories, from name + description.  Each category carries the
# prior exponent for its earners' growth in ln(1 + hours), the largest
# median rate it can plausibly reach by the target, and the earners' jump
# factor at a refresh (how much of a refresh's new players already hold it).
# ---------------------------------------------------------------------------
# (prior exponent, median cap in percent, refresh earners base)
CATEGORIES = {
    "platinum": (0.35, 2.0, 0.25),
    "completion": (0.30, 2.0, 0.25),
    "chapter": (1.35, 45.0, 1.00),
    "story_final": (1.25, 18.0, 0.90),
    "story_mission": (1.25, 22.0, 0.90),
    "stranger": (0.95, 14.0, 0.70),
    "grind": (0.55, 6.0, 0.40),
    "misc": (0.85, 16.0, 0.60),
}

CHAPTER_CAPS = {1: 45.0, 2: 35.0, 3: 30.0, 4: 25.0}

GRIND_WORDS = (
    "stunts", "gold medal", "win 20", "mission strands", "collectible card",
    "100 locations", "50 challenges", "bank vaults", "discover 100",
    "earn $250,000", "spend $100,000", "purchase all", "register 10",
    "meditate at every", "unique animal", "top fun", "shooting range",
)


def classify(trophy: dict) -> str:
    """Category of a trophy from its tier, name and description."""
    tier = trophy.get("tier", "")
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in text:
        return "completion"
    m = re.search(r"chapter (\d)", text)
    if m:
        return "chapter"
    if "final mission" in text and "for " not in text.split("final mission", 1)[1][:12]:
        return "story_final"
    if "final mission for " in text:
        return "stranger"
    if any(word in text for word in GRIND_WORDS):
        return "grind"
    if text.startswith("complete ") or "complete " in text:
        return "story_mission"
    return "misc"


def chapter_cap(trophy: dict) -> float:
    text = trophy.get("description", "").lower()
    m = re.search(r"chapter (\d)", text)
    if m:
        return CHAPTER_CAPS.get(int(m.group(1)), 30.0)
    return CATEGORIES["chapter"][1]


def distinct_reads(history: list[dict]) -> list[dict]:
    """Reads with a distinct (players_tracked, rates) signature, in order."""
    seen: set = set()
    out = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key in seen:
            continue
        seen.add(key)
        out.append(read)
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    history = case["history"]
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z"
                         if len(case["game"]["release_date"]) == 10
                         else case["game"]["release_date"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    reads = distinct_reads(history)
    last = history[-1]
    last_players = last["players_tracked"]
    last_at = parse_time(last["at"])
    hours_last = max(0.0, (last_at - release).total_seconds() / 3600.0)
    x_last = math.log1p(hours_last)
    x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
    dx = max(0.0, x_target - x_last)

    # Time coordinates of the distinct reads.
    xs: list[float] = []
    ns: list[float] = []
    for read in reads:
        at = parse_time(read["at"])
        h = max(0.0, (at - release).total_seconds() / 3600.0)
        xs.append(math.log1p(h))
        ns.append(float(read["players_tracked"]))
    x_span = (max(xs) - min(xs)) if len(xs) > 1 else 0.0
    n_distinct = len(reads)
    players_varied = len(set(ns)) > 1

    # --- players_tracked growth -------------------------------------------
    log_ns = [math.log(max(1.0, n)) for n in ns]
    slope_n_fit = ols_slope(xs, log_ns) if players_varied else None
    if slope_n_fit is None:
        slope_n = PRIOR_N_EXP
    else:
        w = n_distinct / (n_distinct + 2.0)
        slope_n = w * slope_n_fit + (1.0 - w) * PRIOR_N_EXP
    slope_n = min(2.5, max(0.0, slope_n))

    # How much the data fit is trusted: 0 for a flat history, near 1 for a
    # history with many distinct reads.
    w_data = 0.0 if not players_varied else n_distinct / (n_distinct + 2.0)

    # --- refresh prior (used when the history is flat) ---------------------
    # Hours the page sat unchanged before the checkpoint: the whole span of
    # the history when every read is identical.
    if players_varied:
        staleness = 0.0
    else:
        first_at = parse_time(history[0]["at"])
        staleness = max(0.0, (last_at - first_at).total_seconds() / 3600.0)
    # Probability that a refresh lands before the target: the horizon plus
    # the time the page is already stale, against one refresh interval.
    # A page stale for hours is due one, so the stale time counts too.
    p_refresh = 1.0 - math.exp(-(horizon_hours + min(staleness, 4.0)) / REFRESH_HOURS)
    # A refresh's player jump decays with time since release: the launch
    # wave is biggest in the first hours.  Calibrated on the house's own
    # recording of the first refresh (638 -> 3,868 players tracked, while
    # the trophies earned by tracked players went 943 -> 4,276, so the
    # non-100% earners grew only ~1.3x against a ~6x player jump).
    jump_n = 1.0 + 6.0 * math.exp(-hours_last / 12.0)
    log_jump_n = math.log(jump_n)
    # The expected log move is the refresh probability times one refresh's
    # log move; with a varied history this term is tiny (the jumps have
    # decayed away by then) and the trend fit below dominates instead.
    expected_jumps = p_refresh

    # --- spread components --------------------------------------------------
    rw_sigma = 0.5 * math.sqrt(horizon_days)
    sigma_extra = 0.45 + 0.40 * (1.0 - w_data)

    rows = []
    for i, trophy in enumerate(trophies):
        cat = classify(trophy)
        exp_prior, cap, jump_e_base = CATEGORIES[cat]
        if cat == "chapter":
            cap = chapter_cap(trophy)

        # Last non-null rate for this trophy.
        rate_last = None
        for read in reversed(history):
            r = read["rates"][i]
            if r is not None:
                rate_last = float(r)
                break
        if rate_last is None:
            rate_last = {"platinum": 0.1, "completion": 0.1, "grind": 0.3}.get(cat, 0.8)

        # --- saturated trophy: already at (or near) 100% -------------------
        if rate_last >= 99.5:
            median = 100.0 - 0.2 - 0.15 * horizon_days
            median = min(99.9, max(90.0, median))
            sigma = 0.5
            rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
            continue

        # --- earners trend fit ----------------------------------------------
        exs: list[float] = []
        eys: list[float] = []
        for read, x in zip(reads, xs):
            r = read["rates"][i]
            if r is None:
                continue
            earners = max(0.5, float(r) * float(read["players_tracked"]) / 100.0)
            exs.append(x)
            eys.append(math.log(earners))
        slope_e_fit = ols_slope(exs, eys) if len(set(eys)) > 1 else None
        resid = 0.0
        if slope_e_fit is not None and len(exs) >= 3:
            resid = ols_resid_std(exs, eys, slope_e_fit)
        if slope_e_fit is None:
            slope_e = exp_prior
        else:
            w = len(exs) / (len(exs) + 3.0)
            slope_e = w * slope_e_fit + (1.0 - w) * exp_prior
        slope_e = min(3.0, max(-0.5, slope_e))

        # Trend forecast, anchored at the last read.
        log_ratio_trend = (slope_e - slope_n) * dx
        log_ratio_trend = max(-MAX_LOG_MOVE, min(MAX_LOG_MOVE, log_ratio_trend))
        rate_trend = rate_last * math.exp(log_ratio_trend)

        # --- flat-history refresh prior --------------------------------------
        jump_e = 1.0 + jump_e_base * math.exp(-hours_last / 12.0)
        log_ratio_prior = expected_jumps * (math.log(jump_e) - log_jump_n)
        log_ratio_prior = max(-MAX_LOG_MOVE, min(MAX_LOG_MOVE, log_ratio_prior))
        rate_prior = rate_last * math.exp(log_ratio_prior)

        # Blend: data fit where the history varies, refresh prior where flat.
        log_rate = w_data * math.log(max(0.01, rate_trend)) + (1.0 - w_data) * math.log(max(0.01, rate_prior))
        median = math.exp(log_rate)
        median = min(cap, max(0.01, median))

        # --- spread -----------------------------------------------------------
        sigma_fit = min(0.9, max(0.15, resid)) * (1.0 + dx / max(x_span, 0.5)) if slope_e_fit is not None else 0.5
        sigma = math.sqrt(rw_sigma ** 2 + sigma_fit ** 2 + sigma_extra ** 2)
        sigma = min(1.8, max(0.4, sigma))

        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def quantiles(median: float, sigma: float) -> list:
    """Five quantiles, symmetric in logit space around the median."""
    centre = logit(median)
    qs = [round4(inv_logit(centre + z * sigma)) for z in Z]
    # Rounding can only break monotonicity at the extremes; restore it.
    fixed: list = []
    prev = 0.0
    for q in qs:
        q = max(float(q), prev)
        fixed.append(round4(q))
        prev = float(fixed[-1])
    return [min(100.0, max(0.01, float(q))) for q in fixed]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
