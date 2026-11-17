"""Is work spread fairly, and is any nurse always picked first?

Run every care request through the engine and look at who lands at #1 and in the
top 5. Two modes:

* score only: each request is ranked on its own, nothing remembers earlier picks.
* with workload balancing: requests are handled in order, and a nurse who has
  already been picked first scores a little lower on the next request.

    python -m matcher.fairness
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import MAX_DISTANCE_KM
from .dataset import load_dataset
from .engine import eligible_nurses, match
from .models import CareRequest, Nurse
from .scoring import DEFAULT_WEIGHTS, Weights, score_factors, weighted_contributions

# A nurse needs at least this many eligible requests before "always first" means anything.
MIN_ELIGIBLE_FOR_ALWAYS_FIRST = 5


def gini(values: Sequence[float]) -> float:
    """0 means everyone got the same amount of work, close to 1 means one person got it all."""
    x = np.sort(np.asarray(values, dtype=float))
    total = x.sum()
    if total == 0:
        return 0.0
    n = len(x)
    cumulative = np.cumsum(x)
    return float((n + 1 - 2 * cumulative.sum() / total) / n)


@dataclass
class FairnessReport:
    per_nurse: pd.DataFrame  # nurse_id, name, eligible_requests, top1, top5
    requests_total: int
    requests_unmatched: int
    gini_top1: float
    gini_top1_random: float  # same statistic if each request picked a random eligible nurse
    max_top1_count: int
    max_top1_share: float  # busiest nurse's share of all first picks
    top10_share: float  # share of first picks that went to the 10 busiest nurses
    nurses_never_top5: int
    always_first: list[str]
    mean_top1_score: float  # judged with the normal weights, whatever weights picked the nurse
    mean_top1_distance_km: float


def _random_baseline_gini(
    eligible_ids: list[list[str]], nurse_ids: list[str], trials: int = 200, seed: int = 0
) -> float:
    rng = np.random.default_rng(seed)
    pools = [p for p in eligible_ids if p]
    ginis = []
    for _ in range(trials):
        counts = dict.fromkeys(nurse_ids, 0)
        for pool in pools:
            counts[pool[int(rng.integers(len(pool)))]] += 1
        ginis.append(gini(list(counts.values())))
    return float(np.mean(ginis))


def evaluate(
    nurses: Sequence[Nurse],
    requests: Sequence[CareRequest],
    base_weights: Weights = DEFAULT_WEIGHTS,
    balance_weight: float = 0.0,
    max_distance_km: float = MAX_DISTANCE_KM,
) -> FairnessReport:
    """balance_weight = 0 ranks each request independently; above 0 turns on workload balancing."""
    weights = replace(base_weights, workload=balance_weight)
    nurse_ids = [n.nurse_id for n in nurses]
    top1 = dict.fromkeys(nurse_ids, 0)
    top5 = dict.fromkeys(nurse_ids, 0)
    eligible_count = dict.fromkeys(nurse_ids, 0)
    eligible_pools: list[list[str]] = []
    load: dict[str, int] = {}
    chosen_scores, chosen_distances = [], []
    unmatched = 0

    for request in requests:
        pool = eligible_nurses(request, nurses, max_distance_km)
        eligible_pools.append([n.nurse_id for n, _ in pool])
        for n, _ in pool:
            eligible_count[n.nurse_id] += 1

        result = match(request, nurses, weights=weights, top_k=5,
                       max_distance_km=max_distance_km, load=load)
        if not result.matches:
            unmatched += 1
            continue

        for m in result.matches:
            top5[m.nurse.nurse_id] += 1
        first = result.matches[0]
        top1[first.nurse.nurse_id] += 1
        if balance_weight > 0:
            load[first.nurse.nurse_id] = load.get(first.nurse.nurse_id, 0) + 1

        # Quality is always judged on the normal factors, with no workload term.
        factors = score_factors(first.nurse, request, first.distance_km, max_distance_km)
        chosen_scores.append(sum(weighted_contributions(factors, base_weights).values()))
        chosen_distances.append(first.distance_km)

    df = pd.DataFrame(
        {
            "nurse_id": nurse_ids,
            "name": [n.name for n in nurses],
            "eligible_requests": [eligible_count[i] for i in nurse_ids],
            "top1": [top1[i] for i in nurse_ids],
            "top5": [top5[i] for i in nurse_ids],
        }
    )
    matched = len(requests) - unmatched
    top1_sorted = df["top1"].sort_values(ascending=False)
    always_first = df[
        (df["eligible_requests"] >= MIN_ELIGIBLE_FOR_ALWAYS_FIRST)
        & (df["top1"] == df["eligible_requests"])
    ]["nurse_id"].tolist()

    return FairnessReport(
        per_nurse=df,
        requests_total=len(requests),
        requests_unmatched=unmatched,
        gini_top1=gini(df["top1"]),
        gini_top1_random=_random_baseline_gini(eligible_pools, nurse_ids),
        max_top1_count=int(top1_sorted.iloc[0]),
        max_top1_share=float(top1_sorted.iloc[0] / matched) if matched else 0.0,
        top10_share=float(top1_sorted.head(10).sum() / matched) if matched else 0.0,
        nurses_never_top5=int((df["top5"] == 0).sum()),
        always_first=always_first,
        mean_top1_score=float(np.mean(chosen_scores)) if chosen_scores else 0.0,
        mean_top1_distance_km=float(np.mean(chosen_distances)) if chosen_distances else 0.0,
    )


def comparison_table(score_only: FairnessReport, balanced: FairnessReport, balance_weight: float) -> pd.DataFrame:
    def row(name, f):
        return name, f(score_only), f(balanced)

    rows = [
        row("Requests with no eligible nurse", lambda r: f"{r.requests_unmatched}"),
        row("Gini of first picks (random-pick baseline: "
            f"{score_only.gini_top1_random:.2f})", lambda r: f"{r.gini_top1:.2f}"),
        row("Most first picks for one nurse", lambda r: f"{r.max_top1_count} ({r.max_top1_share:.0%})"),
        row("Share of first picks held by 10 busiest nurses", lambda r: f"{r.top10_share:.0%}"),
        row("Nurses never in anyone's top 5", lambda r: f"{r.nurses_never_top5}"),
        row("Nurses always first for every request they qualify for", lambda r: f"{len(r.always_first)}"),
        row("Average match score of the first pick", lambda r: f"{r.mean_top1_score:.3f}"),
        row("Average distance of the first pick (km)", lambda r: f"{r.mean_top1_distance_km:.1f}"),
    ]
    return pd.DataFrame(rows, columns=["Measure", "Score only", f"Workload balancing ({balance_weight:.2f})"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairness evaluation on the generated dataset.")
    parser.add_argument("--balance-weight", type=float, default=0.10)
    parser.add_argument("--write", type=Path, help="also write the table to this markdown file")
    args = parser.parse_args()

    nurses, requests = load_dataset()
    base = evaluate(nurses, requests)
    balanced = evaluate(nurses, requests, balance_weight=args.balance_weight)
    table = comparison_table(base, balanced, args.balance_weight)
    text = table.to_markdown(index=False) if _has_tabulate() else table.to_string(index=False)
    print(text)
    if args.write:
        args.write.write_text(
            "# Fairness evaluation\n\nGenerated by `python -m matcher.fairness`. "
            "Seeded data, so the numbers are repeatable.\n\n" + text + "\n"
        )


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
