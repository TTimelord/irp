#!/usr/bin/env python3
"""
Rank action distances of all parameter pairs relative to a reference pair.

Works with IRP->robomimic HDF5 files that contain:
- data/demo_*/actions
- data/demo_*/obs/param_size_id
- data/demo_*/obs/param_density_id
- data/demo_*.attrs['goal_id']
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


Pair = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank action distance of all parameter pairs from a reference pair across goals."
    )
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 dataset path")
    parser.add_argument("--ref-size-idx", type=int, required=True, help="Reference param_size_id")
    parser.add_argument("--ref-density-idx", type=int, required=True, help="Reference param_density_id")
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["l2", "l1", "linf"],
        help="Distance metric between action vectors (default: l2)",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "median", "max"],
        help="How to aggregate per-goal distances into one score (default: mean)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many rows to print in rankings (default: 10)")
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include reference pair in ranking outputs (default: excluded)",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional output CSV path for aggregate ranking table",
    )
    return parser.parse_args()


def sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    return sorted(list(data_group.keys()), key=lambda x: int(x.split("_")[-1]))


def action_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if metric == "l2":
        return float(np.linalg.norm(diff, ord=2))
    if metric == "l1":
        return float(np.linalg.norm(diff, ord=1))
    if metric == "linf":
        return float(np.linalg.norm(diff, ord=np.inf))
    raise ValueError(f"Unknown metric: {metric}")


def aggregate_distances(dists: List[float], agg: str) -> float:
    arr = np.asarray(dists, dtype=np.float64)
    if agg == "mean":
        return float(np.mean(arr))
    if agg == "median":
        return float(np.median(arr))
    if agg == "max":
        return float(np.max(arr))
    raise ValueError(f"Unknown aggregate: {agg}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    pair_goal_to_action: Dict[Pair, Dict[int, np.ndarray]] = defaultdict(dict)
    action_dim = None

    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise KeyError("Missing top-level 'data' group")
        for demo_key in sorted_demo_keys(f["data"]):
            demo = f["data"][demo_key]
            size_id = int(np.asarray(demo["obs/param_size_id"])[0])
            density_id = int(np.asarray(demo["obs/param_density_id"])[0])
            goal_id = int(demo.attrs["goal_id"])
            action = np.asarray(demo["actions"])[0].astype(np.float64)

            if action_dim is None:
                action_dim = int(action.shape[0])
            if int(action.shape[0]) != action_dim:
                raise ValueError(
                    f"Inconsistent action dimension at {demo_key}: expected {action_dim}, got {action.shape[0]}"
                )

            pair = (size_id, density_id)
            if goal_id in pair_goal_to_action[pair]:
                raise RuntimeError(f"Duplicate (pair,goal) found for pair={pair}, goal_id={goal_id}")
            pair_goal_to_action[pair][goal_id] = action

    ref_pair: Pair = (args.ref_size_idx, args.ref_density_idx)
    if ref_pair not in pair_goal_to_action:
        raise KeyError(f"Reference pair not found: {ref_pair}")

    ref_goals = sorted(pair_goal_to_action[ref_pair].keys())
    if len(ref_goals) == 0:
        raise RuntimeError(f"Reference pair has no goals: {ref_pair}")

    # Keep only pairs that have actions for all reference goals.
    valid_pairs = []
    skipped_pairs = []
    for pair, goal_map in pair_goal_to_action.items():
        missing = [g for g in ref_goals if g not in goal_map]
        if missing:
            skipped_pairs.append((pair, missing))
        else:
            valid_pairs.append(pair)

    if len(valid_pairs) == 0:
        raise RuntimeError("No pairs have complete goal coverage relative to reference pair.")

    per_goal_rankings: Dict[int, List[Tuple[Pair, float]]] = {}
    for goal_id in ref_goals:
        rows = []
        ref_action = pair_goal_to_action[ref_pair][goal_id]
        for pair in valid_pairs:
            if (not args.include_self) and pair == ref_pair:
                continue
            dist = action_distance(pair_goal_to_action[pair][goal_id], ref_action, args.metric)
            rows.append((pair, dist))
        rows.sort(key=lambda x: x[1])
        per_goal_rankings[goal_id] = rows

    aggregate_rows = []
    for pair in valid_pairs:
        if (not args.include_self) and pair == ref_pair:
            continue
        dists = [
            action_distance(pair_goal_to_action[pair][goal_id], pair_goal_to_action[ref_pair][goal_id], args.metric)
            for goal_id in ref_goals
        ]
        score = aggregate_distances(dists, args.aggregate)
        aggregate_rows.append((pair, score, dists))
    aggregate_rows.sort(key=lambda x: x[1])

    print(f"Input: {input_path}")
    print(f"Reference pair: (size_idx={ref_pair[0]}, density_idx={ref_pair[1]})")
    print(f"Action dim: {action_dim}")
    print(f"Goals used: {ref_goals}")
    print(f"Pairs with complete goals: {len(valid_pairs)}")
    if skipped_pairs:
        print(f"Pairs skipped (missing goals): {len(skipped_pairs)}")
    print("")

    top_k = min(args.top_k, len(aggregate_rows))
    print(f"=== Aggregate Ranking ({args.aggregate} {args.metric}) top {top_k} ===")
    for rank, (pair, score, dists) in enumerate(aggregate_rows[:top_k], start=1):
        per_goal_str = ", ".join(f"g{g}:{dists[i]:.6f}" for i, g in enumerate(ref_goals))
        print(
            f"{rank:>3}. pair(size={pair[0]},density={pair[1]}) "
            f"score={score:.6f} | {per_goal_str}"
        )
    print("")

    print(f"=== Per-goal Rankings ({args.metric}) top {top_k} ===")
    for goal_id in ref_goals:
        print(f"goal_id={goal_id}")
        rows = per_goal_rankings[goal_id][:top_k]
        for rank, (pair, dist) in enumerate(rows, start=1):
            print(f"  {rank:>3}. pair(size={pair[0]},density={pair[1]}) dist={dist:.6f}")
        print("")

    if args.csv_out is not None:
        csv_path = Path(args.csv_out).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            header = [
                "size_id",
                "density_id",
                f"{args.aggregate}_{args.metric}",
            ] + [f"goal_{g}_{args.metric}" for g in ref_goals]
            writer.writerow(header)
            for pair, score, dists in aggregate_rows:
                writer.writerow([pair[0], pair[1], score] + [float(x) for x in dists])
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
