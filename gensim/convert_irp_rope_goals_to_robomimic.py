#!/usr/bin/env python3
"""
Convert IRP rope dataset (zarr) to a goal-conditioned robomimic-style HDF5.

For each sampled goal and each (length_idx, density_idx) parameter pair:
- use precomputed control/best_action_coord to select the best action coordinate
  for that goal pixel,
- map action coordinates to action values [speed, j2_delta, j3_delta],
- write one demo with low-dim observation goal_points.

Output demos are in goal-major order:
for goal_0: all parameter pairs, then goal_1, ...

Each output demo contains:
- data/demo_i/obs/goal_points: (1, 1, 2) float32   # one [x, z] world point
- data/demo_i/obs/goal_pix: (1, 2) int32           # image/grid coordinate
- data/demo_i/obs/param_size: (1,) float32         # rope length
- data/demo_i/obs/param_density: (1,) float32      # rope density
- data/demo_i/obs/param_size_id: (1,) int32
- data/demo_i/obs/param_density_id: (1,) int32
- data/demo_i/actions: (1, 3) float32              # [speed, j2_delta, j3_delta]

Compatibility extras:
- data/demo_i/param_size
- data/demo_i/param_density
- data/demo_i/grid_index: [size_id, density_id, speed_id, j2_id, j3_id]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REQUIRED_DIM_KEYS = [
    "length",
    "density",
    "speed",
    "j2_delta",
    "j3_delta",
]


def required_metadata_relpaths() -> List[str]:
    relpaths = [
        ".zgroup",
        ".zattrs",
        "dim_samples/.zgroup",
        "control/.zgroup",
        "control/max_hitrate/.zarray",
        "control/best_action_coord/.zarray",
    ]
    relpaths.extend([f"dim_samples/{k}/.zarray" for k in REQUIRED_DIM_KEYS])
    return relpaths


def missing_metadata_relpaths(zarr_path: Path) -> List[str]:
    return [rel for rel in required_metadata_relpaths() if not (zarr_path / rel).exists()]


def choose_writable_extract_parent(preferred_parent: Path) -> Path:
    preferred_parent = preferred_parent.resolve()
    if preferred_parent.exists() and os.access(str(preferred_parent), os.W_OK):
        return preferred_parent
    fallback = Path("/tmp").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def ensure_complete_zarr_from_tar(tar_path: Path, zarr_path: Path) -> None:
    missing = missing_metadata_relpaths(zarr_path)
    if not missing:
        return

    if not tar_path.exists():
        raise FileNotFoundError(
            "Zarr metadata is incomplete and tar archive was not found.\n"
            f"Expected tar: {tar_path}\nMissing files (examples): {missing[:5]}"
        )

    print(f"Extracting / repairing zarr from tar: {tar_path}")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=zarr_path.parent)

    missing_after = missing_metadata_relpaths(zarr_path)
    if missing_after:
        raise RuntimeError(
            "Metadata repair/extraction failed. Still missing files (examples): "
            f"{missing_after[:5]}"
        )


def resolve_zarr_path(input_path: Path, extract_dir: Path = None) -> Path:
    if input_path.is_dir():
        if not input_path.name.endswith(".zarr"):
            raise ValueError(f"Expected a .zarr directory, got: {input_path}")

        sibling_tar = Path(str(input_path) + ".tar")
        missing = missing_metadata_relpaths(input_path)
        if not missing:
            return input_path

        preferred_parent = extract_dir if extract_dir is not None else input_path.parent
        target_parent = choose_writable_extract_parent(preferred_parent)
        repaired_zarr = target_parent / input_path.name
        if target_parent != input_path.parent:
            print(
                "Input zarr is incomplete and source parent is not writable. "
                f"Using writable extract dir: {target_parent}"
            )
        ensure_complete_zarr_from_tar(sibling_tar, repaired_zarr)
        return repaired_zarr

    if input_path.is_file() and input_path.name.endswith(".zarr.tar"):
        preferred_parent = extract_dir if extract_dir is not None else input_path.parent
        target_parent = choose_writable_extract_parent(preferred_parent)
        target_parent.mkdir(parents=True, exist_ok=True)
        zarr_dir_name = input_path.name[:-4]  # strip ".tar"
        zarr_path = target_parent / zarr_dir_name
        ensure_complete_zarr_from_tar(input_path, zarr_path)
        return zarr_path

    raise ValueError("Input must be either an extracted .zarr directory or a .zarr.tar file.")


def get_h5_compression_kwargs(compression: str, compression_level: int) -> Dict:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {
            "compression": "gzip",
            "compression_opts": int(compression_level),
            "shuffle": True,
        }
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    raise ValueError(f"Unknown compression: {compression}")


def validate_dim_samples(dim_samples: Dict[str, np.ndarray]) -> None:
    missing = [k for k in REQUIRED_DIM_KEYS if k not in dim_samples]
    if missing:
        raise KeyError(f"Missing required dim samples: {missing}")

    for k in REQUIRED_DIM_KEYS:
        if dim_samples[k].ndim != 1:
            raise ValueError(f"Expected 1D dim_samples/{k}, got shape={dim_samples[k].shape}")


def grid_to_world(
    pix_coord_2: np.ndarray,
    low_2: np.ndarray,
    high_2: np.ndarray,
    grid_2: np.ndarray,
) -> np.ndarray:
    """
    Inverse of GridCoordTransformer.to_grid.
    pix = (world - low) * grid / (high - low)
    => world = pix * (high - low) / grid + low
    """
    pix = np.asarray(pix_coord_2, dtype=np.float32)
    low = np.asarray(low_2, dtype=np.float32)
    high = np.asarray(high_2, dtype=np.float32)
    grid = np.asarray(grid_2, dtype=np.float32)
    return pix * (high - low) / grid + low


def choose_goals_for_pair(
    valid_goals_hw2: np.ndarray,
    num_goals: int,
    goal_sampling: str,
    rs: np.random.RandomState,
) -> np.ndarray:
    n_valid = int(valid_goals_hw2.shape[0])
    if n_valid < num_goals:
        raise RuntimeError(
            f"Not enough valid goals: need {num_goals}, found {n_valid}. "
            "Try a lower --num-goals or lower --hitrate-threshold."
        )

    if goal_sampling == "linspace":
        ids = np.linspace(0, n_valid - 1, num=num_goals, dtype=np.int64)
    elif goal_sampling == "random":
        ids = rs.choice(n_valid, size=num_goals, replace=False)
    else:
        raise ValueError(f"Unknown goal_sampling: {goal_sampling}")

    return valid_goals_hw2[ids].astype(np.int32)


def pair_seed(base_seed: int, size_idx: int, density_idx: int) -> int:
    return int((int(base_seed) + size_idx * 1_000_003 + density_idx * 1_000_033) % (2**32 - 1))


def compute_pair_goals_actions_from_arrays(
    hitrate_img_hw: np.ndarray,
    best_action_pair_hw3: np.ndarray,
    size_idx: int,
    density_idx: int,
    num_goals: int,
    hitrate_threshold: float,
    goal_sampling: str,
    seed: int,
    allow_empty: bool = False,
) -> Dict[str, np.ndarray]:
    valid_goals = np.argwhere(hitrate_img_hw > float(hitrate_threshold))  # (N,2) [row, col]
    if valid_goals.shape[0] == 0:
        if allow_empty:
            return {
                "pair": np.array([size_idx, density_idx], dtype=np.int32),
                "goal_pixs": np.zeros((0, 2), dtype=np.int32),
                "action_coords": np.zeros((0, 3), dtype=np.int32),
                "goal_hitrates": np.zeros((0,), dtype=np.float32),
                "is_empty": np.array([1], dtype=np.int32),
            }
        raise RuntimeError(
            f"No valid goals for pair (size_idx={size_idx}, density_idx={density_idx}) "
            f"at threshold {hitrate_threshold}."
        )

    rs = np.random.RandomState(seed=seed)
    goal_pixs = choose_goals_for_pair(
        valid_goals_hw2=valid_goals,
        num_goals=num_goals,
        goal_sampling=goal_sampling,
        rs=rs,
    )
    action_coords = np.asarray(
        best_action_pair_hw3[goal_pixs[:, 0], goal_pixs[:, 1]],
        dtype=np.int32,
    )  # (G,3)
    goal_hitrates = np.asarray(
        hitrate_img_hw[goal_pixs[:, 0], goal_pixs[:, 1]],
        dtype=np.float32,
    )  # (G,)

    return {
        "pair": np.array([size_idx, density_idx], dtype=np.int32),
        "goal_pixs": goal_pixs.astype(np.int32),
        "action_coords": action_coords.astype(np.int32),
        "goal_hitrates": goal_hitrates.astype(np.float32),
        "is_empty": np.array([0], dtype=np.int32),
    }


def compute_pair_goals_actions_worker(
    zarr_path_str: str,
    size_idx: int,
    density_idx: int,
    num_goals: int,
    hitrate_threshold: float,
    goal_sampling: str,
    seed: int,
    allow_empty: bool = False,
) -> Dict[str, np.ndarray]:
    import zarr

    root = zarr.open(zarr_path_str, mode="r")
    hitrate_img = np.asarray(root["control/max_hitrate"][size_idx, density_idx], dtype=np.float32)
    best_action_pair = np.asarray(root["control/best_action_coord"][size_idx, density_idx], dtype=np.int32)
    return compute_pair_goals_actions_from_arrays(
        hitrate_img_hw=hitrate_img,
        best_action_pair_hw3=best_action_pair,
        size_idx=size_idx,
        density_idx=density_idx,
        num_goals=num_goals,
        hitrate_threshold=hitrate_threshold,
        goal_sampling=goal_sampling,
        seed=seed,
        allow_empty=allow_empty,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IRP rope zarr to goal-conditioned robomimic HDF5."
    )
    default_workers = min(16, max(1, (os.cpu_count() or 1)))
    parser.add_argument("--input", type=str, required=True, help="Path to irp_rope.zarr or irp_rope.zarr.tar")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 file path")
    parser.add_argument("--num-goals", type=int, required=True, help="Number of goals to sample per parameter pair")
    parser.add_argument(
        "--hitrate-threshold",
        type=float,
        default=0.95,
        help="Valid-goal threshold over control/max_hitrate (default: 0.95)",
    )
    parser.add_argument(
        "--goal-sampling",
        type=str,
        default="linspace",
        choices=["linspace", "random"],
        help="Goal sampling strategy from valid-goal pixels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --goal-sampling=random",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Optional cap on number of demos written (for debug)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression for larger datasets",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip level (1-9), only used when --compression=gzip",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="If --input is tar (or incomplete zarr), extract/repair under this directory",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        help=f"Parallel workers for pair-level goal/action selection (default: {default_workers})",
    )
    parser.add_argument(
        "--skip-empty-pairs",
        action="store_true",
        help="Skip (size,density) pairs that have no valid goals at the threshold instead of raising an error",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import h5py
    from tqdm import tqdm
    import zarr

    if args.num_goals <= 0:
        raise ValueError(f"--num-goals must be positive, got {args.num_goals}")
    if args.compression == "gzip" and not (1 <= args.compression_level <= 9):
        raise ValueError("--compression-level must be in [1, 9] for gzip")
    if args.num_workers <= 0:
        raise ValueError(f"--num-workers must be positive, got {args.num_workers}")

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve() if args.extract_dir is not None else None

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zarr_path = resolve_zarr_path(input_path, extract_dir=extract_dir)
    print(f"Using zarr dataset: {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")

    dim_samples = {
        "length": np.asarray(root["dim_samples/length"][:], dtype=np.float32),
        "density": np.asarray(root["dim_samples/density"][:], dtype=np.float32),
        "speed": np.asarray(root["dim_samples/speed"][:], dtype=np.float32),
        "j2_delta": np.asarray(root["dim_samples/j2_delta"][:], dtype=np.float32),
        "j3_delta": np.asarray(root["dim_samples/j3_delta"][:], dtype=np.float32),
    }
    validate_dim_samples(dim_samples)

    max_hitrate = root["control/max_hitrate"]  # (size,density,H,W)
    best_action_coord = root["control/best_action_coord"]  # (size,density,H,W,3)

    if max_hitrate.shape[:2] != (len(dim_samples["length"]), len(dim_samples["density"])):
        raise ValueError(
            "Shape mismatch between control/max_hitrate and dim samples. "
            f"max_hitrate.shape[:2]={max_hitrate.shape[:2]}, "
            f"(len(length),len(density))={(len(dim_samples['length']), len(dim_samples['density']))}"
        )
    if best_action_coord.shape[:4] != max_hitrate.shape:
        raise ValueError(
            "Shape mismatch: control/best_action_coord[:4] must equal control/max_hitrate shape. "
            f"best_action_coord.shape={best_action_coord.shape}, max_hitrate.shape={max_hitrate.shape}"
        )
    if best_action_coord.shape[-1] != 3:
        raise ValueError(f"Expected control/best_action_coord last dim = 3, got {best_action_coord.shape[-1]}")

    # transformer metadata for pixel -> world conversion
    transformer_cfg = root.attrs.get("transformer", None)
    if transformer_cfg is None:
        raise KeyError("Missing root .zattrs['transformer'] in zarr dataset.")
    low = np.asarray(transformer_cfg["low"], dtype=np.float32)
    high = np.asarray(transformer_cfg["high"], dtype=np.float32)
    grid_shape = np.asarray(transformer_cfg["grid_shape"], dtype=np.float32)
    if low.shape != (2,) or high.shape != (2,) or grid_shape.shape != (2,):
        raise ValueError(
            "Expected transformer low/high/grid_shape to be length-2 arrays, got "
            f"low={low}, high={high}, grid_shape={grid_shape}"
        )

    size_count = len(dim_samples["length"])
    density_count = len(dim_samples["density"])
    pair_indices = [(si, di) for si in range(size_count) for di in range(density_count)]

    pair_results: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    skipped_pairs: List[Tuple[int, int]] = []
    if args.num_workers == 1:
        for size_idx, density_idx in tqdm(pair_indices, desc="Selecting goals/actions per param pair", unit="pair"):
            seed_this = pair_seed(args.seed, size_idx, density_idx)
            result = compute_pair_goals_actions_from_arrays(
                hitrate_img_hw=np.asarray(max_hitrate[size_idx, density_idx], dtype=np.float32),
                best_action_pair_hw3=np.asarray(best_action_coord[size_idx, density_idx], dtype=np.int32),
                size_idx=size_idx,
                density_idx=density_idx,
                num_goals=args.num_goals,
                hitrate_threshold=float(args.hitrate_threshold),
                goal_sampling=args.goal_sampling,
                seed=seed_this,
                allow_empty=bool(args.skip_empty_pairs),
            )
            pair = tuple(int(x) for x in result["pair"].tolist())
            if int(result["is_empty"][0]) == 1:
                skipped_pairs.append(pair)
            else:
                pair_results[pair] = result
    else:
        max_workers = min(args.num_workers, len(pair_indices))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for size_idx, density_idx in pair_indices:
                seed_this = pair_seed(args.seed, size_idx, density_idx)
                futures.append(
                    executor.submit(
                        compute_pair_goals_actions_worker,
                        str(zarr_path),
                        size_idx,
                        density_idx,
                        args.num_goals,
                        float(args.hitrate_threshold),
                        args.goal_sampling,
                        seed_this,
                        bool(args.skip_empty_pairs),
                    )
                )
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Selecting goals/actions per param pair", unit="pair"):
                result = fut.result()
                pair = tuple(int(x) for x in result["pair"].tolist())
                if int(result["is_empty"][0]) == 1:
                    skipped_pairs.append(pair)
                else:
                    pair_results[pair] = result

    valid_pair_indices = [pair for pair in pair_indices if pair in pair_results]
    if len(valid_pair_indices) == 0:
        raise RuntimeError(
            "No parameter pairs have valid goals. "
            f"threshold={args.hitrate_threshold}, skip_empty_pairs={args.skip_empty_pairs}"
        )
    if skipped_pairs:
        skipped_pairs_sorted = sorted(skipped_pairs)
        print(
            f"Skipping {len(skipped_pairs_sorted)} empty pair(s): "
            + ", ".join(f"({x[0]},{x[1]})" for x in skipped_pairs_sorted)
        )

    demos_target = args.num_goals * len(valid_pair_indices)
    if args.max_demos is not None:
        demos_target = min(demos_target, args.max_demos)

    h5_comp = get_h5_compression_kwargs(args.compression, args.compression_level)
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")

        f.attrs["source_zarr"] = str(zarr_path)
        f.attrs["num_goals"] = int(args.num_goals)
        f.attrs["num_demos"] = int(demos_target)
        f.attrs["goal_sampler"] = args.goal_sampling
        f.attrs["goal_hitrate_threshold"] = float(args.hitrate_threshold)
        f.attrs["goal_major_order"] = 1
        f.attrs["action_keys"] = "speed,j2_delta,j3_delta"
        f.attrs["num_workers"] = int(args.num_workers)
        f.attrs["skip_empty_pairs"] = int(args.skip_empty_pairs)
        f.attrs["num_pairs_total"] = int(len(pair_indices))
        f.attrs["num_pairs_used"] = int(len(valid_pair_indices))
        f.attrs["num_pairs_skipped"] = int(len(skipped_pairs))
        if skipped_pairs:
            f.create_dataset("skipped_pairs", data=np.asarray(sorted(skipped_pairs), dtype=np.int32), dtype=np.int32)

        demo_id = 0
        write_bar = tqdm(total=demos_target, desc="Writing demos", unit="demo")
        for goal_id in range(args.num_goals):
            for size_idx, density_idx in valid_pair_indices:
                if demo_id >= demos_target:
                    break

                result = pair_results[(size_idx, density_idx)]
                goal_pix = result["goal_pixs"][goal_id].astype(np.int32)  # (2,)
                gp0 = int(goal_pix[0])
                gp1 = int(goal_pix[1])

                action_coord = np.asarray(result["action_coords"][goal_id], dtype=np.int32)
                speed_id = int(action_coord[0])
                j2_id = int(action_coord[1])
                j3_id = int(action_coord[2])

                # Bounds check against dim samples.
                if not (0 <= speed_id < len(dim_samples["speed"])):
                    raise IndexError(f"speed_id out of range: {speed_id}")
                if not (0 <= j2_id < len(dim_samples["j2_delta"])):
                    raise IndexError(f"j2_id out of range: {j2_id}")
                if not (0 <= j3_id < len(dim_samples["j3_delta"])):
                    raise IndexError(f"j3_id out of range: {j3_id}")

                action_values = np.array(
                    [
                        dim_samples["speed"][speed_id],
                        dim_samples["j2_delta"][j2_id],
                        dim_samples["j3_delta"][j3_id],
                    ],
                    dtype=np.float32,
                )
                goal_world = grid_to_world(goal_pix.astype(np.float32), low, high, grid_shape).astype(np.float32)

                grid_index = np.array(
                    [size_idx, density_idx, speed_id, j2_id, j3_id],
                    dtype=np.int32,
                )
                param_size = np.array([dim_samples["length"][size_idx]], dtype=np.float32)
                param_density = np.array([dim_samples["density"][density_idx]], dtype=np.float32)
                param_size_id = np.array([size_idx], dtype=np.int32)
                param_density_id = np.array([density_idx], dtype=np.int32)

                demo_key = f"demo_{demo_id}"
                demo_group = data_group.create_group(demo_key)
                obs_group = demo_group.create_group("obs")

                obs_group.create_dataset("goal_points", data=goal_world[None, None, :], dtype=np.float32, **h5_comp)
                obs_group.create_dataset("goal_pix", data=goal_pix[None, :], dtype=np.int32)
                obs_group.create_dataset("param_size", data=param_size, dtype=np.float32)
                obs_group.create_dataset("param_density", data=param_density, dtype=np.float32)
                obs_group.create_dataset("param_size_id", data=param_size_id, dtype=np.int32)
                obs_group.create_dataset("param_density_id", data=param_density_id, dtype=np.int32)

                demo_group.create_dataset("actions", data=action_values[None, :], dtype=np.float32)

                # Compatibility fields.
                demo_group.create_dataset("param_size", data=param_size, dtype=np.float32)
                demo_group.create_dataset("param_density", data=param_density, dtype=np.float32)
                demo_group.create_dataset("grid_index", data=grid_index, dtype=np.int32)
                demo_group.create_dataset("goal_pix", data=goal_pix, dtype=np.int32)
                demo_group.create_dataset(
                    "goal_hitrate",
                    data=np.array([float(result["goal_hitrates"][goal_id])], dtype=np.float32),
                    dtype=np.float32,
                )

                demo_group.attrs["num_samples"] = 1
                demo_group.attrs["goal_id"] = int(goal_id)

                demo_id += 1
                write_bar.update(1)
            if demo_id >= demos_target:
                break
        write_bar.close()

    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    main()
