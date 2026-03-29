#!/usr/bin/env python3
"""
Convert IRP cloth dataset (zarr) to a goal-conditioned robomimic-style HDF5.

For each sampled goal and each (cloth_size_idx, cloth_density_idx) pair:
- search over all action coordinates (duration, gy1, gz1, gy2),
- pick the action whose trajectory occupancy is closest to the goal points
  using the same image-distance objective as cloth eval,
- write one demo with low-dim observation goal_points and selected action.

Output demos are in goal-major order:
for goal_0: 16 demos (all size/density pairs), then goal_1, ...

Each output demo contains:
- data/demo_i/obs/goal_points: (1, 9, 3) float32
- data/demo_i/obs/param_size: (1,) float32
- data/demo_i/obs/param_density: (1,) float32
- data/demo_i/obs/param_size_id: (1,) int32
- data/demo_i/obs/param_density_id: (1,) int32
- data/demo_i/actions: (1, 4) float32, ordered [duration, gy1, gz1, gy2]

Compatibility extras:
- data/demo_i/param_size
- data/demo_i/param_density
- data/demo_i/grid_index: full selected coord [size, density, duration, gy1, gz1, gy2]

Notes:
- Standalone script: no imports from IRP repo code.
- Input can be either:
  1) extracted zarr directory (.../irp_cloth.zarr), or
  2) tar archive (.../irp_cloth.zarr.tar).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REQUIRED_DIM_KEYS = [
    "cloth_size",
    "cloth_density",
    "duration",
    "gy1",
    "gz1",
    "gy2",
]


class GridCoordTransformer:
    """
    Minimal local reimplementation of IRP's GridCoordTransformer.
    """

    def __init__(self, low: Tuple[float, float], high: Tuple[float, float], grid_shape: Tuple[int, int]):
        low_arr = np.asarray(low, dtype=np.float32)
        high_arr = np.asarray(high, dtype=np.float32)
        grid_arr = np.asarray(grid_shape, dtype=np.float32)
        self.scale = grid_arr / (high_arr - low_arr)
        self.offset = -low_arr
        self.grid_shape = grid_arr

    @property
    def pix_per_m(self) -> float:
        return float(np.mean(self.scale))

    def to_grid(self, coords: np.ndarray, clip: bool = True) -> np.ndarray:
        result = (coords + self.offset) * self.scale
        if clip:
            result = np.clip(result, a_min=(0.0, 0.0), a_max=self.grid_shape)
        return result


def get_nd_index_volume(shape: Tuple[int, ...]) -> np.ndarray:
    # Equivalent to IRP's get_nd_index_volume.
    return np.moveaxis(
        np.stack(np.meshgrid(*(np.arange(x) for x in shape), indexing="ij")),
        0,
        -1,
    )


def required_metadata_relpaths() -> List[str]:
    relpaths = [
        ".zgroup",
        "dim_keys/.zarray",
        "dim_samples/.zgroup",
        "is_valid/.zarray",
        "traj_occu/.zarray",
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


def load_dim_metadata(zarr_path: Path, zarr) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int]]:
    dim_keys_arr = zarr.open(str(zarr_path / "dim_keys"), mode="r")
    dim_keys = [str(x) for x in dim_keys_arr[:].tolist()]
    missing = [k for k in REQUIRED_DIM_KEYS if k not in dim_keys]
    if missing:
        raise KeyError(f"Missing required dim_keys: {missing}. Found dim_keys={dim_keys}")

    dim_samples = {
        k: np.asarray(zarr.open(str(zarr_path / "dim_samples" / k), mode="r")[:], dtype=np.float32)
        for k in REQUIRED_DIM_KEYS
    }
    dim_to_axis = {k: dim_keys.index(k) for k in REQUIRED_DIM_KEYS}
    return dim_keys, dim_samples, dim_to_axis


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


def get_cloth_goal_points(
    cloth_size: float,
    alpha: float,
    table_y: float = 1.0,
    table_size: float = 1.2,
    table_height: float = 0.0,
    max_reach: float = 1.0,
) -> np.ndarray:
    """
    Reimplementation of TableClothSimEnvironment.get_cloth_goal(alpha), using
    cloth_size directly and cloth_spacing = cloth_size / 12.
    Returns (9, 3) world coordinates.
    """
    cloth_spacing = float(cloth_size) / 12.0
    table_y_start = float(table_y) - float(table_size) / 2.0
    capsule_r = cloth_spacing * 0.2
    offset_range = min(float(table_size) - float(cloth_size), float(max_reach) - table_y_start)
    offset = offset_range * float(np.clip(alpha, 0.0, 1.0))

    base_coords = get_nd_index_volume((3, 3, 1))[:, :, 0, :].astype(np.float32) / 2.0 * float(cloth_size)
    rope_coords = base_coords.copy()
    rope_coords[:, :, 0] -= float(cloth_size) / 2.0
    rope_coords[:, :, 1] += offset + table_y_start
    rope_coords[:, :, 2] = float(table_height) + capsule_r
    return rope_coords.reshape(-1, 3).astype(np.float32)


def evaluate_action_for_all_goals(
    traj_img_9hw: np.ndarray,
    goal_pixs_g92: np.ndarray,
    pix_per_m: float,
) -> np.ndarray:
    """
    Compute cloth eval-style image loss for one action trajectory against all goals.

    traj_img_9hw: (9, H, W) bool/uint8
    goal_pixs_g92: (G, 9, 2) float32
    returns: (G,) float32
    """
    n_goals = goal_pixs_g92.shape[0]
    loss_sum = np.zeros((n_goals,), dtype=np.float32)
    inf_arr = np.full((n_goals,), np.inf, dtype=np.float32)

    for kp_id in range(9):
        mask = traj_img_9hw[kp_id].astype(bool)
        coords = np.argwhere(mask)  # (M,2), [row, col]
        if coords.shape[0] == 0:
            return inf_arr

        # Exact min distance from each goal point to occupied pixels.
        goals_this_kp = goal_pixs_g92[:, kp_id, :].astype(np.float32)  # (G,2)
        diff = coords.astype(np.float32)[:, None, :] - goals_this_kp[None, :, :]  # (M,G,2)
        d2 = np.sum(diff * diff, axis=-1)  # (M,G)
        min_dist = np.sqrt(np.min(d2, axis=0)).astype(np.float32)  # (G,)
        loss_sum += min_dist

    return loss_sum / (9.0 * float(pix_per_m))


def compute_pair_best_actions_worker(
    zarr_path_str: str,
    size_idx: int,
    density_idx: int,
    cloth_size: float,
    cloth_density: float,
    alphas: np.ndarray,
    only_valid: bool,
    transformer_low: Tuple[float, float],
    transformer_high: Tuple[float, float],
    transformer_grid: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """
    Worker for selecting argmin actions for one (size_idx, density_idx) pair.
    """
    import zarr

    zarr_path = Path(zarr_path_str)
    traj_occu = zarr.open(str(zarr_path / "traj_occu"), mode="r")

    transformer = GridCoordTransformer(
        low=transformer_low,
        high=transformer_high,
        grid_shape=transformer_grid,
    )
    pix_per_m = transformer.pix_per_m

    # Goal points in world coordinates: (G,9,3)
    goal_points_world = np.stack(
        [get_cloth_goal_points(cloth_size=cloth_size, alpha=float(alpha)) for alpha in alphas],
        axis=0,
    ).astype(np.float32)
    goal_pix_flat = transformer.to_grid(goal_points_world[:, :, [1, 2]].reshape(-1, 2), clip=True)
    goal_pixs = goal_pix_flat.reshape(len(alphas), 9, 2).astype(np.float32)

    valid_path = zarr_path / "is_valid"
    if only_valid and valid_path.exists():
        valid_pair = np.asarray(
            zarr.open(str(valid_path), mode="r")[size_idx, density_idx],
            dtype=bool,
        )  # (8,16,16,16)
    else:
        valid_pair = np.ones(traj_occu.shape[2:6], dtype=bool)

    candidate_action_coords = np.argwhere(valid_pair)
    if candidate_action_coords.shape[0] == 0:
        raise RuntimeError(f"No candidate actions for pair ({size_idx},{density_idx}).")

    best_losses = np.full((len(alphas),), np.inf, dtype=np.float32)
    best_action_coords = np.zeros((len(alphas), 4), dtype=np.int32)

    for action_coord in candidate_action_coords:
        action_coord_t = tuple(int(x) for x in action_coord.tolist())  # (4,)
        traj_img = traj_occu[(size_idx, density_idx) + action_coord_t]  # (9,H,W)
        losses = evaluate_action_for_all_goals(
            traj_img_9hw=np.asarray(traj_img),
            goal_pixs_g92=goal_pixs,
            pix_per_m=pix_per_m,
        )
        improved = losses < best_losses  # strict: deterministic first-tie behavior
        if np.any(improved):
            best_losses[improved] = losses[improved]
            best_action_coords[improved] = action_coord

    return {
        "pair": (size_idx, density_idx),
        "goal_points_world": goal_points_world,  # (G,9,3)
        "best_action_coords": best_action_coords,  # (G,4)
        "best_losses": best_losses,  # (G,)
        "cloth_size": np.array([cloth_size], dtype=np.float32),
        "cloth_density": np.array([cloth_density], dtype=np.float32),
        "size_id": np.array([size_idx], dtype=np.int32),
        "density_id": np.array([density_idx], dtype=np.int32),
        "num_candidates": np.array([candidate_action_coords.shape[0]], dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IRP cloth zarr to goal-conditioned robomimic HDF5."
    )
    default_workers = min(16, max(1, (os.cpu_count() or 1)))
    parser.add_argument("--input", type=str, required=True, help="Path to irp_cloth.zarr or irp_cloth.zarr.tar")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 file path")
    parser.add_argument("--num-goals", type=int, required=True, help="Number of sampled goals (alpha linspace in [0,1])")
    parser.add_argument(
        "--only-valid",
        dest="only_valid",
        action="store_true",
        default=True,
        help="Use only is_valid actions in argmin search (default: enabled)",
    )
    parser.add_argument(
        "--include-invalid",
        dest="only_valid",
        action="store_false",
        help="Search over all actions (including invalid entries)",
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
        help=f"Parallel workers for pair-level action search (default: {default_workers})",
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

    traj_path = zarr_path / "traj_occu"
    if not traj_path.exists():
        raise KeyError(f"Expected array path not found: {traj_path}")

    traj_occu = zarr.open(str(traj_path), mode="r")
    _, dim_samples, dim_to_axis = load_dim_metadata(zarr_path, zarr)

    size_axis = dim_to_axis["cloth_size"]
    density_axis = dim_to_axis["cloth_density"]
    duration_axis = dim_to_axis["duration"]
    gy1_axis = dim_to_axis["gy1"]
    gz1_axis = dim_to_axis["gz1"]
    gy2_axis = dim_to_axis["gy2"]

    # Cloth eval transformer defaults (obs_topdown=False path).
    transformer_low = (-0.1, -0.7)
    transformer_high = (1.8, 1.1)
    transformer_grid = (256, 256)

    size_count = traj_occu.shape[size_axis]
    density_count = traj_occu.shape[density_axis]
    if size_count != 4 or density_count != 4:
        print(
            f"Warning: expected 4x4 size/density grid, got {size_count}x{density_count}. "
            "Proceeding with discovered shape."
        )

    # Build alpha goals.
    all_alphas = np.linspace(0.0, 1.0, args.num_goals, dtype=np.float32)
    if args.max_demos is not None:
        goals_needed = int(np.ceil(args.max_demos / max(1, size_count * density_count)))
        num_goals_compute = min(args.num_goals, goals_needed)
    else:
        num_goals_compute = args.num_goals
    alphas = all_alphas[:num_goals_compute]

    valid_path = zarr_path / "is_valid"
    if args.only_valid and not valid_path.exists():
        print("Warning: is_valid not found, falling back to include all actions.")

    # Precompute best action per pair and goal.
    pair_indices = [(si, di) for si in range(size_count) for di in range(density_count)]
    pair_results: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    if args.num_workers == 1:
        for size_idx, density_idx in tqdm(pair_indices, desc="Selecting actions per param pair", unit="pair"):
            cloth_size = float(dim_samples["cloth_size"][size_idx])
            cloth_density = float(dim_samples["cloth_density"][density_idx])
            result = compute_pair_best_actions_worker(
                zarr_path_str=str(zarr_path),
                size_idx=size_idx,
                density_idx=density_idx,
                cloth_size=cloth_size,
                cloth_density=cloth_density,
                alphas=alphas,
                only_valid=args.only_valid,
                transformer_low=transformer_low,
                transformer_high=transformer_high,
                transformer_grid=transformer_grid,
            )
            pair_results[result["pair"]] = result
    else:
        max_workers = min(args.num_workers, len(pair_indices))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for size_idx, density_idx in pair_indices:
                cloth_size = float(dim_samples["cloth_size"][size_idx])
                cloth_density = float(dim_samples["cloth_density"][density_idx])
                fut = executor.submit(
                    compute_pair_best_actions_worker,
                    str(zarr_path),
                    size_idx,
                    density_idx,
                    cloth_size,
                    cloth_density,
                    alphas,
                    args.only_valid,
                    transformer_low,
                    transformer_high,
                    transformer_grid,
                )
                futures.append(fut)

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Selecting actions per param pair", unit="pair"):
                result = fut.result()
                pair_results[result["pair"]] = result

    # Write demos in goal-major order.
    h5_comp = get_h5_compression_kwargs(args.compression, args.compression_level)
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    demos_target = num_goals_compute * size_count * density_count
    if args.max_demos is not None:
        demos_target = min(demos_target, args.max_demos)

    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")

        f.attrs["source_zarr"] = str(zarr_path)
        f.attrs["num_goals_requested"] = int(args.num_goals)
        f.attrs["num_goals_used"] = int(num_goals_compute)
        f.attrs["only_valid"] = int(args.only_valid)
        f.attrs["num_demos"] = int(demos_target)
        f.attrs["action_keys"] = "duration,gy1,gz1,gy2"
        f.attrs["goal_sampler"] = "alpha_linspace"
        f.attrs["goal_major_order"] = 1
        f.attrs["num_workers"] = int(args.num_workers)

        demo_id = 0
        write_bar = tqdm(total=demos_target, desc="Writing demos", unit="demo")
        for goal_id in range(num_goals_compute):
            for size_idx in range(size_count):
                for density_idx in range(density_count):
                    if demo_id >= demos_target:
                        break

                    result = pair_results[(size_idx, density_idx)]
                    goal_points = result["goal_points_world"][goal_id].astype(np.float32)  # (9,3)
                    action_coord = result["best_action_coords"][goal_id]  # (4,)
                    best_loss = float(result["best_losses"][goal_id])

                    # Convert selected action indices to action values.
                    duration_idx = int(action_coord[0])
                    gy1_idx = int(action_coord[1])
                    gz1_idx = int(action_coord[2])
                    gy2_idx = int(action_coord[3])
                    action_values = np.array(
                        [
                            dim_samples["duration"][duration_idx],
                            dim_samples["gy1"][gy1_idx],
                            dim_samples["gz1"][gz1_idx],
                            dim_samples["gy2"][gy2_idx],
                        ],
                        dtype=np.float32,
                    )

                    grid_index = np.array(
                        [size_idx, density_idx, duration_idx, gy1_idx, gz1_idx, gy2_idx],
                        dtype=np.int32,
                    )

                    demo_key = f"demo_{demo_id}"
                    demo_group = data_group.create_group(demo_key)
                    obs_group = demo_group.create_group("obs")

                    obs_group.create_dataset("goal_points", data=goal_points[None, ...], dtype=np.float32, **h5_comp)
                    obs_group.create_dataset("param_size", data=result["cloth_size"], dtype=np.float32)
                    obs_group.create_dataset("param_density", data=result["cloth_density"], dtype=np.float32)
                    obs_group.create_dataset("param_size_id", data=result["size_id"], dtype=np.int32)
                    obs_group.create_dataset("param_density_id", data=result["density_id"], dtype=np.int32)

                    demo_group.create_dataset("actions", data=action_values[None, ...], dtype=np.float32)

                    # Compatibility fields.
                    demo_group.create_dataset("param_size", data=result["cloth_size"], dtype=np.float32)
                    demo_group.create_dataset("param_density", data=result["cloth_density"], dtype=np.float32)
                    demo_group.create_dataset("grid_index", data=grid_index, dtype=np.int32)
                    demo_group.create_dataset("best_img_loss_m", data=np.array([best_loss], dtype=np.float32), dtype=np.float32)

                    demo_group.attrs["num_samples"] = 1
                    demo_group.attrs["goal_id"] = int(goal_id)
                    demo_group.attrs["goal_alpha"] = float(alphas[goal_id])

                    demo_id += 1
                    write_bar.update(1)
                if demo_id >= demos_target:
                    break
            if demo_id >= demos_target:
                break
        write_bar.close()

    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    main()
