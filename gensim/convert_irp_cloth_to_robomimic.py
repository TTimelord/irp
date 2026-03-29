#!/usr/bin/env python3
"""
Convert IRP cloth dataset (zarr) to a robomimic-style HDF5 dataset.

Each output demo contains:
- data/demo_i/obs/target_trajectory: (1, 9, 256, 256)
- data/demo_i/obs/param_size: (1,)
- data/demo_i/obs/param_density: (1,)
- data/demo_i/obs/param_size_id: (1,)
- data/demo_i/obs/param_density_id: (1,)
- data/demo_i/actions: (1, 4), ordered as [duration, gy1, gz1, gy2]
- data/demo_i/param_size: (1,)
- data/demo_i/param_density: (1,)

Notes:
- This script does NOT import code from the IRP repo.
- It reads only from zarr metadata / arrays directly.
- Input can be either:
  1) extracted zarr directory (.../irp_cloth.zarr), or
  2) tar archive (.../irp_cloth.zarr.tar), which will be extracted first.
"""

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IRP cloth zarr dataset to robomimic HDF5 format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to irp_cloth.zarr or irp_cloth.zarr.tar",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--param-indices",
        type=int,
        nargs=2,
        default=None,
        metavar=("SIZE_IDX", "DENSITY_IDX"),
        help=(
            "Optional filter for a single cloth parameter pair using indices, "
            "e.g. --param-indices 2 3."
        ),
    )
    parser.add_argument(
        "--only-valid",
        dest="only_valid",
        action="store_true",
        default=True,
        help="Keep only entries where is_valid == True (default: enabled)",
    )
    parser.add_argument(
        "--include-invalid",
        dest="only_valid",
        action="store_false",
        help="Also include entries where is_valid == False",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle selected grid entries before assigning demo IDs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used with --shuffle)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Optional cap on number of demos written",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression for large arrays",
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
        help=(
            "If --input is .tar, extract it under this directory. "
            "Default: same directory as tar."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    return parser.parse_args()


def resolve_zarr_path(input_path: Path, extract_dir: Path = None) -> Path:
    if input_path.is_dir():
        if input_path.name.endswith(".zarr"):
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
        raise ValueError(f"Expected a .zarr directory, got: {input_path}")

    if input_path.is_file() and input_path.name.endswith(".zarr.tar"):
        preferred_parent = extract_dir if extract_dir is not None else input_path.parent
        target_parent = choose_writable_extract_parent(preferred_parent)
        target_parent.mkdir(parents=True, exist_ok=True)

        zarr_dir_name = input_path.name[:-4]  # strip ".tar"
        zarr_path = target_parent / zarr_dir_name

        ensure_complete_zarr_from_tar(input_path, zarr_path)
        return zarr_path

    raise ValueError(
        "Input must be either an extracted .zarr directory or a .zarr.tar file."
    )


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


def load_dim_metadata(zarr_path: Path, zarr) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int]]:
    dim_keys_arr = zarr.open(str(zarr_path / "dim_keys"), mode="r")
    dim_keys = [str(x) for x in dim_keys_arr[:].tolist()]
    missing = [k for k in REQUIRED_DIM_KEYS if k not in dim_keys]
    if missing:
        raise KeyError(
            f"Missing required dim_keys: {missing}. Found dim_keys={dim_keys}"
        )

    dim_samples = {
        k: np.asarray(zarr.open(str(zarr_path / "dim_samples" / k), mode="r")[:], dtype=np.float32)
        for k in REQUIRED_DIM_KEYS
    }
    dim_to_axis = {k: dim_keys.index(k) for k in REQUIRED_DIM_KEYS}
    return dim_keys, dim_samples, dim_to_axis


def build_selection_mask(
    valid_mask: np.ndarray,
    dim_to_axis: Dict[str, int],
    param_indices: Tuple[int, int] = None,
) -> np.ndarray:
    mask = valid_mask.copy()
    size_axis = dim_to_axis["cloth_size"]
    density_axis = dim_to_axis["cloth_density"]
    if param_indices is None:
        return mask

    size_idx, density_idx = param_indices
    if size_idx < 0 or size_idx >= mask.shape[size_axis]:
        raise ValueError(
            f"cloth_size index out of range: {size_idx}, valid [0, {mask.shape[size_axis] - 1}]"
        )
    if density_idx < 0 or density_idx >= mask.shape[density_axis]:
        raise ValueError(
            f"cloth_density index out of range: {density_idx}, valid [0, {mask.shape[density_axis] - 1}]"
        )

    selected_param_mask = np.zeros_like(mask, dtype=bool)
    indexer = [slice(None)] * mask.ndim
    indexer[size_axis] = size_idx
    indexer[density_axis] = density_idx
    selected_param_mask[tuple(indexer)] = True
    mask &= selected_param_mask
    return mask


def main() -> None:
    args = parse_args()
    import h5py
    from tqdm import tqdm
    import zarr
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    extract_dir = (
        Path(args.extract_dir).expanduser().resolve()
        if args.extract_dir is not None
        else None
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zarr_path = resolve_zarr_path(input_path, extract_dir=extract_dir)
    print(f"Using zarr dataset: {zarr_path}")

    traj_path = zarr_path / "traj_occu"
    if not traj_path.exists():
        raise KeyError(f"Expected array path not found: {traj_path}")
    traj_occu = zarr.open(str(traj_path), mode="r")

    dim_keys, dim_samples, dim_to_axis = load_dim_metadata(zarr_path, zarr)

    valid_path = zarr_path / "is_valid"
    if args.only_valid and valid_path.exists():
        valid_mask = np.asarray(zarr.open(str(valid_path), mode="r")[:], dtype=bool)
    else:
        if args.only_valid and not valid_path.exists():
            print("Warning: is_valid not found, falling back to include all entries.")
        valid_mask = np.ones(traj_occu.shape[: len(dim_keys)], dtype=bool)

    select_mask = build_selection_mask(
        valid_mask=valid_mask,
        dim_to_axis=dim_to_axis,
        param_indices=tuple(args.param_indices) if args.param_indices is not None else None,
    )

    selected_coords = np.argwhere(select_mask)
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(selected_coords)

    if args.max_demos is not None:
        selected_coords = selected_coords[: args.max_demos]

    n_demos = int(len(selected_coords))
    if n_demos == 0:
        raise RuntimeError(
            "No entries selected. Check --only-valid / --include-invalid and --param-indices."
        )

    print(f"Selected demos: {n_demos}")

    h5_comp = get_h5_compression_kwargs(args.compression, args.compression_level)
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")

        # lightweight metadata
        f.attrs["source_zarr"] = str(zarr_path)
        f.attrs["only_valid"] = int(args.only_valid)
        f.attrs["num_demos"] = n_demos
        f.attrs["action_keys"] = "duration,gy1,gz1,gy2"
        f.attrs["param_indices"] = (
            "all"
            if args.param_indices is None
            else f"{int(args.param_indices[0])},{int(args.param_indices[1])}"
        )

        duration_axis = dim_to_axis["duration"]
        gy1_axis = dim_to_axis["gy1"]
        gz1_axis = dim_to_axis["gz1"]
        gy2_axis = dim_to_axis["gy2"]
        size_axis = dim_to_axis["cloth_size"]
        density_axis = dim_to_axis["cloth_density"]

        for demo_i, coord in enumerate(
            tqdm(selected_coords, total=n_demos, desc="Writing demos", unit="demo")
        ):
            coord_tuple = tuple(int(x) for x in coord.tolist())

            traj = traj_occu[coord_tuple]  # (9, 256, 256), bool
            traj_u8 = np.asarray(traj, dtype=np.uint8)

            action = np.array(
                [
                    dim_samples["duration"][coord_tuple[duration_axis]],
                    dim_samples["gy1"][coord_tuple[gy1_axis]],
                    dim_samples["gz1"][coord_tuple[gz1_axis]],
                    dim_samples["gy2"][coord_tuple[gy2_axis]],
                ],
                dtype=np.float32,
            )

            param_size = np.array(
                [dim_samples["cloth_size"][coord_tuple[size_axis]]], dtype=np.float32
            )
            param_density = np.array(
                [dim_samples["cloth_density"][coord_tuple[density_axis]]],
                dtype=np.float32,
            )
            param_size_id = np.array([coord_tuple[size_axis]], dtype=np.int32)
            param_density_id = np.array([coord_tuple[density_axis]], dtype=np.int32)

            demo_key = f"demo_{demo_i}"
            demo_group = data_group.create_group(demo_key)
            obs_group = demo_group.create_group("obs")

            # T = 1 sample per demo
            obs_group.create_dataset(
                "target_trajectory",
                data=traj_u8[None, ...],  # (1, 9, 256, 256)
                **h5_comp,
            )
            obs_group.create_dataset("param_size", data=param_size, dtype=np.float32)
            obs_group.create_dataset("param_density", data=param_density, dtype=np.float32)
            obs_group.create_dataset("param_size_id", data=param_size_id, dtype=np.int32)
            obs_group.create_dataset(
                "param_density_id", data=param_density_id, dtype=np.int32
            )
            demo_group.create_dataset(
                "actions",
                data=action[None, ...],  # (1, 4)
                dtype=np.float32,
            )
            demo_group.create_dataset("param_size", data=param_size, dtype=np.float32)
            demo_group.create_dataset(
                "param_density", data=param_density, dtype=np.float32
            )
            demo_group.create_dataset(
                "grid_index",
                data=np.asarray(coord_tuple, dtype=np.int32),
                dtype=np.int32,
            )
            demo_group.attrs["num_samples"] = 1

    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    main()
