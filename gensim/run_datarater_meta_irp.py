#!/usr/bin/env python3
"""
Meta-gradient DataRater pipeline for IRP HDF5 datasets.

Goal:
- Inner model predicts action from goal positions.
- DataRater assigns per-sample weights on source data.
- DataRater is optimized to minimize target validation loss after inner updates.
- Outputs a 2D score landscape over (param_size_id, param_density_id).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.patches import Rectangle
from torch.func import functional_call


Pair = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DataRater meta-learning on IRP HDF5 datasets.")
    parser.add_argument("--source-hdf5", type=str, required=True, help="Source dataset (train/meta-train)")
    parser.add_argument("--target-hdf5", type=str, required=True, help="Target dataset (meta-val)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    parser.add_argument(
        "--target-train-ratio",
        type=float,
        default=0.0,
        help="Fraction of target samples in the mixed inner-train set (e.g. 0.1 for 90/10 source-target)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=0,
        help="Total samples in mixed inner-train set (<=0 uses source dataset size)",
    )
    parser.add_argument(
        "--source-replace",
        action="store_true",
        help="Sample source portion with replacement when constructing mixed train set",
    )
    parser.add_argument(
        "--target-no-replace",
        action="store_true",
        help="Disable replacement for target portion when constructing mixed train set",
    )

    parser.add_argument("--outer-steps", type=int, default=500, help="Meta (outer) steps")
    parser.add_argument("--inner-steps", type=int, default=1, help="Inner update steps per outer step")
    parser.add_argument("--n-models", type=int, default=10, help="Number of parallel functional inner models")
    parser.add_argument("--inner-lr", type=float, default=0.005, help="Inner learning rate")
    parser.add_argument("--meta-lr", type=float, default=0.01, help="DataRater optimizer learning rate")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Source minibatch size for inner updates (<=0 means full batch)",
    )
    parser.add_argument(
        "--score-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature over DataRater scores",
    )
    parser.add_argument(
        "--weight-uniform-mix",
        type=float,
        default=0.0,
        help="Mix softmax weights with uniform weights to prevent collapse (0..1)",
    )
    parser.add_argument(
        "--score-clip",
        type=float,
        default=0.0,
        help="If > 0, clip DataRater logits to [-score_clip, score_clip] before softmax",
    )
    parser.add_argument(
        "--meta-entropy-reg",
        type=float,
        default=0.0,
        help="Outer objective entropy bonus weight (encourages non-collapsed weights)",
    )
    parser.add_argument(
        "--meta-score-l2-reg",
        type=float,
        default=0.0,
        help="Outer objective L2 regularization on DataRater logits",
    )
    parser.add_argument("--inner-hidden", type=int, default=256, help="Inner model hidden dim")
    parser.add_argument("--datarater-hidden", type=int, default=64, help="DataRater hidden dim")
    parser.add_argument("--normalize", action="store_true", help="Apply z-score normalization to x and y")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--log-every", type=int, default=10, help="Print frequency for outer loop")
    parser.add_argument("--top-k-pairs", type=int, default=8, help="How many top parameter pairs to print")
    return parser.parse_args()


def sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    return sorted(list(data_group.keys()), key=lambda x: int(x.split("_")[-1]))


def load_hdf5_samples(hdf5_path: Path) -> Dict[str, np.ndarray]:
    x_list = []
    y_list = []
    size_ids = []
    density_ids = []
    size_vals = []
    density_vals = []
    goal_ids = []

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Missing 'data' group in {hdf5_path}")

        for demo_key in sorted_demo_keys(f["data"]):
            demo = f["data"][demo_key]
            if "obs" not in demo:
                raise KeyError(f"Missing obs group in {demo_key}")
            if "goal_points" not in demo["obs"]:
                raise KeyError(f"Missing obs/goal_points in {demo_key}")
            if "actions" not in demo:
                raise KeyError(f"Missing actions in {demo_key}")

            x = np.asarray(demo["obs/goal_points"])[0].astype(np.float32).reshape(-1)
            y = np.asarray(demo["actions"])[0].astype(np.float32).reshape(-1)

            size_id = int(np.asarray(demo["obs/param_size_id"])[0])
            density_id = int(np.asarray(demo["obs/param_density_id"])[0])
            size_val = float(np.asarray(demo["obs/param_size"])[0])
            density_val = float(np.asarray(demo["obs/param_density"])[0])
            goal_id = int(demo.attrs["goal_id"]) if "goal_id" in demo.attrs else -1

            x_list.append(x)
            y_list.append(y)
            size_ids.append(size_id)
            density_ids.append(density_id)
            size_vals.append(size_val)
            density_vals.append(density_val)
            goal_ids.append(goal_id)

    if len(x_list) == 0:
        raise RuntimeError(f"No demos found in {hdf5_path}")

    return {
        "x": np.stack(x_list, axis=0),
        "y": np.stack(y_list, axis=0),
        "size_id": np.asarray(size_ids, dtype=np.int32),
        "density_id": np.asarray(density_ids, dtype=np.int32),
        "size_val": np.asarray(size_vals, dtype=np.float32),
        "density_val": np.asarray(density_vals, dtype=np.float32),
        "goal_id": np.asarray(goal_ids, dtype=np.int32),
    }


def zscore_train_apply(train_arr: np.ndarray, val_arr: np.ndarray, eps: float = 1e-6):
    mu = train_arr.mean(axis=0, keepdims=True)
    sigma = train_arr.std(axis=0, keepdims=True)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return (train_arr - mu) / sigma, (val_arr - mu) / sigma, mu, sigma


def sample_indices(n_total: int, n_pick: int, replace: bool, rs: np.random.RandomState) -> np.ndarray:
    if n_pick <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_total <= 0:
        raise ValueError("Cannot sample from empty dataset")
    if (not replace) and n_pick > n_total:
        raise ValueError(
            f"Requested {n_pick} samples without replacement from dataset of size {n_total}. "
            "Use replacement or reduce requested size."
        )
    return rs.choice(n_total, size=n_pick, replace=replace).astype(np.int64)


def build_mixed_train_data(
    src: Dict[str, np.ndarray],
    tgt: Dict[str, np.ndarray],
    target_train_ratio: float,
    train_size: int,
    seed: int,
    source_replace: bool,
    target_replace: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    if not (0.0 <= target_train_ratio < 1.0):
        raise ValueError(f"--target-train-ratio must be in [0,1), got {target_train_ratio}")

    n_src_total = int(src["x"].shape[0])
    n_tgt_total = int(tgt["x"].shape[0])
    if train_size <= 0:
        n_total = n_src_total
    else:
        n_total = int(train_size)
    if n_total <= 0:
        raise ValueError("Mixed train size must be positive.")

    n_tgt = int(round(n_total * target_train_ratio))
    if target_train_ratio > 0.0:
        n_tgt = max(1, n_tgt)
    n_tgt = min(n_tgt, n_total)
    n_src = n_total - n_tgt

    rs = np.random.RandomState(seed=seed + 12345)
    src_idx = sample_indices(n_src_total, n_src, replace=source_replace, rs=rs)
    tgt_idx = sample_indices(n_tgt_total, n_tgt, replace=target_replace, rs=rs)

    def cat_field(k: str) -> np.ndarray:
        a = src[k][src_idx] if n_src > 0 else src[k][:0]
        b = tgt[k][tgt_idx] if n_tgt > 0 else tgt[k][:0]
        return np.concatenate([a, b], axis=0)

    train = {
        "x": cat_field("x").astype(np.float32),
        "y": cat_field("y").astype(np.float32),
        "size_id": cat_field("size_id").astype(np.int32),
        "density_id": cat_field("density_id").astype(np.int32),
        "size_val": cat_field("size_val").astype(np.float32),
        "density_val": cat_field("density_val").astype(np.float32),
        "goal_id": cat_field("goal_id").astype(np.int32),
        "origin": np.concatenate(
            [
                np.zeros((n_src,), dtype=np.int32),  # 0 = source
                np.ones((n_tgt,), dtype=np.int32),   # 1 = target
            ],
            axis=0,
        ),
    }
    info = {
        "n_total": n_total,
        "n_source_part": n_src,
        "n_target_part": n_tgt,
        "target_train_ratio_effective": float(n_tgt / max(1, n_total)),
        "source_replace": bool(source_replace),
        "target_replace": bool(target_replace),
    }
    return train, info


def compute_weights_from_scores(
    scores: torch.Tensor,
    score_temp: float,
    uniform_mix: float = 0.0,
    score_clip: float = 0.0,
) -> torch.Tensor:
    if score_temp <= 0:
        raise ValueError(f"score_temp must be positive, got {score_temp}")
    if not (0.0 <= uniform_mix <= 1.0):
        raise ValueError(f"uniform_mix must be in [0,1], got {uniform_mix}")

    logits = scores
    if score_clip > 0:
        logits = torch.clamp(logits, min=-score_clip, max=score_clip)

    weights = torch.softmax(logits / score_temp, dim=0)
    if uniform_mix > 0:
        n = max(1, int(weights.shape[0]))
        weights = (1.0 - uniform_mix) * weights + uniform_mix * (1.0 / n)
    return weights


class ActionRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class DataRater(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(x_dim + y_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(torch.cat((x, y), dim=1))))  # (N,1)


def init_functional_params(template: nn.Module, n_models: int, device: torch.device):
    params_list = []
    buffers_list = []
    for _ in range(n_models):
        m = ActionRegressor(
            input_dim=template.fc1.in_features,
            output_dim=template.fc3.out_features,
            hidden_dim=template.fc1.out_features,
        ).to(device)
        params = {n: p.detach().clone().requires_grad_(True) for n, p in m.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in m.named_buffers()}
        params_list.append(params)
        buffers_list.append(buffers)
    return params_list, buffers_list


def update_inner_model(
    functional_params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    template: nn.Module,
    datarater: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    inner_steps: int,
    inner_lr: float,
    batch_size: int,
    score_temp: float,
    weight_uniform_mix: float,
    score_clip: float,
) -> Dict[str, torch.Tensor]:
    n = x_train.shape[0]
    keys = list(functional_params.keys())

    for _ in range(inner_steps):
        if batch_size <= 0 or batch_size >= n:
            xb = x_train
            yb = y_train
        else:
            idx = torch.randint(0, n, (batch_size,), device=x_train.device)
            xb = x_train[idx]
            yb = y_train[idx]

        scores = datarater(xb, yb).squeeze(-1)  # (B,)
        weights = compute_weights_from_scores(
            scores=scores,
            score_temp=score_temp,
            uniform_mix=weight_uniform_mix,
            score_clip=score_clip,
        )

        preds = functional_call(template, (functional_params, buffers), (xb,))  # (B, action_dim)
        per_sample_loss = ((preds - yb) ** 2).mean(dim=1)  # (B,)
        loss = torch.sum(weights * per_sample_loss)

        vals = [functional_params[k] for k in keys]
        grads = torch.autograd.grad(loss, vals, create_graph=True)
        functional_params = {k: functional_params[k] - inner_lr * g for k, g in zip(keys, grads)}
    return functional_params


def build_pair_landscape(
    size_ids: np.ndarray,
    density_ids: np.ndarray,
    size_vals: np.ndarray,
    density_vals: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
):
    pair_to_idx = defaultdict(list)
    pair_to_size_val = {}
    pair_to_density_val = {}
    for i, (sid, did) in enumerate(zip(size_ids, density_ids)):
        pair = (int(sid), int(did))
        pair_to_idx[pair].append(i)
        pair_to_size_val[pair] = float(size_vals[i])
        pair_to_density_val[pair] = float(density_vals[i])

    uniq_size = sorted(set(int(x) for x in size_ids.tolist()))
    uniq_density = sorted(set(int(x) for x in density_ids.tolist()))
    s_map = {sid: i for i, sid in enumerate(uniq_size)}
    d_map = {did: i for i, did in enumerate(uniq_density)}

    score_mat = np.full((len(uniq_size), len(uniq_density)), np.nan, dtype=np.float32)
    weight_mat = np.full((len(uniq_size), len(uniq_density)), np.nan, dtype=np.float32)
    count_mat = np.zeros((len(uniq_size), len(uniq_density)), dtype=np.int32)

    pair_rows = []
    for pair, idxs in pair_to_idx.items():
        sid, did = pair
        ii = s_map[sid]
        jj = d_map[did]
        idx_arr = np.asarray(idxs, dtype=np.int64)
        mean_score = float(np.mean(scores[idx_arr]))
        mean_weight = float(np.mean(weights[idx_arr]))
        score_mat[ii, jj] = mean_score
        weight_mat[ii, jj] = mean_weight
        count_mat[ii, jj] = int(len(idxs))
        pair_rows.append(
            {
                "size_id": sid,
                "density_id": did,
                "size_value": pair_to_size_val[pair],
                "density_value": pair_to_density_val[pair],
                "count": int(len(idxs)),
                "mean_score": mean_score,
                "mean_weight": mean_weight,
            }
        )

    pair_rows_sorted = sorted(pair_rows, key=lambda r: r["mean_weight"], reverse=True)
    return uniq_size, uniq_density, score_mat, weight_mat, count_mat, pair_rows_sorted


def save_landscape_plot(
    out_png: Path,
    uniq_size: List[int],
    uniq_density: List[int],
    score_mat: np.ndarray,
    weight_mat: np.ndarray,
    count_mat: np.ndarray,
    target_pair: Pair,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ims = []
    titles = ["Mean DataRater Score", "Mean DataRater Weight"]
    mats = [score_mat, weight_mat]
    cmaps = ["viridis", "magma"]

    for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ims.append(im)
        ax.set_title(title)
        ax.set_xlabel("density_id")
        ax.set_ylabel("size_id")
        ax.set_xticks(np.arange(len(uniq_density)))
        ax.set_xticklabels(uniq_density)
        ax.set_yticks(np.arange(len(uniq_size)))
        ax.set_yticklabels(uniq_size)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.3f}\n(n={count_mat[i, j]})",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        if target_pair is not None:
            target_sid, target_did = target_pair
            if target_sid in uniq_size and target_did in uniq_density:
                ti = uniq_size.index(target_sid)
                tj = uniq_density.index(target_did)
                ax.add_patch(Rectangle((tj - 0.5, ti - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=2))

    fig.colorbar(ims[0], ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(ims[1], ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("DataRater Landscape over (size_id, density_id)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.score_temperature <= 0:
        raise ValueError("--score-temperature must be positive")
    if not (0.0 <= args.weight_uniform_mix <= 1.0):
        raise ValueError("--weight-uniform-mix must be in [0,1]")
    if args.score_clip < 0:
        raise ValueError("--score-clip must be >= 0")
    if args.meta_entropy_reg < 0:
        raise ValueError("--meta-entropy-reg must be >= 0")
    if args.meta_score_l2_reg < 0:
        raise ValueError("--meta-score-l2-reg must be >= 0")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    source_path = Path(args.source_hdf5).expanduser().resolve()
    target_path = Path(args.target_hdf5).expanduser().resolve()
    src = load_hdf5_samples(source_path)
    tgt = load_hdf5_samples(target_path)

    train_data, mix_info = build_mixed_train_data(
        src=src,
        tgt=tgt,
        target_train_ratio=float(args.target_train_ratio),
        train_size=int(args.train_size),
        seed=int(args.seed),
        source_replace=bool(args.source_replace),
        target_replace=not bool(args.target_no_replace),
    )
    x_train_np = train_data["x"].astype(np.float32)
    y_train_np = train_data["y"].astype(np.float32)
    x_val_np = tgt["x"].astype(np.float32)
    y_val_np = tgt["y"].astype(np.float32)

    print(
        "Mixed inner-train composition: "
        f"source={mix_info['n_source_part']} target={mix_info['n_target_part']} "
        f"(target ratio={mix_info['target_train_ratio_effective']:.3f})"
    )

    norm_stats = {}
    x_mu = x_std = y_mu = y_std = None
    if args.normalize:
        x_train_np, x_val_np, x_mu, x_std = zscore_train_apply(x_train_np, x_val_np)
        y_train_np, y_val_np, y_mu, y_std = zscore_train_apply(y_train_np, y_val_np)
        norm_stats = {
            "x_mean": x_mu.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mu.tolist(),
            "y_std": y_std.tolist(),
        }

    device = torch.device(args.device)
    x_train = torch.from_numpy(x_train_np).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_val = torch.from_numpy(x_val_np).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)

    # For landscape plotting: score each source sample exactly once.
    x_src_eval_np = src["x"].astype(np.float32)
    y_src_eval_np = src["y"].astype(np.float32)
    if args.normalize:
        x_src_eval_np = (x_src_eval_np - x_mu) / x_std
        y_src_eval_np = (y_src_eval_np - y_mu) / y_std
    x_src_eval = torch.from_numpy(x_src_eval_np).to(device)
    y_src_eval = torch.from_numpy(y_src_eval_np).to(device)

    x_dim = int(x_train.shape[1])
    y_dim = int(y_train.shape[1])

    inner_template = ActionRegressor(input_dim=x_dim, output_dim=y_dim, hidden_dim=args.inner_hidden).to(device)
    datarater = DataRater(x_dim=x_dim, y_dim=y_dim, hidden_dim=args.datarater_hidden).to(device)
    meta_opt = optim.Adam(datarater.parameters(), lr=args.meta_lr)

    params_list, buffers_list = init_functional_params(inner_template, args.n_models, device)
    loss_fn = nn.MSELoss()

    meta_loss_hist = []
    meta_val_hist = []
    entropy_hist = []
    score_l2_hist = []
    for k in range(args.outer_steps):
        meta_opt.zero_grad()
        meta_losses = []
        next_params_list = []

        for i in range(args.n_models):
            updated_params = update_inner_model(
                functional_params=params_list[i],
                buffers=buffers_list[i],
                template=inner_template,
                datarater=datarater,
                x_train=x_train,
                y_train=y_train,
                inner_steps=args.inner_steps,
                inner_lr=args.inner_lr,
                batch_size=args.batch_size,
                score_temp=args.score_temperature,
                weight_uniform_mix=args.weight_uniform_mix,
                score_clip=args.score_clip,
            )
            preds_val = functional_call(inner_template, (updated_params, buffers_list[i]), (x_val,))
            val_loss = loss_fn(preds_val, y_val)
            meta_losses.append(val_loss)
            next_params_list.append(updated_params)

        meta_val_loss = torch.stack(meta_losses).mean()
        meta_loss = meta_val_loss
        reg_entropy = torch.tensor(0.0, device=device)
        reg_score_l2 = torch.tensor(0.0, device=device)

        if args.meta_entropy_reg > 0 or args.meta_score_l2_reg > 0:
            train_scores_full = datarater(x_train, y_train).squeeze(-1)
            train_weights_full = compute_weights_from_scores(
                scores=train_scores_full,
                score_temp=args.score_temperature,
                uniform_mix=args.weight_uniform_mix,
                score_clip=args.score_clip,
            )
            if args.meta_entropy_reg > 0:
                reg_entropy = -torch.sum(train_weights_full * torch.log(train_weights_full + 1e-12))
                meta_loss = meta_loss - args.meta_entropy_reg * reg_entropy
            if args.meta_score_l2_reg > 0:
                reg_score_l2 = torch.mean(train_scores_full ** 2)
                meta_loss = meta_loss + args.meta_score_l2_reg * reg_score_l2

        meta_loss.backward()
        meta_opt.step()
        meta_loss_hist.append(float(meta_loss.detach().cpu().item()))
        meta_val_hist.append(float(meta_val_loss.detach().cpu().item()))
        entropy_hist.append(float(reg_entropy.detach().cpu().item()))
        score_l2_hist.append(float(reg_score_l2.detach().cpu().item()))

        params_list = [{k: p.detach().requires_grad_(True) for k, p in fp.items()} for fp in next_params_list]
        if (k + 1) % args.log_every == 0 or k == 0:
            print(
                f"[{k+1:04d}/{args.outer_steps}] "
                f"meta_val_loss={meta_val_loss.item():.6f} "
                f"meta_obj={meta_loss.item():.6f} "
                f"entropy={reg_entropy.item():.6f} "
                f"score_l2={reg_score_l2.item():.6f}"
            )

    datarater.eval()
    with torch.no_grad():
        # Diagnostics on mixed inner-train distribution.
        train_scores_t = datarater(x_train, y_train).squeeze(-1)
        train_weights_t = compute_weights_from_scores(
            scores=train_scores_t,
            score_temp=args.score_temperature,
            uniform_mix=args.weight_uniform_mix,
            score_clip=args.score_clip,
        )
        # Landscape scores on source-only distribution (each source sample scored once).
        source_scores_t = datarater(x_src_eval, y_src_eval).squeeze(-1)
        source_weights_t = compute_weights_from_scores(
            scores=source_scores_t,
            score_temp=args.score_temperature,
            uniform_mix=args.weight_uniform_mix,
            score_clip=args.score_clip,
        )
        final_meta_val = loss_fn(
            functional_call(inner_template, (params_list[0], buffers_list[0]), (x_val,)),
            y_val,
        ).item()

    source_scores = source_scores_t.detach().cpu().numpy()
    source_weights = source_weights_t.detach().cpu().numpy()

    target_pairs = sorted(set(zip(tgt["size_id"].tolist(), tgt["density_id"].tolist())))
    target_pair = target_pairs[0] if len(target_pairs) > 0 else None
    if len(target_pairs) > 1:
        print(f"Warning: target dataset has multiple parameter pairs: {target_pairs}. Highlighting first one.")

    # Landscape is source-only: each source sample contributes exactly once.
    uniq_size, uniq_density, score_mat, weight_mat, count_mat, pair_rows_sorted = build_pair_landscape(
        size_ids=src["size_id"],
        density_ids=src["density_id"],
        size_vals=src["size_val"],
        density_vals=src["density_val"],
        scores=source_scores,
        weights=source_weights,
    )

    print("")
    print("Top parameter pairs by mean DataRater weight (source-only landscape):")
    for i, row in enumerate(pair_rows_sorted[: min(args.top_k_pairs, len(pair_rows_sorted))], start=1):
        print(
            f"{i:>2}. (size_id={row['size_id']}, density_id={row['density_id']}) "
            f"weight={row['mean_weight']:.6f}, score={row['mean_score']:.6f}, count={row['count']}"
        )

    out_png = out_dir / "datarater_landscape_size_density.png"
    save_landscape_plot(
        out_png=out_png,
        uniq_size=uniq_size,
        uniq_density=uniq_density,
        score_mat=score_mat,
        weight_mat=weight_mat,
        count_mat=count_mat,
        target_pair=target_pair,
    )

    out_csv = out_dir / "datarater_pair_scores.csv"
    with open(out_csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "size_id",
                "density_id",
                "size_value",
                "density_value",
                "count",
                "mean_score",
                "mean_weight",
            ],
        )
        writer.writeheader()
        for row in pair_rows_sorted:
            writer.writerow(row)

    out_meta = out_dir / "run_summary.json"
    final_entropy_source = float(-(source_weights_t * torch.log(source_weights_t + 1e-12)).sum().cpu().item())
    final_effective_samples_source = float(np.exp(final_entropy_source))
    final_entropy_train_mix = float(-(train_weights_t * torch.log(train_weights_t + 1e-12)).sum().cpu().item())
    final_effective_samples_train_mix = float(np.exp(final_entropy_train_mix))
    summary = {
        "source_hdf5": str(source_path),
        "target_hdf5": str(target_path),
        "n_source_dataset_samples": int(src["x"].shape[0]),
        "n_target_dataset_samples": int(tgt["x"].shape[0]),
        "n_train_samples": int(x_train.shape[0]),
        "n_target_samples": int(x_val.shape[0]),
        "x_dim": x_dim,
        "y_dim": y_dim,
        "target_pair": list(target_pair) if target_pair is not None else None,
        "final_meta_val_loss": float(final_meta_val),
        "landscape_source_only": True,
        "final_weight_entropy_source": final_entropy_source,
        "final_effective_samples_source": final_effective_samples_source,
        "final_weight_entropy_train_mix": final_entropy_train_mix,
        "final_effective_samples_train_mix": final_effective_samples_train_mix,
        "meta_loss_history": meta_loss_hist,
        "meta_val_history": meta_val_hist,
        "entropy_reg_history": entropy_hist,
        "score_l2_reg_history": score_l2_hist,
        "mix_info": mix_info,
        "normalize": bool(args.normalize),
        "norm_stats": norm_stats,
        "args": vars(args),
    }
    out_meta.write_text(json.dumps(summary, indent=2))

    print("")
    print(f"Saved plot: {out_png}")
    print(f"Saved pair CSV: {out_csv}")
    print(f"Saved summary: {out_meta}")


if __name__ == "__main__":
    main()
