#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d
from figure8_dataset import Figure8ConditionalDiffusionDataset


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_training_config(run_dir):
    path = os.path.join(run_dir, "train_config.json")
    if not os.path.exists(path):
        raise RuntimeError(f"Missing config: {path}")
    with open(path, "r") as f:
        return json.load(f)


def find_checkpoint(run_dir, ckpt_name=None):
    if ckpt_name is not None:
        p = os.path.join(run_dir, ckpt_name)
        if not os.path.exists(p):
            raise RuntimeError(f"Checkpoint not found: {p}")
        return p

    for name in ["diffusion_ckpt_best.pt", "diffusion_ckpt_latest.pt", "diffusion_ckpt_final.pt"]:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p

    raise RuntimeError(f"No checkpoint found in {run_dir}")


def load_normalizer(run_dir):
    path = os.path.join(run_dir, "normalizer.npz")
    if not os.path.exists(path):
        raise RuntimeError(f"Missing normalizer: {path}")
    d = np.load(path)
    norm = {
        "obs_mean": d["obs_mean"].astype(np.float32),
        "obs_std": d["obs_std"].astype(np.float32),
        "act_mean": d["act_mean"].astype(np.float32),
        "act_std": d["act_std"].astype(np.float32),
    }
    if "cond_mean" in d and "cond_std" in d:
        norm["cond_mean"] = d["cond_mean"].astype(np.float32)
        norm["cond_std"] = d["cond_std"].astype(np.float32)
    return norm


def denormalize_act(x, norm):
    return x * norm["act_std"] + norm["act_mean"]


def build_agent_from_config(config, obs_dim, act_dim, cond_dim, device):
    horizon = int(config["horizon"])
    model_dim = int(config["model_dim"])
    dim_mult = list(config["dim_mult"])
    diffusion_steps = int(config["diffusion_steps"])
    ema_rate = float(config["ema_rate"])
    predict_noise = bool(config["predict_noise"])
    action_loss_weight = float(config.get("action_loss_weight", 1.0))

    traj_dim = obs_dim + act_dim + cond_dim

    nn_diffusion = JannerUNet1d(
        in_dim=traj_dim,
        model_dim=model_dim,
        emb_dim=model_dim,
        dim_mult=dim_mult,
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    ).to(device)

    fix_mask = torch.zeros((horizon, traj_dim), device=device)
    fix_mask[0, :obs_dim] = 1.0
    fix_mask[:, obs_dim + act_dim:] = 1.0

    loss_weight = torch.ones((horizon, traj_dim), device=device)
    loss_weight[:, obs_dim:obs_dim + act_dim] = action_loss_weight
    loss_weight[:, obs_dim + act_dim:] = 0.0

    agent = DiscreteDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=None,
        ema_rate=ema_rate,
        device=device,
        diffusion_steps=diffusion_steps,
        predict_noise=predict_noise,
    )
    return agent


@torch.no_grad()
def sample_chunk_from_dataset_item(
    agent,
    obs0_norm,
    cond_seq_norm,
    horizon,
    obs_dim,
    act_dim,
    cond_dim,
    device,
    sample_steps,
    use_ema,
    num_samples=1,
    temperature=1.0,
):
    """
    Use exactly the dataset-provided obs0 and cond_seq.
    Returns:
        pred_act_mean: [H, act_dim] normalized
        pred_act_all:  [N, H, act_dim] normalized
    """
    traj_dim = obs_dim + act_dim + cond_dim

    prior = np.zeros((num_samples, horizon, traj_dim), dtype=np.float32)
    prior[:, 0, :obs_dim] = obs0_norm[None, :]
    prior[:, :, obs_dim + act_dim:] = cond_seq_norm[None, :, :]

    prior_t = torch.tensor(prior, device=device)

    out = agent.sample(
        prior=prior_t,
        n_samples=num_samples,
        sample_steps=sample_steps,
        solver="ddpm",
        use_ema=use_ema,
        temperature=temperature,
        preserve_history=False,
    )

    if isinstance(out, tuple):
        sampled = out[0]
    else:
        sampled = out

    sampled = sampled.detach().cpu().numpy()  # [N, H, D]
    pred_act_all = sampled[:, :, obs_dim:obs_dim + act_dim]  # [N, H, act_dim]
    pred_act_mean = pred_act_all.mean(axis=0)  # [H, act_dim]

    return pred_act_mean, pred_act_all


def plot_chunk_yz(gt_act, pred_act, save_path, title_suffix=""):
    plt.figure(figsize=(6, 6))
    plt.plot(gt_act[:, 1], gt_act[:, 2], linewidth=3, label="GT")
    plt.plot(pred_act[:, 1], pred_act[:, 2], linewidth=2, linestyle="--", label="Pred")

    plt.scatter(gt_act[0, 1], gt_act[0, 2], s=60, marker="o", label="GT start")
    plt.scatter(gt_act[-1, 1], gt_act[-1, 2], s=60, marker="s", label="GT end")
    plt.scatter(pred_act[0, 1], pred_act[0, 2], s=60, marker="x", label="Pred start")
    plt.scatter(pred_act[-1, 1], pred_act[-1, 2], s=60, marker="^", label="Pred end")

    plt.xlabel("y")
    plt.ylabel("z")
    plt.title(f"Short-Horizon Chunk in y-z plane{title_suffix}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_chunk_timeseries(gt_act, pred_act, save_path, title_suffix=""):
    t = np.arange(len(gt_act))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t, gt_act[:, 1], linewidth=3, label="GT y")
    axes[0].plot(t, pred_act[:, 1], linewidth=2, linestyle="--", label="Pred y")
    axes[0].set_ylabel("y")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, gt_act[:, 2], linewidth=3, label="GT z")
    axes[1].plot(t, pred_act[:, 2], linewidth=2, linestyle="--", label="Pred z")
    axes[1].set_ylabel("z")
    axes[1].set_xlabel("step in chunk")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(f"Short-Horizon Chunk Time Series{title_suffix}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--sample_idx", type=int, default=None, help="Index inside selected dataset split")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of diffusion samples")
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    set_seed(args.seed)

    config = load_training_config(args.run_dir)
    ckpt_path = find_checkpoint(args.run_dir, args.ckpt)
    normalizer = load_normalizer(args.run_dir)

    if args.outdir is None:
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        outdir = os.path.join(args.run_dir, f"short_chunk_debug_{args.split}_{ckpt_stem}")
    else:
        outdir = args.outdir
    ensure_dir(outdir)

    dataset = Figure8ConditionalDiffusionDataset(
        npz_path=args.data,
        horizon=int(config["horizon"]),
        split=args.split,
        split_mode=config["split_mode"],
        val_ratio=float(config["val_ratio"]),
        seed=int(config["seed"]),
        normalize=True,
        speed_threshold=config.get("speed_threshold", None),
        scale_threshold=config.get("scale_threshold", None),
        center_threshold=config.get("center_threshold", None),
    )

    if len(dataset) == 0:
        raise RuntimeError(f"{args.split} dataset is empty.")

    if args.sample_idx is None:
        sample_idx = np.random.randint(len(dataset))
    else:
        sample_idx = args.sample_idx

    item = dataset[sample_idx]
    traj = item["traj"].numpy()   # [H, obs+act+cond], normalized
    start_idx = int(item["start_idx"])

    obs_dim = dataset.o_dim
    act_dim = dataset.a_dim
    cond_dim = dataset.c_dim
    horizon = int(config["horizon"])
    sample_steps = int(config.get("sampling_steps", config["diffusion_steps"]))

    obs = traj[:, :obs_dim]
    act = traj[:, obs_dim:obs_dim + act_dim]
    cond_seq = traj[:, obs_dim + act_dim:]

    agent = build_agent_from_config(
        config=config,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cond_dim=cond_dim,
        device=args.device,
    )
    agent.load(ckpt_path)
    agent.eval()

    pred_act_norm, pred_act_all_norm = sample_chunk_from_dataset_item(
        agent=agent,
        obs0_norm=obs[0],
        cond_seq_norm=cond_seq,
        horizon=horizon,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cond_dim=cond_dim,
        device=args.device,
        sample_steps=sample_steps,
        use_ema=args.use_ema,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )

    gt_act = denormalize_act(act, normalizer)
    pred_act = denormalize_act(pred_act_norm, normalizer)

    yz_path = os.path.join(outdir, f"chunk_yz_idx_{sample_idx}.png")
    ts_path = os.path.join(outdir, f"chunk_timeseries_idx_{sample_idx}.png")

    title_suffix = f"\n(sample_idx={sample_idx}, start_idx={start_idx}, temp={args.temperature}, N={args.num_samples})"
    plot_chunk_yz(gt_act, pred_act, yz_path, title_suffix=title_suffix)
    plot_chunk_timeseries(gt_act, pred_act, ts_path, title_suffix=title_suffix)

    rmse = np.sqrt(np.mean((gt_act - pred_act) ** 2))
    first_err = np.linalg.norm(gt_act[0] - pred_act[0])
    last_err = np.linalg.norm(gt_act[-1] - pred_act[-1])

    if len(gt_act) > 1:
        first_err_to_next = np.linalg.norm(gt_act[1] - pred_act[0])
    else:
        first_err_to_next = np.inf

    if horizon > 1:
        rmse_align_t = np.sqrt(np.mean((pred_act[:-1] - gt_act[:-1]) ** 2))
        rmse_align_t1 = np.sqrt(np.mean((pred_act[:-1] - gt_act[1:]) ** 2))
    else:
        rmse_align_t = rmse
        rmse_align_t1 = np.inf

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Split: {args.split}")
    print(f"sample_idx: {sample_idx}")
    print(f"dataset start_idx: {start_idx}")
    print(f"chunk horizon: {horizon}")
    print(f"temperature: {args.temperature}")
    print(f"num_samples: {args.num_samples}")
    print()

    print("=== First-step alignment debug ===")
    print(f"Pred first action: {pred_act[0]}")
    print(f"GT   first action: {gt_act[0]}")
    if len(gt_act) > 1:
        print(f"GT second action: {gt_act[1]}")
    print(f"Err(pred[0], gt[0]): {first_err:.6f}")
    if len(gt_act) > 1:
        print(f"Err(pred[0], gt[1]): {first_err_to_next:.6f}")
    print()

    print("=== Chunk-level alignment debug ===")
    print(f"RMSE vs gt[t]   : {rmse_align_t:.6f}")
    if len(gt_act) > 1:
        print(f"RMSE vs gt[t+1] : {rmse_align_t1:.6f}")
    print()

    print("=== Standard metrics ===")
    print(f"RMSE over action chunk: {rmse:.6f}")
    print(f"First-step action error: {first_err:.6f}")
    print(f"Last-step action error: {last_err:.6f}")
    print(f"Saved: {yz_path}")
    print(f"Saved: {ts_path}")


if __name__ == "__main__":
    main()