#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_training_config(run_dir: str):
    config_path = os.path.join(run_dir, "train_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Cannot find training config: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def find_checkpoint(run_dir: str, ckpt_name=None):
    if ckpt_name is not None:
        ckpt_path = os.path.join(run_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    candidates = [
        "diffusion_ckpt_best.pt",
        "diffusion_ckpt_latest.pt",
        "diffusion_ckpt_final.pt",
    ]
    for name in candidates:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p

    raise RuntimeError(f"No checkpoint found in {run_dir}")


def load_normalizer(run_dir: str):
    path = os.path.join(run_dir, "normalizer.npz")
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot find normalizer: {path}")
    d = np.load(path)

    return {
        "obs_mean": d["obs_mean"].astype(np.float32),
        "obs_std": d["obs_std"].astype(np.float32),
        "act_mean": d["act_mean"].astype(np.float32),
        "act_std": d["act_std"].astype(np.float32),
        "cond_mean": d["cond_mean"].astype(np.float32),
        "cond_std": d["cond_std"].astype(np.float32),
    }


def normalize_obs(x, norm):
    return (x - norm["obs_mean"]) / norm["obs_std"]


def normalize_cond(x, norm):
    return (x - norm["cond_mean"]) / norm["cond_std"]


def denormalize_act(x, norm):
    return x * norm["act_std"] + norm["act_mean"]


def load_npz(npz_path: str):
    if not os.path.exists(npz_path):
        raise RuntimeError(f"Dataset not found: {npz_path}")
    return np.load(npz_path)


def load_episode_range(d, episode_idx=0):
    episode_ends = d["episode_ends"]
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(
            f"episode_idx={episode_idx} out of range, total episodes={len(episode_ends)}"
        )
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return start, end


def load_full_episode(d, episode_idx=0):
    start, end = load_episode_range(d, episode_idx)
    obs = d["observations"][start:end].astype(np.float32)
    act = d["actions"][start:end].astype(np.float32)
    figure8_params = d["figure8_params"][start].astype(np.float32)  # [7]
    return obs, act, figure8_params, start, end


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


def build_global_progress_seq(window_start_rel, episode_len, horizon):
    """
    window_start_rel: start index within current episode
    returns [H, 1]
    """
    if episode_len <= 1:
        progress = np.zeros((horizon, 1), dtype=np.float32)
    else:
        progress = ((window_start_rel + np.arange(horizon)) / (episode_len - 1)).astype(np.float32)[:, None]
    return np.clip(progress, 0.0, 1.0)


def build_chunk_weights(horizon: int, mode: str = "linear", decay: float = 3.0):
    """
    Returns [H]

    mode:
      - uniform
      - linear
      - exp
    """
    if horizon <= 1:
        return np.ones((1,), dtype=np.float32)

    x = np.linspace(0.0, 1.0, horizon, dtype=np.float32)

    if mode == "uniform":
        w = np.ones_like(x)
    elif mode == "linear":
        w = 1.0 - x
    elif mode == "exp":
        w = np.exp(-decay * x)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    return w.astype(np.float32)


@torch.no_grad()
def sample_chunk_avg(
    agent,
    obs0_norm,
    cond_base_raw,
    window_start_rel,
    episode_len,
    normalizer,
    horizon,
    obs_dim,
    act_dim,
    cond_dim,
    device,
    sample_steps,
    use_ema,
    num_samples=8,
):
    """
    Sample one full chunk and average across num_samples.

    Returns:
        pred_act_mean: [H, act_dim] in normalized space
    """
    traj_dim = obs_dim + act_dim + cond_dim

    progress_seq = build_global_progress_seq(
        window_start_rel=window_start_rel,
        episode_len=episode_len,
        horizon=horizon,
    )  # [H,1]

    cond_seq = np.zeros((horizon, cond_dim), dtype=np.float32)
    cond_seq[:, :-1] = cond_base_raw[None, :]
    cond_seq[:, -1:] = progress_seq
    cond_seq_norm = normalize_cond(cond_seq, normalizer)

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
        temperature=0.0,
        preserve_history=False,
    )

    if isinstance(out, tuple):
        sampled = out[0]
    else:
        sampled = out

    sampled = sampled.detach().cpu().numpy()                  # [N, H, D]
    pred_act = sampled[:, :, obs_dim:obs_dim + act_dim]       # [N, H, act_dim]
    pred_act_mean = pred_act.mean(axis=0)                     # [H, act_dim]

    return pred_act_mean


def reconstruct_full_trajectory_overlap_avg(
    agent,
    full_obs,
    cond_base_raw,
    normalizer,
    horizon,
    obs_dim,
    act_dim,
    cond_dim,
    device,
    sample_steps,
    use_ema,
    num_samples=8,
    stride=1,
    max_len=None,
    weight_mode="linear",
    weight_decay=3.0,
):
    """
    Weighted overlap-average reconstruction.

    Returns:
        pred_full:       [T, act_dim]
        pred_count:      [T]
        pred_weight_sum: [T]
        starts:          [num_windows]
    """
    episode_len = len(full_obs)

    if max_len is not None:
        full_obs = full_obs[:max_len]
        episode_len = len(full_obs)

    full_obs_norm = normalize_obs(full_obs, normalizer)

    T = len(full_obs)
    pred_sum = np.zeros((T, act_dim), dtype=np.float64)
    pred_weight_sum = np.zeros((T,), dtype=np.float64)
    pred_count = np.zeros((T,), dtype=np.int64)

    max_start = T - horizon
    if max_start < 0:
        raise RuntimeError(f"Episode length {T} is shorter than horizon {horizon}")

    starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
    chunk_weights = build_chunk_weights(horizon, mode=weight_mode, decay=weight_decay)

    for s in starts:
        obs0_norm = full_obs_norm[s]

        pred_chunk_norm = sample_chunk_avg(
            agent=agent,
            obs0_norm=obs0_norm,
            cond_base_raw=cond_base_raw,
            window_start_rel=s,
            episode_len=T,
            normalizer=normalizer,
            horizon=horizon,
            obs_dim=obs_dim,
            act_dim=act_dim,
            cond_dim=cond_dim,
            device=device,
            sample_steps=sample_steps,
            use_ema=use_ema,
            num_samples=num_samples,
        )  # [H, act_dim], normalized

        pred_chunk = denormalize_act(pred_chunk_norm, normalizer)  # [H, act_dim], raw

        for k in range(horizon):
            t = s + k
            w = float(chunk_weights[k])
            pred_sum[t] += w * pred_chunk[k]
            pred_weight_sum[t] += w
            pred_count[t] += 1

    valid = pred_weight_sum > 0
    pred_full = np.zeros((T, act_dim), dtype=np.float32)
    pred_full[valid] = (pred_sum[valid] / pred_weight_sum[valid, None]).astype(np.float32)

    return pred_full, pred_count, pred_weight_sum, starts


def plot_full_trajectory(gt_act, pred_act, counts, weight_sums, out_path, episode_idx):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(gt_act[:, 1], gt_act[:, 2], linewidth=3, label="GT")
    ax.plot(pred_act[:, 1], pred_act[:, 2], linewidth=2, linestyle="--", label="Pred weighted overlap-avg")

    ax.scatter(gt_act[0, 1], gt_act[0, 2], s=60, marker="o", label="GT start")
    ax.scatter(gt_act[-1, 1], gt_act[-1, 2], s=60, marker="s", label="GT end")
    ax.scatter(pred_act[0, 1], pred_act[0, 2], s=60, marker="x", label="Pred start")
    ax.scatter(pred_act[-1, 1], pred_act[-1, 2], s=60, marker="^", label="Pred end")

    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(
        f"Full Figure-8 Weighted Overlap-Average Reconstruction (episode {episode_idx})\n"
        f"mean count={counts.mean():.2f}, mean weight={weight_sums.mean():.2f}"
    )
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_overlap_avg_gif(gt_act, pred_act, outdir, episode_idx, fps=4, show_error=True):
    """
    GIF with:
      - full GT trajectory as background
      - Pred trajectory revealed progressively
      - current GT point
      - current Pred point
    """
    frames = []

    all_y = np.concatenate([gt_act[:, 1], pred_act[:, 1]])
    all_z = np.concatenate([gt_act[:, 2], pred_act[:, 2]])

    y_margin = 0.02
    z_margin = 0.02
    ymin, ymax = all_y.min() - y_margin, all_y.max() + y_margin
    zmin, zmax = all_z.min() - z_margin, all_z.max() + z_margin

    T = len(gt_act)

    for t in range(1, T + 1):
        fig, ax = plt.subplots(figsize=(7, 7))

        # full GT trajectory background
        ax.plot(gt_act[:, 1], gt_act[:, 2], linewidth=3, label="GT full")

        # GT evolution up to current t
        ax.plot(
            gt_act[:t, 1],
            gt_act[:t, 2],
            linewidth=2,
            alpha=0.8,
            label="GT evolution",
        )

        # Pred evolution up to current t
        ax.plot(
            pred_act[:t, 1],
            pred_act[:t, 2],
            linewidth=2,
            linestyle="--",
            label="Pred weighted overlap-avg",
        )

        # start/end markers
        ax.scatter(gt_act[0, 1], gt_act[0, 2], s=60, marker="o", label="GT start")
        ax.scatter(gt_act[-1, 1], gt_act[-1, 2], s=60, marker="s", label="GT end")

        # current points
        ax.scatter(gt_act[t - 1, 1], gt_act[t - 1, 2], s=100, marker="o", label="GT current")
        ax.scatter(pred_act[t - 1, 1], pred_act[t - 1, 2], s=100, marker="x", label="Pred current")

        err = np.linalg.norm(gt_act[t - 1] - pred_act[t - 1])

        title = (
            f"Full Figure-8 Weighted Overlap-Average Reconstruction (episode {episode_idx})\n"
            f"Step {t}/{T}"
        )
        if show_error:
            title += f" | point error = {err:.4f}"

        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title(title)
        ax.set_xlim([ymin, ymax])
        ax.set_ylim([zmin, zmax])
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()

        frame_path = os.path.join(outdir, f"_gif_frame_{t:04d}.png")
        fig.savefig(frame_path, dpi=140)
        plt.close(fig)

        frames.append(imageio.imread(frame_path))

    gif_path = os.path.join(outdir, f"full_figure8_weighted_overlap_avg_episode_{episode_idx}.gif")
    imageio.mimsave(gif_path, frames, fps=fps)

    for t in range(1, T + 1):
        frame_path = os.path.join(outdir, f"_gif_frame_{t:04d}.png")
        if os.path.exists(frame_path):
            os.remove(frame_path)

    return gif_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--stride", type=int, default=1, help="Window stride for overlap reconstruction")
    parser.add_argument("--max_len", type=int, default=None, help="Only reconstruct first max_len steps of the episode")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of diffusion samples averaged per chunk")
    parser.add_argument("--fps", type=int, default=4, help="GIF frame rate")

    parser.add_argument(
        "--weight_mode",
        type=str,
        default="linear",
        choices=["uniform", "linear", "exp"],
        help="How to weight chunk timesteps in overlap reconstruction",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=3.0,
        help="Decay factor when weight_mode=exp",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    config = load_training_config(args.run_dir)
    ckpt_path = find_checkpoint(args.run_dir, args.ckpt)
    normalizer = load_normalizer(args.run_dir)

    if args.outdir is None:
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        outdir = os.path.join(
            args.run_dir,
            f"weighted_overlap_avg_episode_{args.episode_idx}_{ckpt_stem}_avg{args.num_samples}_{args.weight_mode}"
        )
    else:
        outdir = args.outdir
    ensure_dir(outdir)

    print("===== Loaded Training Config =====")
    for k, v in config.items():
        print(f"{k}: {v}")

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Output dir: {outdir}")
    print(f"num_samples per chunk: {args.num_samples}")
    print(f"window stride: {args.stride}")
    print(f"weight_mode: {args.weight_mode}")
    print(f"weight_decay: {args.weight_decay}")

    d = load_npz(args.data)
    full_obs, full_act, cond_base_raw, start, end = load_full_episode(d, args.episode_idx)

    if args.max_len is not None:
        full_obs = full_obs[:args.max_len]
        full_act = full_act[:args.max_len]

    obs_dim = full_obs.shape[1]
    act_dim = full_act.shape[1]
    cond_dim = int(config.get("cond_dim", 8))
    horizon = int(config["horizon"])
    sample_steps = int(config.get("sampling_steps", config["diffusion_steps"]))

    agent = build_agent_from_config(
        config=config,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cond_dim=cond_dim,
        device=args.device,
    )
    agent.load(ckpt_path)
    agent.eval()

    pred_full, counts, weight_sums, starts = reconstruct_full_trajectory_overlap_avg(
        agent=agent,
        full_obs=full_obs,
        cond_base_raw=cond_base_raw,
        normalizer=normalizer,
        horizon=horizon,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cond_dim=cond_dim,
        device=args.device,
        sample_steps=sample_steps,
        use_ema=args.use_ema,
        num_samples=args.num_samples,
        stride=args.stride,
        max_len=None,
        weight_mode=args.weight_mode,
        weight_decay=args.weight_decay,
    )

    fig_path = os.path.join(outdir, f"full_figure8_weighted_overlap_avg_episode_{args.episode_idx}.png")
    plot_full_trajectory(
        gt_act=full_act,
        pred_act=pred_full,
        counts=counts,
        weight_sums=weight_sums,
        out_path=fig_path,
        episode_idx=args.episode_idx,
    )

    gif_path = make_overlap_avg_gif(
        gt_act=full_act,
        pred_act=pred_full,
        outdir=outdir,
        episode_idx=args.episode_idx,
        fps=args.fps,
        show_error=True,
    )

    np.savez(
        os.path.join(outdir, f"full_figure8_weighted_overlap_avg_episode_{args.episode_idx}.npz"),
        gt_act=full_act.astype(np.float32),
        pred_act=pred_full.astype(np.float32),
        counts=counts.astype(np.int64),
        weight_sums=weight_sums.astype(np.float32),
        starts=starts.astype(np.int64),
        cond_base_raw=cond_base_raw.astype(np.float32),
    )

    rmse = np.sqrt(np.mean((full_act - pred_full) ** 2))
    first_err = np.linalg.norm(full_act[0] - pred_full[0])
    last_err = np.linalg.norm(full_act[-1] - pred_full[-1])

    print(f"Episode range in dataset: [{start}, {end})")
    print(f"Episode length used: {len(full_act)}")
    print(f"Number of chunk starts: {len(starts)}")
    print(f"Mean coverage count: {counts.mean():.3f}")
    print(f"Min/Max coverage count: {counts.min()} / {counts.max()}")
    print(f"Mean effective weight sum: {weight_sums.mean():.6f}")
    print(f"Min/Max effective weight sum: {weight_sums.min():.6f} / {weight_sums.max():.6f}")
    print(f"Full-trajectory RMSE: {rmse:.6f}")
    print(f"First-step error: {first_err:.6f}")
    print(f"Last-step error: {last_err:.6f}")
    print(f"Saved plot: {fig_path}")
    print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()