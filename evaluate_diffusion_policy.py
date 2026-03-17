import os
import argparse
import numpy as np
import torch

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d

from figure8_dataset import Figure8DiffusionDataset


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_normalizer(path):
    d = np.load(path)
    return {
        "obs_mean": d["obs_mean"].astype(np.float32),
        "obs_std": d["obs_std"].astype(np.float32),
        "act_mean": d["act_mean"].astype(np.float32),
        "act_std": d["act_std"].astype(np.float32),
    }


def denormalize_obs(x, norm):
    return x * norm["obs_std"] + norm["obs_mean"]


def denormalize_act(x, norm):
    return x * norm["act_std"] + norm["act_mean"]


def build_model(obs_dim, act_dim, horizon, model_dim, diffusion_steps, ema_rate, device, predict_noise):
    traj_dim = obs_dim + act_dim

    nn_diffusion = JannerUNet1d(
        in_dim=traj_dim,
        model_dim=model_dim,
        emb_dim=model_dim,
        dim_mult=[1, 2, 4],
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    ).to(device)

    fix_mask = torch.zeros((horizon, traj_dim), device=device)
    fix_mask[0, :obs_dim] = 1.0

    loss_weight = torch.ones((horizon, traj_dim), device=device)

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


def sample_one(agent, obs0_norm, horizon, obs_dim, act_dim, device, sample_steps, use_ema):
    traj_dim = obs_dim + act_dim

    prior = np.zeros((1, horizon, traj_dim), dtype=np.float32)
    prior[0, 0, :obs_dim] = obs0_norm
    prior_t = torch.tensor(prior, device=device)

    with torch.no_grad():
        sampled, _ = agent.sample(
            prior=prior_t,
            n_samples=1,
            sample_steps=sample_steps,
            solver="ddpm",
            use_ema=use_ema,
            temperature=1.0,
        )

    sampled = sampled[0].detach().cpu().numpy()  # [H, D]
    pred_obs = sampled[:, :obs_dim]
    pred_act = sampled[:, obs_dim:]
    return pred_obs, pred_act


def mse(x, y):
    return np.mean((x - y) ** 2)


def mae(x, y):
    return np.mean(np.abs(x - y))


def smoothness_l2(x):
    """
    x: [H, act_dim]
    """
    if x.shape[0] < 2:
        return 0.0
    dx = np.diff(x, axis=0)
    return np.mean(np.sum(dx * dx, axis=-1))


def evaluate(agent, dataset, normalizer, device, n_eval, horizon, sample_steps, use_ema):
    obs_dim = dataset.o_dim
    act_dim = dataset.a_dim

    idxs = np.random.choice(len(dataset), size=min(n_eval, len(dataset)), replace=False)

    metrics = {
        "first_action_mse": [],
        "full_action_mse": [],
        "first_xyz_mse": [],
        "full_xyz_mse": [],
        "first_xyz_mae": [],
        "full_xyz_mae": [],
        "smoothness_l2": [],
    }

    for idx in idxs:
        item = dataset[idx]

        obs_norm = item["obs"]["state"].numpy()  # [H, obs_dim]
        act_norm = item["act"].numpy()           # [H, act_dim]

        pred_obs_norm, pred_act_norm = sample_one(
            agent=agent,
            obs0_norm=obs_norm[0],
            horizon=horizon,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            sample_steps=sample_steps,
            use_ema=use_ema,
        )

        real_obs = denormalize_obs(obs_norm, normalizer)
        real_act = denormalize_act(act_norm, normalizer)
        pred_obs = denormalize_obs(pred_obs_norm, normalizer)
        pred_act = denormalize_act(pred_act_norm, normalizer)

        # full action
        metrics["full_action_mse"].append(mse(pred_act, real_act))
        metrics["first_action_mse"].append(mse(pred_act[0], real_act[0]))

        # xyz only
        real_xyz = real_act[:, :3]
        pred_xyz = pred_act[:, :3]

        metrics["full_xyz_mse"].append(mse(pred_xyz, real_xyz))
        metrics["first_xyz_mse"].append(mse(pred_xyz[0], real_xyz[0]))

        metrics["full_xyz_mae"].append(mae(pred_xyz, real_xyz))
        metrics["first_xyz_mae"].append(mae(pred_xyz[0], real_xyz[0]))

        # smoothness of predicted actions
        metrics["smoothness_l2"].append(smoothness_l2(pred_act))

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary_std = {k + "_std": float(np.std(v)) for k, v in metrics.items()}

    return metrics, summary, summary_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--normalizer", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--ema_rate", type=float, default=0.995)
    parser.add_argument("--predict_noise", action="store_true")

    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--n_eval", type=int, default=200)

    parser.add_argument("--save_metrics", type=str, default="")

    args = parser.parse_args()

    set_seed(args.seed)

    dataset = Figure8DiffusionDataset(args.data, horizon=args.horizon, normalize=True)
    normalizer = load_normalizer(args.normalizer)

    obs_dim = dataset.o_dim
    act_dim = dataset.a_dim

    agent = build_model(
        obs_dim=obs_dim,
        act_dim=act_dim,
        horizon=args.horizon,
        model_dim=args.model_dim,
        diffusion_steps=args.diffusion_steps,
        ema_rate=args.ema_rate,
        device=args.device,
        predict_noise=args.predict_noise,
    )
    agent.load(args.ckpt)
    agent.eval()

    metrics, summary, summary_std = evaluate(
        agent=agent,
        dataset=dataset,
        normalizer=normalizer,
        device=args.device,
        n_eval=args.n_eval,
        horizon=args.horizon,
        sample_steps=args.sample_steps,
        use_ema=args.use_ema,
    )

    print("===== Evaluation Summary =====")
    for k, v in summary.items():
        print(f"{k:20s}: {v:.8f}")
    for k, v in summary_std.items():
        print(f"{k:20s}: {v:.8f}")

    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics) or ".", exist_ok=True)
        save_dict = {}
        # Per-episode arrays (length = n_eval)
        save_dict.update({f"per_episode_{k}": np.asarray(v, dtype=np.float32) for k, v in metrics.items()})
        # Scalar summaries
        save_dict.update({f"mean_{k}": np.asarray(v, dtype=np.float32) for k, v in summary.items()})
        save_dict.update({f"std_{k}": np.asarray(v, dtype=np.float32) for k, v in summary_std.items()})
        np.savez(args.save_metrics, **save_dict)
        print(f"Saved metrics to: {args.save_metrics}")


if __name__ == "__main__":
    main()