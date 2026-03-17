import os
import json
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters

from figure8_dataset import Figure8ConditionalDiffusionDataset


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_metadata(save_dir, train_dataset, val_dataset, args):
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, "normalizer.npz"),
        **train_dataset.get_normalizer(),
    )

    split_info = train_dataset.get_split_info()
    np.savez(
        os.path.join(save_dir, "split_info.npz"),
        **split_info,
    )

    with open(os.path.join(save_dir, "dataset_summary.txt"), "w") as f:
        f.write(f"train_windows: {len(train_dataset)}\n")
        f.write(f"val_windows: {len(val_dataset)}\n")
        f.write(f"train_episodes: {len(train_dataset.train_episode_ids)}\n")
        f.write(f"val_episodes: {len(val_dataset.val_episode_ids)}\n")
        f.write(f"obs_dim: {train_dataset.o_dim}\n")
        f.write(f"act_dim: {train_dataset.a_dim}\n")
        f.write(f"cond_dim: {train_dataset.c_dim}\n")
        f.write(f"split_mode: {train_dataset.split_mode}\n")

    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def init_csv_logger(csv_path, fieldnames):
    file_exists = os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        f.flush()
    return f, writer


@torch.no_grad()
def evaluate(agent, val_loader, device):
    agent.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        traj = batch["traj"].to(device)
        loss = agent.loss(traj)
        total_loss += float(loss.item())
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    agent.train()
    return avg_loss

def save_checkpoint_bundle(agent, save_dir, ckpt_name, train_dataset, args, step, best_val_loss=None):
    ckpt_path = os.path.join(save_dir, ckpt_name)
    agent.save(ckpt_path)

    np.savez(
        os.path.join(save_dir, "normalizer.npz"),
        **train_dataset.get_normalizer(),
    )

    state = {
        "step": step,
        "best_val_loss": best_val_loss,
        "checkpoint": ckpt_name,
        "args": vars(args),
    }
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/figure8_diffusion_cond")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--dim_mult", type=int, nargs="+", default=[1, 2, 2])

    # D4RL-style hybrid setup:
    # train with x0 prediction, sample with DDPM solver later
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--solver", type=str, default="ddpm")
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--predict_noise", action="store_true",
                        help="If set, train epsilon-prediction. Leave unset for D4RL-style x0-prediction.")
    parser.add_argument("--gradient_steps", type=int, default=15000)
    parser.add_argument("--ema_rate", type=float, default=0.995)

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--val_interval", type=int, default=1000)

    parser.add_argument("--action_loss_weight", type=float, default=5.0)

    # split config
    parser.add_argument(
        "--split_mode",
        type=str,
        default="holdout_speed",
        choices=["random_episode", "holdout_speed", "holdout_scale", "holdout_center"],
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--speed_threshold", type=float, default=None)
    parser.add_argument("--scale_threshold", type=float, default=None)
    parser.add_argument("--center_threshold", type=float, default=None)


    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.predict_noise:
        print("WARNING: predict_noise=True. This is NOT the D4RL-style hybrid setup.")
    else:
        print("Using D4RL-style hybrid setup: x0 prediction in training, DDPM solver for inference.")

    train_dataset = Figure8ConditionalDiffusionDataset(
        npz_path=args.data,
        horizon=args.horizon,
        split="train",
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=True,
        speed_threshold=args.speed_threshold,
        scale_threshold=args.scale_threshold,
        center_threshold=args.center_threshold,
    )

    val_dataset = Figure8ConditionalDiffusionDataset(
        npz_path=args.data,
        horizon=args.horizon,
        split="val",
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=True,
        speed_threshold=args.speed_threshold,
        scale_threshold=args.scale_threshold,
        center_threshold=args.center_threshold,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    print("===== Dataset =====")
    print(f"split_mode:      {args.split_mode}")
    print(f"train_episodes:  {len(train_dataset.train_episode_ids)}")
    print(f"val_episodes:    {len(train_dataset.val_episode_ids)}")
    print(f"train_windows:   {len(train_dataset)}")
    print(f"val_windows:     {len(val_dataset)}")
    print(f"obs_dim:         {train_dataset.o_dim}")
    print(f"act_dim:         {train_dataset.a_dim}")
    print(f"cond_dim:        {train_dataset.c_dim}")

    save_metadata(args.save_dir, train_dataset, val_dataset, args)

    train_log_file, train_log_writer = init_csv_logger(
        os.path.join(args.save_dir, "train_log.csv"),
        fieldnames=["step", "train_loss", "lr"],
    )
    val_log_file, val_log_writer = init_csv_logger(
        os.path.join(args.save_dir, "val_log.csv"),
        fieldnames=["step", "val_loss"],
    )

    obs_dim = train_dataset.o_dim
    act_dim = train_dataset.a_dim
    cond_dim = train_dataset.c_dim
    traj_dim = obs_dim + act_dim + cond_dim

    nn_diffusion = JannerUNet1d(
        in_dim=traj_dim,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.dim_mult,
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    ).to(args.device)

    print("===== Diffusion Model =====")
    report_parameters(nn_diffusion)

    # Conditioning:
    # - fix first observation
    # - fix condition at all timesteps
    fix_mask = torch.zeros((args.horizon, traj_dim), device=args.device)
    fix_mask[0, :obs_dim] = 1.0
    fix_mask[:, obs_dim + act_dim:] = 1.0

    # Weighted loss:
    # - action dims weighted higher
    # - condition dims zero-weighted (inputs only)
    loss_weight = torch.ones((args.horizon, traj_dim), device=args.device)
    loss_weight[:, obs_dim:obs_dim + act_dim] = args.action_loss_weight
    loss_weight[:, obs_dim + act_dim:] = 0.0

    agent = DiscreteDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=None,
        ema_rate=args.ema_rate,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
        predict_noise=args.predict_noise,  # False by default => D4RL-style x0 prediction
    )

    lr_scheduler = CosineAnnealingLR(agent.optimizer, args.gradient_steps)

    agent.train()

    step = 0
    running_train_loss = 0.0
    best_val_loss = float("inf")

    while step < args.gradient_steps:
        for batch in train_loader:
            traj = batch["traj"].to(args.device)

            out = agent.update(traj)
            loss = float(out["loss"])
            running_train_loss += loss
            lr_scheduler.step()

            step += 1

            if step % args.log_interval == 0:
                avg_train_loss = running_train_loss / args.log_interval
                current_lr = float(agent.optimizer.param_groups[0]["lr"])

                print({
                    "step": step,
                    "train_loss": avg_train_loss,
                    "lr": current_lr,
                })

                train_log_writer.writerow({
                    "step": step,
                    "train_loss": avg_train_loss,
                    "lr": current_lr,
                })
                train_log_file.flush()

                running_train_loss = 0.0

            if step % args.val_interval == 0:
                val_loss = evaluate(agent, val_loader, args.device)

                print({
                    "step": step,
                    "val_loss": val_loss,
                })

                val_log_writer.writerow({
                    "step": step,
                    "val_loss": val_loss,
                })
                val_log_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint_bundle(
                        agent=agent,
                        save_dir=args.save_dir,
                        ckpt_name="diffusion_ckpt_best.pt",
                        train_dataset=train_dataset,
                        args=args,
                        step=step,
                        best_val_loss=best_val_loss,
                    )

            if step % args.save_interval == 0:
                save_checkpoint_bundle(
                    agent=agent,
                    save_dir=args.save_dir,
                    ckpt_name=f"diffusion_ckpt_{step}.pt",
                    train_dataset=train_dataset,
                    args=args,
                    step=step,
                    best_val_loss=best_val_loss,
                )
                save_checkpoint_bundle(
                    agent=agent,
                    save_dir=args.save_dir,
                    ckpt_name="diffusion_ckpt_latest.pt",
                    train_dataset=train_dataset,
                    args=args,
                    step=step,
                    best_val_loss=best_val_loss,
                )

            if step >= args.gradient_steps:
                break

    save_checkpoint_bundle(
        agent=agent,
        save_dir=args.save_dir,
        ckpt_name="diffusion_ckpt_final.pt",
        train_dataset=train_dataset,
        args=args,
        step=step,
        best_val_loss=best_val_loss,
    )

    train_log_file.close()
    val_log_file.close()


if __name__ == "__main__":
    main()