import numpy as np
import torch
from torch.utils.data import Dataset


class Figure8ConditionalDiffusionDataset(Dataset):
    """
    Returns:
        traj: [H, obs_dim + act_dim + cond_dim]
              where cond includes:
              [cx, cy, cz, ry, rz, speed, phase, global_progress]

    Assumptions about the NPZ:
        observations:   [N, obs_dim]
        actions:        [N, act_dim]
        episode_ids:    [N]
        figure8_params: [N, 7] = [cx, cy, cz, ry, rz, speed, phase]

    Important:
        - windows never cross episode boundaries
        - normalization statistics are computed from TRAIN EPISODES ONLY
    """

    BASE_PARAM_NAMES = ["cx", "cy", "cz", "ry", "rz", "speed", "phase"]
    PARAM_NAMES = ["cx", "cy", "cz", "ry", "rz", "speed", "phase", "global_progress"]

    def __init__(
        self,
        npz_path,
        horizon=32,
        split="train",                    # "train" or "val"
        split_mode="holdout_speed",       # "random_episode", "holdout_speed", "holdout_scale", "holdout_center"
        val_ratio=0.2,
        seed=0,
        normalize=True,
        speed_threshold=None,
        scale_threshold=None,
        center_threshold=None,
        eps=1e-6,
    ):
        super().__init__()
        assert split in ["train", "val"]

        data = np.load(npz_path)

        self.obs = data["observations"].astype(np.float32)
        self.act = data["actions"].astype(np.float32)
        self.episode_ids = data["episode_ids"].astype(np.int64)
        self.figure8_params = data["figure8_params"].astype(np.float32)

        self.horizon = int(horizon)
        self.split = split
        self.split_mode = split_mode
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        self.o_dim = self.obs.shape[1]
        self.a_dim = self.act.shape[1]

        # original figure8 param dim = 7
        self.base_c_dim = self.figure8_params.shape[1]

        # add global progress -> cond_dim = 8
        self.c_dim = self.base_c_dim + 1

        self._build_episode_table()
        self._split_episodes(
            speed_threshold=speed_threshold,
            scale_threshold=scale_threshold,
            center_threshold=center_threshold,
        )
        self._build_normalizer_from_train()
        self._build_windows()

    def _build_episode_table(self):
        self.unique_episode_ids = np.unique(self.episode_ids)
        self.episode_table = []
        self.episode_lookup = {}

        for ep in self.unique_episode_ids:
            idx = np.where(self.episode_ids == ep)[0]
            start = int(idx[0])
            end = int(idx[-1]) + 1
            length = end - start

            params = self.figure8_params[start].copy()
            row = {
                "episode_id": int(ep),
                "start": start,
                "end": end,
                "length": length,
                "params": params,
            }
            self.episode_table.append(row)
            self.episode_lookup[int(ep)] = row

    def _split_episodes(
        self,
        speed_threshold=None,
        scale_threshold=None,
        center_threshold=None,
    ):
        rng = np.random.default_rng(self.seed)

        ep_ids = np.array([row["episode_id"] for row in self.episode_table], dtype=np.int64)
        params = np.stack([row["params"] for row in self.episode_table], axis=0)  # [E, 7]

        cx = params[:, 0]
        cy = params[:, 1]
        cz = params[:, 2]
        ry = params[:, 3]
        rz = params[:, 4]
        speed = params[:, 5]

        if self.split_mode == "random_episode":
            perm = rng.permutation(len(ep_ids))
            n_val = max(1, int(round(len(ep_ids) * self.val_ratio)))
            val_idx = perm[:n_val]
            val_mask = np.zeros(len(ep_ids), dtype=bool)
            val_mask[val_idx] = True

        elif self.split_mode == "holdout_speed":
            if speed_threshold is None:
                speed_threshold = np.quantile(speed, 1.0 - self.val_ratio)
            val_mask = speed >= speed_threshold

        elif self.split_mode == "holdout_scale":
            scale = np.sqrt(ry * rz)
            if scale_threshold is None:
                scale_threshold = np.quantile(scale, 1.0 - self.val_ratio)
            val_mask = scale >= scale_threshold

        elif self.split_mode == "holdout_center":
            center = np.stack([cx, cy, cz], axis=1)
            center_mean = center.mean(axis=0, keepdims=True)
            center_dist = np.linalg.norm(center - center_mean, axis=1)
            if center_threshold is None:
                center_threshold = np.quantile(center_dist, 1.0 - self.val_ratio)
            val_mask = center_dist >= center_threshold

        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        if val_mask.sum() == 0:
            perm = rng.permutation(len(ep_ids))
            n_val = max(1, int(round(len(ep_ids) * self.val_ratio)))
            val_idx = perm[:n_val]
            val_mask = np.zeros(len(ep_ids), dtype=bool)
            val_mask[val_idx] = True

        if (~val_mask).sum() == 0:
            val_mask[np.argmax(val_mask)] = False

        self.train_episode_ids = ep_ids[~val_mask]
        self.val_episode_ids = ep_ids[val_mask]

        self.active_episode_ids = (
            self.train_episode_ids if self.split == "train" else self.val_episode_ids
        )
        self.active_episode_set = set(self.active_episode_ids.tolist())

        self.split_info = {
            "split_mode": self.split_mode,
            "train_episode_ids": self.train_episode_ids.copy(),
            "val_episode_ids": self.val_episode_ids.copy(),
        }

    def _build_normalizer_from_train(self):
        train_mask = np.isin(self.episode_ids, self.train_episode_ids)

        obs_train = self.obs[train_mask]
        act_train = self.act[train_mask]
        cond_train = self.figure8_params[train_mask]   # [N_train, 7]

        self.obs_mean = obs_train.mean(axis=0).astype(np.float32)
        self.obs_std = np.clip(obs_train.std(axis=0), self.eps, None).astype(np.float32)

        self.act_mean = act_train.mean(axis=0).astype(np.float32)
        self.act_std = np.clip(act_train.std(axis=0), self.eps, None).astype(np.float32)

        cond_mean_base = cond_train.mean(axis=0).astype(np.float32)
        cond_std_base = np.clip(cond_train.std(axis=0), self.eps, None).astype(np.float32)

        # global progress is in [0,1]
        progress_mean = np.array([0.5], dtype=np.float32)
        progress_std = np.array([0.5], dtype=np.float32)

        self.cond_mean = np.concatenate([cond_mean_base, progress_mean], axis=0).astype(np.float32)
        self.cond_std = np.concatenate([cond_std_base, progress_std], axis=0).astype(np.float32)

    def _build_windows(self):
        starts = []
        for row in self.episode_table:
            ep_id = row["episode_id"]
            if ep_id not in self.active_episode_set:
                continue

            start = row["start"]
            end = row["end"]
            length = row["length"]

            if length < self.horizon:
                continue

            for s in range(start, end - self.horizon + 1):
                starts.append(s)

        self.window_starts = np.asarray(starts, dtype=np.int64)

    def __len__(self):
        return len(self.window_starts)

    def _normalize_obs(self, x):
        return (x - self.obs_mean) / self.obs_std

    def _normalize_act(self, x):
        return (x - self.act_mean) / self.act_std

    def _normalize_cond(self, x):
        return (x - self.cond_mean) / self.cond_std

    def _build_global_progress(self, s, e):
        """
        Global progress within the whole episode, not the chunk.

        For timestep t in [s, e):
            progress_t = (t - episode_start) / (episode_length - 1)
        """
        ep_id = int(self.episode_ids[s])
        row = self.episode_lookup[ep_id]
        ep_start = row["start"]
        ep_length = row["length"]

        if ep_length <= 1:
            progress = np.zeros((self.horizon, 1), dtype=np.float32)
        else:
            progress = ((np.arange(s, e) - ep_start) / (ep_length - 1)).astype(np.float32)[:, None]

        progress = np.clip(progress, 0.0, 1.0)
        return progress

    def __getitem__(self, idx):
        s = int(self.window_starts[idx])
        e = s + self.horizon

        obs = self.obs[s:e].copy()
        act = self.act[s:e].copy()

        cond_base = self.figure8_params[s].copy()                           # [7]
        cond_base_seq = np.repeat(cond_base[None, :], self.horizon, axis=0)  # [H, 7]

        global_progress = self._build_global_progress(s, e)                # [H, 1]
        cond_seq = np.concatenate([cond_base_seq, global_progress], axis=-1)  # [H, 8]

        if self.normalize:
            obs = self._normalize_obs(obs)
            act = self._normalize_act(act)
            cond_seq = self._normalize_cond(cond_seq)

        traj = np.concatenate([obs, act, cond_seq], axis=-1).astype(np.float32)

        return {
            "traj": torch.from_numpy(traj),                         # [H, o_dim + a_dim + c_dim]
            "cond": torch.from_numpy(cond_seq.astype(np.float32)),  # [H, c_dim]
            "start_idx": s,
        }

    def get_normalizer(self):
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "act_mean": self.act_mean,
            "act_std": self.act_std,
            "cond_mean": self.cond_mean,
            "cond_std": self.cond_std,
            "obs_dim": np.array([self.o_dim], dtype=np.int64),
            "act_dim": np.array([self.a_dim], dtype=np.int64),
            "cond_dim": np.array([self.c_dim], dtype=np.int64),
            "param_names": np.array(self.PARAM_NAMES),
        }

    def get_split_info(self):
        out = {}
        for k, v in self.split_info.items():
            if isinstance(v, np.ndarray):
                out[k] = v
            else:
                out[k] = np.array([v], dtype=object)
        return out