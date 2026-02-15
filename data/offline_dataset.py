import numpy as np
import torch
from torch.utils.data import Dataset


def _to_torch(x):
    """Convert numpy array to torch tensor, or recurse into dicts."""
    if isinstance(x, dict):
        return {k: _to_torch(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


class OfflineRLDataset(Dataset):
    """PyTorch Dataset for offline RL transitions loaded from .pt files.

    Each item is a dict with keys: obs, action, reward, next_obs, done.
    Optional keys (if present in data): log_prob, value, terminated, truncated.

    obs and next_obs are dicts with keys:
        "state": float32 state vector
        "rgb": uint8 image tensor (H, W, C)
    """

    def __init__(
        self,
        paths: list[str],
        normalize_obs: bool = False,
        normalize_action: bool = False,
    ):
        """
        Args:
            paths: List of paths to .pt dataset files.
            normalize_obs: If True, normalize state obs to zero mean, unit std.
            normalize_action: If True, normalize actions to [-1, 1] using min-max scaling.
        """
        all_data: dict[str, list] = {}
        for path in paths:
            data = torch.load(path)
            for key, val in data.items():
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(_to_torch(val))

        # Concatenate across files
        def _concat(items):
            if isinstance(items[0], dict):
                return {k: _concat([it[k] for it in items]) for k in items[0]}
            return torch.cat(items, dim=0)

        for key in all_data:
            all_data[key] = _concat(all_data[key])

        # Observations (dict with "state" and "rgb")
        self.obs = all_data["obs"]
        self.next_obs = all_data["next_obs"]

        # State shortcut for convenience
        self.state = self.obs["state"].float()
        self.next_state = self.next_obs["state"].float()
        self.rgb = self.obs["rgb"]
        self.next_rgb = self.next_obs["rgb"]

        # Actions and rewards
        self.actions = all_data["actions"].float()
        self.rewards = all_data["rewards"].float()
        self.dones = all_data["dones"].float()

        # Optional fields
        self.log_probs = (
            all_data["log_probs"].float() if "log_probs" in all_data else None
        )
        self.values = all_data["values"].float() if "values" in all_data else None
        self.terminated = (
            all_data["terminated"].float() if "terminated" in all_data else None
        )
        self.truncated = (
            all_data["truncated"].float() if "truncated" in all_data else None
        )

        # Env simulator states (nested dict of tensors, each shape (T, D))
        # To restore transition i: pass get_env_state(i) to base_env.set_state_dict()
        self.env_states = all_data.get("env_states", None)

        # State normalization (does not affect RGB)
        self.normalize_obs = normalize_obs
        if normalize_obs:
            all_states = torch.cat([self.state, self.next_state], dim=0)
            self.obs_mean = all_states.mean(dim=0)
            self.obs_std = all_states.std(dim=0).clamp(min=1e-6)
            self.state = (self.state - self.obs_mean) / self.obs_std
            self.next_state = (self.next_state - self.obs_mean) / self.obs_std
            self.obs["state"] = self.state
            self.next_obs["state"] = self.next_state

        # Action normalization (min-max to [-1, 1])
        self.normalize_action = normalize_action
        if normalize_action:
            self.action_min = self.actions.min(dim=0).values
            self.action_max = self.actions.max(dim=0).values
            self.actions = (
                2
                * (self.actions - self.action_min)
                / (self.action_max - self.action_min + 1e-6)
                - 1
            )

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        item = {
            "obs": {"state": self.state[idx], "rgb": self.rgb[idx]},
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": {"state": self.next_state[idx], "rgb": self.next_rgb[idx]},
            "done": self.dones[idx],
            "idx": idx,
        }
        if self.log_probs is not None:
            item["log_prob"] = self.log_probs[idx]
        if self.values is not None:
            item["value"] = self.values[idx]
        if self.terminated is not None:
            item["terminated"] = self.terminated[idx]
        if self.truncated is not None:
            item["truncated"] = self.truncated[idx]
        return item

    def get_env_state(self, idx):
        """Extract the env state dict for transition idx.

        Returns a nested dict with tensors of shape (1, D), ready to pass
        directly to base_env.set_state_dict().
        """
        if self.env_states is None:
            raise ValueError("Dataset does not contain env_states")

        def _slice(x):
            if isinstance(x, dict):
                return {k: _slice(v) for k, v in x.items()}
            return x[idx : idx + 1]

        return _slice(self.env_states)

    def extract_trajectories(
        self, num_envs: int = 1, gamma: float = 0.8
    ) -> list[dict]:
        """Extract per-episode trajectories from the flat dataset.

        When data is collected with parallel environments, transitions are
        interleaved: at each timestep, there are ``num_envs`` consecutive
        transitions (one per env).  This method de-interleaves by env index,
        splits at episode boundaries (``done`` flags), and computes discounted
        Monte Carlo returns within each episode.

        Args:
            num_envs: Number of parallel envs used during data collection.
            gamma: Discount factor for MC return computation.

        Returns:
            List of trajectory dicts, each containing:
                states:       (T, state_dim)
                next_states:  (T, state_dim)
                rgbs:         (T, H, W, C)  uint8
                next_rgbs:    (T, H, W, C)  uint8
                rewards:      (T,)
                mc_returns:   (T,)  discounted MC returns
                dones:        (T,)
                terminated:   (T,)
                flat_indices: (T,)  original indices into the flat dataset
        """
        N = len(self)
        assert N % num_envs == 0, (
            f"Dataset size {N} not divisible by num_envs {num_envs}"
        )

        terminated = (
            self.terminated if self.terminated is not None else self.dones
        )

        trajectories: list[dict] = []
        for env_idx in range(num_envs):
            # Chronological indices for this env
            indices = torch.arange(env_idx, N, num_envs)

            env_states = self.state[indices]
            env_next_states = self.next_state[indices]
            env_rgbs = self.rgb[indices]
            env_next_rgbs = self.next_rgb[indices]
            env_rewards = self.rewards[indices]
            env_dones = self.dones[indices]
            env_terminated = terminated[indices]

            # Split at episode boundaries (done == 1)
            done_positions = torch.where(env_dones > 0.5)[0].tolist()

            start = 0
            for done_pos in done_positions:
                end = done_pos + 1
                trajectories.append(
                    self._build_trajectory(
                        env_states[start:end],
                        env_next_states[start:end],
                        env_rgbs[start:end],
                        env_next_rgbs[start:end],
                        env_rewards[start:end],
                        env_dones[start:end],
                        env_terminated[start:end],
                        indices[start:end],
                        gamma,
                    )
                )
                start = end

            # Trailing partial trajectory (env didn't finish before collection ended)
            if start < len(indices):
                trajectories.append(
                    self._build_trajectory(
                        env_states[start:],
                        env_next_states[start:],
                        env_rgbs[start:],
                        env_next_rgbs[start:],
                        env_rewards[start:],
                        env_dones[start:],
                        env_terminated[start:],
                        indices[start:],
                        gamma,
                    )
                )

        return trajectories

    @staticmethod
    def _build_trajectory(
        states, next_states, rgbs, next_rgbs,
        rewards, dones, terminated, flat_indices, gamma,
    ):
        traj_len = states.shape[0]
        mc_returns = torch.zeros(traj_len)
        running_return = 0.0
        for t in reversed(range(traj_len)):
            running_return = rewards[t].item() + gamma * running_return
            mc_returns[t] = running_return
        return {
            "states": states,
            "next_states": next_states,
            "rgbs": rgbs,
            "next_rgbs": next_rgbs,
            "rewards": rewards,
            "mc_returns": mc_returns,
            "dones": dones,
            "terminated": terminated,
            "flat_indices": flat_indices,
        }

    def get_normalization_params(self):
        """Return normalization parameters as a dict.

        Keys (present only when the corresponding normalization is enabled):
            obs_mean, obs_std: state observation stats
            action_min, action_max: action min-max bounds
        """
        params = {}
        if self.normalize_obs:
            params["obs_mean"] = self.obs_mean
            params["obs_std"] = self.obs_std
        if self.normalize_action:
            params["action_min"] = self.action_min
            params["action_max"] = self.action_max
        return params
