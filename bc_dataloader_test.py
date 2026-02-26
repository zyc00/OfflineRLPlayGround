"""Original BC with ORIGINAL DataLoader (no GPU preload), to verify training equivalence."""
import os, random, time, math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, tyro
from mani_skill.utils.io_utils import load_json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm

@dataclass
class Args:
    seed: int = 1
    total_iters: int = 100_000
    batch_size: int = 1024
    lr: float = 3e-4
    demo_path: str = os.path.expanduser("~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5")
    exp_name: str = "bc_dataloader_test"
    log_freq: int = 25000

class IterationBasedBatchSampler(BatchSampler):
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch
    def __len__(self):
        return self.num_iterations

def load_h5_data(data):
    out = {}
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

class ManiSkillDataset(Dataset):
    def __init__(self, dataset_file, device, load_count=-1):
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.observations, self.actions = [], []
        self.device = device
        if load_count is None:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count), desc="Loading"):
            eps = self.episodes[eps_id]
            traj = load_h5_data(self.data[f"traj_{eps['episode_id']}"])
            self.observations.append(traj["obs"][:-1])
            self.actions.append(traj["actions"])
        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)
    def __len__(self):
        return len(self.observations)
    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx]).float().to(device=self.device)
        action = torch.from_numpy(self.actions[idx]).float().to(device=self.device)
        return obs, action

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim),
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    args = tyro.cli(Args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    ds = ManiSkillDataset(args.demo_path, device=device)
    sampler = RandomSampler(ds)
    batchsampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    itersampler = IterationBasedBatchSampler(batchsampler, args.total_iters)
    dataloader = DataLoader(ds, batch_sampler=itersampler, num_workers=0)

    actor = Actor(ds.observations.shape[1], ds.actions.shape[1]).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)

    for iteration, (obs, action) in enumerate(dataloader):
        pred = actor(obs)
        optimizer.zero_grad()
        loss = F.mse_loss(pred, action)
        loss.backward()
        optimizer.step()
        if iteration % args.log_freq == 0:
            print(f"Iteration {iteration}, loss: {loss.item():.6e}")

    # Full dataset loss
    all_obs = torch.from_numpy(ds.observations).float().to(device)
    all_act = torch.from_numpy(ds.actions).float().to(device)
    with torch.no_grad():
        full_loss = F.mse_loss(actor(all_obs), all_act).item()
    print(f"Full dataset loss: {full_loss:.6e}")

    os.makedirs(f"runs/{args.exp_name}/checkpoints", exist_ok=True)
    torch.save({"actor": actor.state_dict()}, f"runs/{args.exp_name}/checkpoints/final.pt")
    print("Saved.")
