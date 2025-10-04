"""
PPO benchmark for the custom UAV_LQDRL_Environment.

Requirements
------------
- Python 3.10+
- gymnasium
- numpy
- scipy
- stable-baselines3 >= 2.0.0 (Gymnasium-compatible)
- matplotlib
- pandas

Usage
-----
1) Run training:
       python ppo_benchmark_uav.py --timesteps 200_000 --seed 42
3) Outputs:
   - ./ppo_uav_logs/  (CSV logs)
   - ./ppo_uav_plots/ (PNG plots)
   - ./ppo_uav_models/uav_ppo.zip (trained model)

Notes
-----
- This script queries metrics directly from the environment via getters:
  `get_sum_rates()`, `get_secrecy_rates()`, `get_energy_efficiency()`,
  `get_uav_position()`, `get_energy_cons()`.
"""
from __future__ import annotations
import os
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ====== Import your custom env ======
from eve_uav_lqdrl_env import UAV_LQDRL_Environment
from prioritised_experience_replay import SumTree, Memory
from replay_buffer import ReplayBuffer

# -------------------------------
# Helper: make a monitored VecEnv
# -------------------------------
def make_env(seed: int | None = None):
    def _init():
        env = UAV_LQDRL_Environment()
        env = Monitor(env)  # records episode rewards/lengths
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


# ------------------------------------
# Custom callback to record extra stats
# ------------------------------------
class MetricsCallback(BaseCallback):
    """Collects custom per-step metrics from the env and stores per-episode logs."""

    def __init__(self):
        super().__init__()
        # Episode-level containers
        self.episodes: List[Dict[str, Any]] = []

        # Running buffers for current episode (reset at each done)
        self._buf_sum_rates: List[List[float]] = []  # list of per-step sum_rate arrays
        self._buf_secrecy_rates: List[List[float]] = []
        self._buf_energy_eff: List[float] = []
        self._buf_energy_cons: List[float] = []
        self._buf_rewards: List[float] = []
        self._buf_uav_pos: List[List[float]] = []  # [x,y,z] per step
        self._buf_dist_to_centroid: List[float] = []

        self.n_episodes = 0

    def _on_step(self) -> bool:
        # VecEnv with 1 env
        env = self.training_env.envs[0].unwrapped

        # Pull metrics from env getters
        try:
            sum_rates = env.get_sum_rates()  # list length = num_legit_users
        except Exception:
            sum_rates = []
        try:
            secrecy_rates = env.get_secrecy_rates()  # list length = num_legit_users
        except Exception:
            secrecy_rates = []
        try:
            energy_eff = float(env.get_energy_efficiency())
        except Exception:
            energy_eff = np.nan
        try:
            energy_cons = float(env.get_energy_cons())
        except Exception:
            energy_cons = np.nan
        try:
            uav_pos = env.get_uav_position().tolist()
        except Exception:
            uav_pos = [np.nan, np.nan, np.nan]
        try:
            uav_pos = env.get_uav_position().tolist()
            gu_centroid = np.mean([gu.position for gu in env.legit_users], axis=0)
            gu_centroid[2] += 10
            dist_to_centroid = np.linalg.norm(uav_pos - gu_centroid)
        except Exception:
            dist_to_centroid = np.nan

        # Current step reward for this env
        rewards = self.locals.get("rewards", None)
        step_rew = float(rewards[0]) if rewards is not None else np.nan

        # Append to buffers
        self._buf_sum_rates.append(list(map(float, sum_rates)) if sum_rates else [])
        self._buf_secrecy_rates.append(list(map(float, secrecy_rates)) if secrecy_rates else [])
        self._buf_energy_eff.append(energy_eff)
        self._buf_energy_cons.append(energy_cons)
        self._buf_rewards.append(step_rew)
        self._buf_uav_pos.append(uav_pos)
        self._buf_dist_to_centroid.append(dist_to_centroid)

        # Episode boundary?
        dones = self.locals.get("dones", None)
        if dones is not None and bool(dones[0]):
            self.n_episodes += 1
            episode: Dict[str, Any] = {
                "sum_rates": self._buf_sum_rates,
                "secrecy_rates": self._buf_secrecy_rates,
                "energy_eff": self._buf_energy_eff,
                "energy_cons": self._buf_energy_cons,
                "rewards": self._buf_rewards,
                "uav_pos": self._buf_uav_pos,
                "dist_to_centroid": self._buf_dist_to_centroid,
            }
            self.episodes.append(episode)

            # reset buffers
            self._buf_sum_rates = []
            self._buf_secrecy_rates = []
            self._buf_energy_eff = []
            self._buf_energy_cons = []
            self._buf_rewards = []
            self._buf_uav_pos = []
            self._buf_dist_to_centroid = []

        return True

    # Convenience: export episodes to tidy DataFrames
    def episodes_to_dataframes(self) -> Dict[str, pd.DataFrame]:
        rows_sum = []
        rows_sec = []
        rows_eff = []
        rows_ec = []
        rows_rew = []
        rows_pos = []
        rows_dist = []

        for ep_idx, ep in enumerate(self.episodes):
            # Sum rates and secrecy rates are per-step arrays (one per GU)
            for t, arr in enumerate(ep["sum_rates"]):
                for k, v in enumerate(arr):
                    rows_sum.append({"episode": ep_idx, "t": t, "gu": k, "sum_rate": v})
            for t, arr in enumerate(ep["secrecy_rates"]):
                for k, v in enumerate(arr):
                    rows_sec.append({"episode": ep_idx, "t": t, "gu": k, "secrecy_rate": v})

            for t, v in enumerate(ep["energy_eff"]):
                rows_eff.append({"episode": ep_idx, "t": t, "energy_eff": v})
            for t, v in enumerate(ep["energy_cons"]):
                rows_ec.append({"episode": ep_idx, "t": t, "energy_cons": v})
            for t, v in enumerate(ep["rewards"]):
                rows_rew.append({"episode": ep_idx, "t": t, "reward": v})
            for t, pos in enumerate(ep["uav_pos"]):
                rows_pos.append({"episode": ep_idx, "t": t, "x": pos[0], "y": pos[1], "z": pos[2]})
            for t, dist in enumerate(ep["dist_to_centroid"]):
                rows_dist.append({"episode": ep_idx, "t": t, "distance_to_centroid": dit})

        df_sum = pd.DataFrame(rows_sum)
        df_sec = pd.DataFrame(rows_sec)
        df_eff = pd.DataFrame(rows_eff)
        df_ec = pd.DataFrame(rows_ec)
        df_rew = pd.DataFrame(rows_rew)
        df_pos = pd.DataFrame(rows_pos)
        df_dist = pd.DataFrame(rows_dist)
        return {
            "sum_rates": df_sum,
            "secrecy_rates": df_sec,
            "energy_eff": df_eff,
            "energy_cons": df_ec,
            "rewards": df_rew,
            "uav_pos": df_pos,
            "dist_to_centroid": df_dist,
        }


# ----------------------
# Plotting (matplotlib)
# ----------------------
def plot_sum_rates(df_sum: pd.DataFrame, out_dir: str):
    if df_sum.empty:
        return
    # one figure with multiple lines (episode/gu pairs) — no custom colors
    fig = plt.figure()
    # Aggregate by episode & GU
    for (ep, gu), group in df_sum.groupby(["episode", "gu" ]):
        group = group.sort_values("t")
        plt.plot(group["t"].values, group["sum_rate"].values, label=f"ep{ep}-gu{gu}")
    plt.xlabel("timestep")
    plt.ylabel("sum rate (bps)")
    plt.title("Sum rates per GU over time")
    plt.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sum_rates.png"), dpi=150)
    plt.close(fig)


def plot_secrecy_rates(df_sec: pd.DataFrame, out_dir: str):
    if df_sec.empty:
        return
    fig = plt.figure()
    for (ep, gu), group in df_sec.groupby(["episode", "gu" ]):
        group = group.sort_values("t")
        plt.plot(group["t"].values, group["secrecy_rate"].values, label=f"ep{ep}-gu{gu}")
    plt.xlabel("timestep")
    plt.ylabel("secrecy rate (bps)")
    plt.title("Secrecy rates per GU over time")
    plt.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "secrecy_rates.png"), dpi=150)
    plt.close(fig)


def plot_energy_eff(df_eff: pd.DataFrame, out_dir: str):
    if df_eff.empty:
        return
    fig = plt.figure()
    for ep, group in df_eff.groupby("episode"):
        group = group.sort_values("t")
        plt.plot(group["t"].values, group["energy_eff"].values, label=f"ep{ep}")
    plt.xlabel("timestep")
    plt.ylabel("energy efficiency (bps/J)")
    plt.title("Energy efficiency over time")
    plt.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "energy_efficiency.png"), dpi=150)
    plt.close(fig)


def plot_energy_cons(df_ec: pd.DataFrame, out_dir: str):
    if df_ec.empty:
        return
    fig = plt.figure()
    for ep, group in df_ec.groupby("episode"):
        group = group.sort_values("t")
        plt.plot(group["t"].values, group["energy_cons"].values, label=f"ep{ep}")
    plt.xlabel("timestep")
    plt.ylabel("energy consumed so far (J)")
    plt.title("Energy consumption over time")
    plt.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "energy_consumption.png"), dpi=150)
    plt.close(fig)


def plot_rewards(df_rew: pd.DataFrame, out_dir: str):
    if df_rew.empty:
        return
    fig = plt.figure()
    for ep, group in df_rew.groupby("episode"):
        group = group.sort_values("t")
        plt.plot(group["t"].values, group["reward"].values, label=f"ep{ep}")
    plt.xlabel("timestep")
    plt.ylabel("reward")
    plt.title("Step rewards over time")
    plt.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rewards.png"), dpi=150)
    plt.close(fig)


def plot_uav_altitude(df_pos: pd.DataFrame, out_dir: str):
    if df_pos.empty:
        return
    fig = plt.figure()
    for ep, group in df_pos.groupby("episode"):
        group = group.sort_values("t")
        plt.plot(group["t"].values, group["z"].values, label=f"ep{ep}")
    plt.xlabel("timestep")
    plt.ylabel("UAV altitude (m)")
    plt.title("UAV altitude over time")
    plt.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uav_altitude.png"), dpi=150)
    plt.close(fig)


# ----------------------
# Save CSV helper
# ----------------------
def save_csvs(dfs: Dict[str, pd.DataFrame], out_dir: str):
    for name, df in dfs.items():
        if not df.empty:
            df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    #parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--n_steps", type=int, default=2048)
    #parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    args = parser.parse_args()

    # Output dirs
    models_dir = os.path.join("ppo_uav_models")
    logs_dir = os.path.join("ppo_uav_logs")
    plots_dir = os.path.join("ppo_uav_plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Build env
    env = DummyVecEnv([make_env(seed=args.seed)])

    # PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=logs_dir,
    )

    # Train with metrics callback
    metrics_cb = MetricsCallback()
    model.learn(total_timesteps=args.timesteps, callback=metrics_cb, progress_bar=True)

    # Save model
    model_path = os.path.join(models_dir, "uav_ppo")
    model.save(model_path)

    # Export logs to CSV
    dfs = metrics_cb.episodes_to_dataframes()
    save_csvs(dfs, logs_dir)

    # Plots
    plot_sum_rates(dfs["sum_rates"], plots_dir)
    plot_secrecy_rates(dfs["secrecy_rates"], plots_dir)
    plot_energy_eff(dfs["energy_eff"], plots_dir)
    plot_energy_cons(dfs["energy_cons"], plots_dir)
    plot_rewards(dfs["rewards"], plots_dir)
    plot_uav_altitude(dfs["uav_pos"], plots_dir)

    print("\n=== Training complete ===")
    print(f"Model saved to: {model_path}.zip")
    print(f"CSV logs in: {logs_dir}")
    print(f"Plots in: {plots_dir}")


if __name__ == "__main__":
    main()
