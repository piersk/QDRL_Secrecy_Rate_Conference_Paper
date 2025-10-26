import os
import pandas as pd
import matplotlib.pyplot as plt

def load_metric(file_path):
    """Loads a PPO/QDRL metric CSV into a DataFrame."""
    return pd.read_csv(file_path)

def average_by_timestep(df, value_col):
    """Computes average per timestep across episodes."""
    return df.groupby("t")[value_col].mean()

def per_episode_stats(df, value_col):
    """Compute total, avg, min, max per episode for a metric."""
    grouped = df.groupby("episode")[value_col]
    return pd.DataFrame({
        "total": grouped.sum(),
        "average": grouped.mean(),
        "min": grouped.min(),
        "max": grouped.max(),
        "std": grouped.std(),
        "median": grouped.median()
    })

def plot_comparison(ppo_df, qdrl_df, value_col, ylabel, out_path, title=None):
    """Plots average per-timestep metric for PPO vs QDRL."""
    ppo_avg = average_by_timestep(ppo_df, value_col)
    qdrl_avg = average_by_timestep(qdrl_df, value_col)

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_avg.index[:218], ppo_avg.values[:218], label="PPO", linewidth=2)
    #plt.plot(ppo_avg.index, ppo_avg.values, label="PPO", linewidth=2)
    plt.plot(qdrl_avg.index, qdrl_avg.values, label="QDRL", linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(out_path)
    plt.close()

def plot_episode_stats(ppo_df, qdrl_df, value_col, ylabel, out_path_prefix, title):
    """Plots total/avg/min/max per episode for PPO vs QDRL."""
    ppo_stats = per_episode_stats(ppo_df, value_col)
    qdrl_stats = per_episode_stats(qdrl_df, value_col)


    metrics = ["total", "average", "min", "max", "std", "median"]
    for metric in metrics:
        ppo_stats.to_csv(f"metric_files/ppo_{value_col}_{metric}.csv", index=True)
        plt.figure(figsize=(10, 6))
        print("======================")
        print("PPO ", metric, " for ", value_col)
        print(ppo_stats[metric])
        print("Q-DRL ", metric, " for ", value_col)
        print(qdrl_stats[metric])
        print("======================")
        plt.plot(ppo_stats.index, ppo_stats[metric], label=f"PPO {metric}", linewidth=2)
        plt.plot(qdrl_stats.index, qdrl_stats[metric], label=f"QDRL {metric}", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"{title} ({metric.capitalize()} per Episode)")
        plt.legend(prop={'size': 12})
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f"{out_path_prefix}_{metric}.png")
        plt.close()

def plot_constraint_violations(ppo_pos_df, qdrl_pos_df, bounds, out_path):
    """Plots a barchart of constraint violations (UAV out of bounds)."""
    def count_violations(df):
        x, y, z = df["x"], df["y"], df["z"]
        return ((x < 0) | (x > bounds[0]) |
                (y < 0) | (y > bounds[1]) |
                (z < 0) | (z > bounds[2])).sum()

    ppo_violations = count_violations(ppo_pos_df)
    qdrl_violations = count_violations(qdrl_pos_df)
    print("PPO Flight Zone Constraint Violations: ", ppo_violations)
    print("Q-DRL Flight Zone Constraint Violations: ", qdrl_violations)

    plt.figure(figsize=(6, 6))
    plt.bar(["PPO", "QDRL"], [ppo_violations, qdrl_violations], color=["tab:blue", "tab:orange"])
    plt.ylabel("Constraint Violations (count)")
    plt.title("Number of UAV Constraint Violations")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_distance_to_centroid(ppo_dist_df, qdrl_dist_df, out_path):
    """Plots average distance to centroid over timesteps."""
    ppo_avg = average_by_timestep(ppo_dist_df, "distance")
    qdrl_avg = average_by_timestep(qdrl_dist_df, "distance")

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_avg.index[:240], ppo_avg.values[:240], label="PPO", linewidth=2)
    #plt.plot(ppo_avg.index, ppo_avg.values, label="PPO", linewidth=2)
    plt.plot(qdrl_avg.index, qdrl_avg.values, label="QDRL", linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Distance to Centroid (m)")
    plt.title("Average UAV Distance to GU Centroid Over Timesteps")
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    #ppo_dir = "sonic_benchmark_outputs/benchmark_logs/ppo"
    ppo_dir = "sonic_benchmark_outputs/ppo"
    sac_dir = "sonic_benchmark_outputs/benchmark_logs/sac"
    td3_dir = "sonic_benchmark_outputs/benchmark_logs/td3"
    qdrl_dir = "sonic_qdrl_outputs/qdrl_outputs/test7/qdrl_uav_logs"
    ppo_out_dir = "data_analysis_plots/sonic25oct/ppo"
    sac_out_dir = "data_analysis_plots/sonic25oct/sac"
    td3_out_dir = "data_analysis_plots/sonic25oct/td3"
    out_dir = "data_analysis_plots/sonic26oct"
    os.makedirs(out_dir, exist_ok=True)

    bounds = [150, 150, 122]  # flight zone limits

    # === ENERGY CONSUMPTION ===
    ppo_energy_cons = load_metric(os.path.join(ppo_dir, "energy_cons.csv"))
    #ppo_energy_cons = ppo_energy_cons[ppo_energy_cons["episode"] < 29]
    sac_energy_cons = load_metric(os.path.join(sac_dir, "energy_cons.csv"))
    qdrl_energy_cons = load_metric(os.path.join(qdrl_dir, "energy_consumption.csv"))
    plot_comparison(ppo_energy_cons, qdrl_energy_cons, "energy_consumption",
                    "Energy (J)", os.path.join(out_dir, "ppo_energy_consumption_comparison.png"),
                    "Average Energy Consumption per Timestep")
    #plot_episode_stats(ppo_energy_cons, qdrl_energy_cons, "energy_consumption",
    #                   "Energy (J)", os.path.join(ppo_out_dir, "energy_consumption"), "Energy Consumption")

    plot_comparison(sac_energy_cons, qdrl_energy_cons, "energy_consumption",
                    "Energy (J)", os.path.join(out_dir, "sac_energy_consumption_comparison.png"),
                    "Average Energy Consumption per Timestep")
    #plot_episode_stats(sac_energy_cons, qdrl_energy_cons, "energy_consumption",
    #                   "Energy (J)", os.path.join(out_dir, "sac_energy_consumption"), "Energy Consumption")


    # === ENERGY EFFICIENCY ===
    ppo_energy_eff = load_metric(os.path.join(ppo_dir, "energy_eff.csv"))
    #ppo_energy_eff = ppo_energy_eff[ppo_energy_eff["episode"] < 29]
    qdrl_energy_eff = load_metric(os.path.join(qdrl_dir, "energy_efficiency.csv"))
    plot_comparison(ppo_energy_eff, qdrl_energy_eff, "energy_efficiency",
                    "EE (bps/J)", os.path.join(out_dir, "energy_efficiency_comparison.png"),
                    "Average Energy Efficiency per Timestep")
    plot_episode_stats(ppo_energy_eff, qdrl_energy_eff, "energy_efficiency",
                       "EE (bps/J)", os.path.join(out_dir, "energy_efficiency"), "Energy Efficiency")

    # === SUM RATES ===
    ppo_sum_rates = load_metric(os.path.join(ppo_dir, "sum_rates.csv"))
    #ppo_sum_rates = ppo_sum_rates[ppo_sum_rates["episode"] < 29]
    qdrl_sum_rates = load_metric(os.path.join(qdrl_dir, "sum_rates.csv"))
    plot_comparison(ppo_sum_rates, qdrl_sum_rates, "sum_rate",
                    "Sum Rate (bps)", os.path.join(out_dir, "sum_rate_comparison.png"),
                    "Average Sum Rate per Timestep")
    plot_episode_stats(ppo_sum_rates, qdrl_sum_rates, "sum_rate",
                       "Sum Rate (bps)", os.path.join(out_dir, "sum_rate"), "Sum Rate")

    # === SECRECY RATES ===
    ppo_secrecy = load_metric(os.path.join(ppo_dir, "secrecy_rates.csv"))
    #ppo_secrecy = ppo_secrecy[ppo_secrecy["episode"] < 29]
    sac_secrecy = load_metric(os.path.join(sac_dir, "secrecy_rates.csv"))
    qdrl_secrecy = load_metric(os.path.join(qdrl_dir, "secrecy_rates.csv"))
    plot_comparison(ppo_secrecy, qdrl_secrecy, "secrecy_rate",
                    "Secrecy Rate (bps)", os.path.join(out_dir, "secrecy_rate_comparison.png"),
                    "Average Secrecy Rate per Timestep")
    plot_comparison(sac_secrecy, qdrl_secrecy, "secrecy_rate",
                    "Secrecy Rate (bps)", os.path.join(out_dir, "sac_secrecy_rate_comparison.png"),
                    "Average Secrecy Rate per Timestep")

    #plot_episode_stats(ppo_secrecy, qdrl_secrecy, "secrecy_rate",
    #                   "Secrecy Rate (bps)", os.path.join(out_dir, "secrecy_rate"), "Secrecy Rate")

    # === REWARDS ===
    ppo_rewards = load_metric(os.path.join(ppo_dir, "rewards.csv"))
    #ppo_rewards = ppo_rewards[ppo_rewards["episode"] < 29]
    sac_rewards = load_metric(os.path.join(sac_dir, "rewards.csv"))
    td3_rewards = load_metric(os.path.join(td3_dir, "rewards.csv"))
    qdrl_rewards = load_metric(os.path.join(qdrl_dir, "rewards.csv"))
    plot_comparison(ppo_rewards, qdrl_rewards, "reward",
                    "Reward", os.path.join(out_dir, "reward_comparison.png"),
                    "Average Reward per Timestep")
    plot_comparison(sac_rewards, qdrl_rewards, "reward",
                    "Reward", os.path.join(out_dir, "sac_reward_comparison.png"),
                    "Average Reward per Timestep")
    plot_comparison(td3_rewards, qdrl_rewards, "reward",
                    "Reward", os.path.join(out_dir, "td3_reward_comparison.png"),
                    "Average Reward per Timestep")
    plot_episode_stats(ppo_rewards, qdrl_rewards, "reward",
                       "Reward", os.path.join(out_dir, "reward"), "Rewards")

    # === DISTANCE TO CENTROID ===
    ppo_dist = load_metric(os.path.join(ppo_dir, "dist_to_centroid.csv"))
    #ppo_dist = ppo_dist[ppo_dist["episode"] < 29]
    qdrl_dist = load_metric(os.path.join(qdrl_dir, "dist_to_centroid.csv"))
    plot_distance_to_centroid(ppo_dist, qdrl_dist, os.path.join(out_dir, "distance_to_centroid.png"))

    # === CONSTRAINT VIOLATIONS ===
    ppo_uav_pos = load_metric(os.path.join(ppo_dir, "uav_pos.csv"))
    #ppo_uav_pos = ppo_uav_pos[ppo_uav_pos["episode"] < 29]
    ppo_uav_pos = ppo_uav_pos[ppo_uav_pos["episode"] > 811]
    qdrl_uav_pos = load_metric(os.path.join(qdrl_dir, "uav_position.csv"))
    plot_constraint_violations(ppo_uav_pos, qdrl_uav_pos, bounds,
                               os.path.join(out_dir, "constraint_violations.png"))
