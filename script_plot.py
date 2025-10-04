import os
import pandas as pd
import matplotlib.pyplot as plt

def load_metric(file_path):
    """Loads a PPO/QDRL metric CSV into a DataFrame."""
    return pd.read_csv(file_path)

def per_episode_stats(df, value_col):
    """Compute total, avg, min, max per episode for a metric."""
    grouped = df.groupby("episode")[value_col]
    return pd.DataFrame({
        "total": grouped.sum(),
        "average": grouped.mean(),
        "min": grouped.min(),
        "max": grouped.max()
    })

def average_by_timestep(df, value_col):
    """Computes average per timestep across episodes."""
    return df.groupby("t")[value_col].mean()

def plot_constraint_violations(ppo_pos_df, qdrl_pos_df, bounds, out_path):
    """Plots a barchart of constraint violations (UAV out of bounds)."""
    def count_violations(df):
        x, y, z = df["x"], df["y"], df["z"]
        return ((x < 0) | (x > bounds[0]) |
                (y < 0) | (y > bounds[1]) |
                (z < 0) | (z > bounds[2])).sum()

    ppo_violations = count_violations(ppo_pos_df)
    qdrl_violations = count_violations(qdrl_pos_df)

    plt.figure(figsize=(6, 6))
    plt.bar(["PPO", "QDRL"], [ppo_violations, qdrl_violations], color=["tab:blue", "tab:orange"])
    plt.ylabel("Constraint Violations (count)")
    plt.title("Number of UAV Constraint Violations")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_comparison(ppo_df, qdrl_df, value_col, ylabel, out_path, title=None):
    """Plots average per-timestep metric for PPO vs QDRL."""
    ppo_avg = average_by_timestep(ppo_df, value_col)
    qdrl_avg = average_by_timestep(qdrl_df, value_col)

    plt.figure(figsize=(10, 6))
    #plt.plot(ppo_avg.index, ppo_avg.values, label="PPO", linewidth=2, linestyle='-')
    #plt.plot(qdrl_avg.index, qdrl_avg.values, label="QDRL", linewidth=2, linestyle='--')
    plt.plot(ppo_avg.index, ppo_avg.values, label="PPO", linewidth=2)
    plt.plot(qdrl_avg.index, qdrl_avg.values, label="QDRL", linewidth=2)
    plt.xlabel("Timestep Over 30 Episodes")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(prop={'size': 16})
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    ppo_dir = "ppo_uav_logs"
    qdrl_dir = "sonic_qdrl_outputs/qdrl_outputs/test7/qdrl_uav_logs"
    out_dir = "data_analysis_plots/test7"
    os.makedirs(out_dir, exist_ok=True)

    # Energy consumption
    ppo_energy_cons = load_metric(os.path.join(ppo_dir, "energy_cons.csv"))
    qdrl_energy_cons = load_metric(os.path.join(qdrl_dir, "energy_consumption.csv"))
    plot_comparison(
        ppo_energy_cons, qdrl_energy_cons, "energy_consumption",
        "Energy (J)", os.path.join(out_dir, "energy_consumption_comparison.png"),
        "Average Energy Consumption per Timestep Over 30 Episodes"
    )

    # Energy efficiency
    ppo_energy_eff = load_metric(os.path.join(ppo_dir, "energy_eff.csv"))
    qdrl_energy_eff = load_metric(os.path.join(qdrl_dir, "energy_efficiency.csv"))
    plot_comparison(
        ppo_energy_eff, qdrl_energy_eff, "energy_efficiency",
        "EE (bps/J)", os.path.join(out_dir, "energy_efficiency_comparison.png"),
        "Average Energy Efficiency per Timestep Over 30 Episodes"
    )

    # Sum rates
    ppo_sum_rates = load_metric(os.path.join(ppo_dir, "sum_rates.csv"))
    qdrl_sum_rates = load_metric(os.path.join(qdrl_dir, "sum_rates.csv"))
    plot_comparison(
        ppo_sum_rates, qdrl_sum_rates, "sum_rate",
        "Sum Rate (bps)", os.path.join(out_dir, "sum_rate_comparison.png"),
        "Average Sum Rate per Timestep Over 30 Episodes"
    )

    # Secrecy rates
    ppo_secrecy = load_metric(os.path.join(ppo_dir, "secrecy_rates.csv"))
    qdrl_secrecy = load_metric(os.path.join(qdrl_dir, "secrecy_rates.csv"))
    plot_comparison(
        ppo_secrecy, qdrl_secrecy, "secrecy_rate",
        "Secrecy Rate (bps)", os.path.join(out_dir, "secrecy_rate_comparison.png"),
        "Average Secrecy Rate per Timestep Over 30 Episodes"
    )

    # Rewards
    ppo_rewards = load_metric(os.path.join(ppo_dir, "rewards.csv"))
    qdrl_rewards = load_metric(os.path.join(qdrl_dir, "rewards.csv"))
    plot_comparison(
        ppo_rewards, qdrl_rewards, "reward",
        "Reward", os.path.join(out_dir, "reward_comparison.png"),
        "Average Reward per Timestep Over 30 Episodes"
    )
