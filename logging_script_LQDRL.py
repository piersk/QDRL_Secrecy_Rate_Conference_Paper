#script_PL_LQDRL.py
from __future__ import annotations
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import numpy as np
from pennylane import draw
from pennylane import grad
from pennylane.fourier import circuit_spectrum
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
from collections import deque
import random
import time
import math
import os
#import argparse
import numpy as np
import pandas as pd

# Function to plot the latest UAV position as a point as well as it's previous course as a line
def plot_uav_trajectory(env, uav_trajectory, layer, ep, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for gu in env.legit_users:
        gu_pos = gu.position
        ax.scatter(gu_pos[0], gu_pos[1], gu_pos[2], label=f'GU {gu.id}', color="green")
    centroid = np.mean([gu.position for gu in env.legit_users], axis=0)

    for eve in env.eves:
        eve_pos = eve.position
        ax.scatter(eve_pos[0], eve_pos[1], eve_pos[2], label=f'Eve {eve.id}', color="magenta")

    uav_positions = np.array(env.uavs[0].history)
    ax.plot(uav_positions[:,0], uav_positions[:,1], uav_positions[:,2], label="UAV Path", color="blue")
        
    for uav in env.uavs:
        uav_position = uav.position
        ax.scatter(uav_position[0], uav_position[1], uav_position[2], label="UAV Positions", color="cyan")
    ax.scatter(*centroid, label="GU Centroid", color="red", marker="X", s=100)
    plt.legend()
    plt.savefig(f'eve_outputs/plots_eve_outputs/test1/{layer+1}_layers_uav_trajectory_{ep}_timestep_{t}.png')
    plt.close()

logs_dir = os.path.join("qdrl_outputs", "qdrl_uav_logs")
os.makedirs(logs_dir, exist_ok=True)

# === Convert arrays into DataFrames ===
def arrays_to_dataframes(ep_sum_rate_arr, ep_energy_eff_arr, ep_secrecy_rates_arr, ep_energy_cons_arr, ep_rewards_arr, ep_positions_arr, ep_dist_arr):
    dfs = {}
    dfs["sum_rates"] = pd.DataFrame(ep_sum_rate_arr, columns=["episode", "t", "user", "sum_rate"])
    dfs["energy_efficiency"] = pd.DataFrame(ep_energy_eff_arr, columns=["episode", "t", "energy_efficiency"])
    dfs["secrecy_rates"] = pd.DataFrame(ep_secrecy_rates_arr, columns=["episode", "t", "user", "secrecy_rate"])
    dfs["energy_consumption"] = pd.DataFrame(ep_energy_cons_arr, columns=["episode", "t", "energy_consumption"])
    dfs["rewards"] = pd.DataFrame(ep_rewards_arr, columns=["episode", "t", "total_reward"])
    dfs["uav_position"] = pd.DataFrame(ep_positions_arr, columns=["episode", "t", "x", "y", "z"])
    dfs["dist_to_centroid"] = pd.DataFrame(ep_dist_arr, columns=["episode", "t", "distance"])
    return dfs

# === Save CSVs ===
def save_csvs(dfs, logs_dir):
    for name, df in dfs.items():
        if not df.empty:
            path = os.path.join(logs_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"[LOG] Saved {path}")

def gradient_norm(grad):
    return jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in grad]))

# Importing modules required for experiments
from eve_uav_lqdrl_env import UAV_LQDRL_Environment
from quantum_models import QuantumActor, QuantumCritic
from replay_buffer import ReplayBuffer
from pennylane.optimize import AdamOptimizer
from prioritised_experience_replay import SumTree, Memory

overall_start_time = time.time()

all_uav_pos = []
all_secrecy_rates = []
all_sum_rates = []
all_energy_eff = []
all_energy_cons = []
all_rewards = []
all_dist_to_centroid = []

m_layers = 1
for m in range(m_layers):
    print(f"============ Experiment with {m+1} Layers in Ansatz ============")
    env = UAV_LQDRL_Environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = QuantumActor(n_qubits=state_dim, m_layers=m+1)
    critic = QuantumCritic(n_qubits=state_dim + action_dim, m_layers=m+1)

    capacity = 100000
    buffer = Memory(capacity)

    import optax

    actor_opt = optax.adam(learning_rate=0.01)
    critic_opt = optax.adam(learning_rate=0.01)

    actor_opt_state = actor_opt.init(actor.theta)
    critic_opt_state = critic_opt.init(critic.theta)

    episodes = 30
    #episodes = 5
    batch_size = 30
    gamma = 0.99
    max_act_scale = 1e15
    #max_act_scale = 1

    time_step = 1
    diff = 0

    time_arr = []

    tot_reward_arr = []
    rewards_across_eps_arr = []
    actor_losses = []
    critic_losses = []
    ep_distances_to_centroid = []
    ep_sum_rate_arr = []
    ep_energy_eff_arr = []
    ep_timesteps_arr = []
    ep_remaining_energy_arr = []
    ep_energy_cons_arr = []
    ep_secrecy_rates_arr = []

    uav_pos_arr = []

    total_runtime_start = time.time()

    for ep in range(episodes):
        ep_start_time = time.time()
        time_var = 0
        i = 0
        state, _ = env.reset()
        done = False
        total_reward = 0
        ep_uav_trajectory = []
        dist_to_centroid_arr = []
        step_rewards_arr = []
        step_sum_rate_arr = []
        step_energy_eff_arr = []
        step_remaining_energy_arr = []
        step_energy_cons_arr = []
        step_secrecy_rates_arr = []
        break_var = 0

        while not done:
            step_start_time = time.time()
            uav_pos = env.get_uav_position()
            ep_uav_trajectory.append(uav_pos)
            all_uav_pos.append([ep, i, uav_pos[0], uav_pos[1], uav_pos[2]])

            uav_energy = env.get_remaining_energy()
            energy_cons = env.E_MAX - uav_energy 
            step_energy_cons_arr.append(energy_cons)
            all_energy_cons.append([ep, i, energy_cons])

            uav_energy_perc = uav_energy / env.E_MAX

            print("Remaining UAV Energy: ", uav_energy, " J")

            print("Percentage of Remaining UAV Energy: ", uav_energy_perc * 100, "%")

            print("UAV Position Co-ordinates: ", uav_pos)

            gu_centroid = np.mean([gu.position for gu in env.legit_users], axis=0)
            gu_centroid[2] += 10
            print("GU Centroid Co-ordinates: ", gu_centroid)

            dist_to_centroid = np.linalg.norm(uav_pos - gu_centroid)
            dist_to_centroid_arr.append(dist_to_centroid)
            print("Distance of UAV from GU Centroid: ", dist_to_centroid, "m")
            all_dist_to_centroid.append([ep, i, dist_to_centroid])

            energy_efficiency = env.get_energy_efficiency()
            print(f"Energy Efficiency for step {i}: ", energy_efficiency)
            all_energy_eff.append([ep, i, energy_efficiency])

            step_remaining_energy_arr.append(uav_energy)
            step_energy_eff_arr.append(energy_efficiency)

            #state_tensor = np.array(state, requires_grad=False)
            state_tensor = jnp.array(state)
            action = actor(state_tensor)
            print("Action: ", action)
            action = jnp.tanh(jnp.array(action)) * max_act_scale
            print("Action Scaled along Hyperbolic Tangent: ", action)
            action = np.clip(np.array(action), -1, 1)
            print("Clipped Action: ", action)

            next_state, reward, done, _, _ = env.step(action)
            #buffer.push(state, action, reward, next_state, done)
            buffer.store([state, action, reward, next_state, done])
            total_reward += reward
            state = next_state

            all_rewards.append([ep, i, reward])

            step_rewards_arr.append(reward)

            sum_rates = env.get_sum_rates()
            step_sum_rate_arr.append(sum_rates)
            for gu in range(env.num_legit_users):
                all_sum_rates.append([ep, i, gu, sum_rates[gu]])

            secrecy_rates = [0.0, 0.0, 0.0, 0.0]
            secrecy_rates = env.get_secrecy_rates()
            for eve_m in range(len(secrecy_rates)):
                sr = secrecy_rates[eve_m]
                if math.isnan(sr) is True:
                    secrecy_rates[eve_m] = 0.0
            secrecy_rates = [float(sr) if np.isfinite(sr) else 0.0 for sr in secrecy_rates]
            step_secrecy_rates_arr.append(secrecy_rates)
            print(f"Secrecy Rates for Step {i}: ", secrecy_rates)

            for gu in range(env.num_legit_users):
                all_secrecy_rates.append([ep, i, gu, secrecy_rates[gu]])

            if len(buffer) >= batch_size:
                b_idx, batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

                def critic_loss(theta):
                    q_vals = []
                    targets = []
                    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                        sa = jnp.concatenate([jnp.array(s), jnp.array(a)])
                        q_val = critic.qnode(sa, theta)
                        q_val = critic.decode_op(q_val)

                        na = actor(jnp.array(ns))
                        nsa = jnp.concatenate([jnp.array(ns), jnp.array(na)])
                        q_val_next = critic.qnode(nsa, theta)
                        q_val_next = critic.decode_op(q_val_next)

                        target = r + gamma * q_val_next * (1 - d)
                        q_vals.append(q_val)
                        targets.append(target)
                    q_vals = jnp.array(q_vals)
                    targets = jnp.array(targets)
                    td_err = abs(q_vals - targets)
                    loss_val = jnp.mean((q_vals - targets) ** 2)
                    return loss_val, td_err

                def actor_loss(theta):
                    q_vals = []
                    for s in states:
                        a = actor(jnp.array(s), theta)
                        sa = jnp.concatenate([jnp.array(s), jnp.array(a)])
                        q_val = critic.qnode(sa, critic.theta)  # Use critic's current theta
                        q_val = critic.decode_op(q_val)
                        q_vals.append(q_val)
                    return -jnp.mean(jnp.array(q_vals))

                loss_val, td_errors = critic_loss(critic.theta)
                critic_losses.append(loss_val)
                actor_losses.append(actor_loss(actor.theta))
                buffer.batch_update(b_idx, np.array(td_errors))

                time_var += time_step
            time_arr.append(time_var)
            #if i % 10 == 0 or done:
                #plot_uav_trajectory(env, ep_uav_trajectory, m, ep, i)
            step_end_time = time.time()
            step_time = step_start_time - step_end_time 
            print(f"Time taken for step {i} to execute: ", abs(step_time), " seconds")
            i += 1
            # Break out of episode early (for debugging purposes)
            if i == 10:
                break
            if dist_to_centroid_arr[i-2] is not None:
                diff = dist_to_centroid_arr[i-2] - dist_to_centroid_arr[i-1]
                if (diff <= 0.1):
                    break_var += 1
            if break_var >= 250:
                break

        ep_end_time = time.time()
        ep_time = ep_start_time - ep_end_time
        print("Time taken for episode to execute: ", abs(ep_time), " seconds")
        #plot_uav_trajectory(env, ep_uav_trajectory, m, ep, i)
        print(f"Episode {ep} | Total reward: {total_reward:.10f}")
        tot_reward_arr.append(total_reward)
        ep_sum_rate_arr.append(step_sum_rate_arr)
        uav_pos_arr.append(ep_uav_trajectory)
        ep_energy_eff_arr.append(step_energy_eff_arr)
        ep_distances_to_centroid.append(dist_to_centroid_arr)
        rewards_across_eps_arr.append(step_rewards_arr)
        ep_timesteps_arr.append(i)
        ep_remaining_energy_arr.append(step_remaining_energy_arr)
        ep_energy_cons_arr.append(step_energy_cons_arr)
        ep_secrecy_rates_arr.append(step_secrecy_rates_arr)

    print("All good so far")
    total_runtime_end = time.time()
    total_runtime = abs(total_runtime_end - total_runtime_start)
    print(f"Total Time Taken for Experiment with {m+1} Layers to Run: ", total_runtime)

#logs_dir = os.path.join("qdrl_outputs/qdrl_uav_logs/test2")
logs_dir = os.path.join("local_test_outputs/qdrl_uav_logs/test5")
#plots_dir = os.path.join("qdrl_outputs/qdrl_uav_plots/test2")
plots_dir = os.path.join("local_test_outputs/qdrl_uav_plots/test5")

dfs = arrays_to_dataframes(all_sum_rates, all_energy_eff, all_secrecy_rates, all_energy_cons, all_rewards, all_uav_pos, all_dist_to_centroid)

save_csvs(dfs, logs_dir)

overall_end_time = time.time()
overall_time = abs(overall_end_time - overall_start_time)
print("Total Time Taken for Experiment to Run: ", overall_time)
