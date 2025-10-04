import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import random
import time
import math
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import numpy as np
from pennylane import draw
from pennylane import grad
from pennylane.fourier import circuit_spectrum
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import csv
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.special import iv  # Modified Bessel function of the first kind
from scipy.stats import rice

from eve_uav_lqdrl_env import UAV_LQDRL_Environment
from quantum_models import QuantumActor, QuantumCritic
from replay_buffer import ReplayBuffer
from prioritised_experience_replay import SumTree, Memory
from pennylane.optimize import AdamOptimizer

env = UAV_LQDRL_Environment()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = QuantumActor(n_qubits=state_dim, m_layers=1)
critic = QuantumCritic(n_qubits=state_dim+action_dim, m_layers=1)

capacity = 100000
buffer = Memory(capacity)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=500)
model.save('ppo_lqdrl_secrecy_1')

# NOTE: SIMULATION PARAMETERS
# COULD TINKER WITH THESE AND RUN EXPERIMENTS WITH DIFFERENT ONES FOR EACH 
m_layers = 1
episodes = 30
batch_size = 30
gamma = 0.99
max_act_scale = 1e15

time_step = 1
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
    plt.savefig(f'ppo_outputs/{layer+1}_layers_uav_trajectory_{ep}_timestep_{t}.png')
    plt.close()

def gradient_norm(grad):
    return jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in grad]))

def evaluate_agent(model, env, num_episodes):
    episode_rewards = []
    masr_values = []

    for ep in range(num_episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        episode_masr = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Compute MASR from the reward function
            masr_values.append(reward)  # Approximate since MASR is used in reward

        episode_rewards.append(total_reward)

    return episode_rewards, masr_values

for ep in range(episodes):
    state, _ = env.reset()
    ep_start_time = time.time()
    ep_uav_traj = []
    dist_to_centroid_arr = []
    step_rewards_arr = []
    step_sum_rate_arr = []
    step_energy_eff_arr = []
    step_remaining_energy_arr = []
    step_energy_cons_arr = []
    step_secrecy_rates_arr = []
    break_var = 0
    step = 0
    done = False
    tot_reward = 0

    while not done:
        step_start_time = time.time()
        uav_pos = env.get_uav_position()
        uav_energy = env.get_remaining_energy()
        energy_cons = env.E_MAX - uav_energy 
        ep_uav_traj.append(uav_pos)
        uav_energy_perc = uav_energy / env.E_MAX

        with open(f'{ep}_uav_energy.txt', 'w') as f:
            f.write(uav_energy)
            f.write("\n")

        with open(f'{ep}_uav_energy_perc.txt', 'w') as f:
            f.write(uav_energy_perc)
            f.write("\n")

        with open(f'{ep}_uav_position.txt', 'w') as f:
            f.write(uav_pos)
            f.write("\n")

        gu_centroid = np.mean([gu.position for gu in env.legit_users], axis=0)
        gu_centroid[2] += 10

        dist_to_centroid = np.linalg.norm(uav_pos - gu_centroid)

        with open(f'{ep}_dist_to_centroid.txt', 'w') as f:
            f.write(dist_to_centroid)
            f.write("\n")

        uav_energy_eff = env.get_energy_efficiency()
        with open(f'{ep}_uav_energy_efficiency.txt', 'w') as f:
            f.write(uav_energy_eff)
            f.write("\n")

        step_remaining_energy_arr.append(uav_energy)
        step_energy_eff_arr.append(uav_energy_eff)

        sum_rates = env.get_sum_rates()
        step_sum_rate_arr.append(sum_rates)

        with open(f'{ep}_uav_sum_rates.txt', 'w') as f:
            f.write(sum_rates)
            f.write("\n")

        secrecy_rates = [0 for gu in env.num_legit_users]
        secrecy_rates = env.get_secrecy_rates()
        for eve_m in range(len(secrecy_rates)):
            sr = secrecy_rates[eve_m]
            if math.isnan(sr) is True:
                secrecy_rates[eve_m] = 0.0

        with open(f'{ep}_uav_secrecy_rates.txt', 'w') as f:
            f.write(secrecy_rates)
            f.write("\n")

        step_secrecy_rates_arr.append(secrecy_rates)

        # TODO: DEFINE ACTION & ACTOR
        # POSSIBLY DEFINE RANDOM ACTION INPUT VECTOR WITH ASSOCIATED POLICY GRADIENTS
        state_tensor = jnp.array(state)
        uav = env.uavs[0] # UAV-BS
        #action = uav.move()
        next_state, reward, done, _, _ = env.step(action)
        buffer.store([state, action, reward, next_state, done])
        tot_reward += reward
        state = next_state

        if len(buffer) >= batch_size:
            b_idx, batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

            # TODO: ACTOR & CRITIC LOSS HERE 
            # Policy gradient calculation happens here

        step += 1

        # TODO: INTERFACE THE PER WITH THE PPO ALGORITHM

        # TODO: Gather UAV data and store it in text files with a newline delimiter
        # PLOT THIS DATA IN SEPARATE SCRIPTS 
        # WRITE BASH SCRIPT TO RUN ALL OF THE SCRIPTS CONSECUTIVELY 

total_runtime_end = time.time()

# TODO: STORE STEP-WISE DATA IN RESPECTIVE TEXT FILES
# IN PLOTTING SCRIPT, TAKE THE SUM OF THE DATA AND/OR OTHER OPERATIONS FOR FURTHER ANALYSIS

total_runtime = total_runtime_end - total_runtime_start
