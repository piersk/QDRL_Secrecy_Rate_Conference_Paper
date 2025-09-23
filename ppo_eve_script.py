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
model.learn(total_timesteps=10000)
model.save('ppo_lqdrl_secrecy_1')

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
    plt.savefig(f'eve_outputs/plots_eve_outputs/test3/{layer+1}_layers_uav_trajectory_{ep}_timestep_{t}.png')
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


total_runtime_end = time.time()

total_runtime = total_runtime_end - total_runtime_start
