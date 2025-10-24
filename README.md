# QDRL_Secrecy_Rate_Conference_Paper
## Project Description
The work outlined in this repository is the simulation of a Quantum Deep Reinforcement Learning (QDRL) algorithm for the optimisation of secret key rate exchange between a UAV base station (UAV-BS) and a set of legitimate, authenticated ground users that are subject to eavesdropping. 
### Joint Optimisation Problem 
This problem has been modelled as a joint optimisation problem with a set of subproblems. 
The objective of the joint optimisation problem is to maximise the secrecy rate of the UAV-LU communications, subject to a set of constraints.
### Objectives
- Maximisation of Secrecy Rate
- Maximisation of Data Exchange Rate
- Maximisation of Energy Efficiency
- Minimisation of Energy Consumption
- Minimisation of UAV Trajectory to Optimal Location in 3-D Cartesian Space
## The Code
To run these simulations for yourself, follow these instructions.

``git clone https://github.com/piersk/QDRL_Secrecy_Rate_Conference_Paper.git``

Once the repository has been cloned, enter the project directory.

``cd qdrl_code``

Create a and activate Python virtual environment using pip.

``python3 -m venv ppo_env``

``source ppo_env/bin/activate``

Install the required dependencies within this virtual environment.

``pip install -r requirements.txt``

Run the PPO-based simulation, ensuring you have created the required directory structure used in the program. I may add a bash script to do this automatically for end users.

``python3 ppo_script.py --timesteps 200_000 --seed 42``

Run the QDRL-based simulation, again ensuring that you have the required directory structure as used in the program.

``python3 logging_script_LQDRL.py``

To visualise the results from these experiments, run the plotting utilities program.

``python3 ep_time_plot_script.py``
