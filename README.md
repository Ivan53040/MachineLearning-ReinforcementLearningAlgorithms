Reinforcement Learning Algorithms in Python

📌 About

This repository demonstrates core reinforcement learning algorithms implemented in Python, including both classic prediction methods and real-environment training.

It covers:

- Monte Carlo (MC) prediction

- Temporal Difference (TD) learning

Random walk simulations

- LunarLander training (OpenAI Gym)

The LunarLander integration showcases RL agents learning to control a physics-based environment, bridging theory with practical application. This makes the repository a solid portfolio example for applied machine learning and RL.

📁 Repository Structure

- File / Folder	Description

- monte_carlo.py	Monte Carlo prediction implementation

- td_prediction.py	Temporal Difference learning

- random_walk.py	Simple environment simulation

- lunar_lander_train.py	LunarLander environment training with RL agent

- utils.py	Helper functions and shared utilities

- notebooks/	Interactive demos and visualisations

- results/	Plots, learning curves, and screenshots

- requirements.txt	Python dependencies
 
🛠 Installation & Setup

1. Clone the repository:

git clone https://github.com/Ivan53040/MachineLearning-ReinforcementLearningAlgorithms.git
cd MachineLearning-ReinforcementLearningAlgorithms

2. Ensure Python 3.7+ is installed:

python3 --version

3. Install dependencies:

pip install -r requirements.txt

Note: LunarLander requires gym:

pip install gym[box2d] matplotlib numpy

📌 Running the Code

Classic RL Scripts

Run MC, TD or random walk simulations:

python monte_carlo.py

python td_prediction.py

python random_walk.py

LunarLander Training

Train an agent in the LunarLander environment:

python lunar_lander_train.py

- The script runs episodes of LunarLander, learning via RL.

- Visual output and plots of rewards or performance are saved to results/.

Jupyter Notebooks

Open notebooks for interactive demonstrations:

jupyter notebook

🔍 Algorithms Implemented

🔹 Monte Carlo Prediction

- Learns state values using complete episodes

- Estimates value function by averaging returns

🔹 Temporal Difference (TD) Prediction

- Learns value function step-by-step

- Combines ideas from MC and dynamic programming

🔹 Random Walk

- Simple environment illustrating value updates and convergence

🔹 LunarLander Agent Training

- RL agent interacts with the LunarLander environment from OpenAI Gym

- Learns to control landing using reward feedback

- Demonstrates applied reinforcement learning in a physics-based simulation

📊 Results & Visualisations

For LunarLander:

- Plot cumulative reward vs. episodes

- Optionally save videos of agent performance

- Compare learning curves for different hyperparameters

For simulations:

- Plot value functions for MC vs TD learning

- Compare predicted vs. true state values

🎯 Learning Outcomes

By completing this project, you demonstrate:

✔ Understanding of core RL algorithms (MC, TD)

✔ Implementing RL agents from scratch

✔ Training agents in real environments (LunarLander)

✔ Analysing learning performance and convergence

✔ Bridging theory with practical applications


