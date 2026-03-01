#import ploting library
import matplotlib.pyplot as plt
import numpy as np

# Store rewards per episode
all_rewards = []

# compute the r100 of the rewards per episode
def compute_r100(rewards):
    r100 = []
    for i in range(len(rewards)):
        start = max(0, i - 99)
        r100.append(np.mean(rewards[start:i+1])) #R100 [i] = mean(all_rewards[max (0, i-99) : i + 1])
    return r100

# plot the rewards and the r100
def plot_rewards_and_r100(rewards, env_name, net_name):
    r100 = compute_r100(rewards)
    plt.figure(figsize=(8,5))
    plt.plot(rewards, label="Reward", alpha=0.3)
    plt.plot(r100, label="R100 (moving average)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{env_name} – {net_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{env_name}_{net_name}_R100.png", dpi=150)
