import argparse
import os

import gymnasium as gym
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn_common import epsilon_by_frame, DqnNetSingleLayer, DqnNetTwoLayers, alpha_sync, DuellingDqn
from lib.experience_buffer import ExperienceBuffer, Experience
import yaml

# === BEGIN ADD: logging & plotting helpers (REPLACE WHOLE CLASS) ===
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 無視窗後端，直接存圖
import matplotlib.pyplot as plt

class MetricsLogger:
    def __init__(self, run_name: str, out_dir: str, learning_rate: float, env_name: str, save_every: int = 20):
        self.run = run_name
        self.lr = learning_rate
        self.env = env_name
        self.save_every = save_every
        self.ep = []
        self.rew = []

        # 統一到 out_dir
        self.out_dir = out_dir
        results_dir = os.path.join(out_dir, "results")
        plots_dir   = os.path.join(out_dir, "plots")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        self.csv_path = os.path.join(results_dir, f"{self.run}_metrics.csv")
        self.png_path = os.path.join(plots_dir,   f"{self.run}_R100.png")

    def log(self, episode_idx: int, episode_reward: float):
        self.ep.append(episode_idx)
        self.rew.append(episode_reward)
        if episode_idx % self.save_every == 0:
            self._save_csv_and_plot()

    def finalize(self):
        self._save_csv_and_plot()

    def _save_csv_and_plot(self):
        if not self.ep:
            return
        df = pd.DataFrame({
            "episode": self.ep,
            "reward": self.rew,
        })
        df["R100"] = df["reward"].rolling(100, min_periods=1).mean()
        df["learning_rate"] = self.lr
        df.to_csv(self.csv_path, index=False)

        plt.figure(figsize=(8, 4.5))
        plt.plot(df["episode"], df["reward"], alpha=0.35, label="Reward")
        plt.plot(df["episode"], df["R100"], linewidth=2.0, label="R100 (rolling mean)")
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.title(f"{self.env} | {self.run} | lr={self.lr:g}")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(self.png_path, dpi=160)
        plt.close()
# === END ADD ===

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", default="CartPole-v1", help="Full name of the environment, e.g. CartPole-v1, LunarLander-v3, etc.")
parser.add_argument("-c", "--config_file", default="config/dqn.yaml", help="Config file with hyper-parameters")
parser.add_argument("-n", "--network", default='single-hidden',
                    help="DQN network architecture `single-hidden` for single hidden layer, `two-hidden` for 2 hidden layers and `duelling-dqn` for duelling DQN",
                    choices=['single-hidden', 'two-hidden', 'duelling-dqn'])
parser.add_argument("-s", "--seed", type=int, help="Manual seed (leave blank for random seed)")
parser.add_argument("--lr", type=float, help="override learning_rate")
parser.add_argument("--run_name", default=None, help="Optional tag for this run (auto-filled if None)")
parser.add_argument("--eps-decay", type=int, help="override epsilon_decay")
parser.add_argument("--out_dir", default="runs_q7_ll", help="Where to save CSV/plots")
# --- new arg
parser.add_argument("--target-sync", type=int, help="override target_net_sync")  # e.g., 500 / 1000 / 5000

args = parser.parse_args()

# Hyperparameters for the requried environment
hypers = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

if args.env not in hypers:
    raise Exception(f'Hyper-parameters not found for env {args.env} - please add it to the config file (config/dqn.yaml)')
params = hypers[args.env]
# --- allow CLI override of key hypers ---
if args.lr is not None:
    params['learning_rate'] = args.lr
if args.eps_decay is not None:
    params['epsilon_decay'] = args.eps_decay
if args.target_sync is not None:
    params['target_net_sync'] = args.target_sync

env = gym.make(args.env)

# Set seeds
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on GPU")
# elif torch.backends.mps.is_available(): #Mac computers; (sometimes slower, e.g. 200fps vs 700fps)
#     device = torch.device("mps")
#     print("Training on MPS")
else:
    device = torch.device("cpu")
    print("Training on CPU")

if args.network == 'two-hidden':
    net = DqnNetTwoLayers(obs_size=env.observation_space.shape[0],
                          hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
                          n_actions=env.action_space.n).to(device)
    target_net = DqnNetTwoLayers(obs_size=env.observation_space.shape[0],
                                 hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
                                 n_actions=env.action_space.n).to(device)
elif args.network == 'single-hidden':
    net = DqnNetSingleLayer(obs_size=env.observation_space.shape[0],
                            hidden_size=params['hidden_size'],
                            n_actions=env.action_space.n).to(device)
    target_net = DqnNetSingleLayer(obs_size=env.observation_space.shape[0],
                                   hidden_size=params['hidden_size'],
                                   n_actions=env.action_space.n).to(device)
else:
    net = DuellingDqn(obs_size=env.observation_space.shape[0],
                      hidden_size=params['hidden_size'],
                      n_actions=env.action_space.n).to(device)
    target_net = DuellingDqn(obs_size=env.observation_space.shape[0],
                             hidden_size=params['hidden_size'],
                             n_actions=env.action_space.n).to(device)


print(net)
# --- tag run name with overrides so files are self-descriptive ---
suffix = []
if args.lr is not None:
    suffix.append(f"lr{args.lr:g}")
if args.eps_decay is not None:
    suffix.append(f"ed{args.eps_decay}")
if args.target_sync is not None:
    suffix.append(f"ts{args.target_sync}")   # <— add this
tag = ("_" + "_".join(suffix)) if suffix else ""
run_name = f"{args.env}_{args.network}{tag}"

# 用 out_dir + 把 lr、env 傳給 logger，CSV 會多一欄 learning_rate
logger = MetricsLogger(
    run_name=run_name,
    out_dir=args.out_dir,
    learning_rate=params['learning_rate'],
    env_name=args.env,
    save_every=20
)
# writer = SummaryWriter(comment="-CartPoleScratch")

buffer = ExperienceBuffer(int(params['replay_size']), device)

optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

frame_idx = 0
max_reward = -math.inf
all_rewards = []
losses = []
episode_reward = 0
r100 = -math.inf
episode_start = time.time()
start = time.time()
episode_frame = 0
episode_no = 0
visualizer_on = False

state, _ = env.reset()

def calculate_loss(net, target_net):
    states_v, actions_v, rewards_v, dones_v, next_states_v = buffer.sample(params['batch_size'])

    # get the Q value of the state - i.e. Q value for each action possible in that state
    # in CartPole there are 2 actions so this will be tensor of (2, BatchSize)
    Q_s = net.forward(states_v)

    # now we need the state_action_values for the actions that were selected (i.e. the action from the tuple)
    # actions tensor is already {100, 1}, i.e. unsqeezed so we don't need to unsqueeze it again
    # because the Q_s has one row per sample and the actions will be use as indices to choose the value from each row
    # lastly, because the gather will return a column and we need a row, we will squeeze it
    # gather on dim 1 means on rows
    state_action_values = Q_s.gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)

    # now we need Q_s_prime_a - i.e. the next state values
    # we get them from the target net
    # because there are 2 actions, we get a tensor of (2, BatchSize)
    # and because it's Sarsa max, we are taking the max
    # .max(1) will find maximum for each row and return a tuple (values, indices) - we need values so get<0>
    next_state_values = target_net.forward(next_states_v).max(1)[0]

    # calculate expected action values - discounted value of the state + reward
    expected_state_action_values = rewards_v + next_state_values.detach() * params['gamma'] * (1 - dones_v)

    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()

    if params['clip_gradient']:
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)

    optimizer.step()

    return loss


while True:
    frame_idx += 1

    # calculate the value of decaying epsilon
    epsilon = epsilon_by_frame(frame_idx, params)
    if np.random.random() < epsilon:
        # explore
        action = env.action_space.sample()
    else:
        # exploit
        state_a = np.asarray([state])
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = act_v.item()
        # print(action)

    # take step in the environment
    new_state, reward, terminated, truncated, _ = env.step(action)
    is_done = terminated or truncated
    episode_reward += reward

    # store the transition in the experience replay buffer
    exp = Experience(state, action, reward, is_done, new_state)
    buffer.append(exp)
    state = new_state

    # when the episode is done, reset and update progress
    if is_done:
        done_reward = episode_reward
        all_rewards.append(episode_reward)
        episode_no += 1

        state, _ = env.reset()
        if episode_reward > max_reward:
            max_reward = episode_reward

        if len(all_rewards) > 10 and len(losses) > 10:
            r100 = np.mean(all_rewards[-100:])
            l100 = np.mean(losses[-100:])
            dt = time.time() - episode_start
            if dt <= 0:
                dt = 1e-6  # 防止除以 0（Windows timer 精度）
            steps = max(1, frame_idx - episode_frame)
            fps = steps / dt
            print(f"Frame: {frame_idx}: Episode: {episode_no}, R100: {r100: .2f}, MaxR: {max_reward: .2f}, R: {episode_reward: .2f}, FPS: {fps: .1f}, L100: {l100: .2f}, Epsilon: {epsilon: .4f}")


            # visualize the training when reached 95% of the target R100; you should comment this out to speed up training
            if not visualizer_on and r100 > 0.95 * params['stopping_reward']:
                env = gym.make(args.env, render_mode='human')
                env.reset()
                env.render()
                visualizer_on = True

        # === ADD: 記錄本集總回報（放在 if is_done: 內）===
        logger.log(episode_no, done_reward)
        episode_reward = 0
        episode_frame = frame_idx
        episode_start = time.time()

    if len(buffer) < params['replay_size_start']:
        continue

    # do the learning
    loss = calculate_loss(net, target_net)
    losses.append(loss.item())


    if params['alpha_sync']:
        alpha_sync(net, target_net, alpha=1 - params['tau'])
    elif frame_idx % params['target_net_sync'] == 0:
        target_net.load_state_dict(net.state_dict())

    if r100 > params['stopping_reward']:
        print("Finished training")
        logger.finalize()  # === ADD ===
        

        name = f"{args.env}_{args.network}_nn_DQN_act_net_%+.3f_%d.dat" % (r100, frame_idx)
        if not os.path.exists(params['save_path']):
            os.makedirs(params['save_path'])
        torch.save(net.state_dict(), os.path.join(params['save_path'], name))

        break

    if frame_idx > params['max_frames']:
        print(f"Ran out of time at {time.time() - start}")
        logger.finalize()  # === ADD ===
        break

print(f"Completed training in {time.time() - start}")
