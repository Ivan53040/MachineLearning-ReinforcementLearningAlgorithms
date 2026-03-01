# plot_r100.py
import sys, re, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 如果你的 dqn_gym.py 會把每個 episode 的回報印出來到終端，
# 可以把訓練時的輸出重導到檔案：  python dqn_gym.py ... > train.log
# 這支腳本就從 train.log 抓每個 episode reward 來畫 R100。

log_file = sys.argv[1] if len(sys.argv) > 1 else "train.log"
ep, rewards = [], []
pat = re.compile(r"Episode\s+(\d+).*(?:reward|Return)\s*[:=]\s*([-+]?\d+\.?\d*)", re.I)

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            ep.append(int(m.group(1)))
            rewards.append(float(m.group(2)))

if not rewards:
    print("No rewards found in", log_file)
    sys.exit(1)

df = pd.DataFrame({"episode": ep, "reward": rewards}).sort_values("episode")
df["R100"] = df["reward"].rolling(100, min_periods=1).mean()

plt.figure(figsize=(8,4.5))
plt.plot(df["episode"], df["reward"], alpha=0.3, label="Reward")
plt.plot(df["episode"], df["R100"], linewidth=2, label="R100 (rolling mean)")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("CartPole-v1 — single-hidden")
plt.legend(); plt.tight_layout()
plt.savefig("R100_cartpole_v1_single-hidden.png", dpi=160)
print("Saved: R100_cartpole_v1_single-hidden.png")
