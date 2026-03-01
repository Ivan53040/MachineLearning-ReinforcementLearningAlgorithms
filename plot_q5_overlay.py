import os, glob, csv
import argparse
import matplotlib.pyplot as plt

def load_csv(path):
    eps, r100, lr = [], [], None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(int(row["episode"]))
            r100.append(float(row["R100"]))
            if lr is None:
                lr = float(row["learning_rate"])
    return eps, r100, lr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="runs_q5", help="The same out_dir used during training")
    parser.add_argument("--env", default=None, help="Optional: filter filenames containing env (e.g., CartPole-v1)")
    args = parser.parse_args()

    results_dir = os.path.join(args.out_dir, "results")
    files = sorted(glob.glob(os.path.join(results_dir, "*_metrics.csv")))

    plt.figure(figsize=(8, 4.5))
    used = 0
    for fp in files:
        if args.env and (args.env not in os.path.basename(fp)):
            continue
        eps, r100, lr = load_csv(fp)
        if len(eps) == 0:
            continue
        label = f"lr={lr:g}"
        plt.plot(eps, r100, label=label)
        used += 1

    if used == 0:
        print("No CSV found. Check --out_dir and naming.")
        return

    plt.xlabel("Episode")
    plt.ylabel("R100 (100-episode moving avg)")
    plt.title("DQN: R100 vs Episodes (different learning rates)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)
    outpath = os.path.join(args.out_dir, "plots", "q5_lr_overlay.png")
    plt.savefig(outpath, dpi=200)
    print(f"Saved overlay plot to {outpath}")

if __name__ == "__main__":
    main()
