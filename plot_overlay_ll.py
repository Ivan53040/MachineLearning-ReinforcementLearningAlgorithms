import os, glob, re, csv, argparse
import matplotlib.pyplot as plt

def load_csv(path):
    eps, r100 = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(int(row["episode"]))
            r100.append(float(row["R100"]))
    return eps, r100

# parse run tags like: LunarLander-v3_single-hidden_lr0.001_ed5000_ts1000_metrics.csv
TAG_RE = re.compile(r".*_(?P<net>single-hidden|two-hidden|duelling-dqn)(?:_lr(?P<lr>[\d\.e-]+))?(?:_ed(?P<ed>\d+))?(?:_ts(?P<ts>\d+))?_metrics\.csv$")

def label_for(path, compare):
    m = TAG_RE.match(os.path.basename(path))
    if not m:
        return os.path.basename(path)
    d = m.groupdict()
    if compare == "lr":
        return f"lr={d.get('lr','?')}"
    if compare == "ed":
        return f"eps_decay={d.get('ed','?')}"
    if compare == "ts":
        return f"target_sync={d.get('ts','?')}"
    if compare == "arch":
        return d.get("net","?")
    return os.path.basename(path)

def keep_file(path, env, net_filter):
    base = os.path.basename(path)
    if env and env not in base:
        return False
    if net_filter and (f"_{net_filter}_" not in base and not base.endswith(f"_{net_filter}_metrics.csv")):
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs_q7_ll")
    ap.add_argument("--env", default="LunarLander-v3")
    ap.add_argument("--compare", choices=["arch","lr","ts","ed"], required=True,
                    help="arch=architectures, lr=learning-rate, ts=target-sync, ed=epsilon-decay")
    ap.add_argument("--network", default=None, help="optional filter when comparing lr/ts/ed (e.g., single-hidden)")
    ap.add_argument("--minlen", action="store_true", help="truncate all curves to common min episode length")
    args = ap.parse_args()

    results_dir = os.path.join(args.out_dir, "results")
    files = [fp for fp in glob.glob(os.path.join(results_dir, "*_metrics.csv"))
             if keep_file(fp, args.env, args.network if args.compare!="arch" else None)]

    if not files:
        print("No matching CSV files found.")
        return

    series = []
    min_len = None
    for fp in sorted(files):
        e, r = load_csv(fp)
        if len(e) < 10:
            continue
        series.append((fp, e, r))
        min_len = len(e) if min_len is None else min(min_len, len(e))

    plt.figure(figsize=(9,5))
    for fp, e, r in series:
        if args.minlen:
            e, r = e[:min_len], r[:min_len]
        plt.plot(e, r, label=label_for(fp, args.compare))

    plt.xlabel("Episode")
    plt.ylabel("R100 (100-episode moving avg)")
    title_map = {"arch":"Architectures","lr":"Learning rates","ts":"Target sync intervals","ed":"Epsilon-decay"}
    tlabel = title_map[args.compare]
    if args.compare != "arch" and args.network:
        tlabel += f" | net={args.network}"
    plt.title(f"{args.env}: {tlabel}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)
    out = os.path.join(args.out_dir, "plots", f"{args.env}_{args.compare}_overlay.png")
    plt.savefig(out, dpi=200)
    print(f"✅ Saved overlay: {out}")

if __name__ == "__main__":
    main()
