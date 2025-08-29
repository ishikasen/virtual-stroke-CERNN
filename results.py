import os, glob, re, sys, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--in", dest="in_dir", default=".", help="Folder with results_*.csv")
p.add_argument("--out", dest="out_dir", default="./results_out", help="Output folder")
args = p.parse_args()

IN_DIR  = os.path.abspath(args.in_dir)
OUT_DIR = os.path.abspath(args.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Scanning for summary CSVs in: {IN_DIR}")
cands = sorted([p for p in glob.glob(os.path.join(IN_DIR, "results_*.csv"))
                if not p.endswith("_per_rule.csv")])
if not cands:
    sys.exit("No summary CSVs found. Expected files like results_300_vis.csv, results_500_vis.csv, etc.")

EPOCH_PAT  = re.compile(r"(?:^|[_-])(300|500)(?:[_-]|$)")
NET_PATTERNS = {
    "vis":    re.compile(r"(?:^|[_-])vis(?:[_-]|$)", re.IGNORECASE),
    "dorsal": re.compile(r"(?:^|[_-])dorsal(?:[_-]|$)", re.IGNORECASE),
    "fpn":    re.compile(r"(?:^|[_-])fpn(?:[_-]|$)", re.IGNORECASE),
}
def detect_epoch(fn): m = EPOCH_PAT.search(fn); return int(m.group(1)) if m else None
def detect_network(fn):
    for k,pat in NET_PATTERNS.items():
        if pat.search(fn): return k
    return "unknown"

rows=[]
for path in cands:
    print("• Found:", os.path.basename(path))
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"healthy_mse","lesioned_mse","delta_mse"}
    if not need.issubset(df.columns):
        print("  ↳ skip, missing columns:", df.columns.tolist()); continue
    ep = detect_epoch(os.path.basename(path))
    net = detect_network(os.path.basename(path))
    if ep is None or net=="unknown":
        print("  ↳ skip, cannot detect epoch/network from filename"); continue
    r = df.iloc[0].to_dict()
    r.update({"epoch":ep,"network":net,"file":os.path.basename(path)})
    rows.append(r)

summary = pd.DataFrame(rows).sort_values(["network","epoch"])
if summary.empty:
    sys.exit("No usable summary rows; check filenames and columns.")
summary_path = os.path.join(OUT_DIR,"summary_all.csv")
summary.to_csv(summary_path, index=False)
print("Saved summary CSV →", summary_path)

# bar plot
order=["vis","dorsal","fpn"]
plt.figure(figsize=(7,4))
for i,net in enumerate(order):
    vals=[]
    for ep in [300,500]:
        row = summary[(summary.network==net)&(summary.epoch==ep)]
        vals.append(float(row["delta_mse"].iloc[0]) if not row.empty else np.nan)
    xs=[i-0.18, i+0.18]
    plt.bar(xs[0], vals[0], width=0.35, label="300" if i==0 else None)
    plt.bar(xs[1], vals[1], width=0.35, label="500" if i==0 else None)
plt.xticks(range(len(order)), ["VIS","DAN","FPN"])
plt.ylabel("ΔMSE (lesioned − healthy)")
plt.title("Lesion deficit by network and training budget")
plt.legend(title="Epochs")
plt.tight_layout()
fig_path = os.path.join(OUT_DIR,"bar_delta_mse.png")
plt.savefig(fig_path, dpi=300); plt.close()
print("Saved", fig_path)
