# lesioning cortical areas and comparing using MSE (healthy vs lesioned)

import os, sys, glob, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.pytorch_models import LightningRNNModule
from src.dataset import YangTasks, collate_fn


# checkpoint helpers

def newest_ckpt_or_none():
    """Prefer newest last.ckpt; else newest *.ckpt under common run folders."""
    roots = ["saved_models", "lightning_logs", "outputs"]
    last_candidates, any_candidates = [], []
    for root in roots:
        last_candidates += glob.glob(os.path.join(root, "**", "last.ckpt"), recursive=True)
        any_candidates  += glob.glob(os.path.join(root, "**", "*.ckpt"),   recursive=True)
    if last_candidates:
        return max(last_candidates, key=os.path.getmtime)
    if any_candidates:
        return max(any_candidates, key=os.path.getmtime)
    return None


def resolve_ckpt_path(arg_path: str | None):
    if arg_path:
        p = os.path.normpath(arg_path)
        if os.path.isdir(p):
            lastp = os.path.join(p, "last.ckpt")
            if os.path.exists(lastp):
                return lastp
            cand = glob.glob(os.path.join(p, "*.ckpt"))
            if cand:
                return max(cand, key=os.path.getmtime)
            raise FileNotFoundError(f"No .ckpt found inside directory: {p}")
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"Checkpoint path not found: {p}")
    ck = newest_ckpt_or_none()
    if ck is None:
        raise FileNotFoundError("No checkpoints found under saved_models/, lightning_logs/, or outputs/.")
    return ck


def try_load_saved_hps(ckpt_path, hp: dict):
    run_dir = os.path.dirname(ckpt_path)
    for fname in ["hp_pl_module.pkl", "task_hp.pkl"]:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "rb") as f:
                    saved = pickle.load(f)
                if isinstance(saved, dict):
                    hp.update(saved)
            except Exception as e:
                print(f"could not load {fname}: {e}")

    if hp.get("rule_trains"):
        print(f"Using rule_trains from run: {hp['rule_trains']}")
    else:
        print("rule_trains missing after merge; defaults will be used.")
    return hp


def apply_defaults(hp: dict):
    """Fill in anything the model/dataset expects, without overwriting known values."""
    defaults = {
        "loss_type": "lsq",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "dt": 20,
        "tau": 100,
        "noise": 0.0,
        "w_rec_init": "diag",
        "sigma_rec": 0.05,
        "activation": "tanh",
        "name": "cernn",
        "species": "human",

        # constraints should be a dict 
        "constraints": {"mask_within_area_weights": True, "zero_weights_thres": 0.0},

        # areas + duplicates
        "sensory": ["L_V1", "L_3b"],
        "motor":   ["L_FEF"],
        "duplicate": {"L_V1": 5, "L_3b": 5, "L_FEF": 5},

        # tasks / data
        "ruleset": "all",
        "rule_trains": [
            "fdgo_s1","fdgo_v1",
            "reactgo_s1","reactgo_v1",
            "delaygo_s1","delaygo_v1",
            "fdanti_s1","fdanti_v1",
            "reactanti_s1","reactanti_v1",
            "delayanti_s1","delayanti_v1",
            "dmsgo","dmsnogo","dmcgo","dmcnogo",
        ],
        "rule_probs": None,
        "trials_per_epoch": 200,
        "batch_size_val": 64,
        "in_type": "normal",

        # YangTasks internals
        "n_eachring": 2,
        "sigma_x": 0.01,
        "alpha": None,   # set below if missing

        # IO (will be overwritten by ckpt shapes if present)
        "n_input": 21,
        "n_output": 3,   # [fix, sin, cos]
    }

    for k, v in defaults.items():
        if k not in hp or hp[k] is None:
            hp[k] = v
            print(f"using default for {k}: {v}")

    # constraints tidy-up
    if isinstance(hp.get("constraints"), bool):
        hp["constraints"] = {
            "mask_within_area_weights": bool(hp["constraints"]),
            "zero_weights_thres": 0.0,
        }
    if isinstance(hp.get("constraints"), dict) and "zero_weiights_thres" in hp["constraints"]:
        hp["constraints"]["zero_weights_thres"] = hp["constraints"].pop("zero_weiights_thres")

    # duplicate can be int → expand to dict
    if isinstance(hp.get("duplicate"), int):
        d = int(hp["duplicate"])
        hp["duplicate"] = {"L_V1": d, "L_3b": d, "L_FEF": d}

    # alpha = dt/tau if not set
    if hp.get("alpha") in (None, 0):
        hp["alpha"] = float(hp["dt"]) / float(hp["tau"])

    # indices for rule one-hot
    if "rule_start" not in hp or hp["rule_start"] is None:
        hp["rule_start"] = 5
    if "n_rule" not in hp or hp["n_rule"] is None:
        hp["n_rule"] = len(hp["rule_trains"])

    # RNG + regularisers
    if "rng" not in hp or hp["rng"] is None:
        hp["rng"] = np.random.RandomState(0)
    if "regularisers" not in hp or hp["regularisers"] is None:
        hp["regularisers"] = {}

    return hp


# evaluation using MSE

@torch.no_grad()
def eval_once_mse(model, loader, device):
    """
    Compute masked mean squared error (overall and per rule).
    """
    # reset per-rule pointer so we iterate rules in order
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "current_rule_index"):
        loader.dataset.current_rule_index = 0

    model.eval()
    mse = nn.MSELoss(reduction="mean")

    total, n = 0.0, 0
    by_rule = {}

    for trial in loader:  
        x = trial.x.to(device)
        y = trial.y.to(device)
        m = trial.c_mask.to(device)
        rule = trial.rule

        # run underlying RNN 
        out, _ = model.model(x)  # (B, T, C)

        # mask loss (comparing only where mask is greater than 0)
        loss = mse(out * m, y * m).item()

        total += loss
        n += 1
        by_rule.setdefault(rule, []).append(loss)

    overall = total / max(n, 1)
    by_rule = {k: float(np.mean(v)) for k, v in by_rule.items()}
    return overall, by_rule


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate healthy vs lesioned performance (MSE).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to a .ckpt OR a run folder. If omitted, auto-picks newest.")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                   help="Device to run on. Default: cuda if available, else cpu.")
    p.add_argument("--areas", nargs="+", default=["L_FEF"],
                   help="Space-separated list of cortical areas to lesion, e.g. --areas L_V1 L_3b L_FEF")
    p.add_argument("--list-areas", action="store_true",
                   help="Print available cortical areas from the checkpoint and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_ckpt_path(args.ckpt)
    print("\nsettings")
    print("ckpt :", ckpt_path)

    # loading checkpoint 
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # hyperparameters
    hp = dict(ckpt.get("hyper_parameters", {}))
    hp = try_load_saved_hps(ckpt_path, hp)
    hp = apply_defaults(hp)

    # infer IO shapes from the checkpoint tensors to prevent size mismatch
    sd = ckpt["state_dict"]
    try:
        hp["n_input"]  = int(sd["model.rnn.rnncell.weight_ih"].shape[1])
        hp["n_output"] = int(sd["model.readout.weight"].shape[0])
    except KeyError as e:
        print(f"Note: could not infer IO shapes from ckpt ({e}); using defaults")

    # building model & loading weights
    device = torch.device(device_str)
    print("dev  :", device)
    lm = LightningRNNModule(hp)
    miss = lm.load_state_dict(sd, strict=False)
    if miss.missing_keys or miss.unexpected_keys:
        print("Note: state_dict loaded with non-strict keys:",
              {"missing": miss.missing_keys, "unexpected": miss.unexpected_keys})
    lm.to(device)

    # data loader ie. one Trial per rule per iteration
    val_ds = YangTasks(hp, mode="val")
    val_ld = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn, shuffle=False)

    # area listing
    ce = lm.model.ce  # HumanCorticalEmbedding
    if args.list_areas:
        print("\nAvailable cortical areas:")
        print(", ".join(sorted(ce.area2idx.keys())))
        return

    # healthy brain
    healthy_mse, healthy_rules = eval_once_mse(lm, val_ld, device)

    # lesioning certain units 
    missing = [a for a in args.areas if a not in ce.area2idx]
    if missing:
        raise ValueError(f"Areas not found in model: {missing}. "
                         f"Available (sample): {sorted(list(ce.area2idx.keys()))[:10]} ...")

    # build index ranges to zero, robust to different 'duplicates' shapes
    ranges = []
    for area in args.areas:
        start = ce.area2idx[area]
        # duplicates may be dict with some keys, or a single int, or missing
        dup = 1
        try:
            if isinstance(ce.duplicates, dict):
                dup = int(ce.duplicates.get(area, 1))
            elif isinstance(ce.duplicates, (int, np.integer)):
                dup = int(ce.duplicates)
        except Exception:
            dup = 1
        end = start + dup
        ranges.append((area, start, end))

    pretty = ", ".join([f"{a}[{s}:{e})" for a, s, e in ranges])
    print(f"\nLesioning areas: {pretty}")

    def lesion_hook(module, inputs, output):
        # rnn forward returns (h_seq, other); zero selected slices in h_seq
        h_seq, other = output
        h_seq = h_seq.clone()
        for _, s, e in ranges:
            h_seq[:, :, s:e] = 0
        return h_seq, other

    handle = lm.model.rnn.register_forward_hook(lesion_hook)
    les_mse, les_rules = eval_once_mse(lm, val_ld, device)
    handle.remove()

    print("\nMSE RESULTS")
    print(f"Healthy  MSE: {healthy_mse:.4f}")
    print(f"Lesioned MSE: {les_mse:.4f}")
    print(f"Δ MSE: {(les_mse - healthy_mse):+.4f}")

    # per-rule deltas
    rules = sorted(set(list(healthy_rules.keys()) + list(les_rules.keys())))
    for r in rules:
        h = healthy_rules.get(r, float('nan'))
        l = les_rules.get(r, float('nan'))
        d = (l - h) if (not np.isnan(h) and not np.isnan(l)) else float('nan')
        print(f"  {r:15s} {h:.4f} → {l:.4f} (Δ {d:+.4f})")


if __name__ == "__main__":
    main()
