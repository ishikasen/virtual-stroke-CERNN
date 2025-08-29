"""
inspect_ckpt.py

Run:
  python inspect_ckpt.py                                  # auto-pick newest .ckpt
  python inspect_ckpt.py path/to/model.ckpt               # inspect specific .ckpt
"""

import sys, os, glob
import torch
from typing import Optional

# --- allowlist OmegaConf DictConfig so PyTorch 2.6 can unpickle your hparams safely ---
try:
    from omegaconf import DictConfig  # Hydra/omegaconf used in your training
    from torch.serialization import add_safe_globals
    add_safe_globals([DictConfig])
except Exception:
    pass  # if omegaconf isn't installed here, we’ll still try a best-effort load

SEARCH_ROOTS = ["saved_models", "lightning_logs"]

def find_newest_ckpt(roots=SEARCH_ROOTS) -> Optional[str]:
    candidates = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        patterns = [
            os.path.join(root, "**", "*.ckpt"),
            os.path.join(root, "**", "checkpoints", "*.ckpt"),
        ]
        for pat in patterns:
            candidates.extend(glob.glob(pat, recursive=True))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def pretty_print_hp(ckpt_path: str, hp: dict):
    print(f"\nCheckpoint: {ckpt_path}")
    print("---- Hyperparameters ----")

    species = str(hp.get("species", "unknown"))
    if species.lower() == "human":
        print("Species     : human   ✅ HUMAN CONFIG")
    elif species.lower() == "macaque":
        print("Species     : macaque 🚨 MACAQUE CONFIG")
    else:
        print(f"Species     : {species}")

    yaml_key = None
    for k in ("config_name", "cfg_name", "yaml_file", "config"):
        if k in hp:
            yaml_key = k
            break
    print(f"YAML File   : {hp[yaml_key] if yaml_key else '(not stored in checkpoint)'}")

    for key in ("name", "sensory", "motor", "n_rnn", "learning_rate"):
        if key in hp:
            print(f"{key:12s}: {hp[key]}")

    print("\nOther keys:")
    skip = {"name", "species", "sensory", "motor", "n_rnn", "learning_rate"}
    if yaml_key:
        skip.add(yaml_key)
    for k, v in hp.items():
        if k not in skip:
            print(f"{k:12s}: {v}")

def main():
    if len(sys.argv) >= 2:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = find_newest_ckpt()
        if not ckpt_path:
            print("❌ No .ckpt found. Train a model or pass a path: python inspect_ckpt.py path/to/model.ckpt")
            return
        print(f"(auto) Using newest checkpoint: {ckpt_path}")

    if not os.path.isfile(ckpt_path):
        print(f"❌ Not a file: {ckpt_path}")
        return

    # PyTorch 2.6: set weights_only=False to fully unpickle hparams (safe for your own ckpts)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print("⚠️ First load failed, retrying without weights_only arg…")
        ckpt = torch.load(ckpt_path, map_location="cpu")  # fallback for older versions

    hp = ckpt.get("hyper_parameters")
    if hp is None:
        print("❌ No 'hyper_parameters' found in this checkpoint.")
        return

    pretty_print_hp(ckpt_path, hp)

if __name__ == "__main__":
    main()
