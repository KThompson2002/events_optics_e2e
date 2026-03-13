"""
Dump a handful of Isaac Sim frames directly to PNG using PIL (no cv2).
Also prints per-channel stats to diagnose where the mean=64 is coming from.

Usage:
  python3 dump_isaac_frames.py --data_root /home/lea1212/isaacsim/isaac_flow_data_v3
"""

import os, json, argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True)
parser.add_argument('--out_dir', default='./frame_check')
parser.add_argument('--n', type=int, default=5)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

with open(os.path.join(args.data_root, 'labels.json')) as f:
    labels = json.load(f)

# Sample frames spread across the dataset
indices = [int(i * len(labels) / args.n) for i in range(args.n)]

for idx in indices:
    row  = labels[idx]
    rgba = np.load(os.path.join(args.data_root, row['rgb']))   # [H, W, 4] uint8

    r, g, b, a = rgba[...,0], rgba[...,1], rgba[...,2], rgba[...,3]
    print(f"frame {idx:04d}:")
    print(f"  R  mean={r.mean():.1f}  max={r.max()}")
    print(f"  G  mean={g.mean():.1f}  max={g.max()}")
    print(f"  B  mean={b.mean():.1f}  max={b.max()}")
    print(f"  A  mean={a.mean():.1f}  max={a.max()}")

    # Save RGB only
    Image.fromarray(rgba[..., :3], 'RGB').save(
        os.path.join(args.out_dir, f'rgb_{idx:04d}.png'))

    # Save alpha channel as grayscale to see what it contains
    Image.fromarray(a, 'L').save(
        os.path.join(args.out_dir, f'alpha_{idx:04d}.png'))
