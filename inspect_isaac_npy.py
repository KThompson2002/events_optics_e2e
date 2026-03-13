"""
Quick diagnostic: inspect a few RGB .npy files from an Isaac Sim dataset.
Run on the server where the data lives.

Usage:
  python3 inspect_isaac_npy.py --data_root /home/lea1212/isaacsim/isaac_flow_data_v3
"""

import os, json, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True)
parser.add_argument('--n', type=int, default=5, help='Number of frames to inspect')
args = parser.parse_args()

with open(os.path.join(args.data_root, 'labels.json')) as f:
    labels = json.load(f)

print(f"Total frames in labels.json: {len(labels)}\n")

for row in labels[:args.n]:
    path = os.path.join(args.data_root, row['rgb'])
    arr  = np.load(path)
    print(f"  {row['rgb']}")
    print(f"    shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
