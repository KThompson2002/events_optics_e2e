import os, json
import numpy as np
import torch
from torch.utils.data import Dataset


class IsaacFlowSequence(Dataset):
    """
    Loads sequences of length T from the isaac_flow_data dataset.

    Returns
    -------
    rgb : [T, 3, H, W]  float32 in [0, 1]
    gt_flow : [T-1, 2, H, W]  float32 pixel displacement (or zeros when
              ground-truth is unavailable for a frame pair)
    """

    def __init__(self, root_dir: str, T: int, stride: int = 1):
        self.root_dir = root_dir
        self.T = T

        labels_path = os.path.join(root_dir, "labels.json")
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        # Sequence start indices — each sequence is T consecutive frames.
        # Start from index 1 so every frame in the window has a valid flow
        # (frame 0 in the dataset has flow=null).
        self.starts = list(range(1, len(self.labels) - T + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        chunk = self.labels[s : s + self.T]

        # ---- RGB frames ----
        frames = []
        for row in chunk:
            rgba = np.load(os.path.join(self.root_dir, row["rgb"]))  # [H,W,4]
            frames.append(rgba)

        rgba = np.stack(frames, axis=0)           # [T, H, W, 4]
        rgb = rgba[..., :3]                       # drop alpha
        rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]

        # ---- Ground-truth optical flow ----
        H, W = rgb.shape[2], rgb.shape[3]
        gt_flows = []
        for row in chunk:
            if row["flow"] is not None:
                flow = np.load(os.path.join(self.root_dir, row["flow"]))  # [H,W,2]
                flow = torch.from_numpy(flow).permute(2, 0, 1).float()   # [2,H,W]
            else:
                flow = torch.zeros(2, H, W, dtype=torch.float32)
            gt_flows.append(flow)

        gt_flow = torch.stack(gt_flows, dim=0)    # [T, 2, H, W]

        return rgb, gt_flow
