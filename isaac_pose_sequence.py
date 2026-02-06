import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def quat_wxyz_to_R(q: torch.Tensor) -> torch.Tensor:
    """
    q: (...,4) wxyz
    returns: (...,3,3)
    """
    w, x, y, z = q.unbind(-1)
    # normalize just in case
    norm = torch.sqrt(w*w + x*x + y*y + z*z).clamp_min(1e-8)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1] + (3,3))
    return R

def target_in_camera_frame(cam_t, cam_q, tgt_t):
    """
    cam_t: (3,) world
    cam_q: (4,) wxyz world
    tgt_t: (3,) world
    returns: (3,) in camera frame
    """
    # camera rotation world-from-camera or camera-from-world?
    # With our usage: Rcw maps camera->world. Then world->camera is Rcw^T.
    Rcw = quat_wxyz_to_R(cam_q)           # (3,3)
    pc = (Rcw.transpose(-1,-2) @ (tgt_t - cam_t).unsqueeze(-1)).squeeze(-1)
    return pc

class IsaacPoseSequence(Dataset):
    """
    Produces sequences of length T:
      video: [T,H,W,4] uint8 on disk -> returned as float tensor [T,3,H,W] in [0,1]
      gt:    [3] target position in camera frame for the LAST frame in the sequence
    """
    def __init__(self, root_dir: str, T: int, stride: int = 1):
        self.root_dir = root_dir
        self.T = T
        self.stride = stride

        labels_path = os.path.join(root_dir, "labels.json")
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        # build sequence start indices
        self.starts = list(range(0, len(self.labels) - T + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        chunk = self.labels[s:s+self.T]

        frames = []
        for row in chunk:
            p = os.path.join(self.root_dir, row["rgb"])
            rgba = np.load(p)  # uint8 (H,W,4)
            frames.append(rgba)

        rgba = np.stack(frames, axis=0)  # [T,H,W,4]
        rgb = rgba[..., :3]              # drop alpha
        rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]

        # GT pose from the last frame of the sequence
        last = chunk[-1]
        cam_t = torch.tensor(last["camera_world_t"], dtype=torch.float32)
        cam_q = torch.tensor(last["camera_world_q_wxyz"], dtype=torch.float32)
        tgt_t = torch.tensor(last["target_world_t"], dtype=torch.float32)

        gt_cam = target_in_camera_frame(cam_t, cam_q, tgt_t)  # [3]
        return rgb, gt_cam
