import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Spike-FlowNet'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from isaac_flow_sequence import IsaacFlowSequence
from models.FlowNetS_spike import FlowNetS_spike
from multiscaleloss import warp, charbonnier_loss, smooth_loss


# ---------------------------------------------------------------------------
# Differentiable soft-event generation (from gradient_test.py)
# ---------------------------------------------------------------------------
def soft_events(I, thr=0.1, sharpness=50.0, eps=1e-6):
    """
    I: [F, H, W] float, requires_grad=True
    returns:
      Epos, Eneg: [F-1, H, W] float in [0, 1]
    """
    L = torch.log(I.clamp_min(eps))
    dL = L[1:] - L[:-1]                             # [F-1, H, W]
    Epos = torch.sigmoid(sharpness * (dL - thr))
    Eneg = torch.sigmoid(sharpness * (-dL - thr))
    return Epos, Eneg


# ---------------------------------------------------------------------------
# Convert soft_events → Spike-FlowNet input  (fully differentiable)
# ---------------------------------------------------------------------------
def soft_events_to_spike_input(Epos, Eneg, T_bins=5):
    """
    Epos : [N, H, W]  soft positive-event probabilities  (N = F-1)
    Eneg : [N, H, W]  soft negative-event probabilities
    T_bins : int       temporal bins per half (Spike-FlowNet default = 5)

    Returns
    -------
    [1, 4, H, W, T_bins]  float tensor on same device as Epos
        ch 0 : former ON   (positive events, first half)
        ch 1 : former OFF  (negative events, first half)
        ch 2 : latter ON   (positive events, second half)
        ch 3 : latter OFF  (negative events, second half)
    """
    N, H, W = Epos.shape
    half = N // 2
    bin_size = half // T_bins
    usable = T_bins * bin_size

    former_on  = Epos[:usable].reshape(T_bins, bin_size, H, W).sum(1)
    former_off = Eneg[:usable].reshape(T_bins, bin_size, H, W).sum(1)
    latter_on  = Epos[half:half + usable].reshape(T_bins, bin_size, H, W).sum(1)
    latter_off = Eneg[half:half + usable].reshape(T_bins, bin_size, H, W).sum(1)

    # Spike-FlowNet expects [B, 4, H, W, T]
    inp = torch.stack([
        former_on.permute(1, 2, 0),    # [H, W, T_bins]
        former_off.permute(1, 2, 0),
        latter_on.permute(1, 2, 0),
        latter_off.permute(1, 2, 0),
    ], dim=0)                           # [4, H, W, T_bins]
    return inp.unsqueeze(0)             # [1, 4, H, W, T_bins]


# ---------------------------------------------------------------------------
# Differentiable photometric loss  (replaces cv2.resize version)
# ---------------------------------------------------------------------------
def compute_photometric_loss_diff(prev_gray, next_gray, flows,
                                  weights=(1, 1, 1, 1)):
    """
    prev_gray : [B, 1, H, W]  GPU tensor
    next_gray : [B, 1, H, W]  GPU tensor
    flows     : tuple of [B, 2, h, w] at decreasing resolutions
    weights   : per-scale loss weights (highest-res first)
    """
    total = 0.0
    for i, flow in enumerate(flows):
        h, w = flow.shape[2], flow.shape[3]
        prev_r = F.interpolate(prev_gray, size=(h, w),
                               mode='bilinear', align_corners=True)
        next_r = F.interpolate(next_gray, size=(h, w),
                               mode='bilinear', align_corners=True)
        warped = warp(next_r, flow)
        total += weights[len(weights) - 1 - i] * charbonnier_loss(warped - prev_r)
    return total / len(flows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rgb_to_gray(video_tchw):
    """[T, 3, H, W] → [T, H, W]  (luminance, clamped away from zero)."""
    if video_tchw.shape[1] == 1:
        return video_tchw[:, 0]
    r, g, b = video_tchw[:, 0], video_tchw[:, 1], video_tchw[:, 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(1e-6, 1.0)


IMAGE_SIZE = 256          # Spike-FlowNet default spatial resolution
T_BINS     = 5            # temporal bins per half
SP_THRESH  = 0.75         # spiking threshold


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Data  (Isaac-Sim optical flow dataset)
    # ------------------------------------------------------------------
    DATA_ROOT = "./isaac_flow_data"
    T = 32                       # frames per sequence
    ds = IsaacFlowSequence(DATA_ROOT, T=T, stride=1)
    dl = DataLoader(ds, batch_size=2, shuffle=True,
                    num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # Spike-FlowNet model
    # ------------------------------------------------------------------
    model = FlowNetS_spike(batchNorm=False).to(device)
    model.train()

    # Optimizer (same defaults as Spike-FlowNet main_spike_flow_dt1.py)
    param_groups = [
        {'params': model.bias_parameters(), 'weight_decay': 0},
        {'params': model.weight_parameters(), 'weight_decay': 4e-4},
    ]
    optimizer = torch.optim.Adam(param_groups, lr=5e-5,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5, 10, 20, 30, 40, 50, 70, 90],
        gamma=0.7,
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log = {
        "global_step": [], "loss": [],
        "photo_loss": [], "smooth_loss": [],
        "epoch": [], "step": [],
    }
    global_step = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    NUM_EPOCHS = 100
    SMOOTH_WEIGHT = 10.0
    multiscale_weights = [1, 1, 1, 1]

    for epoch in range(NUM_EPOCHS):
        for step, (rgb_tchw, _gt_pose) in enumerate(dl):
            # rgb_tchw: [B, T, 3, H, W]
            rgb_tchw = rgb_tchw.to(device, non_blocking=True)
            B = rgb_tchw.shape[0]

            optimizer.zero_grad(set_to_none=True)

            losses = []
            photo_losses = []
            smooth_losses = []

            for b in range(B):
                video = rgb_tchw[b]                        # [T, 3, H, W]
                gray = rgb_to_gray(video)                  # [T, H, W]

                # Resize to Spike-FlowNet's expected 256x256
                gray = gray.unsqueeze(1)                   # [T, 1, H, W]
                gray = F.interpolate(gray, size=(IMAGE_SIZE, IMAGE_SIZE),
                                     mode='bilinear', align_corners=True)
                gray = gray.squeeze(1)                     # [T, 256, 256]

                # --- Differentiable soft events ---
                Epos, Eneg = soft_events(gray, thr=0.1, sharpness=50.0)

                # --- Convert to Spike-FlowNet input ---
                spike_in = soft_events_to_spike_input(Epos, Eneg,
                                                     T_bins=T_BINS)
                # spike_in: [1, 4, 256, 256, T_BINS]

                # --- Forward pass ---
                flows = model(spike_in.float(), IMAGE_SIZE, SP_THRESH)
                # flows: (flow1, flow2, flow3, flow4) during training

                # --- Photometric loss ---
                prev_gray = gray[0].unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
                next_gray = gray[-1].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                p_loss = compute_photometric_loss_diff(
                    prev_gray, next_gray, flows,
                    weights=multiscale_weights,
                )

                # --- Smoothness loss ---
                s_loss = smooth_loss(flows)

                loss_b = p_loss + SMOOTH_WEIGHT * s_loss
                losses.append(loss_b)
                photo_losses.append(p_loss.detach())
                smooth_losses.append(s_loss.detach())

            loss = torch.stack(losses).mean()

            loss.backward()
            optimizer.step()

            # Logging
            log["global_step"].append(global_step)
            log["epoch"].append(epoch)
            log["step"].append(step)
            log["loss"].append(loss.detach().float().cpu().item())
            log["photo_loss"].append(
                torch.stack(photo_losses).mean().float().cpu().item())
            log["smooth_loss"].append(
                torch.stack(smooth_losses).mean().float().cpu().item())
            global_step += 1

            if step % 20 == 0:
                print(f"epoch={epoch}  step={step}  "
                      f"loss={loss.item():.4f}  "
                      f"photo={log['photo_loss'][-1]:.4f}  "
                      f"smooth={log['smooth_loss'][-1]:.4f}")

        scheduler.step()

    # ------------------------------------------------------------------
    # Save logs
    # ------------------------------------------------------------------
    out_path = "train_log_flow.npz"
    np.savez(
        out_path,
        global_step=np.array(log["global_step"]),
        loss=np.array(log["loss"], dtype=np.float32),
        photo_loss=np.array(log["photo_loss"], dtype=np.float32),
        smooth_loss=np.array(log["smooth_loss"], dtype=np.float32),
        epoch=np.array(log["epoch"]),
        step=np.array(log["step"]),
    )
    print(f"Saved training logs to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
