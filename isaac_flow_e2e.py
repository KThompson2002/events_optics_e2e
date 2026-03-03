import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Spike-FlowNet'))

import random
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import numpy as np

from isaac_flow_sequence import IsaacFlowSequence
from models.FlowNetS_spike import FlowNetS_spike
from multiscaleloss import charbonnier_loss


# ---------------------------------------------------------------------------
# Differentiable soft-event generation
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
# Supervised GT flow loss — Charbonnier at each output scale
# ---------------------------------------------------------------------------
def compute_gt_flow_loss(flows, gt_at_image_size, weights=(1, 1, 1, 1)):
    """
    flows            : tuple of [1, 2, h, w] at decreasing resolutions
    gt_at_image_size : [2, IS, IS]  GT flow already scaled to IMAGE_SIZE pixels
    weights          : per-scale loss weights (highest-res first)

    Resizes GT to each scale and proportionally adjusts pixel-displacement
    values before computing Charbonnier loss.
    """
    total = 0.0
    gt = gt_at_image_size.unsqueeze(0)   # [1, 2, IS, IS]
    IS_h, IS_w = gt.shape[2], gt.shape[3]
    for i, flow in enumerate(flows):
        h, w = flow.shape[2], flow.shape[3]
        gt_r = F.interpolate(gt, size=(h, w),
                             mode='bilinear', align_corners=True).clone()
        gt_r[:, 0] *= (w / IS_w)    # scale x-displacement proportionally
        gt_r[:, 1] *= (h / IS_h)    # scale y-displacement proportionally
        diff = flow - gt_r
        total += weights[len(weights) - 1 - i] * charbonnier_loss(diff)
    return total / len(flows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rgb_to_gray(video_tchw):
    """[T, 3, H, W] → [T, H, W]  (luminance)."""
    if video_tchw.shape[1] == 1:
        return video_tchw[:, 0]
    r, g, b = video_tchw[:, 0], video_tchw[:, 1], video_tchw[:, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


IMAGE_SIZE         = 256   # Spike-FlowNet default spatial resolution
T_BINS             = 5     # temporal bins per half
SP_THRESH          = 0.75  # spiking threshold
EVENT_THRESH       = 0.1   # lowered from 0.5 → denser events for sparser scenes
MIN_EVENT_ACTIVITY = 10.0  # skip sequences with fewer total binary events


# ---------------------------------------------------------------------------
# Augmentation: h/v flip only — consistent with GT flow transformation.
# Rotation and random crop are excluded because they require rotating/warping
# the GT flow field, which is non-trivial and error-prone.
# ---------------------------------------------------------------------------
def augment_with_gt(video_tchw, gt_flow_hw, image_size):
    """
    Apply identical h-flip and v-flip to every frame in the sequence AND
    transform the GT flow consistently.  Then resize both to image_size.

    video_tchw  : [T, 3, H, W]  float32 in [0,1], any device
    gt_flow_hw  : [2, H, W]     GT accumulated flow in pixel units (same H,W)
    image_size  : int            target spatial size

    Returns
    -------
    video_aug   : [T, 3, IS, IS]
    gt_aug      : [2, IS, IS]   pixel units at image_size scale
    """
    T, C, H, W = video_tchw.shape
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5

    frames = []
    for t in range(T):
        frame = video_tchw[t]                                       # [3, H, W]
        if hflip:
            frame = TF.hflip(frame)
        if vflip:
            frame = TF.vflip(frame)
        frame = F.interpolate(frame.unsqueeze(0),
                              size=(image_size, image_size),
                              mode='bilinear',
                              align_corners=True).squeeze(0)
        frames.append(frame)
    video_aug = torch.stack(frames, dim=0)                          # [T,3,IS,IS]

    # Transform GT flow to match the same spatial flips
    gt = gt_flow_hw.unsqueeze(0)                                    # [1, 2, H, W]
    if hflip:
        # flip pixels left-right → x-component negates
        gt = torch.flip(gt, dims=[-1]).clone()
        gt[:, 0] = -gt[:, 0]
    if vflip:
        # flip pixels top-bottom → y-component negates
        gt = torch.flip(gt, dims=[-2]).clone()
        gt[:, 1] = -gt[:, 1]

    # Resize and scale pixel displacements proportionally
    gt = F.interpolate(gt, size=(image_size, image_size),
                       mode='bilinear', align_corners=True).clone()
    gt[:, 0] *= (image_size / W)
    gt[:, 1] *= (image_size / H)

    return video_aug, gt.squeeze(0)                                 # [2, IS, IS]


# ---------------------------------------------------------------------------
# Validation — Average End-point Error against Isaac Sim GT flow
# ---------------------------------------------------------------------------
def validate(model, val_loader, device):
    """
    Compute mean AEE over the validation set using GT flow supervision.

    GT flow per sample is accumulated across the first half of the T-frame
    window (gt_flow[:,1:half+1,:,:] summed over time), giving approximate
    total pixel displacement from frame[0] to frame[half].
    AEE is averaged only over pixels where the accumulated GT magnitude > 0.
    """
    model.eval()
    aee_total      = 0.0
    aee_gt_total   = 0.0
    pred_mag_total = 0.0
    n_samples      = 0

    with torch.no_grad():
        for rgb_tchw, gt_flow_tchw in val_loader:
            rgb_tchw     = rgb_tchw.to(device, non_blocking=True)
            gt_flow_tchw = gt_flow_tchw.to(device, non_blocking=True)
            B = rgb_tchw.shape[0]

            for b in range(B):
                video = rgb_tchw[b]                         # [T, 3, H, W]
                gray  = rgb_to_gray(video)                  # [T, H, W]

                # Resize (no augmentation during validation)
                gray = gray.unsqueeze(1)
                gray = F.interpolate(gray, size=(IMAGE_SIZE, IMAGE_SIZE),
                                     mode='bilinear', align_corners=True)
                gray = gray.squeeze(1)                      # [T, 256, 256]

                frame_max = gray.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
                gray = (gray / frame_max).clamp(0.0, 1.0)

                Epos, Eneg = soft_events(gray, thr=0.1, sharpness=50.0)

                Epos_bin = (Epos.detach() > EVENT_THRESH).float()
                Eneg_bin = (Eneg.detach() > EVENT_THRESH).float()
                spike_in = soft_events_to_spike_input(Epos_bin, Eneg_bin,
                                                      T_bins=T_BINS)

                if spike_in.sum() < MIN_EVENT_ACTIVITY:
                    continue

                # Eval mode returns only flow1: [1, 2, H, W]
                flow_pred = model(spike_in.float(), IMAGE_SIZE, SP_THRESH)

                # Accumulate GT over first half of window (matches training target)
                T_seq = rgb_tchw.shape[1]
                half  = (T_seq - 1) // 2                           # 15 for T=32
                gt_total = gt_flow_tchw[b, 1:half + 1].sum(dim=0)  # [2, H, W]

                # Resize GT to model output resolution and scale pixel values
                H_orig, W_orig = gt_total.shape[1], gt_total.shape[2]
                H_p, W_p = flow_pred.shape[2], flow_pred.shape[3]
                gt_resized = F.interpolate(
                    gt_total.unsqueeze(0), size=(H_p, W_p),
                    mode='bilinear', align_corners=True
                ).squeeze(0).clone()                        # [2, H_p, W_p]
                gt_resized[0] *= (W_p / W_orig)
                gt_resized[1] *= (H_p / H_orig)

                diff = flow_pred[0] - gt_resized            # [2, H, W]
                ee   = torch.sqrt(diff[0] ** 2 + diff[1] ** 2)  # [H, W]

                gt_mag   = torch.sqrt(gt_resized[0] ** 2 + gt_resized[1] ** 2)
                pred_mag = torch.sqrt(flow_pred[0, 0] ** 2 + flow_pred[0, 1] ** 2)
                mask = gt_mag > 0

                if mask.sum() > 0:
                    aee_total      += ee[mask].mean().item()
                    aee_gt_total   += gt_mag[mask].mean().item()
                    pred_mag_total += pred_mag.mean().item()
                    n_samples      += 1

    model.train()
    n = max(n_samples, 1)
    val_aee    = aee_total    / n
    val_aee_gt = aee_gt_total / n
    pred_mag   = pred_mag_total / n
    ratio      = val_aee / max(val_aee_gt, 1e-6)
    print(f"  [validate] n_samples={n_samples}  "
          f"val_aee={val_aee:.4f}  val_aee_gt={val_aee_gt:.4f}  "
          f"pred_mag={pred_mag:.4f}  ratio={ratio:.3f}")
    return val_aee, val_aee_gt, ratio


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Data  (Isaac-Sim optical flow dataset)
    # ------------------------------------------------------------------
    DATA_ROOT = "/home/lea1212/isaacsim/isaac_flow_data_v2"
    T = 32                       # frames per sequence
    ds_train = IsaacFlowSequence(DATA_ROOT, T=T, stride=1, split='train')
    ds_val   = IsaacFlowSequence(DATA_ROOT, T=T, stride=1, split='val')
    dl       = DataLoader(ds_train, batch_size=2, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=2, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"Dataset split — train: {len(ds_train)}, val: {len(ds_val)}")

    # ------------------------------------------------------------------
    # Spike-FlowNet model
    # ------------------------------------------------------------------
    model = FlowNetS_spike(batchNorm=False).to(device)
    model.train()

    param_groups = [
        {'params': model.bias_parameters(), 'weight_decay': 0},
        {'params': model.weight_parameters(), 'weight_decay': 4e-4},
    ]
    optimizer = torch.optim.Adam(param_groups, lr=2e-5,
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
        "global_step": [], "loss": [], "gt_loss": [],
        "epoch": [], "step": [],
        # validation — one entry per check (every 20 steps + end of epoch)
        "val_global_step": [], "val_aee": [], "val_aee_gt": [], "val_ratio": [],
        # epoch-boundary entries (subset of above, for convenience)
        "val_aee_epoch": [],
    }
    global_step = 0
    best_aee = float('inf')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    NUM_EPOCHS = 5
    multiscale_weights = [1, 1, 1, 1]
    half = (T - 1) // 2     # = 15 for T=32; GT accumulated over frames 0→half

    for epoch in range(NUM_EPOCHS):
        for step, (rgb_tchw, gt_flow_tchw) in enumerate(dl):
            # rgb_tchw     : [B, T, 3, H, W]
            # gt_flow_tchw : [B, T, 2, H, W]  GT per-frame displacement (pixel)
            rgb_tchw     = rgb_tchw.to(device, non_blocking=True)
            gt_flow_tchw = gt_flow_tchw.to(device, non_blocking=True)
            B = rgb_tchw.shape[0]

            optimizer.zero_grad(set_to_none=True)

            losses    = []
            gt_losses = []
            flow_mags = []

            for b in range(B):
                video   = rgb_tchw[b]          # [T, 3, H, W]
                gt_flow = gt_flow_tchw[b]      # [T, 2, H, W]

                # Accumulate GT displacement over first half of window.
                # gt_flow[0] is the pre-window frame (no useful flow);
                # gt_flow[1..half] cover transitions frame[0]→…→frame[half].
                gt_total = gt_flow[1:half + 1].sum(dim=0)   # [2, H, W]

                # Augment video and GT consistently (flip only — no rotation).
                # augment_with_gt also resizes both to IMAGE_SIZE.
                video_aug, gt_aug = augment_with_gt(video, gt_total, IMAGE_SIZE)

                gray = rgb_to_gray(video_aug)              # [T, 256, 256]

                # Per-frame normalize to [0,1]
                frame_max = gray.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
                gray = (gray / frame_max).clamp(0.0, 1.0)

                # Generate soft events and binarize
                Epos, Eneg = soft_events(gray, thr=0.1, sharpness=50.0)
                Epos_bin = (Epos.detach() > EVENT_THRESH).float()
                Eneg_bin = (Eneg.detach() > EVENT_THRESH).float()
                spike_in = soft_events_to_spike_input(Epos_bin, Eneg_bin,
                                                      T_bins=T_BINS)
                # spike_in: [1, 4, 256, 256, T_BINS]

                # Skip sequences with insufficient event activity
                if spike_in.sum() < MIN_EVENT_ACTIVITY:
                    continue

                # Forward pass — returns (flow1, flow2, flow3, flow4) in train
                flows = model(spike_in.float(), IMAGE_SIZE, SP_THRESH)
                flow1_mag = flows[0].detach().norm(dim=1).mean().item()

                # Supervised GT flow loss at all four scales
                gt_loss = compute_gt_flow_loss(flows, gt_aug,
                                               weights=multiscale_weights)

                losses.append(gt_loss)
                gt_losses.append(gt_loss.detach())
                flow_mags.append(flow1_mag)

            if not losses:
                continue

            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

            # Logging
            log["global_step"].append(global_step)
            log["epoch"].append(epoch)
            log["step"].append(step)
            log["loss"].append(loss.detach().float().cpu().item())
            log["gt_loss"].append(
                torch.stack(gt_losses).mean().float().cpu().item())
            global_step += 1

            if step % 20 == 0:
                mean_fmag = sum(flow_mags) / max(len(flow_mags), 1)
                print(f"epoch={epoch}  step={step}  "
                      f"loss={loss.item():.4f}  "
                      f"gt_loss={log['gt_loss'][-1]:.4f}  "
                      f"flow_mag={mean_fmag:.4f}")
                val_aee, val_aee_gt, val_ratio = validate(model, val_loader, device)
                log["val_global_step"].append(global_step)
                log["val_aee"].append(val_aee)
                log["val_aee_gt"].append(val_aee_gt)
                log["val_ratio"].append(val_ratio)

        scheduler.step()

        # Validation at end of epoch
        val_aee, val_aee_gt, val_ratio = validate(model, val_loader, device)
        log["val_global_step"].append(global_step)
        log["val_aee"].append(val_aee)
        log["val_aee_gt"].append(val_aee_gt)
        log["val_ratio"].append(val_ratio)
        log["val_aee_epoch"].append(epoch)
        print(f"[epoch end] epoch={epoch}")

        if val_aee < best_aee:
            best_aee = val_aee
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_aee": val_aee,
                    "spike_thresh": SP_THRESH,
                    "image_size": IMAGE_SIZE,
                },
                "spike_flownet_senpi_best.pth",
            )
            print(f"  => saved best checkpoint (val_aee={best_aee:.4f})")

    # ------------------------------------------------------------------
    # Save logs
    # ------------------------------------------------------------------
    out_path = "train_log_flow.npz"
    np.savez(
        out_path,
        global_step=np.array(log["global_step"]),
        loss=np.array(log["loss"], dtype=np.float32),
        gt_loss=np.array(log["gt_loss"], dtype=np.float32),
        epoch=np.array(log["epoch"]),
        step=np.array(log["step"]),
        val_global_step=np.array(log["val_global_step"]),
        val_aee=np.array(log["val_aee"], dtype=np.float32),
        val_aee_gt=np.array(log["val_aee_gt"], dtype=np.float32),
        val_ratio=np.array(log["val_ratio"], dtype=np.float32),
        val_aee_epoch=np.array(log["val_aee_epoch"]),
    )
    print(f"Saved training logs to {out_path}")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": NUM_EPOCHS - 1,
            "val_aee": log["val_aee"][-1] if log["val_aee"] else float("inf"),
            "spike_thresh": SP_THRESH,
            "image_size": IMAGE_SIZE,
        },
        "spike_flownet_senpi_final.pth",
    )
    print(f"Best val AEE: {best_aee:.4f}  (checkpoint: spike_flownet_senpi_best.pth)")
    print("Done.")


if __name__ == "__main__":
    main()
