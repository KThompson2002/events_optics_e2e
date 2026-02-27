import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIG --------
npz_path = "train_log_flow_2.npz"
# ------------------------

data = np.load(npz_path)

# Training metrics (one entry per optimizer step)
global_step = data["global_step"]
loss        = data["loss"]
photo_loss  = data["photo_loss"]
smooth_loss = data["smooth_loss"]
epoch       = data["epoch"]
step        = data["step"]

# Validation metrics (one entry per validation check: every 20 steps + epoch end)
val_gs      = data["val_global_step"]
val_aee     = data["val_aee"]
val_aee_gt  = data["val_aee_gt"]
val_epoch   = data["val_aee_epoch"]   # epoch indices of epoch-end checks

print("=== Loaded arrays ===")
for k in data.files:
    print(f"  {k}: shape={data[k].shape}  dtype={data[k].dtype}")

# Compute AEE ratio (how much better than predicting zero flow)
# ratio < 1.0 means the model is beating the zero-flow baseline
val_ratio = val_aee / np.maximum(val_aee_gt, 1e-6)

# Identify epoch-boundary validation steps for vertical markers.
# val_aee_epoch stores the epoch index at each epoch-end call; the
# corresponding global steps are the last val_gs entries — we recover
# them by counting: every epoch contributes (n_per_epoch + 1) val checks
# (one per 20 steps plus one at epoch end).  The simplest robust approach:
# mark the steps where the epoch changes in the training log.
epoch_boundary_steps = []
if len(epoch) > 1:
    for i in range(1, len(epoch)):
        if epoch[i] != epoch[i - 1]:
            epoch_boundary_steps.append(global_step[i])

# ================================================================
# Figure 1 — Training losses
# ================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("Training Losses")

ax = axes[0]
ax.plot(global_step, loss, label="total loss", linewidth=1.0)
for gs in epoch_boundary_steps:
    ax.axvline(gs, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
ax.set_ylabel("Loss")
ax.set_title("Total Loss")
ax.legend()
ax.grid(True)

ax = axes[1]
ax.plot(global_step, photo_loss,  label="photometric", linewidth=1.0)
ax.plot(global_step, smooth_loss, label="smoothness",  linewidth=1.0)
for gs in epoch_boundary_steps:
    ax.axvline(gs, color="gray", linestyle="--", linewidth=0.7, alpha=0.6,
               label="_epoch" if gs == epoch_boundary_steps[0] else None)
ax.set_xlabel("Global step")
ax.set_ylabel("Loss")
ax.set_title("Loss Components  (dashed = epoch boundary)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("plot_train_losses.png", dpi=150)
plt.show()

# ================================================================
# Figure 2 — Validation AEE
# ================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("Validation AEE")

ax = axes[0]
ax.plot(val_gs, val_aee,    label="val AEE",         linewidth=1.5, marker="o", markersize=3)
ax.plot(val_gs, val_aee_gt, label="GT mag (baseline)", linewidth=1.0,
        linestyle="--", color="gray")
for gs in epoch_boundary_steps:
    ax.axvline(gs, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
ax.set_ylabel("Pixels")
ax.set_title("AEE vs GT magnitude baseline  (below dashed = model beats zero-flow)")
ax.legend()
ax.grid(True)

ax = axes[1]
ax.plot(val_gs, val_ratio, label="AEE / GT mag", linewidth=1.5,
        marker="o", markersize=3, color="C2")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="zero-flow baseline")
for gs in epoch_boundary_steps:
    ax.axvline(gs, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
ax.set_xlabel("Global step")
ax.set_ylabel("Ratio")
ax.set_title("AEE ratio  (< 1.0 = model is directionally correct)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("plot_val_aee.png", dpi=150)
plt.show()

# ================================================================
# Figure 3 — Combined 2×2 dashboard
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Training Dashboard")

# Top-left: total loss
ax = axes[0, 0]
ax.plot(global_step, loss, linewidth=1.0, color="C0")
for gs in epoch_boundary_steps:
    ax.axvline(gs, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
ax.set_title("Total Loss")
ax.set_xlabel("Global step")
ax.set_ylabel("Loss")
ax.grid(True)

# Top-right: loss components
ax = axes[0, 1]
ax.plot(global_step, photo_loss,  label="photo",  linewidth=1.0, color="C1")
ax.plot(global_step, smooth_loss, label="smooth", linewidth=1.0, color="C3")
for gs in epoch_boundary_steps:
    ax.axvline(gs, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
ax.set_title("Loss Components")
ax.set_xlabel("Global step")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)

# Bottom-left: val AEE vs baseline
ax = axes[1, 0]
ax.plot(val_gs, val_aee,    label="val AEE",    linewidth=1.5, marker="o", markersize=3, color="C0")
ax.plot(val_gs, val_aee_gt, label="GT baseline", linewidth=1.0, linestyle="--", color="gray")
for gs in epoch_boundary_steps:
    ax.axvline(gs, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax.set_title("Validation AEE")
ax.set_xlabel("Global step")
ax.set_ylabel("Pixels")
ax.legend()
ax.grid(True)

# Bottom-right: AEE ratio
ax = axes[1, 1]
ax.plot(val_gs, val_ratio, linewidth=1.5, marker="o", markersize=3, color="C2")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
for gs in epoch_boundary_steps:
    ax.axvline(gs, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax.set_title("AEE Ratio  (< 1.0 = beating zero-flow)")
ax.set_xlabel("Global step")
ax.set_ylabel("AEE / GT mag")
ax.grid(True)

plt.tight_layout()
plt.savefig("plot_dashboard.png", dpi=150)
plt.show()

print("\nSaved: plot_train_losses.png, plot_val_aee.png, plot_dashboard.png")
