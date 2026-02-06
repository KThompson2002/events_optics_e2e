import numpy as np
import matplotlib.pyplot as plt

d = np.load("train_log_pose.npz")
steps = d["global_step"]
loss  = d["loss"]
pred  = d["pred_xyz"]   # [N,3]
gt    = d["gt_xyz"]     # [N,3]

err = pred - gt
err_norm = np.linalg.norm(err, axis=1)

# 1) Loss curve
plt.figure()
plt.plot(steps, loss)
plt.xlabel("global_step")
plt.ylabel("MSE loss (per-sample)")
plt.title("Training loss")
plt.show()

# 2) Error norm curve
plt.figure()
plt.plot(steps, err_norm)
plt.xlabel("global_step")
plt.ylabel("||pred - gt|| (L2)")
plt.title("Position error norm")
plt.show()

# 3) GT vs Pred over time (per axis)
axes = ["x", "y", "z"]
for i, ax in enumerate(axes):
    plt.figure()
    plt.plot(steps, gt[:, i], label=f"gt_{ax}")
    plt.plot(steps, pred[:, i], label=f"pred_{ax}")
    plt.xlabel("global_step")
    plt.ylabel(ax)
    plt.title(f"GT vs Pred ({ax})")
    plt.legend()
    plt.show()

# 4) Pred vs GT scatter (per axis) + y=x line
for i, ax in enumerate(axes):
    plt.figure()
    plt.scatter(gt[:, i], pred[:, i], s=8)
    mn = min(gt[:, i].min(), pred[:, i].min())
    mx = max(gt[:, i].max(), pred[:, i].max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel(f"gt_{ax}")
    plt.ylabel(f"pred_{ax}")
    plt.title(f"Pred vs GT scatter ({ax})")
    plt.show()
