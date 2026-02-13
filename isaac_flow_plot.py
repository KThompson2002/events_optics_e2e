import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIG --------
npz_path = "train_log_flow.npz"
# ------------------------

# Load data
data = np.load(npz_path)

# Unpack
global_step = data["global_step"]
loss        = data["loss"]
photo_loss  = data["photo_loss"]
smooth_loss = data["smooth_loss"]
epoch       = data["epoch"]
step        = data["step"]

print("Loaded keys:")
print("global_step:", global_step.shape)
print("loss:", loss.shape)
print("photo_loss:", photo_loss.shape)
print("smooth_loss:", smooth_loss.shape)
print("epoch:", epoch.shape)
print("step:", step.shape)

# -------- 1. Total loss vs global step --------
plt.figure()
plt.plot(global_step, loss, label="total loss")
plt.xlabel("Global step")
plt.ylabel("Loss")
plt.title("Total Loss vs Global Step")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------- 2. Photometric & smoothness loss --------
plt.figure()
plt.plot(global_step, photo_loss, label="photometric loss")
plt.plot(global_step, smooth_loss, label="smoothness loss")
plt.xlabel("Global step")
plt.ylabel("Loss")
plt.title("Loss Components vs Global Step")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------- 3. Loss vs training step (within epoch) --------
plt.figure()
plt.plot(step, loss)
plt.xlabel("Step (within epoch)")
plt.ylabel("Loss")
plt.title("Loss vs Step")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- 4. Loss vs epoch --------
plt.figure()
plt.plot(epoch, loss, ".")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- 5. Combined diagnostic plot --------
plt.figure(figsize=(10, 6))
plt.plot(global_step, loss, label="total")
plt.plot(global_step, photo_loss, label="photo")
plt.plot(global_step, smooth_loss, label="smooth")
plt.xlabel("Global step")
plt.ylabel("Loss")
plt.title("Training Loss Breakdown")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
