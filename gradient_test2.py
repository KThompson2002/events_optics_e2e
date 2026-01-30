import torch
from senpi.sim.simulator import EventSimulator
from senpi.sim.params import make_params

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

params = make_params()
params["return_events"] = 1
params["return_frame"] = 1  # <-- turn on frames

# reduce noise for clarity
params["sensor_noise"] = 0
params["shot_noise"] = 0
params["sigma_pos"] = 0.0
params["sigma_neg"] = 0.0

sim = EventSimulator(params)

F, H, W = 32, 64, 64
base = torch.rand(1, H, W, device=device)
I = torch.cat([(base + 0.01 * t).clamp(0, 1) for t in range(F)], dim=0)
I.requires_grad_(True)

out = sim.forward(I)
events, frames = out[0], out[1]

print("events:", events.dtype, "requires_grad:", events.requires_grad, "shape:", tuple(events.shape))
print("frames:", frames.dtype, "requires_grad:", frames.requires_grad, "shape:", tuple(frames.shape))

# Gradient test THROUGH FRAMES (not events)
loss = frames.float().mean()
loss.backward()

print("I.grad is None?:", I.grad is None)
print("mean(|grad|):", I.grad.abs().mean().item() if I.grad is not None else None)
print("max(|grad|):", I.grad.abs().max().item() if I.grad is not None else None)

