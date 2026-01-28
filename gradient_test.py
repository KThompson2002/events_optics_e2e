import torch

# --- SENPI IMPORT (ADJUST IF NEEDED) ---
from senpi.simulation.simulator import Simulator

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create simulator
sim = Simulator(
    threshold=0.2,
    refractory_period=1e-3,
    device=device
)

# Fake photometric input: [frames, H, W]
F, H, W = 16, 64, 64
I = torch.rand(F, H, W, device=device, requires_grad=True)

# Forward through SENPI
events = sim.forward(I)   # expected [n,4]

print("Events shape:", events.shape)
print("Requires grad on events?:", events.requires_grad)

# Build a dummy differentiable loss from events
# Use polarity column (p) as float
loss = events[:, 3].float().sum()

print("Loss:", loss.item())

# Backprop
loss.backward()

print("Gradient on input exists?:", I.grad is not None)
print("Mean |grad| on input:", I.grad.abs().mean().item() if I.grad is not None else None)
