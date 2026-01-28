import torch
import senpi

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Create simulator (SENPI EBI API)
sim = senpi.sim.EventSimulator()

# Create photometric input: [f, x, y] = [frames, H, W]
F, H, W = 32, 64, 64

# Force events (a brightness ramp) so n>0
base = torch.rand(1, H, W, device=device)
I = torch.cat([(base + 0.01 * t).clamp(0, 1) for t in range(F)], dim=0)
I.requires_grad_(True)

events = sim.forward(I)  # expected [n,4] in [t,x,y,p]

print("events type:", type(events))
print("events shape:", tuple(events.shape) if isinstance(events, torch.Tensor) else None)
print("events dtype:", events.dtype if isinstance(events, torch.Tensor) else None)
print("events requires_grad:", events.requires_grad if isinstance(events, torch.Tensor) else None)

if not isinstance(events, torch.Tensor) or events.numel() == 0:
    raise RuntimeError("No events tensor produced (or n=0).")

# Differentiable scalar (avoid using events.shape[0])
loss = events.float().sum()
loss.backward()

print("I.grad is None?:", I.grad is None)
print("mean(|grad|):", I.grad.abs().mean().item() if I.grad is not None else None)
print("max(|grad|):", I.grad.abs().max().item() if I.grad is not None else None)
