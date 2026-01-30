import torch
from senpi.sim.simulator import EventSimulator
from senpi.sim.params import make_params


def soft_events(I, thr=0.1, sharpness=50.0, eps=1e-6):
    """
    I: [F,H,W] float, requires_grad=True
    returns:
      Epos, Eneg: [F-1,H,W] float in [0,1]
    """
    L = torch.log(I.clamp_min(eps))
    dL = L[1:] - L[:-1]  # [F-1,H,W]

    # smooth "firing" probabilities
    Epos = torch.sigmoid(sharpness * (dL - thr))
    Eneg = torch.sigmoid(sharpness * (-dL - thr))
    return Epos, Eneg


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Create simulator (SENPI EBI API)
params = make_params()
params["return_events"] = 1
params["return_frame"] = 1   # keep memory low for the gradient test

# (Optional) reduce noise for a cleaner gradient signal
params["sensor_noise"] = 0
params["shot_noise"] = 0
params["sigma_pos"] = 0.0
params["sigma_neg"] = 0.0

sim = EventSimulator(params)

# Create photometric input: [f, x, y] = [frames, H, W]
F, H, W = 32, 64, 64

# Force events (a brightness ramp) so n>0
base = torch.rand(1, H, W, device=device)
I = torch.cat([(base + 0.01 * t).clamp(0, 1) for t in range(F)], dim=0)
I.requires_grad_(True)

out = sim.forward(I)  # expected [n,4] in [t,x,y,p]

if isinstance(out, (tuple, list)):
    events = out[0]
    frames = out[1] if len(out) > 1 else None
else:
    events = out
    frames = None

print("events type:", type(events))
print("events shape:", tuple(events.shape) if isinstance(events, torch.Tensor) else None)
print("events dtype:", events.dtype if isinstance(events, torch.Tensor) else None)
print("events requires_grad:", events.requires_grad if isinstance(events, torch.Tensor) else None)
print("num events:", events.shape[0])

if not isinstance(events, torch.Tensor) or events.numel() == 0:
    raise RuntimeError("No events tensor produced (or n=0).")

# Differentiable scalar (avoid using events.shape[0])
# loss = events.float().sum()
Epos, Eneg = soft_events(I, thr=0.1, sharpness=50.0)
loss = (Epos + Eneg).mean()
loss.backward()
print(I.grad.abs().mean().item())

# loss.backward()

print("I.grad is None?:", I.grad is None)
print("mean(|grad|):", I.grad.abs().mean().item() if I.grad is not None else None)
print("max(|grad|):", I.grad.abs().max().item() if I.grad is not None else None)
