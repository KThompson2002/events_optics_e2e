import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from deeplens import GeoLens
from senpi.sim.simulator import EventSimulator
from senpi.sim.params import make_params
from isaac_pose_sequence import IsaacPoseSequence
from torch.amp import autocast, GradScaler

def make_resnet18_time_as_channels(T: int, out_dim: int = 3):
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(
        in_channels=T, out_channels=64,
        kernel_size=7, stride=2, padding=3, bias=False
    )
    m.fc = nn.Linear(m.fc.in_features, out_dim)
    return m

# reuse rgb_to_gray from your e2e_demo.py
def rgb_to_gray(video_tchw):
    if video_tchw.shape[1] == 1:
        return video_tchw[:, 0]
    r, g, b = video_tchw[:, 0], video_tchw[:, 1], video_tchw[:, 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(1e-6, 1.0)

def make_video_square(video_tchw):
    T, C, H, W = video_tchw.shape
    s = min(H, W) if size is None else int(size)
    # if user asks for crop to larger than min dim, clamp
    s = min(s, H, W)
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    return video_tchw[:, :, y0:y0 + s, x0:x0 + s].contiguous()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ----------------
    # Data
    # ----------------
    DATA_ROOT = "/home/lea1212/isaacsim/isaac_pose_mvi/"
    T = 32
    ds = IsaacPoseSequence(DATA_ROOT, T=T, stride=1)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    # Peek H,W from one sample
    rgb0, _ = ds[0]
    _, _, H, W = rgb0.shape
    S = min(H, W)

    # ----------------
    # DeepLens
    # ----------------
    LENS_PATH = "/home/lea1212/CS496/DeepLens/datasets/lenses/camera/ef50mm_f1.8.json"
    lens = GeoLens(filename=LENS_PATH).to(device)
    lens.set_sensor_res(sensor_res=(S, S))

    # OPTIONAL: train lens too (E2E). If you only want to train CNN first, freeze lens params.
    train_lens = True
    if train_lens:
        lens_optim = lens.get_optimizer(lrs=[1e-4, 1e-4, 1e-2, 1e-4], decay=0.01)
    else:
        lens_optim = None
        for p in getattr(lens, "parameters", lambda: [])():
            p.requires_grad_(False)

    # ----------------
    # SENPI
    # ----------------
    params = make_params()
    params["return_frame"] = 1
    params["return_events"] = 0  # we only need frames for learning
    params["sensor_noise"] = 0
    params["shot_noise"] = 0
    params["sigma_pos"] = 0.0
    params["sigma_neg"] = 0.0
    sim = EventSimulator(params)

    # ----------------
    # CNN (time-as-channels)
    # ----------------
    model = make_resnet18_time_as_channels(T=T, out_dim=3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # If training lens too, either:
    #  - use separate optimizers, or
    #  - step both each iteration.
    # We’ll do both.
    scaler = GradScaler("cuda")

    for epoch in range(5):
        for step, (rgb_tchw, gt_xyz_cam) in enumerate(dl):
            # rgb_tchw: [B,T,3,H,W]
            rgb_tchw = rgb_tchw.to(device, non_blocking=True)
            gt_xyz_cam = gt_xyz_cam.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            if lens_optim is not None:
                lens_optim.zero_grad()

            with autocast("cuda"):
                B, T_, C, H, W = rgb_tchw.shape
                assert T_ == T

                # Flatten batch for lens.render (treat T as batch)
                # We want to render each sequence independently; easiest is loop over batch.
                # (MVI simple; can vectorize later.)
                preds = []
                losses = []
                for b in range(B):
                    video = rgb_tchw[b]  # [T,3,H,W]
                    video = make_video_square(video, mode="crop")



                    rendered = lens.render(video, depth=-10000.0, method="ray_tracing", spp=4).clamp(0, 1)
                    I = rgb_to_gray(rendered)                # [T,H,W]
                    _, eframes = sim.forward(I)              # eframes: [T,H,W] float, differentiable

                    x = eframes.unsqueeze(0)                 # [1,T,H,W]
                    pred_xyz = model(x)                      # [1,3]
                    preds.append(pred_xyz)

                    loss_b = F.mse_loss(pred_xyz.squeeze(0), gt_xyz_cam[b])
                    losses.append(loss_b)

                loss = torch.stack(losses).mean()

            scaler.scale(loss).backward()
            scaler.step(optim)
            if lens_optim is not None:
                scaler.step(lens_optim)
            scaler.update()

            if step % 20 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")

    print("Done.")

if __name__ == "__main__":
    main()