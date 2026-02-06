import torch
import torch.nn.functional as F

from deeplens import GeoLens
from senpi.sim.simulator import EventSimulator
from senpi.sim.params import make_params

import csv


def make_checkerboard(H, W, squares=8, device="cuda", dtype=torch.float32):
    yy = torch.arange(H, device=device).view(H, 1)
    xx = torch.arange(W, device=device).view(1, W)
    cy = (yy // max(H // squares, 1)).clamp_max(squares - 1)
    cx = (xx // max(W // squares, 1)).clamp_max(squares - 1)
    board = ((cx + cy) % 2).to(dtype)
    return (0.15 + 0.85 * board).clamp(0, 1)  # [H,W]

def get_lens_geometry(lens):
    radii = []
    thickness = []

    for surf in lens.surfaces:   # GeoLens stores optical surfaces here
        if hasattr(surf, "radius"):
            radii.append(float(surf.radius))
        if hasattr(surf, "thickness"):
            thickness.append(float(surf.thickness))

    mean_radius = sum(radii)/len(radii) if radii else 0.0
    mean_thickness = sum(thickness)/len(thickness) if thickness else 0.0

    return mean_radius, mean_thickness

def translate_video(img_hw, T=32, dx_per_frame=0.6, dy_per_frame=0.0):
    """
    img_hw: [H,W]
    returns: video [T,1,H,W]
    """
    device, dtype = img_hw.device, img_hw.dtype
    H, W = img_hw.shape

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1,H,W,2]

    dx_norm = 2.0 * dx_per_frame / max(W - 1, 1)
    dy_norm = 2.0 * dy_per_frame / max(H - 1, 1)

    img = img_hw[None, None, ...]  # [1,1,H,W]
    frames = []
    for t in range(T):
        grid = base_grid.clone()
        grid[..., 0] = grid[..., 0] - t * dx_norm
        grid[..., 1] = grid[..., 1] - t * dy_norm
        ft = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        frames.append(ft[0])  # [1,H,W]
    return torch.stack(frames, dim=0)  # [T,1,H,W]


# ---------- Metric on SENPI event frames ----------
def sobel_edge_energy(frames_thw):
    """
    frames_thw: [T,H,W] float
    returns scalar (higher = sharper/more structured)
    """
    device, dtype = frames_thw.device, frames_thw.dtype
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3)
    x = frames_thw.unsqueeze(1)  # [T,1,H,W]
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return (gx.abs() + gy.abs()).mean()


def rgb_to_gray(video_tchw):
    if video_tchw.shape[1] == 1:
        return video_tchw[:, 0]
    r, g, b = video_tchw[:, 0], video_tchw[:, 1], video_tchw[:, 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(1e-6, 1.0)


def main():
    assert torch.cuda.is_available(), "This MVI expects CUDA for DeepLens + speed."
    device = torch.device("cuda")
    torch.manual_seed(0)

    # --------------------------
    # Line 1 (from sample): load lens + setup
    # --------------------------
    # Replace LENS_PATH with your lens json/file used in the sample config
    LENS_PATH = "/home/lea1212/CS496/DeepLens/datasets/lenses/camera/ef50mm_f1.8.json"
    lens = GeoLens(filename=LENS_PATH)
    lens = lens.to(device)

    # Synthetic video settings
    T, H, W = 32, 128, 128
    lens.set_sensor_res(sensor_res=(W, H))  # sample does this too :contentReference[oaicite:5]{index=5}

    # Optional: set constraints (matches sample pattern) :contentReference[oaicite:6]{index=6}
    lens.set_target_fov_fnum(rfov=20, fnum=4.0)

    # --------------------------
    # SENPI simulator
    # --------------------------
    params = make_params()
    params["return_frame"] = 1
    params["return_events"] = 1
    params["sensor_noise"] = 0
    params["shot_noise"] = 0
    params["sigma_pos"] = 0.0
    params["sigma_neg"] = 0.0
    sim = EventSimulator(params)

    # --------------------------
    # Line 2 (from sample): get lens optimizer
    # --------------------------
    # Use the same shape as sample lrs list; tune later :contentReference[oaicite:7]{index=7}
    lens_optim = lens.get_optimizer(lrs=[1e-4, 1e-4, 1e-2, 1e-4], decay=0.01)

    # Build differentiable moving scene (no dataset)
    base = make_checkerboard(H, W, squares=8, device=device)
    video_t1hw = translate_video(base, T=T, dx_per_frame=0.6, dy_per_frame=0.0)  # [T,1,H,W]
    video_tchw = video_t1hw.repeat(1, 3, 1, 1)  # [T,3,H,W] so lens.render works like sample

    lam = 0.05  # activity penalty weight

    import csv
    log_path = "e2e_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["iter", "edge", "activity", "loss", "num_events"])
        writer.writerow(["iter","edge","activity","loss","num_events","mean_radius","mean_thickness"])


    for it in range(60):
        # --------------------------
        # Line 4 (from sample): zero-grad
        # --------------------------
        lens_optim.zero_grad()

        # --------------------------
        # Line 3 (from sample): differentiable render
        # --------------------------
        # Treat T as batch. You can also pass depth/method/spp like in your snippet.
        rendered = lens.render(video_tchw, depth=-10000.0, method="ray_tracing", spp=16).clamp(0, 1)

        # Photometric frames for SENPI: [T,H,W]
        I = rgb_to_gray(rendered)

        # SENPI forward -> (events, eframes)
        events, eframes = sim.forward(I)

        # Metric: sharpness/structure in event frames
        edge = sobel_edge_energy(eframes)
        activity = eframes.abs().mean()
        loss = -(edge - lam * activity)  # maximize edge while discouraging trivial high activity

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            # writer.writerow([it, edge.item(), activity.item(), loss.item(), int(events.shape[0])])
            mean_r, mean_t = get_lens_geometry(lens)
            writer.writerow([it, edge.item(), activity.item(), loss.item(), int(events.shape[0]), mean_r, mean_t])

        if it % 10 == 0:
            lens.draw_layout(filename=f"lens_layout_{it:03d}.png")
        loss.backward()

        # --------------------------
        # Line 5 (from sample): step
        # --------------------------
        lens_optim.step()

        if it % 10 == 0 or it == 59:
            # We can't reliably access "the" lens param tensor here, so we log metric + event count.
            print(
                f"it={it:03d} edge={edge.item():.6f} activity={activity.item():.6f} "
                f"loss={loss.item():.6f} num_events={events.shape[0]}"
            )

    print("Done. If edge increases over iterations, DeepLens→SENPI E2E gradients are working.")


if __name__ == "__main__":
    main()

