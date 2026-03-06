"""
camera_placement_eval.py
========================
End-to-end camera placement investigation:
  box scene → events (soft_events) → SpikeFlowNet (frozen) → flow features
  → linear regression → GT robot velocity

Produces:
  camera_placement_results.png    — R², MSE, event density bars per mounting
  camera_placement_per_axis.png   — per-axis (x/y/z) R² breakdown
  camera_placement_results.csv    — numeric table for the report

Run on a CUDA machine:
  cd events_optics_e2e
  source venv/bin/activate
  python camera_placement_eval.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Spike-FlowNet'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import models                                          # Spike-FlowNet models
from box_scene import generate_trajectory, MOUNTINGS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGE_SIZE   = 256
T_BINS       = 5
SP_THRESH    = 0.75
WINDOW       = 22     # frames per SpikeFlowNet input window
STRIDE       = 4      # window stride along trajectory
N_FRAMES     = 256    # total frames per mounting (1 full lap)
EVENT_THRESH = 0.10   # soft-event binarisation threshold
CKPT_PATH    = os.path.join(os.path.dirname(__file__),
                            '..', 'Spike-FlowNet', 'pretrain', 'checkpoint_dt1.pth.tar')


# ---------------------------------------------------------------------------
# Event helpers (reused from isaac_flow_e2e.py)
# ---------------------------------------------------------------------------

def soft_events(I, thr=0.1, sharpness=50.0, eps=1e-6):
    """I: [T, H, W] float in [0,1] → Epos, Eneg: [T-1, H, W]"""
    L  = torch.log(I.clamp_min(eps))
    dL = L[1:] - L[:-1]
    return torch.sigmoid(sharpness * (dL - thr)), torch.sigmoid(sharpness * (-dL - thr))


def soft_events_to_spike_input(Epos, Eneg, T_bins=5):
    """→ [1, 4, H, W, T_bins]"""
    N, H, W  = Epos.shape
    half     = N // 2
    bin_size = half // T_bins
    usable   = T_bins * bin_size

    def _bin(E, start): return E[start:start+usable].reshape(T_bins, bin_size, H, W).sum(1)

    inp = torch.stack([
        _bin(Epos, 0).permute(1, 2, 0),
        _bin(Eneg, 0).permute(1, 2, 0),
        _bin(Epos, half).permute(1, 2, 0),
        _bin(Eneg, half).permute(1, 2, 0),
    ], dim=0)
    return inp.unsqueeze(0)                            # [1, 4, H, W, T_bins]


# ---------------------------------------------------------------------------
# Flow feature extraction
# ---------------------------------------------------------------------------

def flow_to_features(flow: torch.Tensor) -> np.ndarray:
    """
    flow: [1, 2, H, W]
    Returns 1-D feature vector capturing spatial flow statistics.
    """
    u = flow[0, 0].cpu().float()
    v = flow[0, 1].cpu().float()
    mag = torch.sqrt(u**2 + v**2)

    def _skew(x):
        s = x.std().item()
        return (((x - x.mean())**3).mean().item()) / (s**3 + 1e-8)

    # Spatial histogram of flow magnitude (4 bins)
    mag_np   = mag.numpy().ravel()
    hist, _  = np.histogram(mag_np, bins=4, range=(0, mag_np.max() + 1e-6), density=True)

    feats = [
        u.mean().item(),   v.mean().item(),
        u.std().item(),    v.std().item(),
        mag.mean().item(), mag.std().item(),
        _skew(u),          _skew(v),
        *hist.tolist(),                                # 4 histogram bins
    ]
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-mounting evaluation
# ---------------------------------------------------------------------------

def evaluate_mounting(name: str, model, device) -> dict:
    print(f"  Generating scene for '{name}' ({N_FRAMES} frames) ...", end=' ', flush=True)
    frames, gt_vel = generate_trajectory(name, n_frames=N_FRAMES, image_size=IMAGE_SIZE)
    print("done")

    # [T, H, W, 3] → grayscale torch [T, H, W]
    rgb_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    gray  = 0.2989 * rgb_t[:, 0] + 0.5870 * rgb_t[:, 1] + 0.1140 * rgb_t[:, 2]
    fmax  = gray.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    gray  = (gray / fmax).clamp(0, 1)

    X_list, y_list, densities = [], [], []

    for s in range(0, N_FRAMES - WINDOW, STRIDE):
        g_win = gray[s:s + WINDOW].to(device)
        Epos, Eneg = soft_events(g_win)
        Ebin = (Epos > EVENT_THRESH).float()
        Nbin = (Eneg > EVENT_THRESH).float()

        total = Ebin.sum() + Nbin.sum()
        if total < 10:
            continue

        spike_in = soft_events_to_spike_input(Ebin, Nbin, T_bins=T_BINS)

        with torch.no_grad():
            flow = model(spike_in, IMAGE_SIZE, SP_THRESH)   # [1, 2, H, W]

        X_list.append(flow_to_features(flow))
        y_list.append(gt_vel[s + WINDOW // 2])
        densities.append(total.item() / (Ebin.numel() + Nbin.numel()))

    if len(X_list) == 0:
        print(f"  WARNING: no valid windows for '{name}'")
        return dict(r2=-1.0, r2_dims=[-1.]*3, mse=np.nan,
                    density=0.0, n=0, name=name)

    X   = np.stack(X_list)           # [N, features]
    y   = np.stack(y_list)           # [N, 3]
    den = float(np.mean(densities))

    # 5-fold cross-validated R² per velocity axis, then mean
    pipe     = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    r2_dims  = []
    for dim in range(3):
        cv_r2 = cross_val_score(pipe, X, y[:, dim], cv=min(5, len(X)),
                                scoring='r2')
        r2_dims.append(float(cv_r2.mean()))

    pipe.fit(X, y)
    mse = float(np.mean((pipe.predict(X) - y) ** 2))

    return dict(
        r2      = float(np.mean(r2_dims)),
        r2_dims = r2_dims,
        mse     = mse,
        density = den,
        n       = len(X),
        name    = name,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(results: list, out_dir: str):
    names  = [r['name']    for r in results]
    r2s    = [r['r2']      for r in results]
    mses   = [r['mse']     for r in results]
    dens   = [r['density'] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # Summary: R², MSE, event density
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, title, ylabel in zip(
        axes,
        [r2s, mses, dens],
        ['Trajectory Predictability (CV R²)',
         'Trajectory MSE (in-sample)',
         'Event Density'],
        ['CV R²  (flow → GT velocity)',
         'MSE  (m/step)²',
         'Mean events / pixel (normalised)'],
    ):
        ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Camera Mounting')
        ax.tick_params(axis='x', rotation=30)
    axes[0].axhline(0, color='k', linewidth=0.7, linestyle='--')

    plt.suptitle('SpikeFlowNet — Camera Placement Evaluation (Box Scene)', fontsize=12)
    plt.tight_layout()
    p1 = os.path.join(out_dir, 'camera_placement_results.png')
    plt.savefig(p1, dpi=150)
    print(f"Saved → {p1}")
    plt.close()

    # Per-axis R² breakdown
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    x = np.arange(len(names))
    w = 0.25
    ax2.bar(x - w, [r['r2_dims'][0] for r in results], w, label='x-vel', color='steelblue')
    ax2.bar(x,     [r['r2_dims'][1] for r in results], w, label='y-vel', color='salmon')
    ax2.bar(x + w, [r['r2_dims'][2] for r in results], w, label='z-vel', color='mediumseagreen')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30)
    ax2.set_ylabel('CV R²')
    ax2.set_title('Per-axis Trajectory Predictability by Camera Mounting')
    ax2.axhline(0, color='k', linewidth=0.7, linestyle='--')
    ax2.legend()
    plt.tight_layout()
    p2 = os.path.join(out_dir, 'camera_placement_per_axis.png')
    plt.savefig(p2, dpi=150)
    print(f"Saved → {p2}")
    plt.close()


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def save_csv(results: list, out_dir: str):
    path = os.path.join(out_dir, 'camera_placement_results.csv')
    lines = ['mounting,r2_mean,r2_x,r2_y,r2_z,mse,event_density,n_windows']
    for r in results:
        dims = r['r2_dims']
        lines.append(f"{r['name']},{r['r2']:.4f},"
                     f"{dims[0]:.4f},{dims[1]:.4f},{dims[2]:.4f},"
                     f"{r['mse']:.6f},{r['density']:.5f},{r['n']}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if str(device) == 'cpu':
        print("WARNING: SpikeFlowNet has hardcoded .cuda() calls — needs a GPU.")

    # Load SpikeFlowNet (frozen)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model = models.spike_flownets(data=ckpt)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("SpikeFlowNet loaded and frozen.\n")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = []

    for name in MOUNTINGS:
        print(f"[{name}]")
        res = evaluate_mounting(name, model, device)
        results.append(res)
        print(f"  n={res['n']}  R²={res['r2']:.3f}  MSE={res['mse']:.6f}  "
              f"density={res['density']:.4f}")
        print(f"  R² (x,y,z): {[f'{d:.3f}' for d in res['r2_dims']]}\n")

    # Sort best-to-worst by mean R²
    results.sort(key=lambda r: -r['r2'])

    # Print table
    print("=" * 65)
    print(f"{'Mounting':<12} {'R²':>7} {'R²x':>7} {'R²y':>7} {'R²z':>7}"
          f" {'MSE':>10} {'density':>9} {'N':>6}")
    print("-" * 65)
    for r in results:
        d = r['r2_dims']
        print(f"{r['name']:<12} {r['r2']:>7.3f} {d[0]:>7.3f} {d[1]:>7.3f}"
              f" {d[2]:>7.3f} {r['mse']:>10.6f} {r['density']:>9.4f} {r['n']:>6}")

    make_plots(results, out_dir)
    save_csv(results, out_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
