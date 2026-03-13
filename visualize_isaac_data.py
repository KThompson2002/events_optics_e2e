"""
visualize_isaac_data.py
=======================
Convert saved Isaac Sim .npy files into PNG images and/or an mp4 video.

Usage
-----
  # PNGs + side-by-side video for the first 120 frames
  python3 visualize_isaac_data.py --data_root /home/lea1212/isaacsim/isaac_flow_data_v3 \
                                  --out_dir ./isaac_vis --max_frames 120 --video

  # Flow only, no video
  python3 visualize_isaac_data.py --data_root /path/to/data --out_dir ./isaac_vis --no_video
"""

import os
import json
import argparse
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Flow → HSV color wheel (same convention as SpikeFlowNet util.py)
# ---------------------------------------------------------------------------
def flow_to_color(flow_hw2: np.ndarray) -> np.ndarray:
    """
    flow_hw2 : [H, W, 2]  float32, pixel displacement (dx, dy)
    returns  : [H, W, 3]  uint8 BGR image
    """
    dx, dy = flow_hw2[..., 0], flow_hw2[..., 1]
    mag = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx)                          # [-π, π]

    # Normalize magnitude to [0, 1] (clip at 95th percentile for robustness)
    mag_max = np.percentile(mag, 95)
    if mag_max < 1e-6:
        mag_max = 1.0
    mag_norm = np.clip(mag / mag_max, 0, 1)

    # Hue encodes direction, saturation=1, value encodes magnitude
    hsv = np.zeros((*flow_hw2.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True,
                        help='Path to isaac_flow_data directory (contains labels.json)')
    parser.add_argument('--out_dir', default='./isaac_vis',
                        help='Output directory for PNGs and video')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Limit to first N frames (default: all)')
    parser.add_argument('--fps', type=int, default=15,
                        help='FPS for output video (default: 15)')
    parser.add_argument('--video', action=argparse.BooleanOptionalAction,
                        default=True, help='Write mp4 video (default: True)')
    args = parser.parse_args()

    labels_path = os.path.join(args.data_root, 'labels.json')
    with open(labels_path) as f:
        labels = json.load(f)

    if args.max_frames is not None:
        labels = labels[:args.max_frames]

    rgb_dir  = os.path.join(args.out_dir, 'rgb')
    flow_dir = os.path.join(args.out_dir, 'flow')
    side_dir = os.path.join(args.out_dir, 'side_by_side')
    os.makedirs(rgb_dir,  exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(side_dir, exist_ok=True)

    writer = None
    print(f"Processing {len(labels)} frames → {args.out_dir}")

    for i, row in enumerate(labels):
        # ---- RGB ----
        rgba = np.load(os.path.join(args.data_root, row['rgb']))  # [H,W,4] uint8
        rgb  = rgba[..., :3]                                       # drop alpha
        bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(rgb_dir, f'{i:05d}.png'), bgr)

        # ---- Flow ----
        if row.get('flow') is not None:
            flow = np.load(os.path.join(args.data_root, row['flow']))  # [H,W,2]
            flow_bgr = flow_to_color(flow)
        else:
            flow_bgr = np.zeros_like(bgr)
        cv2.imwrite(os.path.join(flow_dir, f'{i:05d}.png'), flow_bgr)

        # ---- Side-by-side ----
        H, W = bgr.shape[:2]
        if flow_bgr.shape[:2] != (H, W):
            flow_bgr = cv2.resize(flow_bgr, (W, H))
        side = np.concatenate([bgr, flow_bgr], axis=1)   # [H, 2W, 3]
        cv2.imwrite(os.path.join(side_dir, f'{i:05d}.png'), side)

        # ---- Video writer (lazy init after we know frame size) ----
        if args.video and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = os.path.join(args.out_dir, 'isaac_preview.mp4')
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W * 2, H))

        if args.video and writer is not None:
            writer.write(side)

        if i % 50 == 0:
            print(f"  {i}/{len(labels)}")

    if writer is not None:
        writer.release()
        print(f"Video saved to {os.path.join(args.out_dir, 'isaac_preview.mp4')}")

    print("Done.")


if __name__ == '__main__':
    main()
