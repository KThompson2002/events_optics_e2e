"""
Synthetic box scene renderer for camera placement evaluation.

Pure numpy + cv2 — no external 3D library needed.
The robot moves in a horizontal circle around a unit box at the origin.
Camera mounting is a fixed orientation offset relative to the robot body frame.
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Box geometry
# ---------------------------------------------------------------------------
_HALF = 0.5
BOX_VERTS = np.array([
    [-_HALF, -_HALF, -_HALF], [ _HALF, -_HALF, -_HALF],
    [ _HALF,  _HALF, -_HALF], [-_HALF,  _HALF, -_HALF],
    [-_HALF, -_HALF,  _HALF], [ _HALF, -_HALF,  _HALF],
    [ _HALF,  _HALF,  _HALF], [-_HALF,  _HALF,  _HALF],
], dtype=np.float64)

FACE_INDICES = [
    [0, 1, 2, 3],  # -z  (front in world)
    [5, 4, 7, 6],  # +z  (back)
    [4, 0, 3, 7],  # -x  (left)
    [1, 5, 6, 2],  # +x  (right)
    [3, 2, 6, 7],  # +y  (top)
    [4, 5, 1, 0],  # -y  (bottom)
]

# Distinct solid colors per face so edge-crossings generate strong events
FACE_COLORS = np.array([
    [0.90, 0.20, 0.20],   # red
    [0.20, 0.80, 0.20],   # green
    [0.25, 0.45, 0.90],   # blue
    [0.90, 0.80, 0.10],   # yellow
    [0.90, 0.50, 0.10],   # orange
    [0.60, 0.20, 0.90],   # purple
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Robot kinematics
# ---------------------------------------------------------------------------

def robot_frame(phi: float, radius: float, height: float):
    """
    Returns (position, forward, right, up) of robot at circle angle phi.

    Convention
    ----------
    forward  = tangent to CCW circle (direction of travel)
    right    = radially OUTWARD from box center
    up       = world [0, 1, 0]

    The box is always to the robot's LEFT (-right direction).
    """
    pos     = np.array([radius * np.sin(phi), height, radius * np.cos(phi)])
    forward = np.array([np.cos(phi), 0.0, -np.sin(phi)])
    up      = np.array([0.0, 1.0, 0.0])
    right   = np.cross(forward, up)
    right  /= np.linalg.norm(right)
    return pos, forward, right, up


# ---------------------------------------------------------------------------
# Camera mountings
# ---------------------------------------------------------------------------
# Each value is a callable (forward, right, up) -> (cam_fwd, cam_up)
# where the inputs are robot body-frame axes in world coordinates.
MOUNTINGS = {
    'forward':  lambda f, r, u: (f.copy(),  u.copy()),   # faces direction of travel
    'backward': lambda f, r, u: (-f,        u.copy()),   # rear-facing
    'inward':   lambda f, r, u: (-r,        u.copy()),   # toward box (robot's left)
    'outward':  lambda f, r, u: (r.copy(),  u.copy()),   # away from box
    'downward': lambda f, r, u: (-u,        f.copy()),   # on top, looking at ground
    'upward':   lambda f, r, u: (u.copy(), -f),          # on bottom, looking at sky
}


def _camera_R(cam_fwd: np.ndarray, cam_up: np.ndarray) -> np.ndarray:
    """
    Build [3, 3] camera-to-world rotation matrix.
    Columns: right, up, forward (all in world frame).
    """
    fwd   = cam_fwd / np.linalg.norm(cam_fwd)
    right = np.cross(fwd, cam_up)
    right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)
    return np.column_stack([right, up, fwd])


# ---------------------------------------------------------------------------
# Floor geometry (checkerboard so cameras always have textured features)
# ---------------------------------------------------------------------------
FLOOR_Y      = -0.8    # world y of the floor plane
FLOOR_EXTENT = 5.0     # floor runs ±FLOOR_EXTENT in x and z
FLOOR_N      = 8       # tiles per side  →  8×8 = 64 tiles total
_TILE_COLORS = [np.array([0.72, 0.72, 0.72]),   # light grey
                np.array([0.32, 0.32, 0.40])]   # dark grey-blue


def _floor_tiles():
    """Pre-compute the 4 world-space corners of every floor tile."""
    tile = 2.0 * FLOOR_EXTENT / FLOOR_N
    tiles = []
    for ix in range(FLOOR_N):
        for iz in range(FLOOR_N):
            x0 = -FLOOR_EXTENT + ix * tile
            z0 = -FLOOR_EXTENT + iz * tile
            verts = np.array([
                [x0,        FLOOR_Y, z0],
                [x0 + tile, FLOOR_Y, z0],
                [x0 + tile, FLOOR_Y, z0 + tile],
                [x0,        FLOOR_Y, z0 + tile],
            ])
            color = _TILE_COLORS[(ix + iz) % 2].copy()
            tiles.append((verts, color))
    return tiles


_FLOOR_TILES = _floor_tiles()          # computed once at import time


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_frame(cam_pos: np.ndarray, cam_R: np.ndarray,
                 image_size: int = 256, fov_deg: float = 90.0) -> np.ndarray:
    """
    Render the box using a pinhole camera + painter's algorithm (cv2 fill).

    Parameters
    ----------
    cam_pos    : [3]    camera position in world frame
    cam_R      : [3,3]  camera-to-world rotation; columns = right, up, forward
    image_size : int    output image is square
    fov_deg    : float

    Returns
    -------
    img : [H, W, 3]  float32 RGB in [0, 1]
    """
    H = W = image_size
    f  = W / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx = cy = W / 2.0

    R_wc    = cam_R.T                                       # world-to-camera [3,3]
    verts_c = (R_wc @ (BOX_VERTS - cam_pos).T).T           # [8, 3] camera space

    visible = []
    for idx, color in zip(FACE_INDICES, FACE_COLORS):
        pts_c      = verts_c[idx]                           # [4, 3]
        centroid_z = pts_c[:, 2].mean()
        if centroid_z <= 0.05:                              # behind or too close
            continue

        # Back-face culling: z-component of face normal in camera space
        v1     = pts_c[1] - pts_c[0]
        v2     = pts_c[2] - pts_c[0]
        norm_z = v1[0] * v2[1] - v1[1] * v2[0]
        if norm_z >= 0:
            continue

        # Pinhole projection (y up in camera → flip to image y-down)
        u = f * pts_c[:, 0] / pts_c[:, 2] + cx
        v = -f * pts_c[:, 1] / pts_c[:, 2] + cy
        pts_img = np.column_stack([u, v])

        # Mild diffuse shading: faces closer look brighter
        shade = np.clip(0.4 + 0.6 * np.exp(-0.3 * (centroid_z - 1.0)), 0.35, 1.0)
        visible.append((centroid_z, pts_img, color * shade))

    # Floor tiles — only when camera is above the floor plane
    if cam_pos[1] > FLOOR_Y:
        for tile_verts_w, tile_color in _FLOOR_TILES:
            pts_c      = (R_wc @ (tile_verts_w - cam_pos).T).T   # [4, 3]
            centroid_z = pts_c[:, 2].mean()
            if centroid_z <= 0.05:
                continue
            u = f * pts_c[:, 0] / pts_c[:, 2] + cx
            v = -f * pts_c[:, 1] / pts_c[:, 2] + cy
            pts_img = np.column_stack([u, v])
            # Cull tiles whose projected area is degenerate (all off-screen)
            if pts_img[:, 0].max() < 0 or pts_img[:, 0].min() > W:
                continue
            if pts_img[:, 1].max() < 0 or pts_img[:, 1].min() > H:
                continue
            shade = np.clip(0.3 + 0.7 * np.exp(-0.15 * (centroid_z - 1.0)), 0.25, 1.0)
            visible.append((centroid_z, pts_img, tile_color * shade))

    # Painter's: back-to-front
    img = np.full((H, W, 3), 0.12, dtype=np.float32)       # dark background
    visible.sort(key=lambda x: -x[0])

    for _, pts_img, color in visible:
        pts_int = pts_img.astype(np.int32).reshape((-1, 1, 2))
        bgr = (float(color[2]), float(color[1]), float(color[0]))
        cv2.fillPoly(img, [pts_int], bgr)
        cv2.polylines(img, [pts_int], True, (0.0, 0.0, 0.0), 1)

    return img                                              # [H, W, 3] float32


# ---------------------------------------------------------------------------
# Trajectory generator
# ---------------------------------------------------------------------------

def generate_trajectory(mounting_name: str,
                        n_frames:   int   = 256,
                        radius:     float = 3.0,
                        height:     float = 0.3,
                        n_laps:     float = 1.0,
                        image_size: int   = 256,
                        fov_deg:    float = 90.0):
    """
    Render a full robot orbit around the box with the given camera mounting.

    Returns
    -------
    frames       : [T, H, W, 3]  float32 RGB in [0, 1]
    gt_vel_world : [T, 3]        float32 robot velocity in world frame (m/step)
    """
    phis     = np.linspace(0, 2 * np.pi * n_laps, n_frames, endpoint=False)
    dphi     = phis[1] - phis[0]
    mount_fn = MOUNTINGS[mounting_name]

    frames   = []
    gt_vels  = []

    for phi in phis:
        pos, fwd, right, up = robot_frame(phi, radius, height)
        cam_fwd, cam_up     = mount_fn(fwd, right, up)
        cam_R               = _camera_R(cam_fwd, cam_up)
        frames.append(render_frame(pos, cam_R, image_size=image_size, fov_deg=fov_deg))

        # Analytical GT world-frame velocity: d(pos)/d(phi) * dphi
        dpos = np.array([radius * np.cos(phi), 0.0, -radius * np.sin(phi)]) * dphi
        gt_vels.append(dpos)

    return (np.stack(frames, axis=0),
            np.stack(gt_vels, axis=0).astype(np.float32))
