"""
Isaac Sim optical-flow dataset v3 — multi-trajectory diverse data
=================================================================
Improvements over v2
────────────────────
* Five camera trajectory segments with distinct orbital radii, speeds,
  radial amplitudes, heights, and phase offsets.  All segments share the
  same textured-box scene (no restart between segments) so generation is
  fast.
* A ``segment_id`` field is written to every label entry so that
  IsaacFlowSequence can filter out sequences that straddle a segment
  boundary (the camera teleports between segments, so boundary sequences
  would contain a discontinuous jump).
* Total frames: 5 × 600 = 3 000 (vs 1 500 in v2).

Trajectory parameters
─────────────────────
Each dict in TRAJECTORIES controls one segment:
  N          – frames in this segment
  N_ORBITS   – full turns around the box during the segment
  N_RADIAL   – radial oscillations per segment
  R0         – mean orbit radius (m)
  AR         – radial oscillation amplitude (m)
  CAM_Z      – mean camera height (m)
  CAM_Z_AMP  – vertical oscillation amplitude (m)
  phase      – starting angle offset (rad) — gives different initial views
"""

import os
import json
import math
import numpy as np

from isaacsim import SimulationApp

_ISAAC_ROOT = os.path.expandvars(
    os.environ.get("ISAAC_SIM_PATH", "/home/lea1212/isaacsim")
)
_KIT = os.path.join(
    _ISAAC_ROOT, "apps",
    "isaacsim.exp.action_and_event_data_generation.base.kit",
)

simulation_app = SimulationApp({
    "headless":   True,
    "experience": _KIT,
})

import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, Sdf, Gf, UsdLux


# ── Helpers (inlined to avoid double-SimulationApp crash) ─────────────────────

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def set_xform(prim, t_xyz, quat_wxyz):
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if not ops:
        xform.AddXformOp(UsdGeom.XformOp.TypeTranslate)
        xform.AddXformOp(UsdGeom.XformOp.TypeOrient)
        ops = xform.GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(*t_xyz))
    w, x, y, z = quat_wxyz
    ops[1].Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def set_scale(prim, sx, sy, sz):
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    has_scale = any(op.GetOpType() == UsdGeom.XformOp.TypeScale for op in ops)
    if not has_scale:
        xform.AddXformOp(UsdGeom.XformOp.TypeScale)
        ops = xform.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3d(sx, sy, sz))
            break


def get_world_transform(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(xform.GetPrim())
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    w = q.GetReal()
    return (
        np.array([t[0], t[1], t[2]], np.float32),
        np.array([w, imag[0], imag[1], imag[2]], np.float32),
    )


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    eye    = np.asarray(eye,    dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up     = np.asarray(up,     dtype=np.float64)
    fwd   = target - eye;  fwd   /= np.linalg.norm(fwd)   + 1e-12
    right = np.cross(fwd, up);  right /= np.linalg.norm(right) + 1e-12
    new_up = np.cross(right, fwd)
    R  = np.stack([right, new_up, -fwd], axis=1)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s;  x = (R[2,1]-R[1,2])*s;  y = (R[0,2]-R[2,0])*s;  z = (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1]-R[1,2])/s;  x = 0.25*s;  y = (R[0,1]+R[1,0])/s;  z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2]-R[2,0])/s;  x = (R[0,1]+R[1,0])/s;  y = 0.25*s;  z = (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0]-R[0,1])/s;  x = (R[0,2]+R[2,0])/s;  y = (R[1,2]+R[2,1])/s;  z = 0.25*s
    q = np.array([w, x, y, z]);  q /= np.linalg.norm(q)
    return tuple(q.tolist())


# ── Material / scene helpers ──────────────────────────────────────────────────

def make_color_material(stage, path: str, rgb: tuple) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    # OmniPBR is Isaac Sim's native shader — renders correctly in all kit modes
    # including headless data-generation kits that don't run full RTX.
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_color_constant",
                       Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("reflection_roughness_constant",
                       Sdf.ValueTypeNames.Float).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    return mat


def bind_material(prim, mat):
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


def add_checkerboard_face(stage, path_prefix, mat_a, mat_b,
                          center_xyz, right_axis, up_axis,
                          face_size=2.0, n_grid=6, panel_depth=0.02):
    step = face_size / n_grid
    nx = right_axis[1]*up_axis[2] - right_axis[2]*up_axis[1]
    ny = right_axis[2]*up_axis[0] - right_axis[0]*up_axis[2]
    nz = right_axis[0]*up_axis[1] - right_axis[1]*up_axis[0]
    normal = (nx, ny, nz)
    panel_idx = 0
    for row in range(n_grid):
        for col in range(n_grid):
            u = (col - n_grid / 2.0 + 0.5) * step
            v = (row - n_grid / 2.0 + 0.5) * step
            cx = center_xyz[0] + u*right_axis[0] + v*up_axis[0] + panel_depth*normal[0]
            cy = center_xyz[1] + u*right_axis[1] + v*up_axis[1] + panel_depth*normal[1]
            cz = center_xyz[2] + u*right_axis[2] + v*up_axis[2] + panel_depth*normal[2]
            prim_path = f"{path_prefix}/Panel_{panel_idx:04d}"
            panel = UsdGeom.Cube.Define(stage, prim_path).GetPrim()
            set_xform(panel, t_xyz=(cx, cy, cz), quat_wxyz=(1, 0, 0, 0))
            scale_x = step/2 if right_axis[0] != 0 or up_axis[0] != 0 else panel_depth/2
            scale_y = step/2 if right_axis[1] != 0 or up_axis[1] != 0 else panel_depth/2
            scale_z = step/2 if right_axis[2] != 0 or up_axis[2] != 0 else panel_depth/2
            set_scale(panel, scale_x, scale_y, scale_z)
            bind_material(panel, mat_a if (row + col) % 2 == 0 else mat_b)
            panel_idx += 1


def add_bumps_to_face(stage, path_prefix, mat, center_xyz, right_axis, up_axis,
                      face_size=2.0, n_bumps=3, bump_size=0.18, bump_depth=0.15):
    """
    Add a grid of n_bumps × n_bumps protruding cubes to a box face.
    Bumps create depth discontinuities — sharp flow boundaries at their edges
    and parallax against the flat face behind them.
    """
    step = face_size / (n_bumps + 1)
    # Outward normal = right × up
    nx = right_axis[1]*up_axis[2] - right_axis[2]*up_axis[1]
    ny = right_axis[2]*up_axis[0] - right_axis[0]*up_axis[2]
    nz = right_axis[0]*up_axis[1] - right_axis[1]*up_axis[0]

    for idx, (row, col) in enumerate(
            [(r, c) for r in range(n_bumps) for c in range(n_bumps)]):
        u = (col - (n_bumps - 1) / 2.0) * step
        v = (row - (n_bumps - 1) / 2.0) * step
        # Centre of bump sits half its depth past the face surface
        cx = center_xyz[0] + u*right_axis[0] + v*up_axis[0] + (bump_depth/2)*nx
        cy = center_xyz[1] + u*right_axis[1] + v*up_axis[1] + (bump_depth/2)*ny
        cz = center_xyz[2] + u*right_axis[2] + v*up_axis[2] + (bump_depth/2)*nz

        prim = UsdGeom.Cube.Define(stage, f"{path_prefix}/Bump_{idx:03d}").GetPrim()
        set_xform(prim, t_xyz=(cx, cy, cz), quat_wxyz=(1, 0, 0, 0))

        # Along the normal axis → bump_depth/2; along face axes → bump_size/2
        scale_x = bump_size/2 if (right_axis[0] != 0 or up_axis[0] != 0) else bump_depth/2
        scale_y = bump_size/2 if (right_axis[1] != 0 or up_axis[1] != 0) else bump_depth/2
        scale_z = bump_size/2 if (right_axis[2] != 0 or up_axis[2] != 0) else bump_depth/2
        set_scale(prim, scale_x, scale_y, scale_z)
        bind_material(prim, mat)


# Colorful objects scattered inside the camera orbit at varying depths and
# heights.  Objects at different distances from the camera move at different
# rates in the image plane, giving clearly non-uniform flow fields.
#   (prim_type, USD_path, xyz, half_scales, rgb_color)
SCENE_OBJECTS = [
    ("Sphere", "/World/Objects/S0", ( 0.8,  0.0,  0.40), (0.20, 0.20, 0.20), (0.90, 0.15, 0.15)),
    ("Sphere", "/World/Objects/S1", ( 0.0,  0.8,  0.50), (0.18, 0.18, 0.18), (0.15, 0.75, 0.20)),
    ("Sphere", "/World/Objects/S2", (-0.7,  0.3,  0.35), (0.15, 0.15, 0.15), (0.15, 0.40, 0.90)),
    ("Sphere", "/World/Objects/S3", ( 0.4, -0.8,  0.45), (0.22, 0.22, 0.22), (0.10, 0.80, 0.80)),
    ("Sphere", "/World/Objects/S4", (-0.5, -0.5,  0.30), (0.17, 0.17, 0.17), (0.90, 0.60, 0.10)),
    # Cubes for sharp flow boundaries
    ("Cube",   "/World/Objects/C0", ( 1.1,  0.5,  0.25), (0.20, 0.20, 0.20), (0.80, 0.20, 0.70)),
    ("Cube",   "/World/Objects/C1", (-0.9, -0.7,  0.30), (0.22, 0.22, 0.22), (0.90, 0.90, 0.10)),
    ("Cube",   "/World/Objects/C2", ( 0.5,  1.1,  0.20), (0.18, 0.18, 0.18), (0.10, 0.70, 0.50)),
    ("Cube",   "/World/Objects/C3", (-1.2,  0.2,  0.25), (0.20, 0.20, 0.20), (0.90, 0.40, 0.10)),
    # Tall pillars — strong vertical features visible from all angles
    ("Cube",   "/World/Objects/P0", ( 1.0, -1.0,  0.50), (0.10, 0.10, 0.50), (0.95, 0.85, 0.20)),
    ("Cube",   "/World/Objects/P1", (-1.0,  0.9,  0.45), (0.10, 0.10, 0.45), (0.20, 0.90, 0.60)),
    ("Cube",   "/World/Objects/P2", ( 0.3, -1.2,  0.55), (0.10, 0.10, 0.55), (0.60, 0.20, 0.90)),
]


def add_scene_objects(stage):
    mats = {}
    for prim_type, path, xyz, scales, color in SCENE_OBJECTS:
        mat_path = path + "/Mat"
        mat = make_color_material(stage, mat_path, color)
        mats[path] = mat

        if prim_type == "Sphere":
            prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
            # UsdGeom.Sphere radius = 1 by default; use uniform scale
            set_xform(prim, t_xyz=xyz, quat_wxyz=(1, 0, 0, 0))
            set_scale(prim, scales[0], scales[0], scales[0])
        else:
            prim = UsdGeom.Cube.Define(stage, path).GetPrim()
            set_xform(prim, t_xyz=xyz, quat_wxyz=(1, 0, 0, 0))
            set_scale(prim, scales[0], scales[1], scales[2])
        bind_material(prim, mat)


def add_floor_grid(stage, mat_a, mat_b, grid_half=3.0, n=7, tile_height=0.05):
    """
    Raised tile grid on the floor — gives visible ground texture when the
    camera is at low height and creates clear flow on ground-facing pixels.
    """
    step = (2 * grid_half) / n
    for row in range(n):
        for col in range(n):
            x = -grid_half + (col + 0.5) * step
            y = -grid_half + (row + 0.5) * step
            z = -tile_height / 2          # sits on the floor surface (z ≈ 0)
            path = f"/World/FloorTiles/T_{row:02d}_{col:02d}"
            prim = UsdGeom.Cube.Define(stage, path).GetPrim()
            set_xform(prim, t_xyz=(x, y, z), quat_wxyz=(1, 0, 0, 0))
            set_scale(prim, step/2 * 0.85, step/2 * 0.85, tile_height/2)
            bind_material(prim, mat_a if (row + col) % 2 == 0 else mat_b)


def build_textured_box(stage, box_half=1.0, n_grid=6):
    mat_white  = make_color_material(stage, "/World/Materials/White",  (0.95, 0.95, 0.95))
    mat_dark   = make_color_material(stage, "/World/Materials/Dark",   (0.12, 0.12, 0.12))
    mat_bump   = make_color_material(stage, "/World/Materials/Bump",   (0.85, 0.25, 0.10))
    mat_floor_a = make_color_material(stage, "/World/Materials/FloorA", (0.80, 0.80, 0.80))
    mat_floor_b = make_color_material(stage, "/World/Materials/FloorB", (0.30, 0.30, 0.30))

    # Floor with raised tile grid
    floor = UsdGeom.Cube.Define(stage, "/World/Floor").GetPrim()
    set_xform(floor, t_xyz=(0.0, 0.0, -box_half - 0.03), quat_wxyz=(1, 0, 0, 0))
    set_scale(floor, box_half * 4, box_half * 4, 0.03)
    bind_material(floor, mat_white)
    add_floor_grid(stage, mat_floor_a, mat_floor_b,
                   grid_half=box_half * 3.5, n=9, tile_height=0.06)

    # Base box
    base = UsdGeom.Cube.Define(stage, "/World/BoxBase").GetPrim()
    set_xform(base, t_xyz=(0.0, 0.0, 0.0), quat_wxyz=(1, 0, 0, 0))
    set_scale(base, box_half, box_half, box_half)
    bind_material(base, mat_dark)

    d = box_half
    for face_cfg in [
        ("/World/Panels/FacePX", (d, 0, 0),  (0,1,0), (0,0,1)),
        ("/World/Panels/FaceNX", (-d, 0, 0), (0,1,0), (0,0,1)),
        ("/World/Panels/FacePY", (0, d, 0),  (1,0,0), (0,0,1)),
        ("/World/Panels/FaceNY", (0, -d, 0), (1,0,0), (0,0,1)),
    ]:
        prefix, ctr, ra, ua = face_cfg
        add_checkerboard_face(stage, prefix, mat_white, mat_dark,
                              center_xyz=ctr, right_axis=ra, up_axis=ua,
                              face_size=2*d, n_grid=n_grid)
        add_bumps_to_face(stage, prefix + "/Bumps", mat_bump,
                          center_xyz=ctr, right_axis=ra, up_axis=ua,
                          face_size=2*d, n_bumps=3)

    # Scattered objects at varying depths for parallax
    add_scene_objects(stage)


# ── Trajectory configurations ─────────────────────────────────────────────────

TRAJECTORIES = [
    # Close orbit, slow turns, low height
    dict(N=600, N_ORBITS=2, N_RADIAL=3, R0=2.0, AR=0.5,
         CAM_Z=0.3, CAM_Z_AMP=0.2, phase=0.0),
    # Far orbit, fast turns, medium height
    dict(N=600, N_ORBITS=4, N_RADIAL=2, R0=3.2, AR=0.6,
         CAM_Z=0.5, CAM_Z_AMP=0.3, phase=1.1),
    # Medium orbit, high radial amplitude, high camera
    dict(N=600, N_ORBITS=3, N_RADIAL=5, R0=2.5, AR=1.0,
         CAM_Z=0.7, CAM_Z_AMP=0.4, phase=2.3),
    # Fast orbit, low radial amplitude, low camera
    dict(N=600, N_ORBITS=5, N_RADIAL=3, R0=2.8, AR=0.4,
         CAM_Z=0.2, CAM_Z_AMP=0.1, phase=0.7),
    # Very far orbit, slow, high vertical oscillation
    dict(N=600, N_ORBITS=2, N_RADIAL=4, R0=3.8, AR=0.9,
         CAM_Z=0.4, CAM_Z_AMP=0.6, phase=1.9),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR = "./isaac_flow_data_v4"
    RES     = (320, 240)
    W, H    = RES
    BOX_HALF = 1.0
    N_GRID   = 6
    LOOK_AT  = (0.0, 0.0, 0.0)

    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "rgb"))
    ensure_dir(os.path.join(OUT_DIR, "flow"))

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World", "Xform")
    build_textured_box(stage, box_half=BOX_HALF, n_grid=N_GRID)

    # Dome light — must set intensity explicitly; the default is 0 in this kit.
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(500.0)
    dome.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # Distant light for directional shading so the checkerboard faces are
    # distinguishable from different angles.
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr().Set(1000.0)
    distant.CreateAngleAttr().Set(0.53)
    set_xform(distant.GetPrim(), t_xyz=(5.0, 3.0, 8.0), quat_wxyz=(1, 0, 0, 0))

    cam_prim = UsdGeom.Camera.Define(stage, "/World/Camera").GetPrim()
    set_xform(cam_prim, t_xyz=(TRAJECTORIES[0]['R0'], 0.0, TRAJECTORIES[0]['CAM_Z']),
              quat_wxyz=(1, 0, 0, 0))

    rep.orchestrator.set_capture_on_play(False)
    rp = rep.create.render_product("/World/Camera", RES)
    rgb_anno = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb_anno.attach(rp)
    mv_anno  = rep.AnnotatorRegistry.get_annotator("motion_vectors")
    mv_anno.attach(rp)

    # Warm-up: give the renderer enough steps to fully initialise materials
    # and lighting before the first capture.  One step is not sufficient.
    for _ in range(10):
        rep.orchestrator.step()

    labels     = []
    all_mags   = []
    global_idx = 0

    for seg_id, cfg in enumerate(TRAJECTORIES):
        N         = cfg['N']
        N_ORBITS  = cfg['N_ORBITS']
        N_RADIAL  = cfg['N_RADIAL']
        R0        = cfg['R0']
        AR        = cfg['AR']
        CAM_Z     = cfg['CAM_Z']
        CAM_Z_AMP = cfg['CAM_Z_AMP']
        phase     = cfg['phase']

        seg_mags = []
        # prev_cam_xy = None for detecting the first frame (no valid flow)
        seg_first_global = global_idx

        print(f"\n── Segment {seg_id} ──────────────────────────────────")
        print(f"   N_ORBITS={N_ORBITS}  N_RADIAL={N_RADIAL}  "
              f"R0={R0}  AR={AR}  CAM_Z={CAM_Z}  phase={phase:.2f}")

        for i in range(N):
            t     = i / N
            theta = phase + 2.0 * math.pi * N_ORBITS * t
            r     = R0 + AR * math.sin(2.0 * math.pi * N_RADIAL * t)
            cam_x = r * math.cos(theta)
            cam_y = r * math.sin(theta)
            cam_z = CAM_Z + CAM_Z_AMP * math.sin(2.0 * math.pi * 2 * N_RADIAL * t)

            cam_q = look_at_quat((cam_x, cam_y, cam_z), LOOK_AT)
            set_xform(cam_prim, t_xyz=(cam_x, cam_y, cam_z), quat_wxyz=cam_q)
            rep.orchestrator.step()

            # RGB
            rgb = rgb_anno.get_data()
            if rgb.shape[0] == W and rgb.shape[1] == H:
                rgb = np.transpose(rgb, (1, 0, 2))
            np.save(os.path.join(OUT_DIR, "rgb", f"{global_idx:06d}.npy"), rgb)

            # GT flow — only valid for frames after the first in each segment
            flow_path = None
            is_first_in_seg = (i == 0)
            if not is_first_in_seg:
                mv = mv_anno.get_data()
                if mv.shape[0] == W and mv.shape[1] == H:
                    mv = np.transpose(mv, (1, 0, 2))
                flow_xy = mv[:, :, :2].astype(np.float32)
                np.save(os.path.join(OUT_DIR, "flow", f"{global_idx:06d}.npy"), flow_xy)
                flow_path = f"flow/{global_idx:06d}.npy"

                mag = float(np.linalg.norm(flow_xy, axis=2).mean())
                seg_mags.append(mag)
                all_mags.append(mag)

                if i <= 3 or i % 100 == 0:
                    print(f"  seg={seg_id} frame={i:04d} | "
                          f"flow_mag={mag:.3f} px  "
                          f"{'[OK]' if 1.0 < mag < 15.0 else '[CHECK SCALE]'}")

            cam_t, cam_q_world = get_world_transform("/World/Camera")
            labels.append({
                "i":           global_idx,
                "segment_id":  seg_id,
                "rgb":         f"rgb/{global_idx:06d}.npy",
                "flow":        flow_path,
                "camera_world_t":      cam_t.tolist(),
                "camera_world_q_wxyz": cam_q_world.tolist(),
            })
            global_idx += 1

        if seg_mags:
            print(f"  Segment {seg_id} flow: "
                  f"mean={np.mean(seg_mags):.2f}  "
                  f"min={np.min(seg_mags):.2f}  "
                  f"max={np.max(seg_mags):.2f} px/frame")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total frames : {global_idx}")
    print(f"Flow summary : mean={np.mean(all_mags):.2f}  "
          f"min={np.min(all_mags):.2f}  max={np.max(all_mags):.2f}  "
          f"p95={np.percentile(all_mags, 95):.2f} px/frame")
    if np.mean(all_mags) < 0.5:
        print("WARNING: mean flow < 0.5 px — check NDC conversion.")
    elif np.mean(all_mags) > 30:
        print("WARNING: mean flow > 30 px — reduce N_ORBITS or AR.")

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"\nDone. {global_idx} frames across {len(TRAJECTORIES)} segments → {OUT_DIR}/")
    print("Set DATA_ROOT to this path in isaac_flow_e2e.py")
    simulation_app.close()


if __name__ == "__main__":
    main()
