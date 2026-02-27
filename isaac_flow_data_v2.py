"""
Isaac Sim optical-flow dataset v2 — textured box scene
=======================================================
Scene
─────
A 2 m³ box sits at the origin with a checkerboard pattern of coloured
panels covering all four vertical faces.  The rich texture provides
many pixels with detectable log-intensity changes even for small
per-frame camera motion, giving reliable event generation.

Camera trajectory
─────────────────
The camera orbits the box at a slowly oscillating radius.  Because the
orbital angular rate and the radial oscillation frequency are
incommensurate, the camera spends some intervals moving primarily
sideways (turning / rotational flow) and others moving primarily
inward/outward (forward / looming flow).  This teaches the model to
distinguish the two motion types from the optical-flow field alone.

  r(frame)     = R0 + Ar * sin(2π * N_RADIAL * frame / N)
  theta(frame) = 2π * N_ORBITS * frame / N
  cam_xyz      = (r*cos θ, r*sin θ, cam_z)

GT flow
───────
Omniverse motion_vectors returns NDC values where ±1 spans the full
image width / height.  We convert to pixel units:
  pixel_x = NDC_x * (W / 2)
  pixel_y = NDC_y * (H / 2)
Output is written as float32 .npy files, same format as v1 — fully
compatible with IsaacFlowSequence (just change DATA_ROOT).

Verification
────────────
The script prints mean flow magnitude (px) every 50 frames.
Target: 2–8 px / frame → reliable events AND tractable photometric loss
with T = 16 (half-window 8 frames ≈ 16–64 px accumulated displacement).
If the printed magnitude is near 0.01–0.05, the NDC factor needs to be
doubled:  change (W/2, H/2) → (W, H) in the conversion below.
"""

import os
import json
import math
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, Sdf, Gf, UsdLux

from isaac_flow_data import (
    ensure_dir, set_xform, set_scale, get_world_transform, look_at_quat,
)


# ── Material helpers ─────────────────────────────────────────────────────────

def make_color_material(stage, path: str, rgb: tuple) -> UsdShade.Material:
    """Create a simple diffuse UsdPreviewSurface material."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor",
                       Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness",
                       Sdf.ValueTypeNames.Float).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    return mat


def bind_material(prim, mat: UsdShade.Material):
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


# ── Checkerboard-panel helper ────────────────────────────────────────────────

def add_checkerboard_face(
    stage,
    path_prefix: str,
    mat_a,
    mat_b,
    center_xyz,      # world-space centre of the face
    right_axis,      # unit vector along the "width" direction of the face
    up_axis,         # unit vector along the "height" direction of the face
    face_size=2.0,   # side length in metres
    n_grid=6,        # panels per side  (n_grid² panels per face)
    panel_depth=0.02,
):
    """
    Place n_grid × n_grid thin panels over a face to form a checkerboard.
    panel_depth controls how far the panels protrude (just enough to be
    visible without z-fighting the base cube).
    """
    step = face_size / n_grid
    # normal pointing outward: right × up
    nx = right_axis[1] * up_axis[2] - right_axis[2] * up_axis[1]
    ny = right_axis[2] * up_axis[0] - right_axis[0] * up_axis[2]
    nz = right_axis[0] * up_axis[1] - right_axis[1] * up_axis[0]
    normal = (nx, ny, nz)

    panel_idx = 0
    for row in range(n_grid):
        for col in range(n_grid):
            # offset from face centre in local coordinates
            u = (col - n_grid / 2.0 + 0.5) * step
            v = (row - n_grid / 2.0 + 0.5) * step

            cx = center_xyz[0] + u * right_axis[0] + v * up_axis[0] + panel_depth * normal[0]
            cy = center_xyz[1] + u * right_axis[1] + v * up_axis[1] + panel_depth * normal[1]
            cz = center_xyz[2] + u * right_axis[2] + v * up_axis[2] + panel_depth * normal[2]

            prim_path = f"{path_prefix}/Panel_{panel_idx:04d}"
            panel = UsdGeom.Cube.Define(stage, prim_path).GetPrim()
            set_xform(panel, t_xyz=(cx, cy, cz), quat_wxyz=(1, 0, 0, 0))

            # scale: step × step × panel_depth (half-extents for UsdGeom.Cube)
            sx = step / 2 * abs(right_axis[0]) + panel_depth / 2 * abs(normal[0])
            sy = step / 2 * (abs(right_axis[1]) + abs(up_axis[1])) + panel_depth / 2 * abs(normal[1])
            sz = step / 2 * (abs(right_axis[2]) + abs(up_axis[2])) + panel_depth / 2 * abs(normal[2])

            # Simpler: just scale along each world axis
            # For axis-aligned faces this is clean
            scale_x = step / 2 if right_axis[0] != 0 or up_axis[0] != 0 else panel_depth / 2
            scale_y = step / 2 if right_axis[1] != 0 or up_axis[1] != 0 else panel_depth / 2
            scale_z = step / 2 if right_axis[2] != 0 or up_axis[2] != 0 else panel_depth / 2

            set_scale(panel, scale_x, scale_y, scale_z)

            mat = mat_a if (row + col) % 2 == 0 else mat_b
            bind_material(panel, mat)

            panel_idx += 1


def build_textured_box(stage, box_half=1.0, n_grid=6):
    """
    Build a 2×box_half metre axis-aligned box at the origin with a
    checkerboard pattern on all four vertical faces.
    """
    mat_white = make_color_material(stage, "/World/Materials/White", (0.95, 0.95, 0.95))
    mat_dark  = make_color_material(stage, "/World/Materials/Dark",  (0.12, 0.12, 0.12))

    # Base (thin floor slab)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor").GetPrim()
    set_xform(floor, t_xyz=(0.0, 0.0, -box_half - 0.03), quat_wxyz=(1, 0, 0, 0))
    set_scale(floor, box_half * 3, box_half * 3, 0.03)
    bind_material(floor, mat_white)

    # Base box (solid — panels sit on top)
    base = UsdGeom.Cube.Define(stage, "/World/BoxBase").GetPrim()
    set_xform(base, t_xyz=(0.0, 0.0, 0.0), quat_wxyz=(1, 0, 0, 0))
    set_scale(base, box_half, box_half, box_half)
    bind_material(base, mat_dark)

    d = box_half  # half-extent

    # +X face  (right_axis = Y, up_axis = Z)
    add_checkerboard_face(
        stage, "/World/Panels/FacePX", mat_white, mat_dark,
        center_xyz=(d, 0.0, 0.0),
        right_axis=(0.0, 1.0, 0.0), up_axis=(0.0, 0.0, 1.0),
        face_size=2 * d, n_grid=n_grid,
    )
    # -X face
    add_checkerboard_face(
        stage, "/World/Panels/FaceNX", mat_white, mat_dark,
        center_xyz=(-d, 0.0, 0.0),
        right_axis=(0.0, 1.0, 0.0), up_axis=(0.0, 0.0, 1.0),
        face_size=2 * d, n_grid=n_grid,
    )
    # +Y face  (right_axis = X, up_axis = Z)
    add_checkerboard_face(
        stage, "/World/Panels/FacePY", mat_white, mat_dark,
        center_xyz=(0.0, d, 0.0),
        right_axis=(1.0, 0.0, 0.0), up_axis=(0.0, 0.0, 1.0),
        face_size=2 * d, n_grid=n_grid,
    )
    # -Y face
    add_checkerboard_face(
        stage, "/World/Panels/FaceNY", mat_white, mat_dark,
        center_xyz=(0.0, -d, 0.0),
        right_axis=(1.0, 0.0, 0.0), up_axis=(0.0, 0.0, 1.0),
        face_size=2 * d, n_grid=n_grid,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Config ──────────────────────────────────────────────────────────────
    OUT_DIR    = "./isaac_flow_data_v2"
    N          = 1500            # total frames (more data than v1)
    RES        = (320, 240)      # (width, height) in pixels
    W, H       = RES

    # Camera orbit / radial parameters
    N_ORBITS   = 5               # full turns around the box in N frames
    N_RADIAL   = 3               # radial oscillations in N frames
    #   incommensurate ratio ↑ mixes turning & forward motion smoothly
    R0         = 2.5             # base orbit radius (m)
    AR         = 0.8             # radial oscillation amplitude (m)
    CAM_Z      = 0.4             # camera height (m)
    CAM_Z_AMP  = 0.3             # vertical oscillation amplitude
    CAM_Z_FREQ = 2               # vertical oscillations per radial period
    BOX_HALF   = 1.0             # half-extent of the textured box (m)
    N_GRID     = 6               # checkerboard panels per face side
    LOOK_AT    = (0.0, 0.0, 0.0) # always look at the box centre

    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "rgb"))
    ensure_dir(os.path.join(OUT_DIR, "flow"))

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World", "Xform")

    # ── Scene ────────────────────────────────────────────────────────────────
    build_textured_box(stage, box_half=BOX_HALF, n_grid=N_GRID)

    UsdLux.DomeLight.Define(stage, "/World/DomeLight")

    cam_prim = UsdGeom.Camera.Define(stage, "/World/Camera").GetPrim()
    set_xform(cam_prim, t_xyz=(R0, 0.0, CAM_Z), quat_wxyz=(1, 0, 0, 0))

    # ── Annotators ──────────────────────────────────────────────────────────
    rep.orchestrator.set_capture_on_play(False)
    rp = rep.create.render_product("/World/Camera", RES)

    rgb_anno = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb_anno.attach(rp)
    mv_anno  = rep.AnnotatorRegistry.get_annotator("motion_vectors")
    mv_anno.attach(rp)

    labels = []
    rep.orchestrator.step()   # warm-up

    flow_mags = []

    # ── Capture loop ─────────────────────────────────────────────────────────
    for i in range(N):
        t = i / N               # normalised time in [0, 1)

        # Orbit angle
        theta = 2.0 * math.pi * N_ORBITS * t

        # Radial distance — incommensurate with orbit so motion type varies
        r = R0 + AR * math.sin(2.0 * math.pi * N_RADIAL * t)

        cam_x = r * math.cos(theta)
        cam_y = r * math.sin(theta)
        cam_z = CAM_Z + CAM_Z_AMP * math.sin(2.0 * math.pi * CAM_Z_FREQ * N_RADIAL * t)

        cam_q = look_at_quat((cam_x, cam_y, cam_z), LOOK_AT)
        set_xform(cam_prim, t_xyz=(cam_x, cam_y, cam_z), quat_wxyz=cam_q)

        rep.orchestrator.step()

        # ── RGB ──────────────────────────────────────────────────────────────
        rgb = rgb_anno.get_data()
        if rgb.shape[0] == W and rgb.shape[1] == H:
            rgb = np.transpose(rgb, (1, 0, 2))
        np.save(os.path.join(OUT_DIR, "rgb", f"{i:06d}.npy"), rgb)

        # ── GT optical flow ───────────────────────────────────────────────────
        flow_path = None
        if i > 0:
            mv = mv_anno.get_data()
            if mv.shape[0] == W and mv.shape[1] == H:
                mv = np.transpose(mv, (1, 0, 2))

            flow_ndc = mv[:, :, :2].astype(np.float32)

            # NDC → pixel conversion
            # Omniverse convention: ±1 NDC spans the full image dimension.
            #   pixel_x = NDC_x * (W / 2)
            #   pixel_y = NDC_y * (H / 2)
            # If diagnostics below show magnitude ~0.01-0.05 (still NDC scale),
            # change to:  flow_ndc * np.array([W, H])  and re-generate.
            flow_xy = flow_ndc * np.array([W / 2.0, H / 2.0], dtype=np.float32)
            np.save(os.path.join(OUT_DIR, "flow", f"{i:06d}.npy"), flow_xy)
            flow_path = f"flow/{i:06d}.npy"

            mag = float(np.linalg.norm(flow_xy, axis=2).mean())
            flow_mags.append(mag)

            if i <= 5 or i % 50 == 0:
                ndc_abs_max = float(np.abs(flow_ndc).max())
                print(f"  frame {i:04d} | "
                      f"flow_mag = {mag:.3f} px | "
                      f"max |NDC| = {ndc_abs_max:.5f}  "
                      f"{'[OK]' if 1.0 < mag < 20.0 else '[CHECK SCALE]'}")

        # ── Metadata ─────────────────────────────────────────────────────────
        cam_t, cam_q_world = get_world_transform("/World/Camera")
        labels.append({
            "i":  i,
            "rgb":  f"rgb/{i:06d}.npy",
            "flow": flow_path,
            "camera_world_t":      cam_t.tolist(),
            "camera_world_q_wxyz": cam_q_world.tolist(),
        })

        if i % 100 == 0:
            print(f"Captured {i}/{N}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if flow_mags:
        print(f"\nFlow magnitude summary (px/frame):")
        print(f"  mean  = {np.mean(flow_mags):.3f}")
        print(f"  min   = {np.min(flow_mags):.3f}")
        print(f"  max   = {np.max(flow_mags):.3f}")
        print(f"  p95   = {np.percentile(flow_mags, 95):.3f}")
        if np.mean(flow_mags) < 0.5:
            print("\n  WARNING: mean flow < 0.5 px — NDC conversion may be off.")
            print("  Try changing the conversion to: flow_ndc * [W, H]")
        elif np.mean(flow_mags) > 30:
            print("\n  WARNING: mean flow > 30 px — motion may be too fast.")
            print("  Consider reducing N_ORBITS or AR.")

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"\nDone. {N} frames → {OUT_DIR}/")
    print("Set DATA_ROOT = './isaac_flow_data_v2' in isaac_flow_e2e.py")
    simulation_app.close()


if __name__ == "__main__":
    main()
