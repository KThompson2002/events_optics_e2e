import os
import json
import math
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom, Gf, UsdLux


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
    # Add scale op if not present
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
    """Compute (w, x, y, z) quaternion so camera at *eye* looks toward *target*.

    Isaac Sim cameras look along the local -Z axis with +Y up.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12

    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12

    new_up = np.cross(right, fwd)

    # Rotation matrix: columns are right, new_up, -fwd (camera convention)
    R = np.stack([right, new_up, -fwd], axis=1)  # 3x3

    # Matrix → quaternion (Shepperd's method)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return tuple(q.tolist())


# -----------------------------------------------------------------------
# Per-object trajectory parameters
# -----------------------------------------------------------------------
# Each object oscillates around a center point.  Parameters:
#   (prim_type, center_xyz, amplitude_xyz, freq_xyz, phase_xyz)
OBJ_DEFS = [
    ("Cube",   ( 0.5,  0.3, 0.15), (0.20, 0.10, 0.05), (1.0, 2.0, 0.5), (0.0,  0.0, 0.0)),
    ("Cube",   (-0.3,  0.5, 0.25), (0.10, 0.15, 0.08), (1.5, 1.0, 1.0), (1.0,  0.5, 0.3)),
    ("Cube",   ( 0.0, -0.4, 0.10), (0.15, 0.20, 0.04), (2.0, 0.8, 1.5), (0.5,  1.5, 1.0)),
    ("Sphere", ( 0.4, -0.2, 0.20), (0.12, 0.18, 0.06), (1.2, 1.8, 0.7), (2.0,  0.3, 0.8)),
    ("Sphere", (-0.5,  0.0, 0.30), (0.18, 0.12, 0.10), (0.8, 1.3, 1.2), (0.7,  2.5, 1.5)),
    ("Sphere", ( 0.2,  0.6, 0.12), (0.14, 0.08, 0.07), (1.7, 0.6, 2.0), (1.3,  1.0, 2.0)),
]


def obj_position(obj_idx, angle):
    """Return (x, y, z) for object *obj_idx* at orbit angle *angle* (radians)."""
    _, center, amp, freq, phase = OBJ_DEFS[obj_idx]
    return tuple(
        center[a] + amp[a] * math.sin(freq[a] * angle + phase[a])
        for a in range(3)
    )


def main():
    # ----- Config -----
    OUT_DIR = "./isaac_flow_data"
    N = 1000                   # total frames
    RES = (320, 240)           # (width, height)
    ORBIT_RADIUS = 2.0
    CAM_HEIGHT = 0.8
    LOOK_AT = (0.0, 0.0, 0.3) # scene center

    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "rgb"))
    ensure_dir(os.path.join(OUT_DIR, "flow"))

    stage = omni.usd.get_context().get_stage()

    # ----- World root -----
    stage.DefinePrim("/World", "Xform")

    # ----- Ground plane (large flat cube) -----
    ground_prim = UsdGeom.Cube.Define(stage, "/World/Ground").GetPrim()
    set_xform(ground_prim, t_xyz=(0.0, 0.0, -0.025), quat_wxyz=(1, 0, 0, 0))
    set_scale(ground_prim, 5.0, 5.0, 0.05)

    # ----- Scene objects -----
    obj_prims = []
    obj_paths = []
    for k, (prim_type, center, _, _, _) in enumerate(OBJ_DEFS):
        path = f"/World/Obj{k}"
        if prim_type == "Cube":
            prim = UsdGeom.Cube.Define(stage, path).GetPrim()
        else:
            prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
        set_xform(prim, t_xyz=center, quat_wxyz=(1, 0, 0, 0))
        set_scale(prim, 0.1, 0.1, 0.1)
        obj_prims.append(prim)
        obj_paths.append(path)

    # ----- Camera -----
    cam_prim = UsdGeom.Camera.Define(stage, "/World/Camera").GetPrim()
    set_xform(cam_prim, t_xyz=(ORBIT_RADIUS, 0.0, CAM_HEIGHT),
              quat_wxyz=(1, 0, 0, 0))

    # ----- Dome light -----
    UsdLux.DomeLight.Define(stage, "/World/DomeLight")

    # ----- Replicator render products -----
    rep.orchestrator.set_capture_on_play(False)
    rp = rep.create.render_product("/World/Camera", RES)

    rgb_anno = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb_anno.attach(rp)

    mv_anno = rep.AnnotatorRegistry.get_annotator("motion_vectors")
    mv_anno.attach(rp)

    labels = []

    # Warm-up step so annotators initialize
    rep.orchestrator.step()

    for i in range(N):
        angle = 2.0 * math.pi * i / N  # one full orbit

        # ---- Camera orbit ----
        cam_x = ORBIT_RADIUS * math.cos(angle)
        cam_y = ORBIT_RADIUS * math.sin(angle)
        cam_z = CAM_HEIGHT + 0.15 * math.sin(3.0 * angle)
        cam_q = look_at_quat((cam_x, cam_y, cam_z), LOOK_AT)
        set_xform(cam_prim, t_xyz=(cam_x, cam_y, cam_z), quat_wxyz=cam_q)

        # ---- Object trajectories ----
        for k in range(len(OBJ_DEFS)):
            pos = obj_position(k, angle)
            set_xform(obj_prims[k], t_xyz=pos, quat_wxyz=(1, 0, 0, 0))

        rep.orchestrator.step()

        # ---- Capture RGB ----
        rgb = rgb_anno.get_data()  # uint8 RGBA
        if rgb.shape[0] == RES[0] and rgb.shape[1] == RES[1]:
            rgb = np.transpose(rgb, (1, 0, 2))
        np.save(os.path.join(OUT_DIR, "rgb", f"{i:06d}.npy"), rgb)

        # ---- Capture motion vectors (optical flow GT) ----
        flow_path = None
        if i > 0:
            mv = mv_anno.get_data()  # float32 [H, W, 4]
            if mv.shape[0] == RES[0] and mv.shape[1] == RES[1]:
                mv = np.transpose(mv, (1, 0, 2))
            flow_xy = mv[:, :, :2].astype(np.float32)  # keep only x, y
            flow_path = f"flow/{i:06d}.npy"
            np.save(os.path.join(OUT_DIR, flow_path), flow_xy)

        # ---- Metadata ----
        cam_t, cam_q_world = get_world_transform("/World/Camera")
        obj_entries = []
        for k, path in enumerate(obj_paths):
            ot, _ = get_world_transform(path)
            obj_entries.append({"path": path, "world_t": ot.tolist()})

        labels.append({
            "i": i,
            "rgb": f"rgb/{i:06d}.npy",
            "flow": flow_path,
            "camera_world_t": cam_t.tolist(),
            "camera_world_q_wxyz": cam_q_world.tolist(),
            "objects": obj_entries,
        })

        if i % 100 == 0:
            print(f"Captured {i}/{N}")

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"Done. Wrote labels.json with {len(labels)} entries to {OUT_DIR}/")
    simulation_app.close()


if __name__ == "__main__":
    main()
