import os
import json
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})  # set False to see UI

import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom, Gf


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def set_xform(prim, t_xyz, quat_wxyz):
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if not ops:
        xform.AddXformOp(UsdGeom.XformOp.TypeTranslate)
        xform.AddXformOp(UsdGeom.XformOp.TypeOrient)
        ops = xform.GetOrderedXformOps()

    # Translate
    ops[0].Set(Gf.Vec3d(*t_xyz))
    # Orient (wxyz)
    w, x, y, z = quat_wxyz
    ops[1].Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


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
        np.array([w, imag[0], imag[1], imag[2]], np.float32)
    )



def main():
    OUT_DIR = "./isaac_pose_mvi"
    N = 200
    RES = (320, 240)  # (W,H)

    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "rgb"))

    stage = omni.usd.get_context().get_stage()

    # --- Create fixed-path prims ---
    world = stage.DefinePrim("/World", "Xform")

    cam_prim = UsdGeom.Camera.Define(stage, "/World/Camera").GetPrim()
    tgt_prim = UsdGeom.Cube.Define(stage, "/World/Target").GetPrim()

    # Camera initial pose
    set_xform(cam_prim, t_xyz=(1.0, 0.0, 0.6), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    # Target initial pose
    set_xform(tgt_prim, t_xyz=(0.6, 0.0, 0.4), quat_wxyz=(1.0, 0.0, 0.0, 0.0))

    rep.orchestrator.set_capture_on_play(False)

    # --- Replicator render product from the camera path ---
    rp = rep.create.render_product("/World/Camera", RES)
    rgb_anno = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb_anno.attach(rp)

    labels = []

    rep.orchestrator.step()

    for i in range(N):
        # Randomize target position
        x = float(np.random.uniform(0.3, 1.2))
        y = float(np.random.uniform(-0.4, 0.4))
        z = float(np.random.uniform(0.15, 0.9))
        set_xform(tgt_prim, t_xyz=(x, y, z), quat_wxyz=(1.0, 0.0, 0.0, 0.0))

        rep.orchestrator.step()

        rgb = rgb_anno.get_data()  # uint8 RGBA

        # Normalize shape to (H,W,4) if needed
        if rgb.shape[0] == RES[0] and rgb.shape[1] == RES[1]:
            rgb = np.transpose(rgb, (1, 0, 2))

        np.save(os.path.join(OUT_DIR, "rgb", f"{i:06d}.npy"), rgb)

        cam_t, cam_q = get_world_transform("/World/Camera")
        tgt_t, tgt_q = get_world_transform("/World/Target")

        labels.append({
            "i": i,
            "rgb": f"rgb/{i:06d}.npy",
            "camera_prim_path": "/World/Camera",
            "target_prim_path": "/World/Target",
            "camera_world_t": cam_t.tolist(),
            "camera_world_q_wxyz": cam_q.tolist(),
            "target_world_t": tgt_t.tolist(),
            "target_world_q_wxyz": tgt_q.tolist(),
        })

        if i % 25 == 0:
            print(f"Captured {i}/{N}")

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"Done. Wrote labels.json with {len(labels)} entries.")
    simulation_app.close()

if __name__ == "__main__":
    main()
