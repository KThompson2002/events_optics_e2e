import os
import json
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})  # set False to see UI

import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom, Gf


# ---------- USD helpers ----------
def get_world_transform(prim_path: str):
    """
    Returns (translation_xyz, rotation_quat_wxyz) in world coordinates.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    cache = UsdGeom.XformCache()  # time=default
    mat = cache.GetLocalToWorldTransform(xform.GetPrim())
    # translation
    t = mat.ExtractTranslation()
    # rotation (as quaternion)
    rot = mat.ExtractRotationQuat()  # pxr.Gf.Quatd
    q = rot.GetImaginary()  # xyz
    w = rot.GetReal()
    return (np.array([t[0], t[1], t[2]], dtype=np.float32),
            np.array([w, q[0], q[1], q[2]], dtype=np.float32))


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    # ---------- User config ----------
    OUT_DIR = "./isaac_pose_mvi"
    N = 2000
    RES = (320, 240)  # (W,H) per Replicator render_product API
    CAMERA_PATH = "/World/Camera"
    TARGET_PATH = "/World/Target"

    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "rgb"))

    # Important for Isaac workflows: capture only when we call step() :contentReference[oaicite:2]{index=2}
    rep.orchestrator.set_capture_on_play(False)

    # ---------- Scene setup ----------
    with rep.new_layer():
        # lighting + target object + camera
        rep.create.light(rotation=(315, 0, 0), intensity=3000, light_type="distant")

        # Create a target prim if it doesn't exist (simple cube)
        target = rep.create.cube(position=(0, 0, 0), scale=(0.1, 0.1, 0.1))
        # Camera looking at target
        cam = rep.create.camera(position=(1.0, 0.0, 0.5), look_at=target)

        rp = rep.create.render_product(cam, RES)

        # RGB annotator (LdrColor) :contentReference[oaicite:3]{index=3}
        rgb_anno = rep.AnnotatorRegistry.get_annotator("LdrColor")
        rgb_anno.attach(rp)

        # (Optional) camera params annotator (intrinsics/extrinsics)
        cam_anno = rep.AnnotatorRegistry.get_annotator("camera_params")
        cam_anno.attach(rp)

        # NOTE: We created prims via ReplicatorItem; we’ll use the actual prim paths for labels.
        # If you want deterministic paths, you can spawn USD assets at fixed prim paths instead.

    labels = []

    # ---------- Data loop ----------
    for i in range(N):
        # simple randomization: move target around
        # (keep it in front of camera)
        x = np.random.uniform(0.2, 1.2)
        y = np.random.uniform(-0.4, 0.4)
        z = np.random.uniform(0.1, 0.8)
        rep.modify.pose(position=(x, y, z), input_prims=[TARGET_PATH])

        # Step capture (this triggers annotators) :contentReference[oaicite:4]{index=4}
        rep.orchestrator.step()

        rgb = rgb_anno.get_data()  # uint8 RGBA :contentReference[oaicite:5]{index=5}
        cam_params = cam_anno.get_data()

        # Handle shape: some versions return (H,W,4), docs say (W,H,4) :contentReference[oaicite:6]{index=6}
        if rgb.shape[0] == RES[0] and rgb.shape[1] == RES[1]:
            rgb = np.transpose(rgb, (1, 0, 2))  # -> (H,W,4)

        rgb_path = os.path.join(OUT_DIR, "rgb", f"{i:06d}.npy")
        np.save(rgb_path, rgb)

        cam_t, cam_q = get_world_transform(CAMERA_PATH)
        tgt_t, tgt_q = get_world_transform(TARGET_PATH)

        labels.append({
            "i": i,
            "rgb": f"rgb/{i:06d}.npy",
            "camera_world_t": cam_t.tolist(),
            "camera_world_q_wxyz": cam_q.tolist(),
            "target_world_t": tgt_t.tolist(),
            "target_world_q_wxyz": tgt_q.tolist(),
            "camera_params": cam_params  # json-serializable dict (usually)
        })

        if i % 100 == 0:
            print(f"Captured {i}/{N}")

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)

    print("Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
