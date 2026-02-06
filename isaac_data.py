import os
import json
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})  # set False to see UI

import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom


def get_world_transform(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(xform.GetPrim())
    t = mat.ExtractTranslation()
    rot = mat.ExtractRotationQuat()
    q = rot.GetImaginary()
    w = rot.GetReal()
    return (np.array([t[0], t[1], t[2]], dtype=np.float32),
            np.array([w, q[0], q[1], q[2]], dtype=np.float32))


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def main():
    OUT_DIR = "./isaac_pose_mvi"
    N = 10
    RES = (320, 240)  # (W,H)

    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "rgb"))

    rep.orchestrator.set_capture_on_play(False)

    labels = []

    # Keep scene creation + capture loop in the SAME layer context
    with rep.new_layer():
        rep.create.light(rotation=(315, 0, 0), intensity=3000, light_type="distant")

        target = rep.create.cube(position=(0, 0, 0), scale=(0.1, 0.1, 0.1))
        cam = rep.create.camera(position=(1.0, 0.0, 0.5), look_at=target)

        tgt_path = rep.utils.get_node_targets(target.node, "inputs:prims")[0]
        cam_path = rep.utils.get_node_targets(cam.node, "inputs:prims")[0]
        print("Target prim path:", tgt_path)
        print("Camera prim path:", cam_path)

        rp = rep.create.render_product(cam, RES)

        rgb_anno = rep.AnnotatorRegistry.get_annotator("LdrColor")
        rgb_anno.attach(rp)

        try:
            for i in range(N):
                x = np.random.uniform(0.3, 1.2)
                y = np.random.uniform(-0.4, 0.4)
                z = np.random.uniform(0.15, 0.9)

                # Use the ReplicatorItem here (more robust than a string path)
                rep.modify.pose(position=(x, y, z), input_prims=[target])

                rep.orchestrator.step()

                rgb = rgb_anno.get_data()  # uint8 RGBA

                # normalize shape to (H,W,4)
                if rgb.shape[0] == RES[0] and rgb.shape[1] == RES[1]:
                    rgb = np.transpose(rgb, (1, 0, 2))

                np.save(os.path.join(OUT_DIR, "rgb", f"{i:06d}.npy"), rgb)

                cam_t, cam_q = get_world_transform(cam_path)
                tgt_t, tgt_q = get_world_transform(tgt_path)

                labels.append({
                    "i": i,
                    "rgb": f"rgb/{i:06d}.npy",
                    "camera_prim_path": cam_path,
                    "target_prim_path": tgt_path,
                    "camera_world_t": cam_t.tolist(),
                    "camera_world_q_wxyz": cam_q.tolist(),
                    "target_world_t": tgt_t.tolist(),
                    "target_world_q_wxyz": tgt_q.tolist(),
                })

                if i % 10 == 0:
                    print(f"Captured {i}/{N}")
        finally:
            # Stop orchestrator cleanly before shutdown
            try:
                rep.orchestrator.stop()
            except Exception:
                pass

    # Write labels AFTER leaving the layer (pure Python I/O)
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)
    print(f"Wrote labels.json with {len(labels)} entries.")

    simulation_app.close()


if __name__ == "__main__":
    main()
