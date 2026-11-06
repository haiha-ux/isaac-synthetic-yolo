"""Isaac Sim + Replicator synthetic data generator, tabletop scene, YOLO labels.

Run with Isaac's python:  ~/isaacsim/python.sh gen_data.py --mode rand --n 2000 --seed 0 --out data/rand_train
Modes:
  rand   domain randomization (camera, lighting, table/background, distractors)
  fixed  baseline: one camera, one light setup, one table colour, no distractors
  shift  held-out test: camera/lighting/background/distractor ranges outside the rand ranges
Object poses on the table are randomized in every mode.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["rand", "fixed", "shift"], required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--rt_subframes", type=int, default=4)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.semantics import add_labels  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

from yolo_labels import CLASSES, label_line  # noqa: E402

W, H = 640, 480
YCB = {
    "cracker_box": "003_cracker_box", "sugar_box": "004_sugar_box", "soup_can": "005_tomato_soup_can",
    "mustard_bottle": "006_mustard_bottle", "bowl": "024_bowl",
    "meat_can": "010_potted_meat_can", "wood_block": "036_wood_block",
}
COPIES = 3  # pooled prims per class; each frame shows a random subset
MIN_BOX_PX = 6
MAX_OCCLUSION = 0.8

rng = np.random.default_rng(args.seed)
U = rng.uniform


def log(msg):
    sys.stderr.write(msg + "\n")


# ---------------------------------------------------------------- scene
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
root = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"


def make_material(path, rgb=(0.5, 0.5, 0.5), rough=0.6):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, sh


def set_material(sh, rgb, rough):
    sh.GetInput("diffuseColor").Set(Gf.Vec3f(*[float(c) for c in rgb]))
    sh.GetInput("roughness").Set(float(rough))


def hsv(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


class Obj:
    """Pooled object: Xform (pose/scale) -> child asset or primitive."""

    def __init__(self, name, kind, cls=None):
        self.path = "/World/" + name
        self.cls, self.kind = cls, kind
        self.xf = UsdGeom.Xform.Define(stage, self.path)
        self.api = UsdGeom.XformCommonAPI(self.xf)
        child = self.path + "/geo"
        self.sh = None
        if kind in YCB:
            stage.DefinePrim(child, "Xform").GetReferences().AddReference(root + YCB[kind] + ".usd")
        else:
            prim = {"cube": UsdGeom.Cube, "sphere": UsdGeom.Sphere, "cylinder": UsdGeom.Cylinder,
                    "cone": UsdGeom.Cone, "capsule": UsdGeom.Capsule}[kind].Define(stage, child)
            if kind == "cube":
                prim.GetSizeAttr().Set(1.0)
            else:
                prim.GetRadiusAttr().Set(0.5)
                if kind != "sphere":
                    prim.GetHeightAttr().Set(1.0)
            mat, self.sh = make_material(self.path + "/mat")
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
        if cls is not None:
            add_labels(stage.GetPrimAtPath(self.path), labels=[cls], instance_name="class")
        self.hide()

    def hide(self):
        UsdGeom.Imageable(self.xf).MakeInvisible()
        self.api.SetTranslate(Gf.Vec3d(0, 0, -50))

    def place(self, x, y, yaw, tilt, scale, bbox_cache):
        UsdGeom.Imageable(self.xf).MakeVisible()
        self.api.SetScale(Gf.Vec3f(scale, scale, scale))
        self.api.SetRotate(Gf.Vec3f(tilt, 0, yaw), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        self.api.SetTranslate(Gf.Vec3d(x, y, 0))
        bb = bbox_cache.ComputeWorldBound(self.xf.GetPrim()).ComputeAlignedRange()
        self.api.SetTranslate(Gf.Vec3d(x, y, -bb.GetMin()[2]))
        return 0.5 * max(bb.GetSize()[0], bb.GetSize()[1])  # footprint radius


targets = [Obj("t_%s_%d" % (c, i), c, cls=c) for c in CLASSES for i in range(COPIES)]
distr_kinds = ["cube", "sphere", "cylinder", "meat_can", "wood_block"]
if args.mode == "shift":
    distr_kinds += ["cone", "capsule"]  # shapes never seen in rand training
distractors = [Obj("d_%s_%d" % (k, i), k) for k in distr_kinds for i in range(2)]

floor = UsdGeom.Cube.Define(stage, "/World/floor")
floor.GetSizeAttr().Set(1.0)
UsdGeom.XformCommonAPI(floor).SetScale(Gf.Vec3f(8, 8, 0.02))
UsdGeom.XformCommonAPI(floor).SetTranslate(Gf.Vec3d(0, 0, -0.01))
fmat, fsh = make_material("/World/floor_mat")
UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(fmat)

dome = UsdLux.DomeLight.Define(stage, "/World/dome")
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateAngleAttr(1.0)
sun_api = UsdGeom.XformCommonAPI(sun)
lamp = UsdLux.SphereLight.Define(stage, "/World/lamp")
lamp.CreateRadiusAttr(0.15)
lamp_api = UsdGeom.XformCommonAPI(lamp)

cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.GetHorizontalApertureAttr().Set(20.955)
cam.GetVerticalApertureAttr().Set(20.955 * H / W)
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 50))
cam_op = cam.AddTransformOp()


def point_camera(radius, elev_deg, az_deg, target):
    el, az = math.radians(elev_deg), math.radians(az_deg)
    eye = np.array(target) + radius * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    f = np.array(target) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = Gf.Matrix4d(r[0], r[1], r[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0, eye[0], eye[1], eye[2], 1)
    cam_op.Set(m)


rp = rep.create.render_product("/World/cam", (W, H))
rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
bb_a = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight", init_params={"semanticTypes": ["class"]})
rgb_a.attach([rp])
bb_a.attach([rp])
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=False)


# ---------------------------------------------------------------- randomization
def randomize_env():
    m = args.mode
    if m == "fixed":
        point_camera(1.3, 50, 0, (0, 0, 0.05))
        cam.GetFocalLengthAttr().Set(24.0)
        set_material(fsh, (0.55, 0.55, 0.55), 0.7)
        dome.CreateIntensityAttr(800)
        dome.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        sun.CreateIntensityAttr(2000)
        sun.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        sun_api.SetRotate(Gf.Vec3f(-55, 0, 30), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        lamp.CreateIntensityAttr(0)
        return
    if m == "rand":
        radius, elev, focal = U(1.0, 1.6), U(35, 70), U(20, 28)
        floor_rgb = hsv(U(0, 1), U(0, 0.6), U(0.25, 0.9))
        rough = U(0.4, 1.0)
        dome_i, tint = U(300, 1500), U(0.8, 1.0, 3)
        sun_i, sun_el = U(500, 3000), U(25, 80)
        lamp_i = U(0, 40000) if U() < 0.5 else 0
    else:  # shift: outside every range above
        radius = U(0.7, 0.9) if U() < 0.5 else U(1.9, 2.4)
        elev = U(12, 28) if U() < 0.5 else U(75, 88)
        focal = U(18, 22) if radius < 1 else U(30, 36)
        floor_rgb = hsv(U(0, 1), U(0.7, 1.0), U(0.12, 0.35))
        rough = U(0.05, 0.3)
        dome_i = U(40, 150) if U() < 0.5 else U(2500, 4000)
        tint = [U(0.4, 1.0), U(0.3, 0.8), U(0.2, 0.6)] if U() < 0.5 else [U(0.2, 0.6), U(0.4, 0.8), U(0.7, 1.0)]
        sun_i, sun_el = U(3500, 6000), U(6, 20)
        lamp_i = U(60000, 120000)
    point_camera(radius, elev, U(0, 360), (U(-0.08, 0.08), U(-0.08, 0.08), 0.05))
    cam.GetFocalLengthAttr().Set(float(focal))
    set_material(fsh, floor_rgb, rough)
    dome.CreateIntensityAttr(float(dome_i))
    dome.CreateColorAttr(Gf.Vec3f(*[float(t) for t in tint]))
    sun.CreateIntensityAttr(float(sun_i))
    sun.CreateColorAttr(Gf.Vec3f(*[float(c) for c in hsv(U(0, 1), U(0, 0.3), 1.0)]))
    sun_api.SetRotate(Gf.Vec3f(-float(sun_el), 0, float(U(0, 360))), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    lamp.CreateIntensityAttr(float(lamp_i))
    lamp_api.SetTranslate(Gf.Vec3d(U(-0.8, 0.8), U(-0.8, 0.8), U(0.5, 1.2)))
    lamp.CreateColorAttr(Gf.Vec3f(*[float(c) for c in hsv(U(0, 1), U(0, 0.5), 1.0)]))


def randomize_objects():
    for o in targets + distractors:
        o.hide()
    n_t = int(rng.integers(3, 8))
    n_d = 0 if args.mode == "fixed" else (int(rng.integers(2, 6)) if args.mode == "rand" else int(rng.integers(5, 9)))
    pool_t = list(rng.permutation(len(targets)))[:n_t]
    pool_d = list(rng.permutation(len(distractors)))[:n_d]
    placed = []  # (x, y, r)
    for o in [targets[i] for i in pool_t] + [distractors[i] for i in pool_d]:
        for _ in range(30):  # rejection sampling; allow mild overlap so occlusion happens
            x, y = U(-0.3, 0.3), U(-0.3, 0.3)
            tilt = 90.0 if (o.cls in ("cracker_box", "sugar_box") and U() < 0.25) else 0.0
            scale = 1.0 if o.kind in YCB else float(U(0.05, 0.12))
            r = o.place(x, y, U(0, 360), tilt, scale, bbox_cache)
            if all(math.hypot(x - px, y - py) > 0.6 * (r + pr) for px, py, pr in placed):
                break
        placed.append((x, y, r))
        if o.sh is not None:
            if args.mode == "fixed":
                set_material(o.sh, (0.5, 0.5, 0.5), 0.6)
            else:
                set_material(o.sh, hsv(U(0, 1), U(0.2, 1), U(0.3, 1)), U(0.2, 0.9))


def step():
    rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=True)


# ---------------------------------------------------------------- capture
os.makedirs(args.out + "/images", exist_ok=True)
os.makedirs(args.out + "/labels", exist_ok=True)
for _ in range(3):  # warm-up: first frames carry stale/blank annotator data
    randomize_env()
    randomize_objects()
    step()

n_boxes = {c: 0 for c in CLASSES}
n_empty = 0
t0 = time.time()
for i in range(args.n):
    randomize_env()
    randomize_objects()
    step()
    img = rgb_a.get_data()
    bb = bb_a.get_data()
    id2cls = {int(k): v["class"] for k, v in bb["info"]["idToLabels"].items()}
    lines = []
    for d in bb["data"]:
        c = id2cls.get(int(d["semanticId"]))
        if c not in CLASSES or float(d["occlusionRatio"]) > MAX_OCCLUSION:
            continue
        box = (float(d["x_min"]), float(d["y_min"]), float(d["x_max"]), float(d["y_max"]))
        if box[2] - box[0] < MIN_BOX_PX or box[3] - box[1] < MIN_BOX_PX:
            continue
        lines.append(label_line(CLASSES.index(c), box, W, H))
        n_boxes[c] += 1
    n_empty += not lines
    name = "%05d" % i
    Image.fromarray(np.asarray(img)[..., :3]).save(args.out + "/images/%s.jpg" % name, quality=95)
    with open(args.out + "/labels/%s.txt" % name, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    if i % 100 == 0:
        log("frame %d/%d  %.2f img/s" % (i, args.n, (i + 1) / (time.time() - t0)))
dt = time.time() - t0
stats = {"mode": args.mode, "seed": args.seed, "images": args.n, "seconds": round(dt, 1),
         "images_per_s": round(args.n / dt, 2), "rt_subframes": args.rt_subframes,
         "boxes_per_class": n_boxes, "images_without_boxes": n_empty}
json.dump(stats, open(args.out + "/gen_stats.json", "w"), indent=1)
log("DONE " + json.dumps(stats))
app.close()
