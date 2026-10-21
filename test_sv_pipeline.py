"""End-to-end test: render a scene with a marker + box, then run the real pipeline."""
import numpy as np, cv2
import sv_config as cfg
from sv_geometry import CameraParams, Cuboid, project_cuboid, render_silhouette
from sv_pipeline import measure_image, annotate
from test_sv_geometry import make_pose

W, H = 1280, 720
cam = CameraParams.approximate(W, H, cfg.FALLBACK_HFOV_DEG)

def build_scene(truth: Cuboid, tilt=45, marker_cm=10.0, offset=(-22.0, 0.0)):
    pose = make_pose(tilt)
    scene = np.full((H, W, 3), 190, np.uint8)
    # textured table so GrabCut has something to separate against
    noise = (np.random.default_rng(0).normal(0, 7, (H, W, 1))).astype(np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # marker, placed beside the box on the same plane
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICT))
    mimg = cv2.aruco.generateImageMarker(adict, cfg.ARUCO_ID, 400)
    mimg = cv2.copyMakeBorder(mimg, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    s = marker_cm / 2.0 * (500 / 400)
    ox, oy = offset
    obj = np.array([[ox-s, oy+s, 0], [ox+s, oy+s, 0],
                    [ox+s, oy-s, 0], [ox-s, oy-s, 0]], float)
    dst, _ = cv2.projectPoints(obj, pose.rvec, pose.tvec, cam.K, cam.dist)
    src = np.array([[0,0],[500,0],[500,500],[0,500]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst.reshape(-1,2).astype(np.float32))
    warp = cv2.warpPerspective(cv2.cvtColor(mimg, cv2.COLOR_GRAY2BGR), Hm, (W,H))
    msk = cv2.warpPerspective(np.full((500,500), 255, np.uint8), Hm, (W,H)) > 0
    scene[msk] = warp[msk]

    # the box: draw each visible face with its own shade so it looks solid
    px = project_cuboid(truth, cam, pose).astype(np.int32)
    faces = [([4,5,6,7],(120,70,40)), ([0,1,5,4],(90,55,30)),
             ([1,2,6,5],(70,45,25)), ([2,3,7,6],(85,50,28)), ([3,0,4,7],(75,48,26))]
    sil = render_silhouette(truth, cam, pose, (H,W))
    for idx, col in faces:
        cv2.fillConvexPoly(scene, px[idx], col)
    return scene, pose, sil

truth = Cuboid(cx=6.0, cy=2.0, theta=np.deg2rad(25), L=24.0, W=16.0, H=11.0)
scene, pose, sil = build_scene(truth)
cv2.imwrite("results/_test_scene.jpg", scene)

r = measure_image(scene, cam=cam, seg_method="grabcut")
print(r.summary())
print("seg:", r.seg_method)
if r.ok:
    tL,tW,tH = truth.sorted_dims(); eL,eW,eH = r.cuboid.sorted_dims()
    print(f"\ntrue L={tL:.2f} W={tW:.2f} H={tH:.2f}")
    print(f"est  L={eL:.2f} W={eW:.2f} H={eH:.2f}")
    print(f"ERR  L={abs(eL-tL):.2f} W={abs(eW-tW):.2f} H={abs(eH-tH):.2f}")
    cv2.imwrite("results/_test_annotated.jpg", annotate(scene, r, cam=cam))
    print("wrote results/_test_annotated.jpg")
    # segmentation quality, for context on the error
    inter=np.count_nonzero((r.mask>0)&(sil>0)); union=np.count_nonzero((r.mask>0)|(sil>0))
    print(f"segmentation IoU vs true silhouette: {inter/union:.3f}")
else:
    raise SystemExit("PIPELINE FAILED")
