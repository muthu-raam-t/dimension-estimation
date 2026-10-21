"""
sv_app.py -- browser UI (Gradio). Webcam snapshot or file upload in, dimensions out.

    pip install gradio
    python3 sv_app.py

Then open the printed http://127.0.0.1:7860 link. Use the webcam tab to take a
shot, or drag in a photo. This is the nicer thing to demo to an examiner; use
sv_live.py if you want continuous video or if Gradio will not install in time.
"""
import cv2
import numpy as np

import sv_config as cfg
from sv_pipeline import measure_image, annotate, load_camera

try:
    import gradio as gr
except ImportError:
    raise SystemExit("Gradio is not installed.  pip install gradio\n"
                     "(or just use: python3 sv_live.py)")


def run(image_rgb, method):
    if image_rgb is None:
        return None, "Give me an image first."
    bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    cam = load_camera(bgr.shape)
    r = measure_image(bgr, cam=cam, seg_method=method)
    out = annotate(bgr, r, cam=cam)

    if not r.ok:
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), f"**Failed:** {r.message}"

    L, W, H = r.cuboid.sorted_dims()
    warn = ""
    if not r.trusted:
        warn = ("\n\n> **Low confidence.** Either the fit quality is poor or the "
                "camera is too close to straight-down. Tilt to 30-60 degrees and "
                "make sure the whole parcel and the marker are visible.")
    if not cfg.INTRINSICS_FILE.exists():
        warn += ("\n\n> No camera calibration found - using a guessed field of "
                 "view. Run `sv_calibrate_intrinsics.py` for better accuracy.")

    md = (f"| | cm |\n|---|---|\n"
          f"| **Length** | **{L:.2f}** |\n"
          f"| **Width** | **{W:.2f}** |\n"
          f"| **Height** | **{H:.2f}** |\n"
          f"| Volume | {r.cuboid.volume_cm3():.0f} cm³ |\n"
          f"| Volumetric weight | "
          f"{r.cuboid.volumetric_weight_kg(cfg.VOLUMETRIC_DIVISOR):.2f} kg |\n"
          f"| Fit IoU | {r.iou:.3f} |\n"
          f"| Camera tilt | {r.tilt_deg:.0f}° |\n"
          f"| Segmentation | {r.seg_method} |\n{warn}")
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), md


with gr.Blocks(title="Single-Camera Parcel Dimensioning") as demo:
    gr.Markdown(
        "# Single-Camera 3-D Parcel Dimensioning\n"
        "One webcam, no stereo rig. Place the printed ArUco marker flat on the "
        "same surface as the parcel, keep both fully in frame, and tilt the "
        "camera 30–60° off vertical."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(sources=["webcam", "upload"], type="numpy", label="Input")
            method = gr.Radio(["auto", "yolo_seg", "yolo_det", "grabcut"],
                              value="auto", label="Segmentation")
            btn = gr.Button("Measure", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Fitted cuboid")
            out_md = gr.Markdown()

    btn.click(run, [inp, method], [out_img, out_md])

if __name__ == "__main__":
    demo.launch()
