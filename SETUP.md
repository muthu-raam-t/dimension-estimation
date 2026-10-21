# Single-Camera Parcel Dimensioning — run sheet

Drop every `sv_*.py` file into `~/cvproject`. Nothing here imports your existing
`geometry.py`, `models.py` or `config.py`, and no filename collides with them, so
your previous work is untouched.

---

## The 2-hour path

### 0. Install (2 min)

```bash
cd ~/cvproject
pip install opencv-contrib-python numpy scipy
pip install ultralytics gradio        # optional: detector + web UI
```

`opencv-contrib-python` matters — plain `opencv-python` has no ArUco module.
Check:

```bash
python3 -c "import cv2; print(cv2.__version__, hasattr(cv2,'aruco'))"
```

### 1. Print the marker (10 min, do this first — it has to physically happen)

```bash
python3 sv_make_marker.py
```

Prints to `calibration/marker_to_print.png`.

- Print at **100% / "Actual size"**. Not "Fit to page" — that silently rescales
  and every measurement afterwards is wrong by that factor.
- Measure the printed black square edge to edge with a ruler.
- Put that number in `sv_config.py` → `MARKER_LENGTH_CM`.
- Tape it flat to the table. A curled marker tilts the estimated ground plane.

**This is the single most important step.** Everything is scaled by this number.

### 2. Camera calibration (10 min — optional but roughly halves the error)

Print a 9×6 chessboard, stick it on cardboard, then:

```bash
python3 sv_calibrate_intrinsics.py --capture
```

SPACE to grab a view, Q when done. Take 15–25 views: centre, all four corners,
and several strong tilts. Tilt variety is what actually pins down focal length.
Aim for a reported reprojection error below 0.5 px.

Skipping this is allowed — the code falls back to a guessed 60° field of view
and prints a warning — but expect about double the dimension error.

### 3. Set up the scene (5 min)

- Flat table, even lighting, parcel and marker both fully in frame.
- **Tilt the webcam 30–60° off vertical.** Not straight down.
  Straight down is geometrically degenerate for a single camera: the top face
  maps to the ground plane by a pure scaling, so height becomes weakly
  observable. The live view shows the tilt angle and warns you.
- Keep the marker on the *same surface* the parcel sits on.

### 4. Run it

```bash
python3 sv_live.py          # OpenCV window, zero extra deps — most reliable
python3 sv_app.py           # browser UI at 127.0.0.1:7860 — nicer to demo
```

In `sv_live.py`: **SPACE** measures, **C** toggles continuous, **S** saves,
**Q** quits.

Single image:

```bash
python3 sv_measure.py --image photo.jpg --save results/out.jpg
```

If nothing is detected, force the model-free path:

```bash
python3 sv_live.py --method grabcut
```

---

## Dataset + detector (for the paper comparison, not the demo)

The demo works without any trained model — GrabCut gives a usable mask. Do this
when you want the detector table.

```bash
python3 sv_prepare_dataset.py --zip ~/Downloads/archive.zip
python3 sv_train.py --quick        # ~5 min sanity run
python3 sv_train.py                # the real run; use Colab if you have no GPU
```

I ran `sv_prepare_dataset.py` against your actual archive. It found 104 images,
matched 103 to labels, and dropped `1696670622499.jpg` which has no label file.
It also fixes the `label/` → `labels/` naming, ignores the duplicate
`label/label/` tree, sanitises the `WhatsApp Image ...` filenames with spaces,
and writes a correct single-class yaml — the bundled `coco128.yaml` is the
generic 80-class COCO file and does not describe your data at all.

The split is **group-aware**: images are grouped by perceptual hash so
near-duplicate burst frames cannot straddle train and val. On your archive it
found 103 distinct groups, so the frames are more varied than burst captures —
good news, and now you can say the split is honest rather than hoping.

---

## Accuracy reporting

```bash
python3 sv_eval.py --csv results/ground_truth.csv --images results/shots
```

`ground_truth.csv`:

```
image,length_cm,width_cm,height_cm
box01.jpg,24.0,16.2,11.0
```

It prints MAE, RMSE and mean relative error per axis next to the base paper's
numbers (0.86 / 0.76 / 1.43 cm), plus the "accuracy" figure under an explicit
formula — the paper never defines its 89%, and `1 − mean relative error` is the
definition that reproduces it. State the formula whenever you quote it.

---

## How it works (for your viva)

The paper recovers height from stereo disparity across an 11 cm baseline. With
one camera there is no disparity, so the metric scale has to come from
somewhere else — that is what the marker is for.

1. **ArUco marker → ground plane.** `solvePnP` on the four marker corners gives
   the camera's pose relative to the table, in centimetres.
2. **Segment the parcel.** YOLO mask if a model is available, otherwise YOLO box
   + GrabCut, otherwise plain GrabCut.
3. **Fit a cuboid by silhouette matching.** A box standing on a known plane has
   6 unknowns: centre (cx, cy), yaw, L, W, H. Project a candidate cuboid's 8
   corners, take their convex hull — which *is* the exact silhouette, since the
   image of a convex solid is convex — and score it against the observed mask by
   IoU. Maximise over the 6 parameters.

Optimisation detail worth knowing, because it is where this nearly went wrong:
a joint local search fails. A tall narrow box and a flat wide box cast
confusingly similar silhouettes, so the objective has strong local maxima. The
code runs a bounded global search (differential evolution) and then polishes at
higher resolution.

### Claims you can defend, vs. the paper

- **Height needs no second camera.** The paper's weakest axis (1.43 cm,
  16.6% relative) comes from stereo; here it comes from the plane constraint.
- **Rotated parcels.** The paper uses axis-aligned bounding boxes, which
  systematically over-measure any parcel rotated relative to the image axes.
  Silhouette fitting estimates yaw explicitly, so it does not.
- **One fit yields all three dimensions**, with no separate regression
  compensation stage. The paper needs a piecewise weighted linear regression
  fitted on 37 samples to correct its height.
- **Confidence is reported.** Every measurement carries a fit IoU and a tilt
  angle, and is flagged low-confidence below IoU 0.85 or under 20° tilt. The
  paper reports no per-measurement reliability signal.

### Validation already done

`test_sv_geometry.py` and `test_sv_pipeline.py` run a synthetic scene with a
known camera, known marker pose and a known box, then recover the dimensions:

| case | L err | W err | H err |
|---|---|---|---|
| tilt 35° / 45° / 60°, clean mask | ≤0.07 cm | ≤0.08 cm | ≤0.10 cm |
| noisy mask (imperfect segmentation) | 0.07 | 0.21 | 0.20 |
| flat parcel (H = 4 cm) | 0.09 | 0.32 | 0.27 |
| full pipeline, GrabCut (seg IoU 0.94) | 0.07 | 0.10 | 0.28 |

```bash
python3 test_sv_geometry.py
python3 test_sv_pipeline.py
```

**Read this honestly:** those are *synthetic* numbers. They prove the geometry
and the optimiser are correct — which is exactly what you want nailed down
before hardware arrives — but they are not real-world accuracy. A real webcam
adds lens distortion, rolling shutter, motion blur, segmentation error on real
textured parcels, and marker printing/measurement error. Expect real numbers
meaningfully worse. Do not put the synthetic figures in the paper as results;
put them in as a correctness check, and report `sv_eval.py` numbers from real
parcels as the actual result.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "ArUco marker not found" | Marker partly out of frame, glare, too small, or too oblique. It should span ≥80 px. Check `MARKER_LENGTH_CM` and `ARUCO_DICT` match what you printed. |
| "could not segment the parcel" | Try `--method grabcut`. Make sure the parcel contrasts with the table. |
| Height badly wrong, everything else fine | Camera too close to straight down. Tilt to 30–60°. The live view tells you the angle. |
| All dimensions off by the same factor | `MARKER_LENGTH_CM` is wrong, or the marker printed at the wrong scale. Re-measure the printed square. |
| Low IoU (<0.85) | Segmentation mask is poor, or the parcel is not actually a cuboid. |
| `cv2.aruco` missing | `pip uninstall opencv-python && pip install opencv-contrib-python` |
| Camera won't open | `ls /dev/video*`, then `python3 sv_live.py --camera 1` |

---

## Files

| file | what it does |
|---|---|
| `sv_config.py` | every constant — **edit `MARKER_LENGTH_CM` first** |
| `sv_geometry.py` | the core maths: pose, back-projection, cuboid fitting |
| `sv_segment.py` | mask from YOLO-seg / YOLO-det+GrabCut / GrabCut |
| `sv_pipeline.py` | `measure_image()` — the one function everything wraps |
| `sv_make_marker.py` | generates the marker to print |
| `sv_calibrate_intrinsics.py` | chessboard calibration |
| `sv_measure.py` | CLI, single image |
| `sv_live.py` | live webcam window |
| `sv_app.py` | browser UI |
| `sv_prepare_dataset.py` | cleans the Kaggle archive into a YOLO dataset |
| `sv_train.py` | fine-tunes YOLO11n |
| `sv_eval.py` | accuracy table vs the base paper |
| `test_sv_geometry.py`, `test_sv_pipeline.py` | correctness checks |
