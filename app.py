"""Dimension Estimation - browser UI.

    streamlit run app.py

Camera (or upload) -> object mask -> monocular depth -> marker-anchored scale
-> length and width in cm. Also collects dataset rows for train_correction.py.

Two ways to find the object:
  1. YOLO11n-seg  - knows 80 COCO classes (bottle, cup, book, laptop, ...)
  2. Outline mode - largest shape against a plain background; works on boxes
                    and anything else YOLO does not know.
"""
import csv
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
import geometry

st.set_page_config(page_title="Dimension Estimation", layout="wide")

DATA_COLS = ["object_id", "class", "capture", "true_long", "true_short",
             "raw_long", "raw_short", "distance_cm", "long_px", "short_px",
             "area_px", "yolo_conf", "scale_k", "n_markers"]


@st.cache_resource(show_spinner="Loading models (first run downloads ~100 MB)...")
def load_models():
    import models as _m
    _m.get_yolo()
    _m.get_depth()
    return _m


def load_correction():
    try:
        import joblib
        if os.path.exists(config.CORRECTION_MODEL):
            return joblib.load(config.CORRECTION_MODEL)
    except Exception:
        pass
    return None


def to_bgr(uploaded):
    return cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8),
                        cv2.IMREAD_COLOR)


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def marker_mask(shape, markers):
    """True where a marker sits, so the outline finder ignores it."""
    mask = np.zeros(shape[:2], np.uint8)
    for centre, side in markers:
        x, y = int(centre[0]), int(centre[1])
        r = int(side * 0.9)
        cv2.rectangle(mask, (x - r, y - r), (x + r, y + r), 255, -1)
    return mask


def outline_segment(frame, markers):
    """Largest coherent shape against a plain background. No model needed."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # background is whatever dominates the border
    border = np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]])
    if border.mean() > 127:
        th = cv2.bitwise_not(th)

    th[marker_mask(frame.shape, markers) > 0] = 0
    k = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.02 * h * w]
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return {"mask": mask > 0, "label": "object", "conf": 0.0,
            "area": int((mask > 0).sum())}


def measure(frame, marker_cm, hfov, mode, manual_distance=None):
    """Returns a result dict, or an error string."""
    config.MARKER_CM = marker_cm
    config.ASSUMED_HFOV_DEG = hfov

    K, dist = geometry.load_intrinsics()
    if K is not None:
        frame = cv2.undistort(frame, K, dist)
    f = geometry.focal_px(frame.shape[1], K)
    markers = geometry.detect_markers(frame)

    det, used = None, mode
    if mode in ("Auto", "YOLO (known objects)"):
        det = load_models().segment(frame)
        used = "YOLO"
    if det is None and mode in ("Auto", "Outline (any object)"):
        det = outline_segment(frame, markers)
        used = "Outline"
    if det is None:
        return ("No object found. Make sure the OBJECT and the marker are both "
                "in the shot, the camera is far enough back to see both, and the "
                "background is plain. For boxes, pick Outline mode in the sidebar.")

    depth = load_models().depth_map_cm(frame)
    k, _ = geometry.anchor_scale(depth, markers, f)
    depth = depth * k

    if manual_distance:
        raw = float(np.median(depth[det["mask"]]))
        if raw > 1:
            depth = depth * (manual_distance / raw)

    m = geometry.measure(det["mask"], depth, f)
    if m is None:
        return "Object region too small to measure. Move closer."

    long_cm, short_cm, corrected = m["long_cm"], m["short_cm"], False
    corr = load_correction()
    if corr is not None:
        try:
            feats = {"raw_long": m["long_cm"], "raw_short": m["short_cm"],
                     "distance_cm": m["distance_cm"], "long_px": m["long_px"],
                     "short_px": m["short_px"], "area_px": m["area_px"],
                     "yolo_conf": det["conf"]}
            X = np.array([[feats[c] for c in corr["feats"]]], dtype=float)
            long_cm = m["long_cm"] + float(corr["models"]["true_long"].predict(X)[0])
            short_cm = m["short_cm"] + float(corr["models"]["true_short"].predict(X)[0])
            corrected = True
        except Exception:
            pass

    vis = frame.copy()
    overlay = np.zeros_like(vis)
    overlay[det["mask"]] = (0, 180, 120)
    vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0)
    cv2.drawContours(vis, [cv2.boxPoints(m["rect"]).astype(int)], 0, (0, 255, 255), 3)
    cv2.putText(vis, f"{det['label']}  {long_cm:.1f} x {short_cm:.1f} cm",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    dnorm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {"label": det["label"], "conf": det["conf"], "used": used,
            "long_cm": long_cm, "short_cm": short_cm,
            "raw_long": m["long_cm"], "raw_short": m["short_cm"],
            "distance_cm": m["distance_cm"], "long_px": m["long_px"],
            "short_px": m["short_px"], "area_px": m["area_px"],
            "scale_k": k, "n_markers": len(markers), "corrected": corrected,
            "vis": vis, "depth": cv2.applyColorMap(dnorm, cv2.COLORMAP_INFERNO)}


def append_row(row):
    os.makedirs(os.path.dirname(config.DATASET_CSV) or ".", exist_ok=True)
    new = not os.path.exists(config.DATASET_CSV)
    with open(config.DATASET_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(DATA_COLS)
        w.writerow([row[c] for c in DATA_COLS])


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
mode = st.sidebar.radio(
    "Object detection", ["Auto", "YOLO (known objects)", "Outline (any object)"],
    help="Auto tries YOLO first, then falls back to the outline method. "
         "Use Outline for cardboard boxes - YOLO does not know that class.")
marker_cm = st.sidebar.number_input(
    "Marker size (cm)", 1.0, 40.0, float(config.MARKER_CM), 0.1,
    help="Measure the black square of the printed or on-screen marker.")
hfov = st.sidebar.slider("Assumed horizontal FOV (deg)", 40, 90,
                         int(config.ASSUMED_HFOV_DEG))
use_manual = st.sidebar.checkbox("Override distance manually")
manual_distance = st.sidebar.number_input(
    "Measured distance (cm)", 10.0, 300.0, 60.0, 1.0) if use_manual else None

st.sidebar.markdown("---")
collect = st.sidebar.checkbox("Dataset collection mode")
if collect:
    obj_id = st.sidebar.text_input("Object id", "object01")
    true_a = st.sidebar.number_input("True dimension A (cm)", 0.5, 200.0, 20.0, 0.5)
    true_b = st.sidebar.number_input("True dimension B (cm)", 0.5, 200.0, 10.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Frame the OBJECT and the marker together, camera far enough back to see "
    "both. Plain background, object roughly centred.")

# ----------------------------------------------------------------- main
st.title("Dimension Estimation")
st.caption("Single camera · monocular depth · marker-anchored scale")

if "rows" not in st.session_state:
    st.session_state.rows = []
if "saved" not in st.session_state:
    st.session_state.saved = 0

tab_cam, tab_file = st.tabs(["Camera", "Upload image"])
with tab_cam:
    shot = st.camera_input("Take a photo", label_visibility="collapsed")
with tab_file:
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

source = shot or upload

if source is not None:
    frame = to_bgr(source)
    if frame is None:
        st.error("Could not read that image.")
    else:
        with st.spinner("Measuring..."):
            res = measure(frame, marker_cm, hfov, mode, manual_distance)

        if isinstance(res, str):
            st.warning(res)
        else:
            c1, c2 = st.columns(2)
            c1.image(to_rgb(res["vis"]), caption="Detection and measurement",
                     use_container_width=True)
            c2.image(to_rgb(res["depth"]), caption="Depth map",
                     use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Object", res["label"])
            m2.metric("Length", f"{res['long_cm']:.1f} cm")
            m3.metric("Width", f"{res['short_cm']:.1f} cm")
            m4.metric("Distance", f"{res['distance_cm']:.0f} cm")

            if res["n_markers"] == 0 and not manual_distance:
                st.warning("No marker found - the scale is unanchored, so these "
                           "numbers will drift. Show the marker, or tick the "
                           "distance override in the sidebar.")
            else:
                st.caption(
                    f"{res['used']} · markers {res['n_markers']} · "
                    f"scale x{res['scale_k']:.3f} · "
                    + ("correction model applied" if res["corrected"]
                       else "no correction model trained yet"))

            b1, b2 = st.columns(2)
            if b1.button("Add to measurement list", type="primary"):
                st.session_state.rows.append({
                    "object": res["label"],
                    "length_cm": round(res["long_cm"], 1),
                    "width_cm": round(res["short_cm"], 1),
                    "distance_cm": round(res["distance_cm"], 1),
                    "method": res["used"]})
                st.success("Added.")

            if collect and b2.button("Save dataset sample"):
                st.session_state.saved += 1
                append_row({
                    "object_id": obj_id, "class": res["label"],
                    "capture": st.session_state.saved,
                    "true_long": max(true_a, true_b),
                    "true_short": min(true_a, true_b),
                    "raw_long": round(res["raw_long"], 2),
                    "raw_short": round(res["raw_short"], 2),
                    "distance_cm": round(res["distance_cm"], 1),
                    "long_px": round(res["long_px"], 1),
                    "short_px": round(res["short_px"], 1),
                    "area_px": res["area_px"],
                    "yolo_conf": round(res["conf"], 3),
                    "scale_k": round(res["scale_k"], 3),
                    "n_markers": res["n_markers"]})
                st.success(f"Saved · raw {res['raw_long']:.1f} x "
                           f"{res['raw_short']:.1f} vs true "
                           f"{max(true_a, true_b):.1f} x {min(true_a, true_b):.1f}")

if st.session_state.rows:
    st.markdown("### Measurements")
    df = pd.DataFrame(st.session_state.rows)
    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns([1, 5])
    c1.download_button("Download CSV", df.to_csv(index=False).encode(),
                       "measurements.csv", "text/csv")
    if c2.button("Clear list"):
        st.session_state.rows = []
        st.rerun()

if collect and os.path.exists(config.DATASET_CSV):
    d = pd.read_csv(config.DATASET_CSV)
    st.markdown(f"### Dataset · {len(d)} rows · {d.object_id.nunique()} objects")
    st.dataframe(d.tail(10), use_container_width=True)
