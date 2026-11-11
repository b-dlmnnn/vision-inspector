import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from src.vision_engine import generate_synthetic_pcb, inspect_surface_defects

st.set_page_config(
    page_title="Vision Inspector AI | Optical Defect Detection",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Vision Inspector AI — Real-Time Optical Surface & PCB Defect Detection")
st.caption(
    "Automated Optical Inspection (AOI) prototype using Golden-Template reference differencing, "
    "morphological filtering, and contour geometry analysis."
)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Optical Inspection Controls")
source_mode = st.sidebar.radio(
    "Select Image Source",
    ("Synthetic PCB / Wafer Generator", "Live Webcam Capture", "Upload Surface Image (JPG/PNG)"),
)

sensitivity = st.sidebar.slider("Detection Sensitivity Threshold", 15, 120, 45, 5)
min_area = st.sidebar.slider("Minimum Defect Area (px)", 15, 300, 40, 5)
max_allowed_defects = st.sidebar.number_input("Quality Gate: Max Allowed Defects", 0, 10, 0, 1)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Inspection Pipeline:**\n"
    "1. **Synthetic Mode:** Compares inspected board against an aligned Golden Reference Template.\n"
    "2. **Camera / Upload Mode:** Applies morphological Top-Hat & Black-Hat anomaly filtering.\n"
    "3. Classifies defects by contour area & aspect ratio and evaluates PASS/REJECT gate."
)

# --- IMAGE ACQUISITION ---
image_bgr = None
ref_bgr = None
injected_notes = list()

if source_mode == "Synthetic PCB / Wafer Generator":
    c_gen1, c_gen2 = st.columns(2)
    with c_gen1:
        seed_val = st.slider("Board Batch Seed", 1, 50, 42, 1)
    with c_gen2:
        inject_flag = st.checkbox("Inject Micro-Defects (Scratch, Solder Bridge, Particle)", value=True)
    ref_bgr, _ = generate_synthetic_pcb(seed=seed_val + 100, inject_defects=False)
    image_bgr, injected_notes = generate_synthetic_pcb(seed=seed_val, inject_defects=inject_flag)

elif source_mode == "Live Webcam Capture":
    cam_file = st.camera_input("Capture Live Frame from Webcam")
    if cam_file is not None:
        pil_img = Image.open(cam_file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

else:
    up_img = st.file_uploader("Upload PCB, Wafer, or Surface Photo", type=list(("jpg", "jpeg", "png")))
    if up_img is not None:
        pil_img = Image.open(up_img).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- INSPECTION PIPELINE ---
if image_bgr is not None:
    annotated_bgr, mask_gray, df_defects = inspect_surface_defects(
        image_bgr, sensitivity=sensitivity, min_area=min_area, reference_bgr=ref_bgr
    )

    n_found = len(df_defects)
    total_area = int(df_defects["Area_px"].sum()) if n_found > 0 else 0
    decision_pass = n_found <= max_allowed_defects
    h_px, w_px = image_bgr.shape.item(0) if False else list(image_bgr.shape).pop(0), list(image_bgr.shape).pop(1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Defects Detected", n_found)
    m2.metric("Total Defect Area", f"{total_area} px")
    m3.metric("Frame Resolution", f"{w_px} x {h_px}")
    m4.metric("Quality Gate Decision", "✅ PASS" if decision_pass else "🚨 REJECT (Quarantine)")

    if injected_notes:
        st.caption(f"Ground-Truth Simulated Injections: {', '.join(injected_notes)}")

    col_orig, col_det, col_mask = st.columns(3)
    with col_orig:
        st.subheader("1. Raw Optical Frame")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_det:
        st.subheader("2. Annotated Defect Detection")
        st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_mask:
        st.subheader("3. Binary Anomaly Mask")
        st.image(mask_gray, use_container_width=True)

    st.markdown("### Detected Anomaly Log")
    if n_found > 0:
        st.dataframe(df_defects, use_container_width=True, hide_index=True)
    else:
        st.success("No surface anomalies detected above the configured area and sensitivity thresholds.")
else:
    st.info("Waiting for camera capture or image upload to run optical inspection.")
