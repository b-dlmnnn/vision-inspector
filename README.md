# 📷 Vision Inspector AI — Real-Time Optical Surface & PCB Defect Detection

An Automated Optical Inspection (AOI) prototype built with **OpenCV** and **Streamlit** to detect, localize, and classify surface micro-defects across synthetic PCB/wafer boards, uploaded images, and live webcam captures.

## Key Capabilities
- **Morphological Anomaly Extraction:** Uses Top-Hat and Black-Hat transformations to isolate scratches, solder bridges, and foreign particles under varying illumination.
- **Geometric Defect Classification:** Categorizes detected contours by pixel area and bounding-box aspect ratio (`Linear Scratch`, `Solder Bridge`, `Micro-Particle`).
- **Multi-Source Inspection:** Supports real-time webcam capture (`st.camera_input`), user image uploads, and a deterministic synthetic PCB defect generator.
- **Automated Quality Gate:** Evaluates `PASS` / `REJECT` decisions against configurable defect tolerances.

## Run Locally
```bash
pip install -r requirements.txt
python -m pytest -v
streamlit run app.py
```
