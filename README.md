# 📷 Vision Inspector — Computer Vision & ML Optical Defect Prototype

An Automated Optical Inspection (AOI) prototype combining **OpenCV** morphological/template image processing, **ECC sub-pixel image alignment**, and a **Random Forest** contour geometry classifier. Built as the optical companion to [SiliconGuard AI](https://github.com).

**Live App:** [https://streamlit.app](https://streamlit.app)

## 1. Methodology & Privacy
- **Classical CV + Learned Classifier:** Isolates surface anomalies via Golden-Template differencing (when reference template is available) or Top-Hat/Black-Hat morphology, then classifies defects (`Linear Scratch`, `Solder Bridge`, `Micro-Particle`) using a **Random Forest** trained on contour area, aspect ratio, circularity, solidity, and intensity.
- **Privacy:** Webcam captures and uploaded images are processed transiently in RAM and never stored.

## 2. Quantitative Benchmark (200 Randomized Synthetic Boards)

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Detection Precision / Recall / F1** | `0.7514` / `0.6650` / `0.7056` | Evaluated at IoU >= 0.25 across 600 injected defects |
| **Mean Bounding-Box IoU** | `0.7976` | Localization overlap against ground-truth defect boxes |
| **Rule-Based Classification Accuracy** | `99.00%` | Geometric aspect-ratio & area heuristics |
| **Random Forest Classification Accuracy** | `100.00%` | Trained on 5 contour geometry/intensity features |
| **5px Shift False Alarms (No ECC vs. ECC)** | `183` vs. `8` | ECC translation alignment reduces shift false alarms by ~95.6% |

## 3. Limitations & Scope
- **Synthetic Data Scope:** Quantitative benchmarks above are measured on synthetic PCB boards; real industrial deployment requires calibration on public datasets such as **DeepPCB** or **MVTec AD** under controlled lighting.
- **Simulated Economic Bridge:** Gate risk cost parameters (`$4,500` missed defect, `$400` false alarm) are simulated to illustrate integration with downstream packaging economics.

## 4. Run Locally
```bash
pip install -r requirements.txt
python -m src.evaluate_benchmark
python -m pytest -v
streamlit run app.py
```
