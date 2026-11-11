import cv2
import numpy as np
import pandas as pd


def generate_synthetic_pcb(seed=42, inject_defects=True, width=640, height=480):
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (35, 85, 30)

    for y in range(60, height - 40, 60):
        cv2.line(img, (40, y), (width - 40, y), (60, 150, 70), 6)
        for x in range(80, width - 60, 90):
            cv2.circle(img, (x, y), 10, (180, 210, 220), -1)
            cv2.circle(img, (x, y), 5, (35, 85, 30), -1)

    noise = rng.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    injected_list = list()
    if inject_defects:
        cv2.line(img, (150, 110), (260, 195), (220, 230, 245), 3)
        injected_list.append("Surface Scratch")
        cv2.ellipse(img, (410, 240), (18, 12), 25, 0, 360, (210, 220, 235), -1)
        injected_list.append("Solder Bridge")
        cv2.circle(img, (290, 355), 11, (25, 25, 190), -1)
        injected_list.append("Contamination Particle")

    return img, injected_list


def inspect_surface_defects(image_bgr, sensitivity=45, min_area=40, reference_bgr=None):
    output = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if reference_bgr is not None and reference_bgr.shape == image_bgr.shape:
        ref_gray = cv2.GaussianBlur(cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        diff = cv2.absdiff(blurred, ref_gray)
        _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        combined = cv2.add(tophat, blackhat)
        _, thresh = cv2.threshold(combined, sensitivity, 255, cv2.THRESH_BINARY)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    records = list()
    defect_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / max(float(h), 1.0), 2)

        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            defect_type = "Linear Scratch / Trace Break"
        elif area > 350:
            defect_type = "Solder Bridge / Large Anomaly"
        else:
            defect_type = "Micro-Particle / Pit Defect"

        cv2.rectangle(output, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 2)
        cv2.putText(output, f"#{defect_id}", (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        records.append({
            "Defect_ID": f"DEF-{defect_id:02d}",
            "Classification": defect_type,
            "Area_px": int(area),
            "BBox_XYWH": f"({x}, {y}, {w}, {h})",
            "Severity": "HIGH" if area > 300 else "MEDIUM",
        })
        defect_id += 1

    return output, thresh, pd.DataFrame(records)
