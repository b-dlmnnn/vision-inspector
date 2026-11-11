import time
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (aw * ah) + (bw * bh) - inter
    return round(inter / max(float(union), 1.0), 4)


def generate_synthetic_pcb(seed=42, inject_defects=True, shift_px=0, brightness_delta=0, width=640, height=480):
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (35, 85, 30)

    for y in range(60, height - 40, 60):
        cv2.line(img, (40, y), (width - 40, y), (60, 150, 70), 6)
        for x in range(80, width - 60, 90):
            cv2.circle(img, (x, y), 10, (180, 210, 220), -1)
            cv2.circle(img, (x, y), 5, (35, 85, 30), -1)

    gt_boxes = list()
    if inject_defects:
        # 1. Rastgele konumda Yuzey Cizigi (Scratch)
        sx = int(rng.randint(90, 240))
        sy = int(rng.randint(80, 180))
        ex = sx + int(rng.randint(75, 120))
        ey = sy + int(rng.randint(55, 95))
        cv2.line(img, (sx, sy), (ex, ey), (225, 235, 245), 3)
        gt_boxes.append({"type": "Linear Scratch", "box": (min(sx, ex) - 3, min(sy, ey) - 3, abs(ex - sx) + 6, abs(ey - sy) + 6)})

        # 2. Rastgele konumda Lehim Koprusu (Solder Bridge)
        bx = int(rng.randint(340, 500))
        by = int(rng.randint(150, 290))
        cv2.ellipse(img, (bx, by), (18, 12), int(rng.randint(0, 60)), 0, 360, (215, 225, 240), -1)
        gt_boxes.append({"type": "Solder Bridge", "box": (bx - 20, by - 16, 40, 32)})

        # 3. Rastgele konumda Yabanci Parcacik (Contamination Particle)
        px = int(rng.randint(160, 460))
        py = int(rng.randint(310, 410))
        cv2.circle(img, (px, py), 11, (25, 25, 195), -1)
        gt_boxes.append({"type": "Micro-Particle", "box": (px - 13, py - 13, 26, 26)})

    noise = rng.normal(brightness_delta, 4.0, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if shift_px != 0:
        m = np.float32(list((list((1, 0, shift_px)), list((0, 1, shift_px)))))
        img = cv2.warpAffine(img, m, (width, height), borderMode=cv2.BORDER_REFLECT)

    return img, gt_boxes


def align_to_reference(target_bgr, reference_bgr):
    gray_t = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(gray_r, gray_t, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        aligned = cv2.warpAffine(target_bgr, warp_matrix, (reference_bgr.shape.item(1) if False else list(reference_bgr.shape).pop(1), list(reference_bgr.shape).pop(0)), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        return aligned, True
    except cv2.error:
        return target_bgr, False


def extract_contour_features(cnt, gray_img):
    area = float(cv2.contourArea(cnt))
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = round(w / max(float(h), 1.0), 3)
    perim = max(float(cv2.arcLength(cnt, True)), 1.0)
    circularity = round(4.0 * np.pi * area / (perim * perim), 3)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(cnt))), 1.0)
    solidity = round(area / hull_area, 3)
    roi = gray_img[y:y + h, x:x + w]
    mean_intensity = round(float(np.mean(roi)) if roi.size > 0 else 0.0, 2)
    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "mean_intensity": mean_intensity,
        "bbox": (x, y, w, h),
    }


def rule_based_classify(feat):
    if feat["aspect_ratio"] > 2.3 or feat["aspect_ratio"] < 0.43 or feat["circularity"] < 0.38:
        return "Linear Scratch"
    elif feat["area"] > 340:
        return "Solder Bridge"
    return "Micro-Particle"


def train_contour_ml_classifier(n_boards=120):
    X_rows = list()
    y_rows = list()
    ref_img, _ = generate_synthetic_pcb(seed=1, inject_defects=False)
    for s in range(100, 100 + n_boards):
        img, gt_boxes = generate_synthetic_pcb(seed=s, inject_defects=True)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0), cv2.GaussianBlur(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), (5, 5), 0))
        _, thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 35:
                continue
            feat = extract_contour_features(cnt, gray)
            best_iou = 0.0
            best_label = None
            for gt in gt_boxes:
                iou = compute_iou(feat["bbox"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_label = gt["type"]
            if best_iou >= 0.20 and best_label is not None:
                X_rows.append(list((feat["area"], feat["aspect_ratio"], feat["circularity"], feat["solidity"], feat["mean_intensity"])))
                y_rows.append(best_label)

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(np.array(X_rows), np.array(y_rows))
    return clf


def inspect_surface_defects(image_bgr, sensitivity=45, min_area=40, reference_bgr=None, use_alignment=True, ml_clf=None):
    t0 = time.perf_counter()
    work_bgr = image_bgr.copy()
    aligned_ok = False
    if reference_bgr is not None and use_alignment and reference_bgr.shape == work_bgr.shape:
        work_bgr, aligned_ok = align_to_reference(work_bgr, reference_bgr)

    output = work_bgr.copy()
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if reference_bgr is not None and reference_bgr.shape == work_bgr.shape:
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
        feat = extract_contour_features(cnt, gray)
        x, y, w, h = feat["bbox"]
        rule_lbl = rule_based_classify(feat)
        if ml_clf is not None:
            vec = np.array(list((list((feat["area"], feat["aspect_ratio"], feat["circularity"], feat["solidity"], feat["mean_intensity"])),)))
            ml_lbl = str(ml_clf.predict(vec).take(0))
        else:
            ml_lbl = rule_lbl

        cv2.rectangle(output, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 2)
        cv2.putText(output, f"#{defect_id} {ml_lbl}", (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        records.append({
            "Defect_ID": f"DEF-{defect_id:02d}",
            "ML_Class (RF)": ml_lbl,
            "Rule_Class": rule_lbl,
            "Area_px": int(area),
            "Circularity": feat["circularity"],
            "BBox_XYWH": f"({x}, {y}, {w}, {h})",
        })
        defect_id += 1

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    fps = round(1000.0 / max(elapsed_ms, 0.1), 1)
    return output, thresh, pd.DataFrame(records), {"latency_ms": elapsed_ms, "fps": fps, "aligned": aligned_ok}
