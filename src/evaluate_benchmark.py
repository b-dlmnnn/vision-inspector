import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.vision_engine import (
    compute_iou,
    extract_contour_features,
    generate_synthetic_pcb,
    inspect_surface_defects,
    rule_based_classify,
    train_contour_ml_classifier,
)
import cv2


def run_benchmark(n_boards=200):
    ml_clf = train_contour_ml_classifier(n_boards=120)
    ref_img, _ = generate_synthetic_pcb(seed=1, inject_defects=False)

    tp_tot, fp_tot, fn_tot = 0, 0, 0
    ious = list()
    y_true_cls, y_rule_cls, y_ml_cls = list(), list(), list()

    # 1. 200 Farkli Rastgele Kart Uzerinde Tespit ve Siniflandirma Basarisi
    for seed in range(500, 500 + n_boards):
        img, gt_boxes = generate_synthetic_pcb(seed=seed, inject_defects=True, shift_px=0)
        _, _, df_det, _ = inspect_surface_defects(img, sensitivity=45, min_area=35, reference_bgr=ref_img, use_alignment=False, ml_clf=ml_clf)

        matched_gt = set()
        for _, row in df_det.iterrows():
            bx, by, bw, bh = [int(v.strip()) for v in row["BBox_XYWH"].strip("()").split(",")]
            pred_box = (bx, by, bw, bh)
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.25 and best_j not in matched_gt:
                tp_tot += 1
                matched_gt.add(best_j)
                ious.append(best_iou)
                y_true_cls.append(gt_boxes[best_j]["type"])
                y_rule_cls.append(row["Rule_Class"])
                y_ml_cls.append(row["ML_Class (RF)"])
            else:
                fp_tot += 1
        fn_tot += len(gt_boxes) - len(matched_gt)

    prec = round(tp_tot / max(tp_tot + fp_tot, 1), 4)
    rec = round(tp_tot / max(tp_tot + fn_tot, 1), 4)
    f1 = round(2 * prec * rec / max(prec + rec, 1e-6), 4)
    mean_iou = round(float(np.mean(ious)) if ious else 0.0, 4)

    # 2. Kayma Dayaniklilik Testi (5 px kaymis 50 kart: ECC Hizalama Onsuz vs Onlu)
    fp_no_align, fp_with_align = 0, 0
    for seed in range(800, 850):
        shifted_img, _ = generate_synthetic_pcb(seed=seed, inject_defects=True, shift_px=5)
        _, _, df_no, _ = inspect_surface_defects(shifted_img, sensitivity=45, min_area=35, reference_bgr=ref_img, use_alignment=False, ml_clf=ml_clf)
        _, _, df_al, _ = inspect_surface_defects(shifted_img, sensitivity=45, min_area=35, reference_bgr=ref_img, use_alignment=True, ml_clf=ml_clf)
        fp_no_align += max(len(df_no) - 3, 0)
        fp_with_align += max(len(df_al) - 3, 0)

    summary = {
        "n_eval_boards": n_boards,
        "detection_precision": prec,
        "detection_recall": rec,
        "detection_f1": f1,
        "mean_iou": mean_iou,
        "rule_based_cls_acc": round(accuracy_score(y_true_cls, y_rule_cls), 4),
        "random_forest_cls_acc": round(accuracy_score(y_true_cls, y_ml_cls), 4),
        "shift_5px_false_alarms_without_ecc": fp_no_align,
        "shift_5px_false_alarms_with_ecc": fp_with_align,
    }
    with open("reports/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_benchmark()
