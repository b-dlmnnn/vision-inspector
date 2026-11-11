import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.fusion_engine import compute_multimodal_risk
from src.vision_engine import (
    generate_synthetic_pcb,
    inspect_surface_defects,
    train_contour_ml_classifier,
)

app = FastAPI(
    title="Semiconductor Multi-Modal Quality API",
    description="Production REST API combining Optical Inspection (CV + Random Forest) and Sensor Risk Fusion.",
    version="2.0.0",
)

ML_CLF = train_contour_ml_classifier(n_boards=80)
REF_BOARD, _ = generate_synthetic_pcb(seed=1, inject_defects=False)


class FusionRequest(BaseModel):
    sensor_defect_prob: float = 0.25
    board_seed: int = 42
    inject_defects: bool = True
    shift_px: int = 2
    use_ecc_alignment: bool = True


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "semiconductor-multimodal-quality-api", "version": "2.0.0"}


@app.post("/api/v1/inspect-multimodal")
def inspect_multimodal_wafer(req: FusionRequest):
    if not (0.0 <= req.sensor_defect_prob <= 1.0):
        raise HTTPException(status_code=400, detail="sensor_defect_prob must be between 0.0 and 1.0")

    board_img, gt_boxes = generate_synthetic_pcb(
        seed=req.board_seed,
        inject_defects=req.inject_defects,
        shift_px=req.shift_px,
    )
    _, _, df_defects, perf = inspect_surface_defects(
        board_img,
        sensitivity=45,
        min_area=35,
        reference_bgr=REF_BOARD,
        use_alignment=req.use_ecc_alignment,
        ml_clf=ML_CLF,
    )

    n_defects = len(df_defects)
    total_area = int(df_defects["Area_px"].sum()) if n_defects > 0 else 0
    fusion_out = compute_multimodal_risk(
        sensor_prob=req.sensor_defect_prob,
        vision_defects_count=n_defects,
        vision_total_area_px=total_area,
    )

    return {
        "wafer_batch_seed": req.board_seed,
        "vision_telemetry": {
            "defects_detected": n_defects,
            "ground_truth_injected": len(gt_boxes),
            "total_defect_area_px": total_area,
            "latency_ms": perf["latency_ms"],
            "fps": perf["fps"],
            "ecc_aligned": perf["aligned"],
            "detections": df_defects.to_dict(orient="records"),
        },
        "multimodal_decision": fusion_out,
    }
