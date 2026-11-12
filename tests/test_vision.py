from fastapi.testclient import TestClient
from api import app
from src.fusion_engine import compute_multimodal_risk
from src.vision_engine import generate_synthetic_pcb, inspect_surface_defects, train_contour_ml_classifier

client = TestClient(app)


def test_clean_board_passes():
    ref_img, _ = generate_synthetic_pcb(seed=1, inject_defects=False)
    test_img, gt = generate_synthetic_pcb(seed=11, inject_defects=False)
    assert len(gt) == 0
    _, _, df, _ = inspect_surface_defects(test_img, sensitivity=45, min_area=35, reference_bgr=ref_img, use_alignment=False)
    assert len(df) == 0


def test_defective_board_with_ml_and_ecc():
    clf = train_contour_ml_classifier(n_boards=40)
    ref_img, _ = generate_synthetic_pcb(seed=1, inject_defects=False)
    test_img, gt = generate_synthetic_pcb(seed=42, inject_defects=True, shift_px=4)
    assert len(gt) == 3
    _, _, df, perf = inspect_surface_defects(test_img, sensitivity=45, min_area=35, reference_bgr=ref_img, use_alignment=True, ml_clf=clf)
    assert len(df) >= 2
    assert perf["fps"] > 0


def test_multimodal_fusion_logic():
    low = compute_multimodal_risk(sensor_prob=0.10, vision_defects_count=0, vision_total_area_px=0)
    high = compute_multimodal_risk(sensor_prob=0.80, vision_defects_count=2, vision_total_area_px=500)
    assert low["routing"] == "RELEASE"
    assert high["routing"] == "BLOCK_PACKAGING"


def test_fastapi_endpoints():
    r_health = client.get("/health")
    assert r_health.status_code == 200
    r_inf = client.post("/api/v1/inspect-multimodal", json={"sensor_defect_prob": 0.65, "board_seed": 42, "inject_defects": True})
    assert r_inf.status_code == 200
    body = r_inf.json()
    assert body["multimodal_decision"]["routing"] == "BLOCK_PACKAGING"
