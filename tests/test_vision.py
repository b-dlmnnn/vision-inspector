from src.vision_engine import generate_synthetic_pcb, inspect_surface_defects


def test_clean_board_passes():
    ref_img, _ = generate_synthetic_pcb(seed=10, inject_defects=False)
    test_img, injected = generate_synthetic_pcb(seed=11, inject_defects=False)
    assert len(injected) == 0
    _, _, df = inspect_surface_defects(test_img, sensitivity=45, min_area=40, reference_bgr=ref_img)
    assert len(df) == 0


def test_defective_board_detected():
    ref_img, _ = generate_synthetic_pcb(seed=42, inject_defects=False)
    test_img, injected = generate_synthetic_pcb(seed=42, inject_defects=True)
    assert len(injected) == 3
    _, _, df = inspect_surface_defects(test_img, sensitivity=45, min_area=40, reference_bgr=ref_img)
    assert len(df) == 3
