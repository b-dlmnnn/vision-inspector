from src.vision_engine import generate_synthetic_pcb, inspect_surface_defects, train_contour_ml_classifier


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
