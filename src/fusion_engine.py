import numpy as np
import pandas as pd


def compute_multimodal_risk(
    sensor_prob,
    vision_defects_count,
    vision_total_area_px,
    w_sensor=0.45,
    w_vision=0.45,
    w_synergy=0.10,
):
    sensor_p = float(np.clip(sensor_prob, 0.0, 1.0))
    # Optik kusur sayisi ve alanindan normalize edilmis [0, 1] gorsel risk skoru
    count_score = min(vision_defects_count / 3.0, 1.0)
    area_score = min(vision_total_area_px / 900.0, 1.0)
    vision_p = round(0.65 * count_score + 0.35 * area_score, 4)

    # Late Fusion: Dogrusal agirlik + etkilesim (synergy) terimi
    raw_fused = (w_sensor * sensor_p) + (w_vision * vision_p) + (w_synergy * sensor_p * vision_p)
    fused_prob = round(float(np.clip(raw_fused, 0.0, 1.0)), 4)

    if fused_prob >= 0.65 or vision_defects_count >= 2:
        decision = "🚨 SCRAP / QUARANTINE (Critical Multi-Modal Failure)"
        routing = "BLOCK_PACKAGING"
    elif fused_prob >= 0.35 or sensor_p >= 0.75:
        decision = "⚠️ METROLOGY RE-INSPECTION (Suspect Anomaly)"
        routing = "SECONDARY_LAB_CHECK"
    else:
        decision = "✅ PASS TO PACKAGING (Cleared)"
        routing = "RELEASE_TO_CUSTOMER"

    return {
        "sensor_risk_pct": round(sensor_p * 100, 2),
        "vision_risk_pct": round(vision_p * 100, 2),
        "fused_risk_pct": round(fused_prob * 100, 2),
        "decision": decision,
        "routing": routing,
    }


def simulate_fab_lot_fusion(n_wafers=100, seed=42):
    rng = np.random.RandomState(seed)
    rows = list()
    for i in range(n_wafers):
        true_defect = rng.rand() < 0.12
        if true_defect:
            s_prob = float(np.clip(rng.beta(5, 2), 0.05, 0.99))
            v_cnt = int(rng.choice(list((1, 2, 3)), p=list((0.3, 0.5, 0.2))))
            v_area = int(v_cnt * rng.randint(120, 320))
        else:
            s_prob = float(np.clip(rng.beta(1.5, 8), 0.01, 0.70))
            v_cnt = int(rng.choice(list((0, 1)), p=list((0.92, 0.08))))
            v_area = int(v_cnt * rng.randint(40, 90))

        res = compute_multimodal_risk(s_prob, v_cnt, v_area)
        rows.append({
            "Wafer_ID": f"LOT26-{100 + i}",
            "True_Status": "DEFECT" if true_defect else "PASS",
            "Sensor_Risk_%": res["sensor_risk_pct"],
            "Vision_Risk_%": res["vision_risk_pct"],
            "Fused_Risk_%": res["fused_risk_pct"],
            "Routing": res["routing"],
        })
    return pd.DataFrame(rows)
