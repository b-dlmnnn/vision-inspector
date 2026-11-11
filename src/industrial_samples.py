import cv2
import numpy as np

def get_industrial_sample(sample_name="1. High-Glare Copper Trace (Lighting Reflection)"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (32, 78, 28)
    for y in range(50, 440, 55):
        cv2.line(img, (35, y), (605, y), (58, 145, 68), 5)
        for x in range(70, 590, 85):
            cv2.circle(img, (x, y), 9, (175, 205, 215), -1)
            cv2.circle(img, (x, y), 4, (32, 78, 28), -1)
    if "High-Glare" in sample_name:
        overlay = img.copy()
        cv2.circle(overlay, (510, 110), 85, (235, 245, 240), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.line(img, (180, 160), (295, 245), (225, 235, 245), 3)
        note = "Challenging Case: Specular lighting glare in upper-right alongside a true diagonal scratch."
    elif "Low-Contrast" in sample_name:
        cv2.line(img, (210, 120), (330, 210), (95, 175, 105), 2)
        cv2.circle(img, (420, 310), 9, (45, 65, 135), -1)
        note = "Challenging Case: Faint low-contrast hairline trace crack testing threshold sensitivity."
    elif "Multi-Bridge" in sample_name:
        cv2.ellipse(img, (240, 215), (20, 11), 15, 0, 360, (210, 220, 235), -1)
        cv2.ellipse(img, (395, 270), (19, 13), -20, 0, 360, (210, 220, 235), -1)
        cv2.circle(img, (160, 325), 12, (25, 25, 195), -1)
        note = "Production Defect Cluster: Multiple solder bridges and foreign contamination particles."
    else:
        note = "Clean Calibration Reference: Zero injected defects under nominal factory illumination."
    rng = np.random.RandomState(99)
    noise = rng.normal(0, 4.5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, note
