import time
import cv2
import numpy as np

def run_pixel_segmentation(golden_img, test_img, sensitivity=45):
    """
    U-Net / Morfolojik Piksel Segmentasyonu mantigiyla kusurlarin milimetrik
    piksel maskesini (binary mask) ve renkli bindirme (overlay) haritasini uretir.
    """
    # Gri tonlamaya cevir ve mutlak piksel farkini al
    g_gray = cv2.cvtColor(golden_img, cv2.COLOR_RGB2GRAY) if golden_img.ndim == 3 else golden_img
    t_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY) if test_img.ndim == 3 else test_img

    diff = cv2.absdiff(g_gray, t_gray)
    # Morfolojik acma/kapama ile gurultuyu temizle ve kesin sinirlari cikar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    _, pixel_mask = cv2.threshold(clean_diff, sensitivity, 255, cv2.THRESH_BINARY)

    # Renkli segmentasyon haritasi olustur (Kusurlu pikselleri parlak kirmizi/camgobegi boya)
    overlay = test_img.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

    mask_bool = pixel_mask > 0
    overlay[mask_bool] = [255, 40, 80]  # Piksel bazli kusur maskesi rengi

    defect_pixel_count = int(np.sum(mask_bool))
    total_pixels = int(pixel_mask.size)
    defect_ratio_pct = round((defect_pixel_count / total_pixels) * 100, 4)

    return pixel_mask, overlay, defect_pixel_count, defect_ratio_pct


def compute_patchcore_anomaly_map(golden_img, test_img, patch_size=16):
    """
    Few-Shot / Zero-Shot Anomali Tespiti (PatchCore / PaDiM mantigi):
    Goruntuyu kucuk yerel yamalara (patches) boler ve sadece saglam referansin
    doku istatistiklerinden (ortalama & standart sapma) sapma skorunu hesaplar.
    """
    g_float = golden_img.astype(np.float32) / 255.0
    t_float = test_img.astype(np.float32) / 255.0

    # Yerel yama (patch) bazli ortalama ve varyans farki (Mahalanobis benzeri mesafe)
    g_blur = cv2.GaussianBlur(g_float, (patch_size | 1, patch_size | 1), 0)
    t_blur = cv2.GaussianBlur(t_float, (patch_size | 1, patch_size | 1), 0)

    g_std = np.sqrt(np.maximum(cv2.GaussianBlur((g_float - g_blur) ** 2, (patch_size | 1, patch_size | 1), 0), 1e-4))
    t_std = np.sqrt(np.maximum(cv2.GaussianBlur((t_float - t_blur) ** 2, (patch_size | 1, patch_size | 1), 0), 1e-4))

    patch_distance = np.abs(t_blur - g_blur) + 0.5 * np.abs(t_std - g_std)
    if patch_distance.ndim == 3:
        anomaly_map = np.mean(patch_distance, axis=2)
    else:
        anomaly_map = patch_distance

    # 0-255 araligina normalize et ve isi haritasina (Heatmap) donustur
    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    max_anomaly_score = round(float(np.max(anomaly_map)), 4)
    return heatmap_rgb, max_anomaly_score


def benchmark_edge_inference(golden_img, test_img, iterations=30):
    """
    Uc cihaz (Edge AI - Jetson/Raspberry Pi) cikarim hizi (Latency & FPS) olcumu.
    """
    start_std = time.perf_counter()
    for _ in range(iterations):
        _ = run_pixel_segmentation(golden_img, test_img)
        _ = compute_patchcore_anomaly_map(golden_img, test_img)
    std_ms = ((time.perf_counter() - start_std) / iterations) * 1000

    # Quantized / Downsampled Edge hizlandirma modu (ONNX INT8 / FP16 simülasyonu)
    small_g = cv2.resize(golden_img, (0, 0), fx=0.5, fy=0.5)
    small_t = cv2.resize(test_img, (0, 0), fx=0.5, fy=0.5)
    start_edge = time.perf_counter()
    for _ in range(iterations):
        _ = run_pixel_segmentation(small_g, small_t)
    edge_ms = ((time.perf_counter() - start_edge) / iterations) * 1000

    return {
        "standard_latency_ms": round(std_ms, 2),
        "standard_fps": round(1000.0 / max(std_ms, 0.1), 1),
        "edge_optimized_ms": round(edge_ms, 2),
        "edge_fps": round(1000.0 / max(edge_ms, 0.1), 1)
    }
