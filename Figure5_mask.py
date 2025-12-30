import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

# ==========================================

BASE_PATH = "./plot_data"
OUT_ROOT = "results_watershed_analysis"
MASK_PATH = "synthetic_river_mask.npy"

# 时间参数
INTERVAL_MIN = 10
LEAD_STEPS = 18 

# 模型映射与颜色
MODEL_CONFIG = {
    "WAVE-NowcastNet": "WAVE_NowcastNet",
    "NowcastNet-Orig": "NowcastNet_Orig",
    "ConvLSTM": "convlstm",
    "PredRNN": "predrnn",
    "OpticalFlow": "opticalflow"
}

MODEL_COLORS = {
    "Observations": "#000000",
    "WAVE-NowcastNet": "#1f77b4",
    "NowcastNet-Orig": "#ff7f0e",
    "ConvLSTM": "#2ca02c",
    "PredRNN": "#9467bd",
    "OpticalFlow": "#d62728",
}

# ==========================================
if os.path.exists(MASK_PATH):
    BASIN_MASK = np.load(MASK_PATH).astype(np.float32)
    MASK_SUM = np.sum(BASIN_MASK)
    if MASK_SUM == 0:
        raise ValueError("Error: Mask is empty (all zeros).")
else:
    raise FileNotFoundError(f"Error: Mask file not found at {MASK_PATH}.")

def calculate_masked_mean(data_array, mask, mask_sum):
    if data_array is None: return np.nan
    return np.sum(data_array * mask) / mask_sum

def get_period_stats(metrics_array):
    arr = np.array(metrics_array)
    return {
        "0-1h": np.nanmean(arr[0:6]), 
        "1-2h": np.nanmean(arr[6:12]),
        "2-3h": np.nanmean(arr[12:18]), 
        "0-3h": np.nanmean(arr[0:18])
    }

def calculate_rmse(pred_series, gt_series):
    p, g = np.array(pred_series), np.array(gt_series)
    mask = ~np.isnan(p) & ~np.isnan(g)
    if not np.any(mask): return 0.0
    return np.sqrt(np.mean((p[mask] - g[mask])**2))

def neighborhood_metrics(p_raw, g_raw, thr, neigh_size):
    P = (p_raw >= thr).astype(np.uint8)
    G = (g_raw >= thr).astype(np.uint8)
    kernel = np.ones((neigh_size, neigh_size), dtype=np.uint8)
    p_has = (cv2.filter2D(P, -1, kernel, borderType=cv2.BORDER_CONSTANT) > 0).astype(np.uint8)
    g_has = (cv2.filter2D(G, -1, kernel, borderType=cv2.BORDER_CONSTANT) > 0).astype(np.uint8)
    hits = int(np.logical_and(G == 1, p_has == 1).sum())
    misses = int(np.logical_and(G == 1, p_has == 0).sum())
    false_alarms = int(np.logical_and(P == 1, g_has == 0).sum())
    return hits / (hits + false_alarms + misses + 1e-6)

# ==========================================
def plot_event_rainfall(event_str, rain_data, out_dir):
    plt.figure(figsize=(10, 6), dpi=150)
    time_axis = np.arange(1, LEAD_STEPS + 1) * INTERVAL_MIN
    
    gt_series = rain_data["Observations"]
    plt.plot(time_axis, gt_series, label="Observations", color=MODEL_COLORS["Observations"], 
             linestyle="--", linewidth=3, zorder=10)
    
    for name in MODEL_CONFIG.keys():
        pred_series = rain_data[name]
        rmse = calculate_rmse(pred_series, gt_series)
        plt.plot(time_axis, pred_series, label=f"{name} (RMSE: {rmse:.2f})", 
                 color=MODEL_COLORS.get(name, "gray"), linewidth=1.8, alpha=0.8)

    plt.title(f"Watershed Mean Rainfall: {event_str}", fontsize=14, fontweight='bold')
    plt.xlabel("Lead Time (minutes)", fontsize=12)
    plt.ylabel("Rainfall Intensity (mm/h)", fontsize=12)
    plt.legend(loc='upper left', fontsize=9, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{event_str}_basin_rainfall.png"))
    plt.close()

# ==========================================
def process_event_worker(event_id, thresholds):
    event_str = f"event_{event_id:03d}"
    
    gts = []
    gt_rain_series = []
    for i in range(10, 28):
        path = os.path.join(BASE_PATH, "GroundTruth", event_str, f"gt{i}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        val = img.astype(np.float32) / 10.0
        gts.append(val)
        gt_rain_series.append(calculate_masked_mean(val, BASIN_MASK, MASK_SUM))
    
    event_metrics = {thr: {} for thr in thresholds}
    event_rain_data = {"Observations": gt_rain_series}
    
    for model_name, folder in MODEL_CONFIG.items():
        model_rain_series = []
        model_csi_series = {thr: [] for thr in thresholds}
        
        for i in range(10, 28):
            p_path = os.path.join(BASE_PATH, folder, event_str, f"pd{i}.png")
            p_img = cv2.imread(p_path, cv2.IMREAD_UNCHANGED)
            p_val = p_img.astype(np.float32) / 10.0 if p_img is not None else np.full_like(gts[0], np.nan)
            
            model_rain_series.append(calculate_masked_mean(p_val, BASIN_MASK, MASK_SUM))
            
            for thr in thresholds:
                model_csi_series[thr].append(neighborhood_metrics(p_val, gts[i-10], thr, 32))
        
        event_rain_data[model_name] = model_rain_series
        for thr in thresholds:
            event_metrics[thr][model_name] = {"CSI": np.array(model_csi_series[thr])}

    plot_event_rainfall(event_str, event_rain_data, os.path.join(OUT_ROOT, "plots/events"))
    
    return event_str, event_metrics

# ==========================================

def main():
    thresholds = [4, 16, 32]
    event_range = [17, 39] # 示例事件
    
    # 建立目录
    for sub_dir in ["plots/events", "summary"]:
        os.makedirs(os.path.join(OUT_ROOT, sub_dir), exist_ok=True)

    worker = partial(process_event_worker, thresholds=thresholds)
    
    with Pool(cpu_count()) as p:
        results = [r for r in p.map(worker, event_range) if r is not None]

if __name__ == "__main__":
    main()