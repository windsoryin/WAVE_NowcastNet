import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import cv2
from typing import List, Tuple, Dict
from PIL import Image

def binary_threshold(arr: np.ndarray, thr: float) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim > 2:
        if a.shape[-1] == 1:
            a = np.squeeze(a, -1)
        else:
            a = a.max(axis=-1)
    return (a >= thr).astype(np.uint8)


def neighborhood_metrics(pred: np.ndarray, gt: np.ndarray, thr: float, neigh_size: int) -> Tuple[int,int,int]:
    # Explicit sliding-window check: for each pixel, consider a neigh_size x neigh_size
    # window centered at that pixel; if any matching value exists in window -> match.
    # Implemented via convolution sum: sum>0 <=> exists a positive in the window.
    P = binary_threshold(pred, thr).astype(np.uint8)
    G = binary_threshold(gt, thr).astype(np.uint8)
    if neigh_size <= 1:
        pred_has = P
        gt_has = G
    else:
        kernel = np.ones((neigh_size, neigh_size), dtype=np.uint8)
        # sum of P in window centered at each pixel
        pred_sum = cv2.filter2D(P, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
        gt_sum = cv2.filter2D(G, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
        pred_has = (pred_sum > 0).astype(np.uint8)
        gt_has = (gt_sum > 0).astype(np.uint8)
    # hits: GT pixel is 1 AND there exists a pred in its window
    hits = int(np.logical_and(G == 1, pred_has == 1).sum())
    # misses: GT pixel is 1 AND there is no pred in its window
    misses = int(np.logical_and(G == 1, pred_has == 0).sum())
    # false alarms: pred pixel is 1 AND there is no GT in its window
    false_alarms = int(np.logical_and(P == 1, gt_has == 0).sum())
    return hits, false_alarms, misses


def compute_csi_pod_far(tp: int, fp: int, fn: int) -> Tuple[float,float,float]:
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    pod = tp / (tp + fn + 1e-4) 
    far = fp / (tp + fp+ 1e-4) 
    return csi, pod, far


def fss(pred: np.ndarray, gt: np.ndarray, thr: float, neigh_size: int) -> float:
    P = binary_threshold(pred, thr).astype(np.float32)
    G = binary_threshold(gt, thr).astype(np.float32)
    eps = 1e-8
    if neigh_size <= 1:
        p_frac = P
        g_frac = G
    else:
        kernel = np.ones((neigh_size, neigh_size), dtype=np.float32)
        p_sum = cv2.filter2D(P, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        g_sum = cv2.filter2D(G, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        area = float(neigh_size * neigh_size)
        p_frac = p_sum / area
        g_frac = g_sum / area
    mse = float(np.mean((p_frac - g_frac) ** 2))
    mse_ref = float(np.mean(p_frac ** 2 + g_frac ** 2))
    return 1 - mse / (mse_ref + eps) if (mse_ref + eps) > 0 else np.nan


def evaluate_sequence(preds: List[np.ndarray], gts: List[np.ndarray], thr: float, neigh_size: int, model_name: str = None, diag_dir: str = None):
    import time
    csis, pods, fars, fsses = [], [], [], []
    tp_list, fp_list, fn_list = [], [], []
    pred_pos_list, gt_pos_list = [], []
    total = len(preds)
    t0 = time.time()
    times = []
    for i, (p, g) in enumerate(zip(preds, gts), start=1):
        t1 = time.time()
        tp, fp, fn = neighborhood_metrics(p, g, thr, neigh_size)
        Pmask = binary_threshold(p, thr).astype(np.uint8)
        Gmask = binary_threshold(g, thr).astype(np.uint8)
        pred_pos = int(np.sum(Pmask))
        gt_pos = int(np.sum(Gmask))
        pred_pos_list.append(pred_pos)
        gt_pos_list.append(gt_pos)
        csi, pod, far = compute_csi_pod_far(tp, fp, fn)
        fs = fss(p, g, thr, neigh_size)
        t2 = time.time()
        times.append(t2 - t1)
        csis.append(csi)
        pods.append(pod)
        fars.append(far)
        fsses.append(fs)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        # Diagnostics: if FAR unexpectedly high, save masks and print stats
        try:
            if far is not None and far > 0.9:
                print(f'    DIAG far>0.9 at frame {i}/{total} for model {model_name} (TP={tp}, FP={fp}, FN={fn}, pred_pos={pred_pos}, gt_pos={gt_pos})')
                if diag_dir is not None and model_name is not None:
                    dd = os.path.join(diag_dir, model_name)
                    os.makedirs(dd, exist_ok=True)
                    Pmask = binary_threshold(p, thr).astype(np.uint8)
                    Gmask = binary_threshold(g, thr).astype(np.uint8)
                    if neigh_size > 1:
                        kernel = np.ones((neigh_size, neigh_size), dtype=np.uint8)
                        pred_sum = cv2.filter2D(Pmask, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
                        gt_sum = cv2.filter2D(Gmask, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
                        pred_has = (pred_sum > 0).astype(np.uint8)
                        gt_has = (gt_sum > 0).astype(np.uint8)
                    else:
                        pred_has = Pmask
                        gt_has = Gmask
                    try:
                        cv2.imwrite(os.path.join(dd, f'frame{i:03d}_Pmask_thr{int(thr)}.png'), (Pmask * 255).astype(np.uint8))
                        cv2.imwrite(os.path.join(dd, f'frame{i:03d}_Gmask_thr{int(thr)}.png'), (Gmask * 255).astype(np.uint8))
                        cv2.imwrite(os.path.join(dd, f'frame{i:03d}_pred_has_thr{int(thr)}.png'), (pred_has * 255).astype(np.uint8))
                        cv2.imwrite(os.path.join(dd, f'frame{i:03d}_gt_has_thr{int(thr)}.png'), (gt_has * 255).astype(np.uint8))
                        print(f'      saved diagnostic masks to {dd}')
                    except Exception as e:
                        print('      failed to save diag images', e)
                if fp > pred_pos:
                    print(f'      WARNING: FP({fp}) > pred_pos({pred_pos}) at frame {i} for model {model_name}')
                    try:
                        with open(os.path.join(dd, f'frame{i:03d}_warning.txt'), 'w') as wf:
                            wf.write(f'TP={tp}\nFP={fp}\nFN={fn}\npred_pos={pred_pos}\ngt_pos={gt_pos}\n')
                    except Exception:
                        pass
        except Exception:
            pass
        if (i % 10 == 0) or (i == total):
            avg = sum(times) / len(times)
            remaining = total - i
            eta = remaining * avg
            print(f'    Frame {i}/{total} - avg {avg:.3f}s/frame - ETA {eta:.1f}s')
    return np.array(csis), np.array(pods), np.array(fars), np.array(fsses), np.array(tp_list), np.array(fp_list), np.array(fn_list), np.array(pred_pos_list), np.array(gt_pos_list)


def plot_combined_metrics(all_metrics, out_dir, thr, interval_min=10):
    os.makedirs(out_dir, exist_ok=True)
    model_names = list(all_metrics.keys())
    
    cmap = plt.get_cmap('tab10')
    colors = {name: cmap(i) for i, name in enumerate(model_names)}
    metrics_to_plot = ['CSI', 'FSS']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    lines = []
    labels = []

    for idx, m_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        for model_name in model_names:
            vals = all_metrics[model_name][m_name]
            lead_times = (np.arange(1, 1 + len(vals))) * interval_min
            
            line, = ax.plot(lead_times, vals, label=model_name, 
                            color=colors[model_name], linewidth=2.5, 
                            marker='o', markersize=4, markevery=2)
            
            if idx == 0:
                lines.append(line)
                labels.append(model_name)

        ax.set_title(f'{m_name} Score (Thr ≥ {thr} mm/h)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Lead Time (minutes)', fontsize=12) 
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10)) 

    fig.legend(lines, labels, loc='lower center', ncol=len(model_names), 
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    output_path = os.path.join(out_dir, f'combined_metrics_time_thr{thr}.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f" combined metrics: {output_path}")
    plt.close()



def load_png_as_npy(file_path):
    """读取之前处理好的 16位 PNG 并恢复为物理值"""
    if not os.path.exists(file_path):
        return None
    img = Image.open(file_path)
    data = np.array(img).astype(np.float32) / 10.0
    return data


def plot_single_model_multi_thresholds(model_name, metrics_dict, thresholds, out_dir, interval_min=10):
    """
    metrics_dict 结构: { thr: { 'CSI': [...], 'FSS': [...] } }
    """
    os.makedirs(out_dir, exist_ok=True)
    metric_names = ['CSI', 'FSS']
    
    cmap = plt.get_cmap('tab10')
    colors = {thr: cmap(i) for i, thr in enumerate(thresholds)}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    lines = []
    labels = []

    for idx, m_name in enumerate(metric_names):
        ax = axes[idx]
        for thr in thresholds:
            vals = metrics_dict[thr][m_name]
            lead_times = (np.arange(10, 10 + len(vals))) * interval_min
            
            line, = ax.plot(lead_times, vals, 
                            label=f'Threshold {thr} mm/h', 
                            color=colors[thr], 
                            linewidth=2.5, 
                            marker='o', markersize=4, markevery=2)
            
            if idx == 0:
                lines.append(line)
                labels.append(f'Thr ≥ {thr} mm/h')

        ax.set_title(f'{m_name} Scores', fontsize=14, fontweight='bold')
        ax.set_xlabel('Lead Time (minutes)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

    fig.legend(lines, labels, loc='lower center', ncol=len(thresholds), 
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    output_path = os.path.join(out_dir, f'{model_name}_multi_threshold_comparison.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"single metric saved: {output_path}")
    plt.close()


# ==========================================

def run_evaluation_v2(event_id, base_path, thresholds=[4, 16, 32], neigh=32):
    # 模型映射
    model_dirs = {
        "WAVE-NowcastNet": "WAVE_NowcastNet",
        "NowcastNet-Orig": "NowcastNet_Orig",
        "ConvLSTM": "convlstm",
        "OpticalFlow": "opticalflow",
        "PredRNN": "predrnn"
    }
    gt_dir = os.path.join(base_path, "GroundTruth", f"event_{event_id:03d}")
    
   
    all_results = {thr: {} for thr in thresholds}
    wave_multi_thr = {thr: {} for thr in thresholds}

    for thr in thresholds:
        print(f"--- Processing Threshold: {thr} ---")
        for name, folder in model_dirs.items():
            pred_path = os.path.join(base_path, folder, f"event_{event_id:03d}")
            
            preds = []
            gts = []
            for i in range(10, 30):
                p = load_png_as_npy(os.path.join(pred_path, f"pd{i}.png"))
                g = load_png_as_npy(os.path.join(gt_dir, f"gt{i}.png"))
                if p is not None and g is not None:
                    preds.append(p)
                    gts.append(g)
            
            if not preds: continue
            
            csis, _, _, fsses, *_ = evaluate_sequence(preds, gts, thr, neigh)
            
            all_results[thr][name] = {'CSI': csis, 'FSS': fsses}
            if name == "WAVE-NowcastNet":
                wave_multi_thr[thr] = {'CSI': csis, 'FSS': fsses}

    # --- 开始绘图 ---
    out_root = f"results_event_{event_id:03d}"
    
    plot_single_model_multi_thresholds("WAVE-NowcastNet", wave_multi_thr, thresholds, out_root)

    for thr in thresholds:
        thr_dir = os.path.join(out_root, f"comparison_thr_{thr}")
        plot_combined_metrics(all_results[thr], out_root, thr=thr)


if __name__ == "__main__":
    base_data_path = "./plot_data"
    for event_to_process in [17,39]:
        run_evaluation_v2(event_to_process, base_data_path, thresholds=[ 16, 32], neigh=5)