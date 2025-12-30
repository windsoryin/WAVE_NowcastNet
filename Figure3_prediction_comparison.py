# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from PIL import Image

# pd10=T+0, pd16=T+1h, pd22=T+2h, pd28=T+3h
SHOW_INDICES = [10, 16, 22, 28] 
SHOW_LABELS = ["Frame 1(+10min)", "Frame 6(+1h)", "Frame 12(+2h)", "Frame 18(+3h)"]
EXTENT = [-93.5, -83.25, 32.0, 42.23]
BASE_DIR = "./plot_data"
SAVE_DIR = "./comparison_plots"

MODEL_CONFIG = {
    "Observations":    "GroundTruth",
    "WAVE-NowcastNet": "WAVE_NowcastNet",
    "NowcastNet": "NowcastNet_Orig",
    "Optical Flow":    "opticalflow",
    "ConvLSTM":        "convlstm",
    "PredRNN":         "predrnn",

}

# ==========================================

def load_processed_image(model_key, event, frame_idx):
    folder_name = MODEL_CONFIG[model_key]
    prefix = "gt" if model_key == "Observations" else "pd"
    filename = f"{prefix}{frame_idx}.png"
    
    path = os.path.join(BASE_DIR, folder_name, event, filename)
    
    if not os.path.exists(path):
        return np.full((1024, 1024), np.nan)
    
    img = Image.open(path)
    data = np.array(img).astype(np.float32) / 10.0
    if model_key == "PredRNN":
        data[data <= 1] = np.nan 
    else:
        data[data <= 0.1] = np.nan
    return data

# ==========================================

def plot_single_event(event_name):
    nrows = len(MODEL_CONFIG)
    ncols = len(SHOW_INDICES)
    projection = ccrs.PlateCarree()
    
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(3.5 * ncols, 3.2 * nrows), 
                             subplot_kw={'projection': projection},
                             constrained_layout=True)
    
    model_keys = list(MODEL_CONFIG.keys())
    cmap = plt.get_cmap('GnBu').copy()
    cmap.set_bad(alpha=0) 

    last_im = None

    for row, model_key in enumerate(model_keys):
        for col, frame_idx in enumerate(SHOW_INDICES):
            ax = axes[row, col]
            ax.set_extent(EXTENT, crs=projection)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
            
            # 加载当前 event 的数据
            data = load_processed_image(model_key, event_name, frame_idx)
            
            last_im = ax.imshow(data, 
                                origin='upper', 
                                extent=EXTENT, 
                                transform=projection, 
                                cmap=cmap, 
                                vmin=0, vmax=32, # 降雨强度颜色范围
                                zorder=5)
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (col == 0)
            gl.bottom_labels = (row == nrows - 1)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

            if row == 0:
                ax.set_title(SHOW_LABELS[col], fontsize=12, fontweight='bold')
            if col == 0:
                ax.text(-0.3, 0.5, model_key, 
                        transform=ax.transAxes, rotation=90, 
                        va='center', ha='right', fontsize=12, fontweight='bold')

    if last_im:
        cbar = fig.colorbar(last_im, ax=axes, orientation='vertical', 
                            fraction=0.015, pad=0.02, aspect=40)
        cbar.set_label('Precipitation Intensity (mm/h)', fontsize=10)

    output_path = os.path.join(SAVE_DIR, f"comparison_{event_name}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")




# ==========================================

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for i in [17,39]:#range(22, 23):
        current_event = f"event_{i:03d}"
        
        check_path = os.path.join(BASE_DIR, "GroundTruth", current_event)
        if not os.path.exists(check_path):
            print(f"Skipping {current_event}: Directory not found.")
            continue
        try:
            print(f"Processing {current_event}...")
            plot_single_event(current_event)
        except Exception as e:
            print(f"Error processing {current_event}: {e}")
# ==========================================