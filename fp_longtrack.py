import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings

try:
    from sar_irt.utils.core import set_ieee_tgrs_style
except ImportError:
    def set_ieee_tgrs_style():
        plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
                             "legend.fontsize": 11})

warnings.filterwarnings('ignore')


def parse_bbox_array(bbox_val):
    if isinstance(bbox_val, (np.ndarray, list)): return np.array(bbox_val, dtype=float)
    if isinstance(bbox_val, str): return np.array(
        [float(x) for x in bbox_val.replace('[', '').replace(']', '').replace(',', ' ').split()], dtype=float)
    return np.zeros(4)


def extract_fp_geometry(df_fp):
    """Calculates geometric traits dynamically from normalized bbox coordinates."""
    df_fp['bbox_arr'] = df_fp['fp_bbox'].apply(parse_bbox_array)

    # Calculate Width, Height, Area, and Aspect Ratio
    df_fp['w_norm'] = df_fp['bbox_arr'].apply(lambda x: max(1e-6, x[2] - x[0]))
    df_fp['h_norm'] = df_fp['bbox_arr'].apply(lambda x: max(1e-6, x[3] - x[1]))
    df_fp['Area_norm'] = df_fp['w_norm'] * df_fp['h_norm']
    df_fp['Aspect_Ratio'] = df_fp[['w_norm', 'h_norm']].max(axis=1) / df_fp[['w_norm', 'h_norm']].min(axis=1)

    return df_fp


def track_longitudinal_distractors(fp_dir, target_models, max_epoch=500, step=10):
    """
    Tracks how the correlation between confidence and FP geometry evolves over time.
    Samples every `step` epochs to optimize I/O.
    """
    print(f"Tracking longitudinal FP dynamics for: {target_models}")

    trajectory_data = []

    for arch in target_models:
        print(f"Processing {arch}...")
        for epoch in range(0, max_epoch, step):
            search_pattern = os.path.join(fp_dir, f"{arch}_*_epoch{epoch}_fp.feather")
            files = glob.glob(search_pattern)

            if not files:
                continue

            df = pd.read_feather(files[0])
            conf_col = [c for c in df.columns if c not in ['image_path', 'fp_bbox']][0]

            # If no FPs were predicted this epoch, skip to avoid math errors
            if len(df) < 5:
                continue

            df = extract_fp_geometry(df)

            # Calculate Spearman Correlation
            r_area, _ = stats.spearmanr(df['Area_norm'], df[conf_col])
            r_aspect, _ = stats.spearmanr(df['Aspect_Ratio'], df[conf_col])

            trajectory_data.append({
                'Architecture': arch,
                'Epoch': epoch,
                'Total_FP_Count': len(df),
                'Correlation_Area': r_area,
                'Correlation_Aspect_Ratio': r_aspect
            })

    return pd.DataFrame(trajectory_data)


def plot_fp_dynamics(df_traj):
    set_ieee_tgrs_style()

    architectures = df_traj['Architecture'].unique()

    # Plot 1: Total False Positive Count Over Time
    plt.figure(figsize=(12, 6))
    for arch in architectures:
        subset = df_traj[df_traj['Architecture'] == arch]
        plt.plot(subset['Epoch'], subset['Total_FP_Count'], label=arch, linewidth=2, alpha=0.8)

    plt.title('Evolution of Clutter Susceptibility (Total FP Count)', fontsize=16)
    plt.xlabel('Training Epoch', fontsize=14)
    plt.ylabel('Number of False Positives Generated', fontsize=14)
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('fp_evolution_count.pdf', format='pdf', dpi=300)

    # Plot 2: Correlation with Aspect Ratio Over Time
    plt.figure(figsize=(12, 6))
    for arch in architectures:
        subset = df_traj[df_traj['Architecture'] == arch]
        # Smooth the line for readability
        smoothed_corr = subset['Correlation_Aspect_Ratio'].rolling(window=3, min_periods=1).mean()
        plt.plot(subset['Epoch'], smoothed_corr, label=arch, linewidth=2, alpha=0.8)

    plt.title('Feature Refinement: Sensitivity to FP Aspect Ratio Over Time', fontsize=16)
    plt.xlabel('Training Epoch', fontsize=14)
    plt.ylabel('Spearman Correlation ($r$) with Aspect Ratio', fontsize=14)
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.legend(loc='lower right', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('fp_evolution_aspect_ratio.pdf', format='pdf', dpi=300)
    print("Saved longitudinal plots to 'fp_evolution_count.pdf' and 'fp_evolution_aspect_ratio.pdf'.")


if __name__ == "__main__":
    FP_DIR = "SSDD_fp_feather"  # Update with your path

    # Track the "Medium" sized models to compare architectural generations fairly
    TARGET_MODELS = ['yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m', 'yolo26m']

    df_trajectory = track_longitudinal_distractors(FP_DIR, TARGET_MODELS, max_epoch=500, step=5)
    df_trajectory.to_csv('fp_longitudinal_trajectory.csv', index=False)

    plot_fp_dynamics(df_trajectory)