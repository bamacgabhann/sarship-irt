import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings

warnings.filterwarnings('ignore')

TINY_MODELS = ['yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo12n', 'yolo26n']
SMALL_MODELS = ['yolov8s', 'yolov9s', 'yolov10s', 'yolo11s', 'yolo12s', 'yolo26s']
MEDIUM_MODELS = ['yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m', 'yolo26m']
LARGE_MODELS = ['yolov8l', 'yolov9c', 'yolov10b', 'yolov10l', 'yolo11l', 'yolo12l', 'yolo26l']
XLARGE_MODELS = ['yolov8x', 'yolov9e', 'yolov10x', 'yolo11x', 'yolo12x', 'yolo26x']

MODELS_BY_SIZE = {
    'tiny': TINY_MODELS,
    'small': SMALL_MODELS,
    'medium': MEDIUM_MODELS,
    'large': LARGE_MODELS,
    'xlarge': XLARGE_MODELS,
}

YOLOv8_MODELS = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
YOLOv9_MODELS = ['yolov9t', 'yolov9s', 'yolov9m', 'yolov9c', 'yolov9e']
YOLOv10_MODELS = ['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x']
YOLO11_MODELS = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
YOLO12_MODELS = ['yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x']
YOLO26_MODELS = ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x']

MODELS_BY_GEN = {
    'YOLOv8': YOLOv8_MODELS,
    'YOLOv9': YOLOv9_MODELS,
    'YOLOv10': YOLOv10_MODELS,
    'YOLO11': YOLO11_MODELS,
    'YOLO12': YOLO12_MODELS,
    'YOLO26': YOLO26_MODELS,
}


def standardize_bbox_key(bbox):
    """
    Robustly parses varying bounding box formats (numpy arrays, space-separated strings,
    comma-separated strings) into a mathematically uniform, hashable string key.
    """
    try:
        # 1. Extract the raw floats depending on input type
        if isinstance(bbox, (np.ndarray, list)):
            coords = [float(x) for x in bbox]
        elif isinstance(bbox, str):
            # Strip brackets and replace commas with spaces just in case
            clean_str = bbox.replace('[', '').replace(']', '').replace(',', ' ')
            # Split on any whitespace and convert to float
            coords = [float(x) for x in clean_str.split()]
        else:
            return str(bbox)  # Fallback

        # 2. Format to 3 decimal places to eliminate floating-point string artifacts
        return "_".join([f"{c:.3f}" for c in coords])
    except Exception:
        # Safe fallback if a completely malformed row exists
        return str(bbox)


def plot_dynamics(smoothed_df, targets, target_name, target_feature):
    plt.figure(figsize=(14, 8))

    # Plot a subset of distinct architectures to keep the graph legible
    # Prioritize plotting the 'm' (medium) variant of different generations
    target_archs = [col for col in smoothed_df.columns if col in targets]

    # If standard 'm' models aren't found, just plot up to 8 models
    if not target_archs:
        target_archs = smoothed_df.columns[:8]

    for arch in target_archs:
        plt.plot(smoothed_df.index, smoothed_df[arch], label=arch, linewidth=2, alpha=0.8)

    # Add a horizontal line at 0 (Perfect Invariance)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Perfect Invariance (r=0)')

    plt.title(f'CNN Learning Dynamics: Sensitivity to {target_feature.replace("_", " ").title()} over 500 Epochs',
              fontsize=16)
    plt.xlabel('Training Epoch', fontsize=14)

    # Dynamic Y-label based on expected correlation direction
    # Most physical traits (variance, aspect ratio) have negative correlations (higher trait = lower confidence)
    plt.ylabel('Spearman Correlation (Sensitivity)', fontsize=14)

    plt.legend(loc='lower right', fontsize=11, ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Invert Y-axis if the correlations are predominantly negative,
    # so that "improving invariance" points upwards.
#    if smoothed_df.mean().mean() < 0:
#        plt.gca().invert_yaxis()
#        plt.ylabel('Spearman Correlation (Inverted: Upwards = Approaching 0)', fontsize=14)

    plt.tight_layout()
    plot_filename = f"learning_dynamics_{target_feature}_{target_name}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved trajectory plot to '{plot_filename}'.")

def analyze_longitudinal_dynamics(confidence_file, characteristics_csv):
    """
    Tracks how different YOLO architectures learn physical invariances over 500 epochs.
    """
    print(f"Loading vast confidence dataset from {confidence_file}...")

    if confidence_file.endswith('.feather'):
        df_conf = pd.read_feather(confidence_file)
    else:
        df_conf = pd.read_csv(confidence_file)

    df_chars = pd.read_csv(characteristics_csv)

    # ==========================================
    # FIX: Robust Mathematical BBox Standardization
    # ==========================================
    print("Standardizing bounding box formats to create safe join keys...")

    if 'gt_bbox' in df_conf.columns:
        df_conf['gt_bbox'] = df_conf['gt_bbox'].apply(standardize_bbox_key)

    if 'gt_bbox' in df_chars.columns:
        df_chars['gt_bbox'] = df_chars['gt_bbox'].apply(standardize_bbox_key)

    # Clean image paths to ensure perfect matching
    if 'image_path' in df_conf.columns and 'image_path' in df_chars.columns:
        df_conf['image_path'] = df_conf['image_path'].astype(str).str.strip()
        df_chars['image_path'] = df_chars['image_path'].astype(str).str.strip()

    print("Executing inner merge on image_path and gt_bbox...")
    df = pd.merge(df_conf, df_chars, on=['image_path', 'gt_bbox', 'dataset'], how='inner')

    print(f"Successfully merged {len(df)} bounding boxes.")

    # Check if the merge catastrophically failed
    if len(df) == 0:
        print("\nERROR: Merge resulted in 0 rows. Please inspect the raw formats:")
        print("Sample from Conf:", df_conf['gt_bbox'].iloc[0] if not df_conf.empty else "Empty")
        print("Sample from Char:", df_chars['gt_bbox'].iloc[0] if not df_chars.empty else "Empty")
        return

    features = [
        'Length_px', 'Width_px', 'Area_px2', 'Aspect_Ratio',
        'Dist_to_Image_Edge_px', 'Nearest_Neighbor_Dist_px',
        'Target_Mean_Intensity', 'Target_Variance',
        'Background_Mean_Intensity', 'Background_Variance', 'Local_SCR_dB'
    ]

    for target_feature in features:

        if target_feature not in df.columns:
            print(f"Error: Feature '{target_feature}' not found in characteristics.")
            print(f"Available columns: {df_chars.columns.tolist()}")
            return

        # Continue with longitudinal correlation...
        df = df.dropna(subset=[target_feature])

        metadata = ['image_path', 'gt_bbox', 'dataset', 'BBox_Normalized_xywh']
        chars_cols = df_chars.columns.tolist()

        # Isolate the model-epoch columns
        model_cols = [col for col in df.columns if col not in metadata and col not in chars_cols]
        print(f"Isolated {len(model_cols)} model-epoch columns.")

        # ... (Rest of the script remains exactly as previously provided)

        # Fast Spearman Correlation: Calculate ranks once for the target feature
        print(f"Computing longitudinal Spearman correlations for: {target_feature}...")
        feature_ranks = df[target_feature].rank()

        # Calculate ranks for all 15,500 model columns simultaneously
        model_ranks = df[model_cols].rank()

        # Compute Pearson correlation on the ranks (which equals Spearman correlation)
        # df.corrwith() computes column-wise correlation efficiently
        correlations = model_ranks.corrwith(feature_ranks)

        # Build a DataFrame to organize the results by Architecture and Epoch
        records = []

        # Regex to extract Architecture (e.g., YOLOv8m) and Epoch (e.g., 250)
        # Assumes naming convention like 'YOLOv8m_SSDD_epoch250'
        regex_pattern = r'(yolo[a-zA-Z0-9]+)_.*[eE]poch_?(\d+)'

        for col_name, corr_val in correlations.items():
            match = re.search(regex_pattern, col_name, re.IGNORECASE)
            if match:
                arch = match.group(1).lower()
                epoch = int(match.group(2))
                records.append({
                    'Architecture': arch,
                    'Epoch': epoch,
                    'Correlation': corr_val
                })

        df_traj = pd.DataFrame(records)

        if df_traj.empty:
            print(
                "Error: Could not parse Architecture and Epoch from column headers. Please check your regex/naming convention.")
            return

        # Sort for clean plotting
        df_traj = df_traj.sort_values(by=['Architecture', 'Epoch'])

        # Pivot the data: Rows = Epochs (0-499), Columns = Architectures
        pivot_df = df_traj.pivot(index='Epoch', columns='Architecture', values='Correlation')

        # Save the raw trajectory data
        output_filename = f"longitudinal_trajectory_{target_feature}.csv"
        pivot_df.to_csv(output_filename)
        print(f"Saved longitudinal data to '{output_filename}'.")

        # ==========================================
        # Visualization: The Learning Dynamics Curve
        # ==========================================
        print("Generating Learning Dynamics plot...")

        # We apply a slight rolling average (e.g., window=10) to smooth epoch-to-epoch noise
        # and reveal the macro learning trend.
        smoothed_df = pivot_df.rolling(window=10, min_periods=1).mean()

        for k, v in MODELS_BY_SIZE.items():
            plot_dynamics(smoothed_df, v, k, target_feature)

        for k, v in MODELS_BY_GEN.items():
            plot_dynamics(smoothed_df, v, k, target_feature)
#    return pivot_df


if __name__ == "__main__":
    # Example usage: Track how models learn to handle sea clutter
    # Replace 'variance_bg' with 'aspect_ratio', 'area', etc.

    analyze_longitudinal_dynamics(
    "ship_detection_ground_truths_ssdd.feather",
    "sarship_dataset_chars_SSDD.csv"
)




