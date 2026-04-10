import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import re
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11
})


def robust_spatial_merge_conf(df_conf, df_chars):
    """Spatial merge logic adapted for the Feather-to-Feather mapping."""
    print("Executing spatial merge between Confidence Data and Characteristics...")

    def extract_centers(bbox):
        try:
            if isinstance(bbox, (np.ndarray, list)):
                coords = [float(x) for x in bbox]
            elif isinstance(bbox, str):
                clean_str = bbox.replace('[', '').replace(']', '').replace(',', ' ')
                coords = [float(x) for x in clean_str.split()]
            else:
                return 0.0, 0.0
            if len(coords) >= 4:
                return (coords[0] + coords[2]) / 2.0, (coords[1] + coords[3]) / 2.0
            return 0.0, 0.0
        except:
            return 0.0, 0.0

    df_conf['temp_cx_cy'] = df_conf['gt_bbox'].apply(extract_centers)
    df_chars['temp_cx_cy'] = df_chars['gt_bbox'].apply(extract_centers)

    conf_idx_list, char_idx_list = [], []

    for img in df_conf['image_path'].unique():
        conf_mask = df_conf['image_path'] == img
        char_mask = df_chars['image_path'] == img

        conf_idx = df_conf.index[conf_mask].tolist()
        char_idx = df_chars.index[char_mask].tolist()

        if not conf_idx or not char_idx: continue

        conf_centers = np.array(df_conf.loc[conf_idx, 'temp_cx_cy'].tolist())
        char_centers = np.array(df_chars.loc[char_idx, 'temp_cx_cy'].tolist())

        dist_matrix = cdist(conf_centers, char_centers)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        for i, j in zip(row_ind, col_ind):
            if dist_matrix[i, j] < 0.2:
                conf_idx_list.append(conf_idx[i])
                char_idx_list.append(char_idx[j])

    df_conf_matched = df_conf.loc[conf_idx_list].reset_index(drop=True)
    df_char_matched = df_chars.loc[char_idx_list].reset_index(drop=True)

    cols_to_drop = [c for c in df_char_matched.columns if c in df_conf_matched.columns]
    df_merged = pd.concat([df_conf_matched, df_char_matched.drop(columns=cols_to_drop)], axis=1)
    df_merged.drop(columns=['temp_cx_cy'], inplace=True, errors='ignore')
    return df_merged

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

    plt.tight_layout()
    plot_filename = f"learning_dynamics_{target_feature}_{target_name}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved trajectory plot to '{plot_filename}'.")

def analyze_learning_dynamics(confidence_feather, chars_feather):
    print(f"Loading raw confidence data and characteristics...")
    df_conf = pd.read_feather(confidence_feather)
    df_chars = pd.read_feather(chars_feather)

    df = robust_spatial_merge_conf(df_conf, df_chars)

    # Isolate Test Split for generalizable dynamics
    df = df[df['split'].str.lower() == 'test'].copy()

    regex_pattern = re.compile(r'(yolo[a-zA-Z0-9]+)_.*[eE]poch_?(\d+)', re.IGNORECASE)
    model_cols = [col for col in df.columns if regex_pattern.search(col)]
    print(f"Tracking trajectories for {len(model_cols)} epochs across Test Split.")

#    target_features = ['seg_variance', 'rotated_bbox_aspect', 'seg_SCR', 'seg_area']
    target_features = [
        'image_w',
        'image_h',
        'image_total_pixels',
        'bbox_w', 'bbox_h',
        'bbox_area',
        'bbox_aspect',
        'nearest_neighbor_px',
        'dist_to_edge_px',
        'rotated_bbox_theta',
        'rotated_bbox_w',
        'rotated_bbox_h',
        'rotated_bbox_area',
        'rotated_bbox_aspect',
        'bbox_intensity',
        'bbox_variance',
        'bbox_bg_intensity',
        'bbox_bg_variance',
        'bbox_SCR',
        'seg_area',
        'seg_intensity',
        'seg_variance',
        'seg_bg_intensity',
        'seg_bg_variance',
        'seg_SCR'
]
    actual_features = [f for f in target_features if f in df.columns]
    df = df.dropna(subset=actual_features)

    # 1. Fully Converged Model Sensitivities (Epochs >= 490)
    converged_models = [col for col in model_cols if int(regex_pattern.search(col).group(2)) >= 490]

    print("Computing converged feature sensitivities (Epochs 490-499)...")
    sens_records = []

    for feature in actual_features:
        feature_ranks = df[feature].rank()
        model_ranks = df[converged_models].rank()
        correlations = model_ranks.corrwith(feature_ranks)

        for model_name, corr_val in correlations.items():
            match = regex_pattern.search(model_name)
            sens_records.append({
                'Model_Name': model_name,
                'Architecture': match.group(1).lower(),
                'Epoch': int(match.group(2)),
                'Feature': feature,
                'Sensitivity_r': corr_val
            })

    df_sens = pd.DataFrame(sens_records)
    # Pivot so features are columns
    df_sens_pivot = df_sens.pivot_table(index=['Model_Name', 'Architecture', 'Epoch'],
                                        columns='Feature', values='Sensitivity_r').reset_index()
    df_sens_pivot.to_csv("obb_model_feature_sensitivities.csv", index=False)
    print("Saved 'obb_model_feature_sensitivities.csv'.")

    # 2. Plotting Longitudinal Dynamics (e.g., seg_SCR)
    primary_feature = 'seg_SCR'
    if primary_feature in actual_features:
        print(f"Computing 500-epoch trajectory for {primary_feature}...")

        feature_ranks = df[primary_feature].rank()
        model_ranks = df[model_cols].rank()
        traj_correlations = model_ranks.corrwith(feature_ranks)

        traj_records = []
        for col_name, corr_val in traj_correlations.items():
            match = regex_pattern.search(col_name)
            traj_records.append({
                'Architecture': match.group(1).lower(),
                'Epoch': int(match.group(2)),
                'Correlation': corr_val
            })

        df_traj = pd.DataFrame(traj_records)
        df_traj = df_traj.sort_values(by=['Architecture', 'Epoch'])
        pivot_traj = df_traj.pivot(index='Epoch', columns='Architecture', values='Correlation')

        # Plot only medium 'm' models for readability
        smoothed_df = pivot_traj.rolling(window=15, min_periods=1).mean()
        target_archs = [col for col in smoothed_df.columns if col.endswith('m')]

        plt.figure(figsize=(12, 7))
        for arch in target_archs:
            plt.plot(smoothed_df.index, smoothed_df[arch], label=arch.upper(), linewidth=2, alpha=0.85)

        plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
        plt.title(f'CNN Learning Dynamics: Sensitivity to Segmentation SCR', fontsize=16)
        plt.xlabel('Training Epoch', fontsize=14)
        plt.ylabel('Spearman Correlation (Invariance = 0)', fontsize=14)

        if smoothed_df.mean().mean() < 0:
            plt.gca().invert_yaxis()
            plt.ylabel('Spearman Correlation (Inverted)', fontsize=14)

        plt.legend(loc='best', fontsize=11, ncol=2)
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"learning_dynamics_{primary_feature}.pdf", format='pdf', dpi=300)
        print(f"Saved IEEE trajectory plot to 'learning_dynamics_{primary_feature}.pdf'.")


if __name__ == "__main__":
    analyze_learning_dynamics("ship_detection_ground_truths_ssdd.feather",
                              "sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.feather")