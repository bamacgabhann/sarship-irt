import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11
})


def robust_spatial_merge_params(df_params, df_chars):
    print("Executing spatial merge...")

    def extract_centers(bbox):
        try:
            if isinstance(bbox, (np.ndarray, list)):
                coords = [float(x) for x in bbox]
            elif isinstance(bbox, str):
                clean_str = bbox.replace('[', '').replace(']', '').replace(',', ' ')
                coords = [float(x) for x in clean_str.split()]
            else:
                return 0.0, 0.0
            if len(coords) >= 4: return (coords[0] + coords[2]) / 2.0, (coords[1] + coords[3]) / 2.0
            return 0.0, 0.0
        except:
            return 0.0, 0.0

    df_params['temp_cx_cy'] = df_params['gt_bbox'].apply(extract_centers)
    df_chars['temp_cx_cy'] = df_chars['gt_bbox'].apply(extract_centers)

    param_idx_list, char_idx_list = [], []
    for img in df_params['image_path'].unique():
        p_mask = df_params['image_path'] == img
        c_mask = df_chars['image_path'] == img

        p_idx, c_idx = df_params.index[p_mask].tolist(), df_chars.index[c_mask].tolist()
        if not p_idx or not c_idx: continue

        p_centers = np.array(df_params.loc[p_idx, 'temp_cx_cy'].tolist())
        c_centers = np.array(df_chars.loc[c_idx, 'temp_cx_cy'].tolist())

        dist_matrix = cdist(p_centers, c_centers)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        for i, j in zip(row_ind, col_ind):
            if dist_matrix[i, j] < 0.2:
                param_idx_list.append(p_idx[i])
                char_idx_list.append(c_idx[j])

    df_p_matched = df_params.loc[param_idx_list].reset_index(drop=True)
    df_c_matched = df_chars.loc[char_idx_list].reset_index(drop=True)
    cols_to_drop = [c for c in df_c_matched.columns if c in df_p_matched.columns]
    df_merged = pd.concat([df_p_matched, df_c_matched.drop(columns=cols_to_drop)], axis=1)
    df_merged.drop(columns=['temp_cx_cy'], inplace=True, errors='ignore')
    return df_merged


def generate_ice_diagnostics(cirt_csv, chars_feather):
    df_params = pd.read_csv(cirt_csv)
    df_chars = pd.read_feather(chars_feather)

    df = robust_spatial_merge_params(df_params, df_chars)
    df_test = df[df['split'].str.lower() == 'test'].copy()

    if 'location_type' in df_test.columns:
        df_test = pd.get_dummies(df_test, columns=['location_type'], drop_first=False)
        loc_cols = [c for c in df_test.columns if c.startswith('location_type_')]
    else:
        loc_cols = []

    # Pruned Features
    pruned_features = [
                          'rotated_bbox_aspect', 'rotated_bbox_area', 'bbox_bg_variance',
                          'seg_SCR', 'seg_variance', 'nearest_neighbor_px', 'dist_to_edge_px'
                      ] + loc_cols

    actual_features = [f for f in pruned_features if f in df_test.columns]
    df_test = df_test.dropna(subset=actual_features + ['CIRT_Raw_Difficulty_b'])

    X = df_test[actual_features]
    y_diff = df_test['CIRT_Raw_Difficulty_b']

    # Fit Random Forest
    rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
    rf.fit(X, y_diff)

    # Generate ICE Plots for the specific anomalies you noted
    target_plots = ['rotated_bbox_aspect', 'bbox_bg_variance', 'seg_SCR', 'rotated_bbox_area']

    print("\nGenerating Individual Conditional Expectation (ICE) Diagnostics...")
    fig, ax = plt.subplots(figsize=(16, 12))

    display = PartialDependenceDisplay.from_estimator(
        rf, X, target_plots,
        kind='both',  # Renders both ICE lines and the PDP average
        grid_resolution=50,
        ax=ax,
        ice_lines_kw={"color": "#b0c4de", "alpha": 0.05, "linewidth": 0.5},  # Faint blue lines for ships
        pd_line_kw={"color": "#d62728", "linewidth": 4}  # Bold red line for the average
    )

    plt.suptitle("Diagnostic ICE Plots: Distinguishing Systemic Physics from Outliers", fontsize=16, y=0.95)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.savefig("diagnostic_ice_plots.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Saved ICE diagnostic plots to 'diagnostic_ice_plots.pdf'.")

    # -------------------------------------------------------------------------
    # Programmatic Outlier Detection (Optional but helpful)
    # Check data density around the spikes
    print("\n--- Data Density Check ---")
    spike_var = len(X[(X['bbox_bg_variance'] > 700) & (X['bbox_bg_variance'] < 800)])
    spike_scr = len(X[(X['seg_SCR'] > 9.5) & (X['seg_SCR'] < 10.5)])

    print(f"Number of test ships with bbox_bg_variance between 700-800: {spike_var}")
    print(f"Number of test ships with seg_SCR between 9.5-10.5: {spike_scr}")


if __name__ == "__main__":
    generate_ice_diagnostics("cirt_ship_parameters.csv", "sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.feather")