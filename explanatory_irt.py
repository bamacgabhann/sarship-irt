import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# IEEE TGRS Formatting Guidelines
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11
})


def robust_spatial_merge_params(df_params, df_chars):
    """Spatially aligns the CSV CIRT parameters with the Feather characteristics."""
    print("Executing spatial merge between Parameters and Characteristics...")

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

    df_params['temp_cx_cy'] = df_params['gt_bbox'].apply(extract_centers)
    df_chars['temp_cx_cy'] = df_chars['gt_bbox'].apply(extract_centers)

    param_idx_list, char_idx_list = [], []

    for img in df_params['image_path'].unique():
        p_mask = df_params['image_path'] == img
        c_mask = df_chars['image_path'] == img

        p_idx = df_params.index[p_mask].tolist()
        c_idx = df_chars.index[c_mask].tolist()

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


def analyze_explanatory_irt(cirt_csv, chars_feather):
    print("Loading datasets...")
    df_params = pd.read_csv(cirt_csv)
    df_chars = pd.read_feather(chars_feather)

    df = robust_spatial_merge_params(df_params, df_chars)

    # METHODOLOGICAL SHIFT: Only analyze generalizable difficulty on unseen data
    df_test = df[df['split'].str.lower() == 'test'].copy()
    print(f"Isolated {len(df_test)} Test Split ships for pure physical evaluation.")

    # 1. Feature Engineering
    # One-hot encode the inshore/offshore categorical variable
    if 'location_type' in df_test.columns:
        df_test = pd.get_dummies(df_test, columns=['location_type'], drop_first=False)

    # features = [
    #     'image_w',
    #     'image_h',
    #     'image_total_pixels',
    #     'bbox_w', 'bbox_h',
    #     'bbox_area',
    #     'bbox_aspect',
    #     'nearest_neighbor_px',
    #     'dist_to_edge_px',
    #     'rotated_bbox_theta',
    #     'rotated_bbox_w',
    #     'rotated_bbox_h',
    #     'rotated_bbox_area',
    #     'rotated_bbox_aspect',
    #     'bbox_intensity',
    #     'bbox_variance',
    #     'bbox_bg_intensity',
    #     'bbox_bg_variance',
    #     'bbox_SCR',
    #     'seg_area',
    #     'seg_intensity',
    #     'seg_variance',
    #     'seg_bg_intensity',
    #     'seg_bg_variance',
    #     'seg_SCR'
    # ]
    features = [
        'bbox_bg_variance',
        'nearest_neighbor_px',
        'rotated_bbox_aspect',
        'bbox_area',
        'seg_SCR',
        'image_total_pixels',
        'dist_to_edge_px',
    ]
    # Add the newly created dummy variables
    loc_cols = [c for c in df_test.columns if c.startswith('location_type_')]
    features.extend(loc_cols)

    # Clean NaNs
    actual_features = [f for f in features if f in df_test.columns]
    df_test = df_test.dropna(subset=actual_features + ['CIRT_Raw_Difficulty_b'])

    X = df_test[actual_features]
    y_diff = df_test['CIRT_Raw_Difficulty_b']

    # 2. Random Forest Regression
    print("Fitting Random Forest...")
    rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
    rf.fit(X, y_diff)

    # Save Feature Importances
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Top Drivers of Generalizable CNN Failure ---")
    print(importances.head(10).to_string(index=False))
    importances.to_csv("eirm_ob_feature_importances_bbox_area.csv", index=False)

    # 3. Partial Dependence Plots (PDPs) for Top 4 Continuous Features
    top_continuous = [f for f in importances['Feature'] if f not in loc_cols and f != 'truncated']
    #[:4]

    print(f"\nGenerating PDPs for top constraints: {top_continuous}")
    fig, ax = plt.subplots(figsize=(14, 10))

    display = PartialDependenceDisplay.from_estimator(
        rf, X, top_continuous,
        kind="average",
        grid_resolution=100,
        ax=ax,
        line_kw={'color': '#1f77b4', 'linewidth': 3}
    )

    plt.suptitle('Physical Constraints on SAR Generalization (Test Split)', fontsize=16, y=0.95)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.savefig("pdp_obb_thresholds_bbox_area.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Saved IEEE-formatted PDP plot to 'pdp_obb_thresholds.pdf'.")


if __name__ == "__main__":
    analyze_explanatory_irt("cirt_ship_parameters.csv", "sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.feather")