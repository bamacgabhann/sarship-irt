import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def generate_advanced_tgrs_figures(cirt_csv, chars_feather):
    df_params = pd.read_csv(cirt_csv)
    df_chars = pd.read_feather(chars_feather)

    df = robust_spatial_merge_params(df_params, df_chars)
    df_test = df[df['split'].str.lower() == 'test'].copy()

    if 'location_type' in df_test.columns:
        df_test = pd.get_dummies(df_test, columns=['location_type'], drop_first=False)
        loc_cols = [c for c in df_test.columns if c.startswith('location_type_')]
    else:
        loc_cols = []

    # --- DOMAIN-DRIVEN FEATURE PRUNING ---
    # We strictly eliminate collinearity to preserve mathematical purity
    pruned_features = [
                          'rotated_bbox_aspect',  # Pure Geometric Elongation
                          'rotated_bbox_theta', # Angle
                          'bbox_area',  # Pure Physical Capacity
                          'bbox_bg_variance',  # Corner Regression Instability
                          'seg_SCR',  # Pure Target Contrast
#                          'seg_variance',  # Internal Target Complexity
                          'image_total_pixels',  # Image size / rescaling
                          'nearest_neighbor_px',  # Spatial NMS Interference
                          'dist_to_edge_px',  # Boundary Padding Constraints
                      ] + loc_cols

    actual_features = [f for f in pruned_features if f in df_test.columns]
    df_test = df_test.dropna(subset=actual_features + ['CIRT_Raw_Difficulty_b'])

    X = df_test[actual_features]
    y_diff = df_test['CIRT_Raw_Difficulty_b']

    # Fit the definitive Random Forest
    rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
    rf.fit(X, y_diff)

    # -------------------------------------------------------------------------
    # FIGURE A: 2D Partial Dependence Contour Plot (Coupled Interaction)
    # -------------------------------------------------------------------------
    print("\nGenerating 2D Partial Dependence Contour Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # We map the interaction between Geometric Elongation and Sea Clutter
    # The tuple in the features list tells sklearn to compute a 2D grid
    display = PartialDependenceDisplay.from_estimator(
        rf, X,
#        features=[('rotated_bbox_aspect', 'bbox_bg_variance')],
        features=[('rotated_bbox_aspect', 'rotated_bbox_theta')],
        kind='average',
        grid_resolution=50,  # Sufficient for smooth contours without taking hours
        ax=ax,
        contour_kw={'cmap': 'magma', 'alpha': 0.8}
    )

    #ax.set_title("Coupled CNN Failure Modes:\nAspect Ratio vs. Corner Background Variance", pad=20, fontsize=15)
    ax.set_title("Coupled CNN Failure Modes:\nAspect Ratio vs. Angle", pad=20, fontsize=15)
    ax.set_xlabel("Ship Aspect Ratio (Length / Width)", fontsize=13)
#    ax.set_ylabel("Bounding Box Corner Variance (Clutter)", fontsize=13)
    ax.set_ylabel("Ship Angle", fontsize=13)

    plt.tight_layout()
#    plt.savefig("fig_2d_pdp_coupled_failure.pdf", format='pdf', dpi=300)
    plt.savefig("fig_2d_pdp_coupled_failure_aspect_angle.pdf", format='pdf', dpi=300)
    print("Saved -> fig_2d_pdp_coupled_failure.pdf")

    # -------------------------------------------------------------------------
    # FIGURE B: Bivariate Density Map (The Architectural Prior Mismatch)
    # -------------------------------------------------------------------------
    print("\nGenerating Spatial Prior Bivariate Density Map...")

    # We use seaborn's JointGrid with a KDE (Kernel Density Estimate)
    # This highlights where the bulk of SAR targets exist geometrically
    plt.figure(figsize=(10, 8))

    # For visualization clarity, we clip extreme outlier areas to focus on the core density
    area_95th = df_test['rotated_bbox_area'].quantile(0.95)
    df_plot = df_test[df_test['rotated_bbox_area'] < area_95th]

    g = sns.jointplot(
        data=df_plot,
        x='rotated_bbox_area',
        y='rotated_bbox_aspect',
        kind="kde",
        cmap="mako_r",
        fill=True,
        thresh=0,
        levels=100,
        height=8,
        space=0
    )

    # Add a dashed line at aspect ratio = 1.0 (perfect square, typical COCO prior)
    g.ax_joint.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optical Square Prior (1:1)')

    g.ax_joint.set_xlabel("Rotated BBox Area (Pixels$^2$)", fontsize=13)
    g.ax_joint.set_ylabel("Rotated Aspect Ratio", fontsize=13)
    g.ax_joint.legend(loc='upper right')

    plt.suptitle("Dataset Cartography: Geometric Density of SAR Targets", y=1.02, fontsize=15)

    plt.savefig("fig_bivariate_density_map.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print("Saved -> fig_bivariate_density_map.pdf")


if __name__ == "__main__":
    generate_advanced_tgrs_figures("cirt_ship_parameters.csv", "sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.feather")