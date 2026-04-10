import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

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


def generate_difficulty_pdps(cirt_csv, characteristics_csv):
    """
    Trains a Random Forest on Ship Difficulty and generates Partial Dependence Plots (PDP)
    to identify the exact physical thresholds where models fail.
    """
    print("\n--- Generating Partial Dependence Plots (PDP) ---")
    df_cirt = pd.read_csv(cirt_csv)
    df_chars = pd.read_csv(characteristics_csv)

    for df in [df_cirt, df_chars]:
        if 'gt_bbox' in df.columns:
            df['gt_bbox'] = df['gt_bbox'].apply(standardize_bbox_key)

    df = pd.merge(df_cirt, df_chars, on=['image_path', 'gt_bbox', 'dataset'], how='inner')

    features = [
        'Length_px', 'Width_px', 'Area_px2', 'Aspect_Ratio',
        'Dist_to_Image_Edge_px', 'Nearest_Neighbor_Dist_px',
        'Target_Mean_Intensity', 'Target_Variance',
        'Background_Mean_Intensity', 'Background_Variance', 'Local_SCR_dB'
    ]

    # We use 'variance_bg' and 'height' based on your previous top RF results,
    # adjust the strings in 'features' array above if your CSV column names differ.
    actual_features = [f for f in features if f in df.columns]

    df = df.dropna(subset=actual_features + ['CIRT_Raw_Difficulty_b'])
    X = df[actual_features]
    y_diff = df['CIRT_Raw_Difficulty_b']

    print("Fitting Random Forest for PDP generation...")
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y_diff)

    # Select the top 4 features from your previous run to plot
    # Change these strings to exactly match your characteristics CSV headers
#    top_features_to_plot = ['Background_Variance', 'Length_px', 'Target_Variance', 'Area_px2', 'Aspect_Ratio']
#    plot_features = [f for f in top_features_to_plot if f in X.columns]
    plot_features = [f for f in actual_features if f in X.columns]

    if plot_features:
        print(f"Plotting PDPs for: {plot_features}")
        fig, ax = plt.subplots(figsize=(14, 10))

        display = PartialDependenceDisplay.from_estimator(
            rf, X, plot_features,
            kind="average",  # 'average' plots the standard PDP line
            grid_resolution=50,
            ax=ax
        )

        plt.suptitle('Partial Dependence of Physical Traits on Ship Difficulty', fontsize=16)
        plt.subplots_adjust(top=0.92, hspace=0.3)
        plt.savefig("pdp_difficulty_thresholds.png", dpi=300, bbox_inches='tight')
        print("Saved PDP plot to 'pdp_difficulty_thresholds.png'.")
    else:
        print("Warning: Top features not found in columns. Check column naming.")


def analyze_model_sensitivities(ground_truth_feather, characteristics_csv):
    """
    Calculates the Spearman Rank Correlation between physical SAR features
    and the raw confidence outputs of every individual YOLO model.
    """
    print("\n--- Generating Individual Model Feature Sensitivities ---")
    df_gt = pd.read_feather(ground_truth_feather)
    df_chars = pd.read_csv(characteristics_csv)

    for df in [df_gt, df_chars]:
        if 'gt_bbox' in df.columns:
            df['gt_bbox'] = df['gt_bbox'].apply(standardize_bbox_key)

    # Merge confidence data with physical characteristics
    df = pd.merge(df_gt, df_chars, on=['image_path', 'gt_bbox', 'dataset'], how='inner')

    # Identify model columns vs metadata/feature columns
    metadata = ['image_path', 'gt_bbox', 'BBox_Normalized_xywh', 'dataset']
    chars_cols = df_chars.columns.drop(['image_path', 'gt_bbox', 'BBox_Normalized_xywh', 'dataset']).tolist()
    model_cols = [col for col in df.columns if col not in metadata and col not in chars_cols]

    # Select key physical features to test
    # (Adjust to your exact column names)
#    target_features = ['Aspect_Ratio', 'Area_px2', 'Background_Variance', 'Local_SCR_dB', 'Dist_to_Image_Edge_px']
    target_features = [
        'Length_px', 'Width_px', 'Area_px2', 'Aspect_Ratio',
        'Dist_to_Image_Edge_px', 'Nearest_Neighbor_Dist_px',
        'Target_Mean_Intensity', 'Target_Variance',
        'Background_Mean_Intensity', 'Background_Variance', 'Local_SCR_dB'
    ]
    actual_features = [f for f in target_features if f in df.columns]

    df = df.dropna(subset=actual_features)

    sensitivity_records = []

    print(f"Analyzing {len(model_cols)} models against {len(actual_features)} physical traits...")
    for model in model_cols:
        model_conf = df[model].fillna(0.0)

        record = {'Model_Name': model}

        for feat in actual_features:
            # We use Spearman rank correlation because confidence scores [0,1]
            # and physical traits are often non-linearly related.
            r, p = stats.spearmanr(df[feat], model_conf)
            record[feat] = r

        sensitivity_records.append(record)

    df_sens = pd.DataFrame(sensitivity_records)

    # Extract architecture family (e.g., 'v8', 'v10') and size ('n', 'm', 'x') for easy sorting
    # Assuming standard naming like 'yolov8m_SSDD_epoch120'
    try:
        df_sens['Architecture'] = df_sens['Model_Name'].str.extract(r'(yolo[v]?\d+)')
        df_sens['Size'] = df_sens['Model_Name'].str.extract(r'yolo[v]?\d+([nsmlxtbe])_')
        df_sens['Training_Set'] = df_sens['Model_Name'].str.extract(r'_([A-Z-]+)_')
    except Exception as e:
        print("Note: Could not automatically parse architecture metadata from column names.")

    df_sens.to_csv("model_feature_sensitivities.csv", index=False)
    print("Saved sensitivity matrix to 'model_feature_sensitivities.csv'.")

    # Print a quick comparison to console (e.g., comparing YOLOv8m to YOLOv10m on Aspect Ratio)
    if 'aspect_ratio' in df_sens.columns and 'Architecture' in df_sens.columns:
        print("\nQuick Compare: Aspect Ratio Sensitivity (More negative = struggles more with elongated ships)")

        # Filter for 'm' sized models to control for parameter count
        m_models = df_sens[df_sens['Size'] == 'm'].copy()
        if not m_models.empty:
            summary = m_models.groupby(['Architecture', 'Training_Set'])['aspect_ratio'].mean().reset_index()
            print(summary.sort_values(by=['Training_Set', 'Architecture']))


if __name__ == "__main__":
    # Ensure you pass your exact characteristics CSV name
    chars_csv = "sarship_dataset_chars_SSDD.csv"

    generate_difficulty_pdps("cirt_ship_parameters.csv", chars_csv)
    analyze_model_sensitivities("ship_detection_ground_truths_ssdd.feather", chars_csv)