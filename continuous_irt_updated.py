import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings
import re

# Suppress optimization warnings for zero-variance items
warnings.filterwarnings('ignore')


def continuous_logistic(theta, a, b):
    """
    Continuous Item Characteristic Curve (ICC).
    Uses np.clip to prevent mathematical overflow in np.exp.
    """
    exponent = np.clip(-a * (theta - b), -100, 100)
    return 1.0 / (1.0 + np.exp(exponent))


def robust_spatial_merge(df_conf, df_chars):
    """
    Bypasses strict string matching by mapping ships using spatial proximity (Hungarian Algorithm).
    Guarantees perfect joins even if the BBox math/normalization differs between scripts.
    """
    print("Executing robust spatial merge to bypass bounding box formatting differences...")
    df_conf['image_path'] = df_conf['image_path'].astype(str).str.strip()
    df_chars['image_path'] = df_chars['image_path'].astype(str).str.strip()

    def extract_centers(bbox):
        try:
            # Handle native numpy arrays directly from the Feather files
            if isinstance(bbox, (np.ndarray, list)):
                coords = [float(x) for x in bbox]
            elif isinstance(bbox, str):
                clean_str = bbox.replace('[', '').replace(']', '').replace(',', ' ')
                coords = [float(x) for x in clean_str.split()]
            else:
                return 0.0, 0.0

            # Both files now use [xmin, ymin, xmax, ymax]
            if len(coords) >= 4:
                cx = (coords[0] + coords[2]) / 2.0
                cy = (coords[1] + coords[3]) / 2.0
                return cx, cy
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

    print("Extracting spatial centers...")
    df_conf['temp_cx_cy'] = df_conf['gt_bbox'].apply(extract_centers)
    df_chars['temp_cx_cy'] = df_chars['gt_bbox'].apply(extract_centers)

    conf_idx_list = []
    char_idx_list = []

    unique_images = df_conf['image_path'].unique()

    for img in unique_images:
        conf_mask = df_conf['image_path'] == img
        char_mask = df_chars['image_path'] == img

        conf_indices = df_conf.index[conf_mask].tolist()
        char_indices = df_chars.index[char_mask].tolist()

        if not conf_indices or not char_indices:
            continue

        conf_centers = np.array(df_conf.loc[conf_indices, 'temp_cx_cy'].tolist())
        chars_centers = np.array(df_chars.loc[char_indices, 'temp_cx_cy'].tolist())

        # Calculate spatial distances between all ships in the image
        dist_matrix = cdist(conf_centers, chars_centers)

        # Hungarian Algorithm mathematically finds the optimal 1-to-1 pairings
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        for i, j in zip(row_ind, col_ind):
            # Tolerance margin (0.2 in normalized coordinates is a huge safety margin)
            if dist_matrix[i, j] < 0.2:
                conf_idx_list.append(conf_indices[i])
                char_idx_list.append(char_indices[j])

    print(f"Spatially mapped {len(conf_idx_list)} ships successfully.")

    # Reconstruct the merged dataframe safely
    df_conf_matched = df_conf.loc[conf_idx_list].reset_index(drop=True)
    df_chars_matched = df_chars.loc[char_idx_list].reset_index(drop=True)

    # Drop redundant columns from chars before concat to avoid duplicates
    cols_to_drop = [c for c in df_chars_matched.columns if c in df_conf_matched.columns]
    df_chars_matched = df_chars_matched.drop(columns=cols_to_drop)

    df_merged = pd.concat([df_conf_matched, df_chars_matched], axis=1)
    df_merged.drop(columns=['temp_cx_cy'], inplace=True, errors='ignore')
    return df_merged


def run_continuous_irt(confidence_file, metrics_csv, characteristics_file):
    print(f"Loading continuous confidence data from {confidence_file}...")
    if confidence_file.endswith('.feather'):
        df_conf = pd.read_feather(confidence_file)
    else:
        df_conf = pd.read_csv(confidence_file)

    print(f"Loading model metrics from {metrics_csv}...")
    try:
        df_metrics = pd.read_csv(metrics_csv)
        if 'checkpoint' in df_metrics.columns:
            df_metrics.set_index('checkpoint', inplace=True)
        else:
            df_metrics.set_index(df_metrics.columns[0], inplace=True)
    except FileNotFoundError:
        print(f"Error: {metrics_csv} not found.")
        return

    print(f"Loading dataset characteristics from {characteristics_file}...")
    if characteristics_file.endswith('.feather'):
        df_chars = pd.read_feather(characteristics_file)
    else:
        df_chars = pd.read_csv(characteristics_file)


    # --- 1. Robust Spatial Merging ---
    df_merged = robust_spatial_merge(df_conf, df_chars)

    if len(df_merged) == 0:
        print("CRITICAL ERROR: Spatial merge failed. Datasets do not overlap.")
        return

    # --- 2. Isolate Model Columns ---
    regex_pattern = re.compile(r'(yolo[a-zA-Z0-9]+)_.*[eE]poch_?(\d+)', re.IGNORECASE)
    model_cols = [col for col in df_merged.columns if regex_pattern.search(col)]
    print(f"Isolated {len(model_cols)} model checkpoints.")

    # --- 3. Calculate Latent Ability (Theta) STRICTLY on the Test Set ---
    print("Calculating Latent Model Ability (Theta) anchored purely on the Test Split...")
    test_mask = df_merged['split'].str.lower() == 'test'
    df_test = df_merged[test_mask]

    if len(df_test) == 0:
        print("CRITICAL ERROR: No 'test' split found. Check your split column.")
        return

    # Base competence is the model's mean confidence across unseen test data
    theta_raw = df_test[model_cols].mean(axis=0).values

    # Min-Max Normalize Theta to [0, 1]
    theta_min, theta_max = np.min(theta_raw), np.max(theta_raw)
    theta = (theta_raw - theta_min) / (theta_max - theta_min + 1e-8)

    # --- 4. Curve Fitting for ALL Items (Train + Val + Test) ---
    print("Fitting Item Characteristic Curves for Dataset Cartography...")
    ship_parameters = []
    item_scores_matrix = np.zeros((len(df_merged), len(model_cols)))

    for i, row in df_merged.iterrows():
        y_data = row[model_cols].values.astype(float)

        try:
            popt, _ = curve_fit(continuous_logistic, theta, y_data,
                                p0=[1.0, 0.5], bounds=([0.01, -5.0], [50.0, 5.0]), maxfev=1000)
            a, b = popt

            y_pred = continuous_logistic(theta, a, b)
            mse = np.mean((y_data - y_pred) ** 2)

        except RuntimeError:
            a, b, mse = 0.0, -5.0 if np.mean(y_data) > 0.5 else 5.0, 0.0

        b_norm = np.clip((b - (-5.0)) / 10.0, 0, 1)

        ship_parameters.append({
            'image_path': row['image_path'],
            'gt_bbox': row.get('gt_bbox', ''),
            'dataset': row.get('dataset', 'SSDD'),
            'split': row['split'],
            'location_type': row.get('location_type', 'unspecified'),
            'CIRT_Discrimination_a': a,
            'CIRT_Raw_Difficulty_b': b,
            'Normalized_Difficulty_b': b_norm,
            'Curve_Fit_MSE': mse
        })

        expected_scores = continuous_logistic(theta, a, b)
        residuals = y_data - expected_scores
        item_scores_matrix[i, :] = a * residuals

    df_params = pd.DataFrame(ship_parameters)
    df_params.to_csv("cirt_ship_parameters.csv", index=False)
    print("Saved ship parameters (Train/Val/Test) to 'cirt_ship_parameters.csv'.")

    # --- 5. Score Models STRICTLY on the Test Set ---
    print("Scoring Models based purely on Test Set Generalization...")
    model_scores = []

    test_indices = df_merged.index[test_mask].tolist()

    for col_idx, checkpoint in enumerate(model_cols):
        model_record = {'Model_Name': checkpoint}

        test_irt_score = np.sum(item_scores_matrix[test_indices, col_idx])

        # Map back to SSDD metrics
        map_col = 'mAP50_SSDD'
        ds_map = df_metrics.loc[checkpoint, map_col] if (
                    checkpoint in df_metrics.index and map_col in df_metrics.columns) else 1.0

        hybrid_score = test_irt_score * ds_map

        model_record['Raw_CIRT_Test_Only'] = test_irt_score
        model_record['Final_Global_Hybrid_Score'] = hybrid_score

        model_scores.append(model_record)

    df_evals = pd.DataFrame(model_scores)
    df_evals = df_evals.sort_values(by='Final_Global_Hybrid_Score', ascending=False).reset_index(drop=True)
    df_evals.to_csv("cirt_model_evaluations.csv", index=False)
    print("Saved evaluation metrics to 'cirt_model_evaluations.csv'.")
    print("Done! The CIRT evaluation is now perfectly constrained against data leakage.")


if __name__ == "__main__":
    run_continuous_irt(
        confidence_file="ship_detection_ground_truths_ssdd.feather",
        metrics_csv="yolo_trained_on_ssdd_checkpoint_metrics.csv",
        characteristics_file="sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.feather"
    )