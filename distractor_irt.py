import os
import glob
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import scipy.stats as stats


try:
    from sar_irt.utils.core import set_ieee_tgrs_style
except ImportError:
    def set_ieee_tgrs_style():
        plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
                             "legend.fontsize": 11})

warnings.filterwarnings('ignore')


def compute_iou(boxA, boxB):
    """Computes Intersection over Union for two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def continuous_logistic(theta, a, b):
    """Continuous Item Characteristic Curve (ICC) adapted for [0,1] confidence scores."""
    exponent = np.clip(-a * (theta - b), -100, 100)
    return 1.0 / (1.0 + np.exp(exponent))


def parse_bbox_array(bbox_val):
    """Safely converts string or array bounding boxes to a numpy array."""
    if isinstance(bbox_val, (np.ndarray, list)):
        return np.array(bbox_val, dtype=float)
    if isinstance(bbox_val, str):
        clean_str = bbox_val.replace('[', '').replace(']', '').replace(',', ' ')
        return np.array([float(x) for x in clean_str.split()], dtype=float)
    return np.zeros(4)


def compile_best_fp_dataset(fp_dir, best_epochs_dict, iou_threshold=0.5):
    """
    Scans the FP directory for the *best* epoch of each model, and uses
    IoU clustering to merge FPs that represent the same physical artifact.
    """
    print("Compiling False Positives using IoU clustering for best-performing epochs...")

    # Load all relevant data into memory
    all_predictions = []
    model_columns = []

    for arch, best_epoch in best_epochs_dict.items():
        search_pattern = os.path.join(fp_dir, f"{arch}_*_epoch{best_epoch}_fp.feather")
        fp_files = glob.glob(search_pattern)

        if not fp_files:
            print(f"Warning: No file found for {arch} at epoch {best_epoch}")
            continue

        df = pd.read_feather(fp_files[0])
        conf_col = [c for c in df.columns if c not in ['image_path', 'fp_bbox']][0]
        model_columns.append(conf_col)

        # Parse bounding boxes
        df['fp_bbox_arr'] = df['fp_bbox'].apply(parse_bbox_array)

        for _, row in df.iterrows():
            all_predictions.append({
                'image_path': row['image_path'],
                'model': conf_col,
                'conf': row[conf_col],
                'bbox': row['fp_bbox_arr']
            })

    # Group by image to perform IoU clustering
    print(f"Loaded {len(all_predictions)} raw FP predictions. Clustering by IoU...")
    df_preds = pd.DataFrame(all_predictions)

    clustered_fps = []
    cluster_id_counter = 0

    for img_path, group in df_preds.groupby('image_path'):
        clusters = []  # List of dicts: {'canonical_bbox': arr, 'confidences': {model: conf}}

        for _, pred in group.iterrows():
            matched = False
            for cluster in clusters:
                if compute_iou(cluster['canonical_bbox'], pred['bbox']) > iou_threshold:
                    # If this model already detected this cluster, keep the highest confidence (NMS style)
                    cluster['confidences'][pred['model']] = max(
                        cluster['confidences'].get(pred['model'], 0.0), pred['conf']
                    )
                    matched = True
                    break

            if not matched:
                clusters.append({
                    'canonical_bbox': pred['bbox'],
                    'confidences': {pred['model']: pred['conf']}
                })

        # Flatten clusters into records
        for c in clusters:
            record = {
                'fp_id': f"FP_{cluster_id_counter}",
                'image_path': img_path,
                'fp_bbox_norm': c['canonical_bbox']
            }
            # Initialize all models to 0.0, then update with actuals
            for model_col in model_columns:
                record[model_col] = c['confidences'].get(model_col, 0.0)

            clustered_fps.append(record)
            cluster_id_counter += 1

    df_fp_matrix = pd.DataFrame(clustered_fps)
    print(f"Clustered into {len(df_fp_matrix)} unique physical artifacts.")
    return df_fp_matrix, model_columns


def calculate_radiometrics(image, mask):
    """Calculates mean intensity and variance for a given masked region."""
    if np.count_nonzero(mask) == 0:
        return np.nan, np.nan
    pixels = image[mask == 255]
    return np.mean(pixels), np.var(pixels)


def extract_fp_characteristics(df_fp, images_base_dir):
    """
    Dynamically calculates physical traits (geometry, radiometry, SCR)
    directly from the image and the clustered canonical bounding box.
    """
    print("\nExtracting physical characteristics for False Positives via OpenCV...")
    characteristics = []

    for _, row in df_fp.iterrows():
        img_path = os.path.join(images_base_dir, row['image_path'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        img_h, img_w = image.shape
        norm_bbox = row['fp_bbox_norm']  # [x1, y1, x2, y2] normalized

        # Convert to absolute pixels
        x1, y1 = int(norm_bbox[0] * img_w), int(norm_bbox[1] * img_h)
        x2, y2 = int(norm_bbox[2] * img_w), int(norm_bbox[3] * img_h)

        # Geometry
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)

        record = {
            'fp_id': row['fp_id'],
            'image_w': img_w,
            'image_h': img_h,
            'image_total_pixels': img_w * img_h,
            'bbox_w_px': bbox_w,
            'bbox_h_px': bbox_h,
            'bbox_area_px2': bbox_w * bbox_h,
            'bbox_aspect': max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + 1e-6),
            'dist_to_edge_px': min(x1, y1, img_w - x2, img_h - y2)
        }

        # Radiometrics (Inner BBox vs 50% Padded Outer Ring)
        bbox_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)

        pad_w, pad_h = int(bbox_w * 0.5), int(bbox_h * 0.5)
        bx1, by1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        bx2, by2 = min(img_w, x2 + pad_w), min(img_h, y2 + pad_h)

        bbox_bg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.rectangle(bbox_bg_mask, (bx1, by1), (bx2, by2), 255, -1)
        bbox_bg_mask = cv2.bitwise_xor(bbox_bg_mask, bbox_mask)  # Subtract inner box to make it a ring

        target_mean, target_var = calculate_radiometrics(image, bbox_mask)
        bg_mean, bg_var = calculate_radiometrics(image, bbox_bg_mask)

        record['target_mean'] = target_mean
        record['target_variance'] = target_var
        record['bg_mean'] = bg_mean
        record['bg_variance'] = bg_var

        # Signal to Clutter Ratio (dB)
        if not pd.isna(target_mean) and not pd.isna(bg_mean) and bg_mean > 1e-6:
            record['SCR_dB'] = 20 * np.log10(target_mean / bg_mean)
        else:
            record['SCR_dB'] = np.nan

        characteristics.append(record)

    df_chars = pd.DataFrame(characteristics)
    print(f"Extracted characteristics for {len(df_chars)} FP artifacts.")
    return df_chars


def run_explanatory_distractor_irt(df_fp, df_chars, model_cols):
    """
    Calculates Distractor Deceptiveness (b) and merges it with the extracted
    physical characteristics for downstream Random Forest/PDP analysis.
    """
    print("\nCalculating Model Susceptibility (\u03B8_FP)...")
    theta_raw = df_fp[model_cols].mean(axis=0).values
    theta_min, theta_max = np.min(theta_raw), np.max(theta_raw)
    theta = (theta_raw - theta_min) / (theta_max - theta_min + 1e-8)

    print("Fitting Distractor Characteristic Curves...")
    distractor_params = []

    for i, row in df_fp.iterrows():
        y_data = row[model_cols].values.astype(float)

        if np.max(y_data) < 0.1:
            continue

        try:
            popt, _ = curve_fit(continuous_logistic, theta, y_data,
                                p0=[1.0, 0.5], bounds=([0.01, -5.0], [50.0, 5.0]), maxfev=1000)
            a, b = popt
        except RuntimeError:
            a, b = 0.0, 5.0

        b_norm = np.clip((b - (-5.0)) / 10.0, 0, 1)

        distractor_params.append({
            'fp_id': row['fp_id'],
            'image_path': row['image_path'],
            'Distractor_Discrimination_a': a,
            'Distractor_Deceptiveness_b': b_norm,
            'Max_Confidence': np.max(y_data),
            'Mean_Confidence': np.mean(y_data)
        })

    df_params = pd.DataFrame(distractor_params)

    # Merge IRT Parameters with Physical Characteristics
    df_explanatory = pd.merge(df_params, df_chars, on='fp_id', how='inner')
    df_explanatory.to_csv("cirt_explanatory_distractors.csv", index=False)

    print(f"Saved Explanatory IRT parameters to 'cirt_explanatory_distractors.csv'.")
    print("This file is now ready for Random Forest and Partial Dependence Plot (PDP) analysis.")
    return df_explanatory, theta_raw


def get_best_epochs(eval_csv_path, score_column='Final_Global_Hybrid_Score'):
    """
    Parses the CIRT evaluations CSV to automatically identify the best performing
    epoch for each model architecture.
    """
    print(f"\nParsing {eval_csv_path} to determine optimal epochs...")
    df_evals = pd.read_csv(eval_csv_path)

    # Regex to extract architecture and epoch (e.g., 'yolov8n_SSDD_epoch490')
    regex_pattern = re.compile(r'(yolo[a-zA-Z0-9]+)_.*[eE]poch_?(\d+)', re.IGNORECASE)

    best_epochs = {}

    def extract_arch(name):
        match = regex_pattern.search(name)
        return match.group(1).lower() if match else None

    def extract_epoch(name):
        match = regex_pattern.search(name)
        return int(match.group(2)) if match else None

    df_evals['Architecture'] = df_evals['Model_Name'].apply(extract_arch)
    df_evals['Epoch'] = df_evals['Model_Name'].apply(extract_epoch)

    df_evals = df_evals.dropna(subset=['Architecture', 'Epoch'])

    # Find index of the maximum score for each architecture
    idx = df_evals.groupby('Architecture')[score_column].idxmax()
    best_rows = df_evals.loc[idx]

    for _, row in best_rows.iterrows():
        best_epochs[row['Architecture']] = int(row['Epoch'])

    print(f"Identified optimal epochs for {len(best_epochs)} architectures.")
    return best_epochs

def analyze_model_sensitivities(df_fp, df_chars):
    """
    Calculates the Spearman Rank Correlation between physical SAR features
    and the raw confidence outputs of every individual YOLO model.
    """
    print("\n--- Generating Individual Model Feature Sensitivities ---")

    # Merge confidence data with physical characteristics
    df = pd.merge(df_fp, df_chars, on=['fp_id'], how='inner')

    # Identify model columns vs metadata/feature columns
    metadata = ['image_path', 'fp_id', 'fp_bbox_norm']
    chars_cols = df_chars.columns.drop(['fp_id']).tolist()
    model_cols = [col for col in df.columns if col not in metadata and col not in chars_cols]

    # Select key physical features to test
#    target_features = ['Aspect_Ratio', 'Area_px2', 'Background_Variance', 'Local_SCR_dB', 'Dist_to_Image_Edge_px']
    target_features = [
        'image_w',
        'image_h',
        'image_total_pixels',
        'bbox_w_px',
        'bbox_h_px',
        'bbox_area_px2',
        'bbox_aspect',
        'dist_to_edge_px',
        'target_mean',
        'target_variance',
        'bg_mean',
        'bg_variance',
        'SCR_dB'
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

    df_sens.to_csv("model_fp_feature_sensitivities.csv", index=False)
    print("Saved sensitivity matrix to 'model_fp_feature_sensitivities.csv'.")

def generate_fp_difficulty_pdps(df):
    """
    Trains a Random Forest on Ship Difficulty and generates Partial Dependence Plots (PDP)
    to identify the exact physical thresholds where models fail.
    """
    print("\n--- Generating Partial Dependence Plots (PDP) ---")

    features = [
        'image_w',
        'image_h',
        'image_total_pixels',
        'bbox_w_px',
        'bbox_h_px',
        'bbox_area_px2',
        'bbox_aspect',
        'dist_to_edge_px',
        'target_mean',
        'target_variance',
        'bg_mean',
        'bg_variance',
        'SCR_dB'
    ]

    actual_features = [f for f in features if f in df.columns]

    df = df.dropna(subset=actual_features + ['Distractor_Deceptiveness_b'])
    X = df[actual_features]
    y_diff = df['Distractor_Deceptiveness_b']

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

        # Extract the computed PDP curve coordinates produced by the Random Forest
        pdp_records = []
        for i, feature_name in enumerate(plot_features):
            # 'values' contains the x-axis grid points
            grid_values = display.pd_results[i]['grid_values'][0]
            # 'average' contains the y-axis partial dependence calculations
            pd_average = display.pd_results[i]['average'][0]

            for val, pdp in zip(grid_values, pd_average):
                pdp_records.append({
                    'Feature': feature_name,
                    'Feature_Value': val,
                    'Partial_Dependence_b': pdp
                })

        df_pdp_curves = pd.DataFrame(pdp_records)
        df_pdp_curves.to_csv("pdp_fp_curves.csv", index=False)
        print("Saved computed PDP curve coordinates to 'pdp_fp_curves.csv'.")

        plt.suptitle('Partial Dependence of Physical Traits on False Positives', fontsize=16)
        plt.subplots_adjust(top=0.92, hspace=0.3)
        plt.savefig("pdp_fp_difficulty_thresholds.png", dpi=300, bbox_inches='tight')
        print("Saved PDP plot to 'pdp_fp_difficulty_thresholds.png'.")
    else:
        print("Warning: Top features not found in columns. Check column naming.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    FP_DIR = "../sarship-yolo26/SSDD_fp_feather"
    IMAGES_BASE_DIR = "../sarship-yolo26/SSDD/images"  # Adjust relative path to images as necessary
    EVALS_CSV = "cirt_model_evaluations.csv"

    # 1. Automatically determine the best epoch for each architecture
    BEST_EPOCHS = get_best_epochs(EVALS_CSV, score_column='Final_Global_Hybrid_Score')

    # 2. Compile clustered dataset using IoU
    df_fp_matrix, models = compile_best_fp_dataset(FP_DIR, BEST_EPOCHS, iou_threshold=0.5)

    # 3. Extract physical radiometrics and geometries dynamically via OpenCV
    df_fp_characteristics = extract_fp_characteristics(df_fp_matrix, IMAGES_BASE_DIR)

    # 4. Run Explanatory IRT linking latent deceptiveness to physics
    df_irt_params, theta = run_explanatory_distractor_irt(df_fp_matrix, df_fp_characteristics, models)

    # 5. Analyze model sensitivities to physical features
    analyze_model_sensitivities(df_fp_matrix, df_fp_characteristics)

    # 6. Generate PDP plots for FP difficulty
    generate_fp_difficulty_pdps(df_irt_params)
