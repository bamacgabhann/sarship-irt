import numpy as np
import pandas as pd
from girth import twopl_mml
import warnings

warnings.filterwarnings('ignore')

def run_calibrated_irt(confidence_csv, metrics_csv):
    """
    Runs an IRT analysis calibrated by dataset-specific model precision.
    Utilizes a robust 2PL binary model for parameter extraction to handle 
    polarized confidence distributions and zero-variance items.
    """
    print(f"Loading confidence data from {confidence_csv}...")
    df_conf = pd.read_csv(confidence_csv)
    
    print(f"Loading model metrics from {metrics_csv}...")
    try:
        df_metrics = pd.read_csv(metrics_csv)
        if 'Model_Name' in df_metrics.columns:
            df_metrics.set_index('Model_Name', inplace=True)
        else:
            df_metrics.set_index(df_metrics.columns[0], inplace=True)
    except FileNotFoundError:
        print(f"Error: {metrics_csv} not found.")
        return

    if 'dataset' in df_conf.columns:
        df_conf['dataset'] = df_conf['dataset'].astype(str).str.strip()
    else:
        print("Error: 'dataset' column not found in the ground truth CSV.")
        return

    metadata_cols = ['image_path', 'gt_bbox', 'dataset']
    actual_metadata_cols = [col for col in df_conf.columns if col.strip() in [m.strip() for m in metadata_cols]]
    model_cols = [col for col in df_conf.columns if col not in actual_metadata_cols]
    
    datasets = df_conf['dataset'].values
    unique_datasets = np.unique(datasets)
    n_ships = len(df_conf)
    n_models = len(model_cols)
    
    print(f"Found {n_ships} ships across datasets: {unique_datasets}")
    print(f"Found {n_models} models to evaluate.")
    
    # 1. Extract and Calibrate Confidence Matrix
    raw_confidence_matrix = df_conf[model_cols].values
    raw_confidence_matrix = np.nan_to_num(raw_confidence_matrix, nan=0.0)
    
    precision_matrix = np.ones((n_ships, n_models))
    for j, model_name in enumerate(model_cols):
        for ds in unique_datasets:
            mask = (datasets == ds)
            precision_col = f'Precision_{ds}'
            if model_name in df_metrics.index and precision_col in df_metrics.columns:
                precision_matrix[mask, j] = df_metrics.loc[model_name, precision_col]
            else:
                precision_matrix[mask, j] = 1.0 
    
    print("Calibrating confidence matrix using dataset-specific Precision...")
    calibrated_confidence_matrix = raw_confidence_matrix * precision_matrix
    
    # 2. Binarize and isolate items with variance
    print("Thresholding calibrated confidence at 0.5 for 2PL extraction...")
    binary_matrix = (calibrated_confidence_matrix > 0.5).astype(int)
    
    # Calculate how many models detected each ship
    detection_counts = np.sum(binary_matrix, axis=1)
    
    # Isolate ships with variance (detected by >0 and <n_models)
    valid_mask = (detection_counts > 0) & (detection_counts < n_models)
    valid_binary_matrix = binary_matrix[valid_mask]
    
    print(f"Excluded {np.sum(detection_counts == 0)} ships with 0 detections.")
    print(f"Excluded {np.sum(detection_counts == n_models)} ships with 100% detection.")
    print(f"Running 2PL MML Optimization on {np.sum(valid_mask)} valid ships...")
    
    # 3. Run 2PL on valid items
    results_2pl = twopl_mml(valid_binary_matrix)
    
    valid_a = results_2pl['Discrimination']
    valid_b = results_2pl['Difficulty']
    
    # 4. Map parameters back to the full dataset size
    # Initialize arrays with default values
    a_full = np.ones(n_ships)  # Default discrimination of 1.0
    b_full = np.zeros(n_ships)
    
    # Impute calculated values
    a_full[valid_mask] = valid_a
    b_full[valid_mask] = valid_b
    
    # Handle extremes for difficulty (b)
    # Highest observed difficulty + margin for ships missed by everyone
    max_b = np.max(valid_b) if len(valid_b) > 0 else 3.0
    min_b = np.min(valid_b) if len(valid_b) > 0 else -3.0
    
    b_full[detection_counts == 0] = max_b + 1.0
    b_full[detection_counts == n_models] = min_b - 1.0
    
    # 5. Normalize difficulty strictly using min/max of the final array
    b_min, b_max = np.min(b_full), np.max(b_full)
    b_norm = (b_full - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(b_full)
    
    ship_metrics = df_conf[actual_metadata_cols].copy()
    ship_metrics['Detections_Count'] = detection_counts
    ship_metrics['Discrimination_a'] = a_full
    ship_metrics['Raw_Difficulty_b'] = b_full
    ship_metrics['Normalized_Difficulty_b'] = b_norm
    ship_metrics.to_csv("calibrated_ship_irt_parameters.csv", index=False)
    print("\nSaved unbiased item parameters to 'calibrated_ship_irt_parameters.csv'.")

    # 6. Evaluate Models using Calibrated Confidence & Dataset-specific mAP
    print("Evaluating dataset-stratified model performance...")
    model_scores = []
    
    for j, model_name in enumerate(model_cols):
        conf = calibrated_confidence_matrix[:, j]
        
        # Scoring equation utilizes the robust 2PL parameters
        rewards = conf * b_norm
        penalties = (1.0 - conf) * (1.0 - b_norm)
        item_scores = a_full * (rewards - penalties)
        
        model_record = {'Model_Name': model_name}
        global_raw_irt = 0.0
        global_hybrid = 0.0
        
        for ds in unique_datasets:
            mask = (datasets == ds)
            ds_irt_score = np.sum(item_scores[mask])
            
            map_col = f'mAP_{ds}'
            ds_map = df_metrics.loc[model_name, map_col] if (model_name in df_metrics.index and map_col in df_metrics.columns) else 1.0
            
            ds_hybrid_score = ds_irt_score * ds_map
            
            model_record[f'Raw_IRT_{ds}'] = ds_irt_score
            model_record[f'Hybrid_Score_{ds}'] = ds_hybrid_score
            
            global_raw_irt += ds_irt_score
            global_hybrid += ds_hybrid_score
            
        model_record['Total_Raw_IRT'] = global_raw_irt
        model_record['Final_Global_Hybrid_Score'] = global_hybrid
        model_scores.append(model_record)
        
    model_eval_df = pd.DataFrame(model_scores)
    
    cols = ['Model_Name', 'Final_Global_Hybrid_Score', 'Total_Raw_IRT']
    ds_cols = [c for c in model_eval_df.columns if c not in cols]
    model_eval_df = model_eval_df[cols + ds_cols]
    
    model_eval_df = model_eval_df.sort_values(by='Final_Global_Hybrid_Score', ascending=False).reset_index(drop=True)
    
    model_eval_df.to_csv("calibrated_model_evaluations.csv", index=False)
    print("Saved evaluations to 'calibrated_model_evaluations.csv'.")
    print("\nTop 5 Models by Final Global Hybrid Score:")
    print(model_eval_df[['Model_Name', 'Final_Global_Hybrid_Score', 'Total_Raw_IRT']].head(5))

if __name__ == "__main__":
    run_calibrated_irt("ship_detection_ground_truths.csv", "model_metrics.csv")
