import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings

# Suppress optimization warnings for zero-variance items
warnings.filterwarnings('ignore')

def continuous_logistic(theta, a, b):
    """
    Continuous Item Characteristic Curve (ICC).
    Uses np.clip to prevent mathematical overflow in np.exp.
    """
    exponent = np.clip(-a * (theta - b), -100, 100)
    return 1.0 / (1.0 + np.exp(exponent))

def run_continuous_irt(confidence_csv, metrics_csv):
    print(f"Loading continuous confidence data from {confidence_csv}...")
    df_conf = pd.read_feather(confidence_csv)
    
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

    # Clean metadata
    if 'dataset' in df_conf.columns:
        df_conf['dataset'] = df_conf['dataset'].astype(str).str.strip()
    
    metadata_cols = ['image_path', 'gt_bbox', 'dataset']
    actual_metadata_cols = [col for col in df_conf.columns if col.strip() in [m.strip() for m in metadata_cols]]
    model_cols = [col for col in df_conf.columns if col not in actual_metadata_cols]
    
    datasets = df_conf['dataset'].values
    unique_datasets = np.unique(datasets)
    n_ships = len(df_conf)
    n_models = len(model_cols)
    
    print(f"Found {n_ships} ships and {n_models} models.")
    
    # 1. Calibrate Confidence and Define Latent Ability (Theta)
    raw_confidence_matrix = df_conf[model_cols].values
    raw_confidence_matrix = np.nan_to_num(raw_confidence_matrix, nan=0.0)
    
    precision_matrix = np.ones((n_ships, n_models))
    theta_matrix = np.zeros((n_ships, n_models))
    
    print("Constructing dataset-stratified Theta (mAP) and Precision matrices...")
    for ds in unique_datasets:
        mask = (datasets == ds)
        precision_col = f'Precision_{ds}'
        map_col = f'mAP50_{ds}'
        
        # Extract mAP for models on this dataset
        ds_maps = []
        for j, checkpoint in enumerate(model_cols):
            # Precision Calibration
            if checkpoint in df_metrics.index and precision_col in df_metrics.columns:
                precision_matrix[mask, j] = df_metrics.loc[checkpoint, precision_col]
            
            # Ability (Theta) extraction
            if checkpoint in df_metrics.index and map_col in df_metrics.columns:
                ds_maps.append(df_metrics.loc[checkpoint, map_col])
            else:
                ds_maps.append(0.5) # Fallback
                
        # Standardize Theta to standard normal distribution (Mean=0, SD=1)
        # This is required for mathematical stability in logistic regressions
        ds_maps = np.array(ds_maps)
        ds_theta_scaled = (ds_maps - np.mean(ds_maps)) / (np.std(ds_maps) + 1e-8)
        
        for j in range(n_models):
            theta_matrix[mask, j] = ds_theta_scaled[j]

    calibrated_confidence_matrix = raw_confidence_matrix * precision_matrix
    
    # 2. Extract Continuous IRT Parameters via Non-Linear Least Squares
    print("Running Continuous Non-Linear Least Squares curve fitting...")
    a_params = np.ones(n_ships)
    b_params = np.zeros(n_ships)
    fit_errors = np.zeros(n_ships) # Track Mean Squared Error of the fit
    
    # Define bounds: a must be positive [0.01 to 10] (better models = higher confidence)
    # b bounded between [-5, 5] (standard normalized theta range)
    bounds = ([0.01, -5.0], [10.0, 5.0])
    
    for i in range(n_ships):
        y_data = calibrated_confidence_matrix[i, :]
        x_data = theta_matrix[i, :]
        
        # Check for zero variance (missed by all or 1.0 confidence by all)
        if np.max(y_data) - np.min(y_data) < 0.05:
            if np.mean(y_data) < 0.1:
                # Universally missed
                a_params[i] = 0.01
                b_params[i] = 5.0 # Max difficulty
            else:
                # Universally detected easily
                a_params[i] = 0.01
                b_params[i] = -5.0 # Min difficulty
            continue
            
        try:
            # Initial guess: a=1.0, b=0.0
            popt, _ = curve_fit(continuous_logistic, x_data, y_data, p0=[1.0, 0.0], bounds=bounds)
            a_params[i] = popt[0]
            b_params[i] = popt[1]
            
            # Calculate fit error (MSE)
            y_pred = continuous_logistic(x_data, *popt)
            fit_errors[i] = np.mean((y_data - y_pred)**2)
            
        except RuntimeError:
            # If optimization fails to converge, fall back to heuristic averages
            a_params[i] = 1.0
            b_params[i] = np.mean(x_data[y_data < 0.5]) if len(x_data[y_data < 0.5]) > 0 else 0.0

    # Normalize difficulty to [0, 1] for evaluation heuristic
    b_min, b_max = np.min(b_params), np.max(b_params)
    b_norm = (b_params - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(b_params)
    
    # Save Ship Parameters
    ship_metrics = df_conf[actual_metadata_cols].copy()
    ship_metrics['CIRT_Discrimination_a'] = a_params
    ship_metrics['CIRT_Raw_Difficulty_b'] = b_params
    ship_metrics['Normalized_Difficulty_b'] = b_norm
    ship_metrics['Curve_Fit_MSE'] = fit_errors
    ship_metrics.to_csv("cirt_ship_parameters.csv", index=False)
    print("Saved continuous item parameters to 'cirt_ship_parameters.csv'.")

    # 3. Evaluate Models (Same mathematical framework as before)
    print("Evaluating models using Continuous IRT parameters...")
    model_scores = []
    
    for j, checkpoint in enumerate(model_cols):
        conf = calibrated_confidence_matrix[:, j]
        
        rewards = conf * b_norm
        penalties = (1.0 - conf) * (1.0 - b_norm)
        item_scores = a_params * (rewards - penalties)
        
        model_record = {'Model_Name': checkpoint}
        global_raw_irt = 0.0
        global_hybrid = 0.0
        
        for ds in unique_datasets:
            mask = (datasets == ds)
            ds_irt_score = np.sum(item_scores[mask])
            
            map_col = f'mAP_{ds}'
            ds_map = df_metrics.loc[checkpoint, map_col] if (checkpoint in df_metrics.index and map_col in df_metrics.columns) else 1.0
            
            ds_hybrid_score = ds_irt_score * ds_map
            
            model_record[f'Raw_CIRT_{ds}'] = ds_irt_score
            model_record[f'Hybrid_Score_{ds}'] = ds_hybrid_score
            global_raw_irt += ds_irt_score
            global_hybrid += ds_hybrid_score
            
        model_record['Total_Raw_CIRT'] = global_raw_irt
        model_record['Final_Global_Hybrid_Score'] = global_hybrid
        model_scores.append(model_record)
        
    model_eval_df = pd.DataFrame(model_scores)
    cols = ['Model_Name', 'Final_Global_Hybrid_Score', 'Total_Raw_CIRT']
    ds_cols = [c for c in model_eval_df.columns if c not in cols]
    model_eval_df = model_eval_df[cols + ds_cols]
    
    model_eval_df = model_eval_df.sort_values(by='Final_Global_Hybrid_Score', ascending=False).reset_index(drop=True)
    model_eval_df.to_csv("cirt_model_evaluations.csv", index=False)
    
    print("\nTop 5 Models by Final Global Hybrid CIRT Score:")
    print(model_eval_df[['Model_Name', 'Final_Global_Hybrid_Score', 'Total_Raw_CIRT']].head(5))

if __name__ == "__main__":
    run_continuous_irt("ship_detection_ground_truths_ssdd.feather", "yolo_trained_on_ssdd_checkpoint_metrics.csv")
