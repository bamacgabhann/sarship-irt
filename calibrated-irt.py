import numpy as np
import pandas as pd
from girth import grm_mml
import warnings

warnings.filterwarnings('ignore')

def polytomize_confidence(confidence_matrix, bins=5):
    """Discretizes continuous [0, 1] confidence scores into integer categories."""
    edges = np.linspace(0, 1, bins + 1)
    categorized = np.digitize(confidence_matrix, edges[:-1])
    categorized = np.clip(categorized, 1, bins)
    return categorized

def run_calibrated_irt(confidence_csv, metrics_csv, n_bins=5):
    """
    Runs an IRT analysis calibrated by model precision to account for False Positives.
    
    Parameters:
    confidence_csv (str): Path to 'ship_detection_ground_truths.csv'.
    metrics_csv (str): Path to a CSV containing overall model metrics. 
                       Must have columns: 'Model_Name', 'Precision', 'mAP50' (or similar).
    """
    print(f"Loading confidence data from {confidence_csv}...")
    df_conf = pd.read_csv(confidence_csv)
    
    print(f"Loading model metrics from {metrics_csv}...")
    try:
        df_metrics = pd.read_csv(metrics_csv)
        # Ensure metrics DataFrame is indexed by Model_Name for easy lookup
        df_metrics.set_index('Model_Name', inplace=True)
    except FileNotFoundError:
        print(f"Error: {metrics_csv} not found. Please create this file with your model mAP/Precision data.")
        return

    metadata_cols = ['image_path', 'gt_bbox']
    model_cols = [col for col in df_conf.columns if col not in metadata_cols]
    
    # 1. Extract and Calibrate Confidence Matrix
    raw_confidence_matrix = df_conf[model_cols].values
    raw_confidence_matrix = np.nan_to_num(raw_confidence_matrix, nan=0.0)
    
    # Extract precision vector corresponding to the order of model_cols
    # Fallback to 1.0 if a model is missing from the metrics CSV to avoid breaking
    precision_vector = np.array([
        df_metrics.loc[m, 'Precision'] if m in df_metrics.index else 1.0 
        for m in model_cols
    ])
    
    print("Calibrating confidence matrix using model Precision...")
    # Broadcast multiplication: scales each column (model) by its precision
    calibrated_confidence_matrix = raw_confidence_matrix * precision_vector
    
    # 2. Run Graded Response Model on Calibrated Data
    print(f"Polytomizing calibrated confidence into {n_bins} categories...")
    ordinal_data = polytomize_confidence(calibrated_confidence_matrix, bins=n_bins)
    
    print("Fitting Graded Response Model (MML). Utilizing CPU vectorization...")
    # This matrix math is highly optimized for your Xeon CPU
    results_grm = grm_mml(ordinal_data)
    
    a_grm = results_grm['Discrimination']
    b_mean_grm = np.mean(results_grm['Difficulty'], axis=1)
    
    # Normalize difficulty
    b_min, b_max = np.min(b_mean_grm), np.max(b_mean_grm)
    b_norm = (b_mean_grm - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(b_mean_grm)
    
    # Save unbiased Ship Parameters
    ship_metrics = df_conf[metadata_cols].copy()
    ship_metrics['Discrimination_a'] = a_grm
    ship_metrics['Normalized_Difficulty_b'] = b_norm
    ship_metrics.to_csv("calibrated_ship_irt_parameters.csv", index=False)
    print("Saved unbiased item parameters to 'calibrated_ship_irt_parameters.csv'.")

    # 3. Evaluate Models using Calibrated Confidence & Global mAP
    print("Evaluating models...")
    model_scores = []
    
    for idx, model_name in enumerate(model_cols):
        # We use the calibrated confidence to calculate the raw IRT score.
        # This inherently penalizes low-precision models because their C drops,
        # increasing the penalty term (1 - C) * (1 - b).
        conf = calibrated_confidence_matrix[:, idx]
        
        rewards = conf * b_norm
        penalties = (1.0 - conf) * (1.0 - b_norm)
        
        item_scores = a_grm * (rewards - penalties)
        raw_irt_score = np.sum(item_scores)
        
        # Further constrain the final score by the model's global mAP 
        # to ensure holistic OD performance is respected.
        model_map = df_metrics.loc[model_name, 'mAP50'] if model_name in df_metrics.index else 1.0
        final_hybrid_score = raw_irt_score * model_map
        
        model_scores.append({
            'Model_Name': model_name,
            'Raw_IRT_Ability': raw_irt_score,
            'mAP50': model_map,
            'Precision': precision_vector[idx],
            'Final_Hybrid_Score': final_hybrid_score
        })
        
    model_eval_df = pd.DataFrame(model_scores)
    model_eval_df = model_eval_df.sort_values(by='Final_Hybrid_Score', ascending=False).reset_index(drop=True)
    
    model_eval_df.to_csv("calibrated_model_evaluations.csv", index=False)
    print("Saved evaluations to 'calibrated_model_evaluations.csv'.")
    print("\nTop 5 Models by Hybrid IRT Score:")
    print(model_eval_df.head(5))

if __name__ == "__main__":
    # Example usage. You must create 'model_metrics.csv' containing the global 
    # Precision and mAP for all 62 models before running this.
    
    # Format of model_metrics.csv should be:
    # Model_Name,Precision,mAP50
    # yolov8n_SSDD_epoch0,0.85,0.72
    # ...
    
    run_calibrated_irt("ship_detection_ground_truths.csv", "model_metrics.csv")
