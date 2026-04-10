import numpy as np
import pandas as pd
from girth import grm_mml
import warnings

# Suppress warnings from girth optimizer if items have zero variance 
# (e.g., ships detected by ALL models with 1.0 confidence)
warnings.filterwarnings('ignore')

def polytomize_confidence(confidence_matrix, bins=5):
    """
    Discretizes continuous [0, 1] confidence scores into integer categories (1 to bins).
    """
    edges = np.linspace(0, 1, bins + 1)
    categorized = np.digitize(confidence_matrix, edges[:-1])
    categorized = np.clip(categorized, 1, bins)
    return categorized

def run_irt_and_evaluate(csv_filepath, n_bins=5):
    """
    Loads SAR ship detection confidence data, calculates IRT item parameters,
    and evaluates model latent ability based on difficulty and discrimination.
    """
    print(f"Loading data from {csv_filepath}...")
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filepath}. Please ensure the file is in the same directory.")
        return
    
    # Extract metadata and model columns
    metadata_cols = ['image_path', 'gt_bbox']
    model_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"Found {len(df)} ships and {len(model_cols)} models.")
    
    # Extract confidence matrix. Shape: (n_ships, n_models)
    confidence_matrix = df[model_cols].values
    
    # Ensure no NaNs exist (fill missing detections with 0.0)
    confidence_matrix = np.nan_to_num(confidence_matrix, nan=0.0)
    
    # 1. Run Graded Response Model (GRM) to get Item Parameters
    print(f"Polytomizing confidence into {n_bins} categories...")
    ordinal_data = polytomize_confidence(confidence_matrix, bins=n_bins)
    
    print("Fitting Graded Response Model (MML). Utilizing CPU vectorization...")
    # Note: With 8846 items, this will utilize your Xeon CPU heavily for a few minutes.
    results_grm = grm_mml(ordinal_data)
    
    # Extract Parameters
    a_grm = results_grm['Discrimination']
    b_thresholds = results_grm['Difficulty']
    
    # Calculate a single mean difficulty for each ship across the ordinal thresholds
    b_mean_grm = np.mean(b_thresholds, axis=1)
    
    # Normalize difficulty to [0, 1] for our continuous scoring heuristic
    # We use min-max scaling. (b - min) / (max - min)
    b_min, b_max = np.min(b_mean_grm), np.max(b_mean_grm)
    b_norm = (b_mean_grm - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(b_mean_grm)
    
    # Save Ship IRT Parameters
    ship_metrics = df[metadata_cols].copy()
    ship_metrics['Discrimination_a'] = a_grm
    ship_metrics['Raw_Difficulty_b'] = b_mean_grm
    ship_metrics['Normalized_Difficulty_b'] = b_norm
    
    ship_metrics.to_csv("ship_irt_parameters.csv", index=False)
    print("Saved ship difficulty parameters to 'ship_irt_parameters.csv'.")

    # 2. Evaluate Models using IRT Parameters and Continuous Confidence
    print("Evaluating model performance using Weighted IRT Scoring...")
    
    model_scores = []
    
    for idx, model_name in enumerate(model_cols):
        model_confidence = confidence_matrix[:, idx]
        
        # Vectorized implementation of the scoring formula:
        # Score = Sum( a * (Confidence * b_norm - (1 - Confidence) * (1 - b_norm)) )
        
        rewards = model_confidence * b_norm
        penalties = (1.0 - model_confidence) * (1.0 - b_norm)
        
        # Apply discrimination weighting
        item_scores = a_grm * (rewards - penalties)
        
        # Aggregate score
        total_score = np.sum(item_scores)
        
        # Also calculate raw detection rate (Confidence > 0.5) for comparison
        raw_detections = np.sum(model_confidence > 0.5)
        detection_rate = raw_detections / len(model_confidence)
        
        model_scores.append({
            'Model_Name': model_name,
            'Total_IRT_Score': total_score,
            'Mean_IRT_Score_Per_Ship': total_score / len(model_confidence),
            'Raw_Detection_Rate': detection_rate
        })
        
    model_eval_df = pd.DataFrame(model_scores)
    
    # Sort by the new IRT score descending
    model_eval_df = model_eval_df.sort_values(by='Total_IRT_Score', ascending=False).reset_index(drop=True)
    
    model_eval_df.to_csv("model_evaluations_irt.csv", index=False)
    print("Saved model evaluations to 'model_evaluations_irt.csv'.")
    
    print("\nTop 10 Models by IRT Latent Ability Score:")
    print(model_eval_df[['Model_Name', 'Total_IRT_Score', 'Raw_Detection_Rate']].head(10))
    
    print("\nBottom 5 Models by IRT Latent Ability Score:")
    print(model_eval_df[['Model_Name', 'Total_IRT_Score', 'Raw_Detection_Rate']].tail(5))

if __name__ == "__main__":
    # Ensure your CSV is named exactly as below and in the same directory
    run_irt_and_evaluate("ship_detection_ground_truths.csv")
