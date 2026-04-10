import pandas as pd
import numpy as np
from scipy import stats

# Load data
metrics_df = pd.read_csv('yolo_trained_on_ssdd_checkpoint_metrics.csv')
sens_df = pd.read_csv('obb_model_feature_sensitivities.csv')

# Merge datasets
metrics_df.rename(columns={'checkpoint': 'Model_Name'}, inplace=True)
#df = pd.merge(metrics_df, sens_df, on='Model_Name', how='inner')
df_final = pd.merge(metrics_df, sens_df, on='Model_Name', how='inner')

# Filter for fully trained models (e.g., last 10 epochs)
#df_final = df[df['epoch'] >= 490].copy()


# Calculate Generalization Gaps (Performance Drop)
# A larger gap means worse OOD generalization
df_final['Gap_HRSID'] = df_final['mAP50_SSDD'] - df_final['mAP50_HRSID']
df_final['Gap_LS-SSDD'] = df_final['mAP50_SSDD'] - df_final['mAP50_LS-SSDD']

output = []
output.append("=== OUT-OF-DISTRIBUTION (OOD) GENERALIZATION ANALYSIS ===")

# Group by Architecture and Size to see average drops
arch_summary = df_final.groupby(['family', 'size'])[['mAP50_SSDD', 'mAP50_HRSID', 'Gap_HRSID', 'mAP50_LS-SSDD', 'Gap_LS-SSDD']].mean().reset_index()
arch_summary = arch_summary.sort_values(by='Gap_HRSID', ascending=True)

output.append("\nAverage Generalization Gap by Architecture (Sorted by Best HRSID Retention):")
output.append(arch_summary.to_string())

# Correlate Physical Sensitivities with the Generalization Gap
sensitivities = [
'rotated_bbox_aspect',  # Pure Geometric Elongation
'rotated_bbox_theta', # Angle
'bbox_area',  # Pure Physical Capacity
'bbox_bg_variance',  # Corner Regression Instability
'seg_SCR',  # Pure Target Contrast
'image_total_pixels',  # Image size / rescaling
'nearest_neighbor_px',  # Spatial NMS Interference
'dist_to_edge_px',  # Boundary Padding Constraints
]
output.append("\n\n=== CORRELATING SENSITIVITIES WITH DOMAIN SHIFT PERFORMANCE DROP ===")
output.append("Note: A positive correlation means higher sensitivity to a feature predicts a LARGER performance drop (worse generalization).")

corr_records = []
for sens in sensitivities:
    # Correlation with HRSID Drop
    r_hrsid, p_hrsid = stats.pearsonr(df_final[sens], df_final['Gap_HRSID'])
    # Correlation with LS-SSDD Drop
    r_ls, p_ls = stats.pearsonr(df_final[sens], df_final['Gap_LS-SSDD'])
    corr_records.append({
        'Physical_Sensitivity': sens,
        'Pearson_r_with_HRSID_Gap': r_hrsid,
        'Pearson_r_with_LS-SSDD_Gap': r_ls
    })

corr_df = pd.DataFrame(corr_records).sort_values(by='Pearson_r_with_HRSID_Gap', ascending=False)
corr_df.to_csv('corr_records.csv', index=False)
output.append("\n" + corr_df.to_string())

print("\n".join(output))