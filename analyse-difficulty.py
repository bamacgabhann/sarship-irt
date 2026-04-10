import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import scipy.stats as stats
import os


def analyze_explanatory_irt(cirt_csv, characteristics_csv):
    """
    Runs Explanatory IRT analysis to predict both Difficulty (b) and Discrimination (a)
    based on physical SAR ship characteristics. Saves outputs to CSV and TXT.
    """
    print("Loading datasets...")
    df_cirt = pd.read_csv(cirt_csv)
    df_chars = pd.read_csv(characteristics_csv)

    # Merge datasets
    df = pd.merge(df_cirt, df_chars, on=['image_path', 'gt_bbox'], how='inner')
    print(f"Successfully merged data for {len(df)} ships.")

    features = [
        'Length_px', 'Width_px', 'Area_px2', 'Aspect_Ratio',
        'Dist_to_Image_Edge_px', 'Nearest_Neighbor_Dist_px',
        'Target_Mean_Intensity', 'Target_Variance',
        'Background_Mean_Intensity', 'Background_Variance', 'Local_SCR_dB'
    ]

    # Clean distance to nearest neighbor (-1 means isolated)
    if 'Nearest_Neighbor_Dist_px' in df.columns:
        max_dist = df['Nearest_Neighbor_Dist_px'].max()
        df['Nearest_Neighbor_Dist_px'] = df['Nearest_Neighbor_Dist_px'].replace(-1, max_dist * 2)

    # Drop rows with NaNs in features or targets
    df = df.dropna(subset=features + ['CIRT_Raw_Difficulty_b', 'CIRT_Discrimination_a'])

    X = df[features]
    y_diff = df['CIRT_Raw_Difficulty_b']
    y_disc = df['CIRT_Discrimination_a']

    # Initialize lists to store results for saving
    correlation_records = []
    importance_records = []
    report_lines = ["=== Explanatory IRT Summary Report ===\n"]

    # ==========================================
    # 1. Pearson Correlation Analysis
    # ==========================================
    print("Running Pearson Correlation Analysis...")
    for feat in features:
        # Difficulty correlations
        r_diff, p_diff = stats.pearsonr(df[feat], y_diff)
        correlation_records.append({
            'Feature': feat, 'Target': 'Difficulty_b',
            'Pearson_r': r_diff, 'P_value': p_diff
        })

        # Discrimination correlations
        r_disc, p_disc = stats.pearsonr(df[feat], y_disc)
        correlation_records.append({
            'Feature': feat, 'Target': 'Discrimination_a',
            'Pearson_r': r_disc, 'P_value': p_disc
        })

    # NEW CODE:
    df_corr = pd.DataFrame(correlation_records)

    # Create a temporary column for absolute magnitude
    df_corr['abs_Pearson_r'] = df_corr['Pearson_r'].abs()

    # Sort first by Target (alphabetically), then by absolute correlation (descending)
    df_corr = df_corr.sort_values(by=['Target', 'abs_Pearson_r'], ascending=[True, False])

    # Drop the temporary column to keep the output clean
    df_corr = df_corr.drop(columns=['abs_Pearson_r'])

    df_corr.to_csv("eirm_correlations.csv", index=False)

    # ==========================================
    # 2. Random Forest Regression (Difficulty & Discrimination)
    # ==========================================
    print("Running Random Forest Regressions...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Utilizing all available cores on your Xeon W-2255
    rf_diff = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf_disc = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

    # Train-test splits
    X_train, X_test, yd_train, yd_test, ya_train, ya_test = train_test_split(
        X_scaled, y_diff, y_disc, test_size=0.2, random_state=42
    )

    # Fit and evaluate Difficulty
    rf_diff.fit(X_train, yd_train)
    yd_pred = rf_diff.predict(X_test)
    r2_diff = r2_score(yd_test, yd_pred)

    # Fit and evaluate Discrimination
    rf_disc.fit(X_train, ya_train)
    ya_pred = rf_disc.predict(X_test)
    r2_disc = r2_score(ya_test, ya_pred)

    # Document R-squared results
    report_lines.append(f"Random Forest R-squared for Difficulty (b): {r2_diff:.4f}")
    report_lines.append(f"Random Forest R-squared for Discrimination (a): {r2_disc:.4f}\n")

    # Extract importances
    for feat, imp_diff, imp_disc in zip(features, rf_diff.feature_importances_, rf_disc.feature_importances_):
        importance_records.append({
            'Feature': feat,
            'Importance_for_Difficulty': imp_diff,
            'Importance_for_Discrimination': imp_disc
        })

    # Save importances to CSV
    df_imp = pd.DataFrame(importance_records)
    df_imp = df_imp.sort_values(by='Importance_for_Difficulty', ascending=False)
    df_imp.to_csv("eirm_rf_importances.csv", index=False)

    # ==========================================
    # 3. Generate and Save Text Report
    # ==========================================
    report_lines.append("--- Top 5 Drivers of Ship Difficulty (b) ---")
    top_diff = df_imp.sort_values(by='Importance_for_Difficulty', ascending=False).head(5)
    for _, row in top_diff.iterrows():
        report_lines.append(f"{row['Feature']:30s}: {row['Importance_for_Difficulty'] * 100:.1f}%")

    report_lines.append("\n--- Top 5 Drivers of Ship Discrimination (a) ---")
    top_disc = df_imp.sort_values(by='Importance_for_Discrimination', ascending=False).head(5)
    for _, row in top_disc.iterrows():
        report_lines.append(f"{row['Feature']:30s}: {row['Importance_for_Discrimination'] * 100:.1f}%")

    with open("eirm_summary_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print("\nAnalysis complete. Outputs saved:")
    print(" - eirm_correlations.csv")
    print(" - eirm_rf_importances.csv")
    print(" - eirm_summary_report.txt")

    # Print summary to console for immediate review
    print("\n" + "\n".join(report_lines))

    return df_corr, df_imp


if __name__ == "__main__":
    analyze_explanatory_irt("cirt_ship_parameters.csv", "sarship_dataset_chars_SSDD.csv")
