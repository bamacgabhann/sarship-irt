import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import warnings

try:
    from sar_irt.utils.core import set_ieee_tgrs_style
except ImportError:
    def set_ieee_tgrs_style():
        plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
                             "legend.fontsize": 11})

warnings.filterwarnings('ignore')


def identify_pareto_frontier(df, x_col, y_col):
    """
    Identifies the Pareto optimal models.
    Goal: Minimize x_col (FP Burden) and Maximize y_col (TP Ability).
    """
    # Sort by FP Burden (ascending), then by TP Ability (descending)
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[True, False])

    pareto_front = []
    max_y_so_far = -np.inf

    for index, row in sorted_df.iterrows():
        # A point is on the front if its TP Ability is strictly greater than
        # the TP Ability of any point with a lower FP Burden.
        if row[y_col] > max_y_so_far:
            pareto_front.append(index)
            max_y_so_far = row[y_col]

    return df.loc[pareto_front]


def plot_pareto_front(evals_csv, robustness_csv):
    """
    Merges TP CIRT and FP Susceptibility, calculates the Pareto Front,
    and plots the optimal trade-off curve for IEEE TGRS.
    """
    print("Loading TP and FP evaluation datasets...")
    df_tp = pd.read_csv(evals_csv)
    df_fp = pd.read_csv(robustness_csv)

    # The FP robustness CSV has 'Raw_Model_Col' (e.g. yolov8n_SSDD_epoch490).
    # We use this to join precisely with 'Model_Name' in the TP evaluations.
    df_merged = pd.merge(df_tp, df_fp, left_on='Model_Name', right_on='Raw_Model_Col', how='inner')

    if len(df_merged) == 0:
        print("Error: Could not merge datasets. Ensure the optimal epochs match exactly.")
        return

    print(f"Successfully unified evaluation for {len(df_merged)} optimal checkpoints.")

    # Normalizing scores for fair visual representation
    # Higher is better for TP Ability
    df_merged['TP_Ability'] = df_merged['Final_Global_Hybrid_Score']
    # Lower is better for FP Susceptibility
    df_merged['FP_Susceptibility'] = df_merged['Total_FP_Volume']

    # Calculate Pareto Front
    pareto_df = identify_pareto_frontier(df_merged, 'FP_Susceptibility', 'TP_Ability')

    # --- PLOTTING ---
    set_ieee_tgrs_style()
    plt.figure(figsize=(12, 9))

    # Color mapping for architecture generations
    colors = {'v8': '#1f77b4', 'v9': '#9467bd', '10': '#d62728', '11': '#ff7f0e', '12': '#2ca02c', '26': '#7f7f7f'}

    def get_color(arch_name):
        for key, color in colors.items():
            if key in arch_name.lower(): return color
        return '#333333'

    df_merged['Color'] = df_merged['Architecture'].apply(get_color)

    # Plot all sub-optimal points
    sub_optimal = df_merged[~df_merged.index.isin(pareto_df.index)]
    plt.scatter(sub_optimal['FP_Susceptibility'], sub_optimal['TP_Ability'],
                c=sub_optimal['Color'], alpha=0.4, s=80, edgecolors='none', label='Sub-optimal Checkpoints')

    # Plot Pareto optimal points
    plt.scatter(pareto_df['FP_Susceptibility'], pareto_df['TP_Ability'],
                c=pareto_df['Color'], alpha=1.0, s=150, edgecolors='black', linewidths=1.5, zorder=5)

    # Draw Pareto Step Curve
    pareto_sorted = pareto_df.sort_values('FP_Susceptibility')
    plt.plot(pareto_sorted['FP_Susceptibility'], pareto_sorted['TP_Ability'],
             color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=4, label='Pareto Frontier')

    # Annotate Pareto Models
    for _, row in pareto_sorted.iterrows():
        # Clean name, e.g., yolov12x_SSDD_epoch490 -> YOLO12x
        clean_name = row['Model_Name'].split('_')[0].upper()
        plt.annotate(clean_name,
                     (row['FP_Susceptibility'], row['TP_Ability']),
                     xytext=(10, -5), textcoords='offset points',
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title('IRT Pareto Frontier: Target Detection vs. Clutter Rejection', fontsize=16, pad=15)
    plt.xlabel('Distractor Susceptibility (Total FP Volume) $\\rightarrow$ Lower is Better', fontsize=14)
    plt.ylabel('True Positive Ability (Hybrid CIRT Score) $\\rightarrow$ Higher is Better', fontsize=14)

    # Highlight the "Ideal" Region
    plt.text(0.05, 0.95, "Ideal Operating Zone\n(High Recall, Zero False Alarms)",
             transform=plt.gca().transAxes, fontsize=12, color='green',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='red', linestyle='--', lw=2, label='Pareto Frontier'),
        Patch(facecolor='#2ca02c', label='YOLO12'),
        Patch(facecolor='#ff7f0e', label='YOLO11'),
        Patch(facecolor='#d62728', label='YOLOv10'),
        Patch(facecolor='#9467bd', label='YOLOv9'),
        Patch(facecolor='#1f77b4', label='YOLOv8'),
        Patch(facecolor='#7f7f7f', label='YOLO26')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('irt_pareto_front_evaluation.pdf', format='pdf', dpi=300)
    print("Saved Unified Pareto Evaluation plot to 'irt_pareto_front_evaluation.pdf'.")

    # Save the Pareto optimal models to a CSV
    pareto_sorted[['Model_Name', 'TP_Ability', 'FP_Susceptibility']].to_csv('irt_pareto_optimal_models.csv',
                                                                            index=False)
    print("Saved list of Pareto Optimal Models to 'irt_pareto_optimal_models.csv'.")


if __name__ == "__main__":
    EVALS_CSV = "cirt_model_evaluations.csv"
    ROBUSTNESS_CSV = "architecture_robustness_ranking.csv"  # The output from distractor_irt.py

    plot_pareto_front(EVALS_CSV, ROBUSTNESS_CSV)