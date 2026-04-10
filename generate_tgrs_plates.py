import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import ast
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SSDD_IMAGE_DIR = "/home/breandan/sarship-yolo26/SSDD/images"  # UPDATE THIS PATH
CIRT_CSV = "cirt_ship_parameters.csv"
CHARS_FEATHER = "sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.feather"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
})


def robust_spatial_merge_params(df_params, df_chars):
    """Merges the psychometric stats with the physical characteristics."""
    print("Merging psychometric and physical dataset characteristics...")

    def extract_normalized_centers(bbox):
        """Extracts the center (cx, cy) from normalized [xmin, ymin, xmax, ymax]"""
        try:
            if isinstance(bbox, str):
                clean_str = bbox.replace('[', '').replace(']', '').replace(',', ' ')
                coords = [float(x) for x in clean_str.split()]
            elif isinstance(bbox, (list, np.ndarray)):
                coords = [float(x) for x in bbox]
            else:
                return 0.0, 0.0

            if len(coords) >= 4:
                # Normalized center: (xmin + xmax) / 2
                return (coords[0] + coords[2]) / 2.0, (coords[1] + coords[3]) / 2.0
            return 0.0, 0.0
        except:
            return 0.0, 0.0

    df_params['temp_norm_cx_cy'] = df_params['gt_bbox'].apply(extract_normalized_centers)
    df_chars['temp_norm_cx_cy'] = df_chars['gt_bbox'].apply(extract_normalized_centers)

    param_idx_list, char_idx_list = [], []
    for img in df_params['image_path'].unique():
        p_mask = df_params['image_path'] == img
        c_mask = df_chars['image_path'] == img

        p_idx, c_idx = df_params.index[p_mask].tolist(), df_chars.index[c_mask].tolist()
        if not p_idx or not c_idx: continue

        p_centers = np.array(df_params.loc[p_idx, 'temp_norm_cx_cy'].tolist())
        c_centers = np.array(df_chars.loc[c_idx, 'temp_norm_cx_cy'].tolist())

        # Because centers are normalized [0, 1], a distance of 0.05 is 5% of the image size
        dist_matrix = cdist(p_centers, c_centers)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        for i, j in zip(row_ind, col_ind):
            if dist_matrix[i, j] < 0.05:  # Tightened threshold for normalized space
                param_idx_list.append(p_idx[i])
                char_idx_list.append(c_idx[j])

    df_p_matched = df_params.loc[param_idx_list].reset_index(drop=True)
    df_c_matched = df_chars.loc[char_idx_list].reset_index(drop=True)
    cols_to_drop = [c for c in df_c_matched.columns if c in df_p_matched.columns]
    df_merged = pd.concat([df_p_matched, df_c_matched.drop(columns=cols_to_drop)], axis=1)
    df_merged.drop(columns=['temp_norm_cx_cy'], inplace=True, errors='ignore')
    return df_merged


def parse_normalized_bbox(bbox_str):
    """Safely extracts 4-point normalized HBB coordinates."""
    try:
        if isinstance(bbox_str, str):
            clean_str = bbox_str.replace('[', '').replace(']', '').replace(',', ' ')
            coords = [float(x) for x in clean_str.split()]
        elif isinstance(bbox_str, (list, np.ndarray)):
            coords = [float(x) for x in bbox_str]
        else:
            return None

        if len(coords) >= 4:
            return coords[:4]  # [xmin, ymin, xmax, ymax]
    except:
        pass
    return None


def plot_ship_chip(ax, row, pad=50):
    """Loads image, denormalizes coordinates, crops around ship, and draws metadata."""
    img_name = os.path.basename(row['image_path'])
    img_path = os.path.join(SSDD_IMAGE_DIR, img_name)

    diff = row.get('CIRT_Raw_Difficulty_b', 0.0)
    disc = row.get('CIRT_Discrimination_a', 0.0)

    try:
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.width, img.height

        norm_coords = parse_normalized_bbox(row['gt_bbox'])

        if norm_coords:
            # 1. Denormalize to absolute pixels
            xmin = norm_coords[0] * img_w
            ymin = norm_coords[1] * img_h
            xmax = norm_coords[2] * img_w
            ymax = norm_coords[3] * img_h

            # 2. Find absolute center for cropping
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0

            # 3. Define crop boundaries (safeguarded against image edges)
            left, right = max(0, int(cx - pad)), min(img_w, int(cx + pad))
            top, bottom = max(0, int(cy - pad)), min(img_h, int(cy + pad))

            img_cropped = img.crop((left, top, right, bottom))
            ax.imshow(img_cropped)

            # 4. Shift bounding box to local crop coordinates
            local_xmin = xmin - left
            local_ymin = ymin - top
            box_w = xmax - xmin
            box_h = ymax - ymin

            # 5. Draw the Horizontal Bounding Box
            rect = patches.Rectangle((local_xmin, local_ymin), box_w, box_h,
                                     linewidth=1.5, edgecolor='#00FF00', facecolor='none', zorder=2)
            ax.add_patch(rect)

        else:
            # Fallback if bbox parsing fails
            ax.imshow(img)
            ax.text(0.5, 0.5, "BBox Error", color="red", ha="center")

        # Psychometric Overlay (Top Left)
        stat_text = f"b: {diff:.2f}\na: {disc:.2f}"
        ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, color='white',
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, pad=0.2))

    except Exception as e:
        ax.text(0.5, 0.5, "Image Missing", color="red", ha="center")

    ax.axis('off')


def generate_ieee_plate(df, queries, labels, filename, title):
    """Generates the 5x2 IEEE single-column figure."""
    print(f"\nGenerating {filename}...")
    fig, axes = plt.subplots(5, 2, figsize=(3.5, 8.0), dpi=300)
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.98)

    for row_idx, (query, label) in enumerate(zip(queries, labels)):
        try:
            subset = df.query(query)
            sort_asc = False if "High" in title else True
            top_ships = subset.sort_values(by='CIRT_Raw_Difficulty_b', ascending=sort_asc).head(2)

            for col_idx in range(2):
                ax = axes[row_idx, col_idx]
                if col_idx < len(top_ships):
                    ship = top_ships.iloc[col_idx]
                    plot_ship_chip(ax, ship)

                    if col_idx == 0:
                        ax.text(0.95, 0.05, label, transform=ax.transAxes, color='yellow',
                                fontsize=7, horizontalalignment='right', fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, pad=0.2))
                else:
                    ax.axis('off')
        except Exception as e:
            print(f"Query Failed: {query} | Error: {e}")
            axes[row_idx, 0].axis('off')
            axes[row_idx, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved -> {filename}")


if __name__ == "__main__":
    df_params = pd.read_csv(CIRT_CSV)
    df_chars = pd.read_feather(CHARS_FEATHER)
    df = robust_spatial_merge_params(df_params, df_chars)

    # -----------------------------------------------------------------
    # FIGURE 1: HIGH DIFFICULTY
    # -----------------------------------------------------------------
    df_hard = df[(df['CIRT_Raw_Difficulty_b'] > 0.6) & (df['CIRT_Discrimination_a'] > 1.0)]

    queries_fig1 = [
        "bbox_bg_variance > 2000",
        "bbox_bg_variance > 700 & bbox_bg_variance < 800",
        "rotated_bbox_aspect > 4.5",
        "nearest_neighbor_px < 50",
        "rotated_bbox_area < 500"
    ]
    labels_fig1 = [
        "Var > 2000",
        "Wake (Var ~750)",
        "Aspect > 4.5",
        "Neighbor < 50px",
        "Area < 500px"
    ]

    generate_ieee_plate(df_hard, queries_fig1, labels_fig1,
                        "fig_high_difficulty_plate.pdf", "Resonant Failure Modes (High Difficulty)")

    # -----------------------------------------------------------------
    # FIGURE 2: LOW DIFFICULTY
    # -----------------------------------------------------------------
    df_easy = df[(df['CIRT_Raw_Difficulty_b'] < -0.2)]

    queries_fig2 = [
        "rotated_bbox_aspect > 1.4 & rotated_bbox_aspect < 1.6",
        "bbox_bg_variance < 200 & seg_SCR > 15",
        "nearest_neighbor_px > 200",
        "rotated_bbox_area > 1500 & rotated_bbox_area < 2000",
        "rotated_bbox_aspect > 1.4 & rotated_bbox_aspect < 1.6 & bbox_bg_variance < 200"
    ]
    labels_fig2 = [
        "Aspect ~ 1.5",
        "High SCR, Low Var",
        "Neighbor > 200px",
        "Optimal Area",
        "Ideal Baseline"
    ]

    generate_ieee_plate(df_easy, queries_fig2, labels_fig2,
                        "fig_low_difficulty_plate.pdf", "Optical Priors (Low Difficulty)")