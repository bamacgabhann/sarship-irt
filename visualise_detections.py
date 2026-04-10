import cv2
import pandas as pd
import os
import ast

# --- Configuration ---
IMAGES_DIR = "/home/breandan/sarship/tests_ssdd_hrsid_ls-ssdd/test/images"
OUTPUT_DIR = "/home/breandan/sarship/test_vis"
GT_CSV = "ship_detection_ground_truths.csv"
FP_CSV = "ship_detection_false_positives.csv"


# The operational confidence threshold for a "successful" detection
CONF_THRESH = 0.5 

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Function: Draw Text with Background ---
def draw_label(img, text, x, y, text_color, bg_color):
    """Draws text with a solid background so it is readable against SAR speckle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    # Get text size to draw the background rectangle
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background box and text
    cv2.rectangle(img, (x, y - text_h - 2), (x + text_w, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    
    # Return the next Y position to stack the next label below/above it
    return y + text_h + 4 

def generate_visualizations():
    # 1. Load the data
    print("Loading CSV data...")
    df_gt = pd.read_csv(GT_CSV)
    df_fp = pd.read_csv(FP_CSV)

    # Dynamically identify the model columns (ignoring the path and bbox columns)
    models_gt = [c for c in df_gt.columns if c not in ['image_path', 'gt_bbox']]
    models_fp = [c for c in df_fp.columns if c not in ['image_path', 'fp_bbox']]
    models = list(set(models_gt) | set(models_fp))

    # Get a unique list of all images referenced in either CSV
    all_images = set(df_gt['image_path']).union(set(df_fp['image_path']))
    
    saved_count = 0

    print(f"Analyzing {len(all_images)} images for visualization...")
    
    for img_name in all_images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        img_gt_data = df_gt[df_gt['image_path'] == img_name]
        img_fp_data = df_fp[df_fp['image_path'] == img_name]
        
        needs_saving = False
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue
            
        h, w, _ = img.shape
        
        # --- Task 1: Ground Truths & Failed Detections (White) ---
        for _, row in img_gt_data.iterrows():
            # Convert string representation of list back to actual list
            bbox = ast.literal_eval(row['gt_bbox'])
            
            # Convert normalized xyxy coordinates to actual pixel values
            x1, y1 = int(bbox * w), int(bbox[1] * h)
            x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
            
            failed_models =
            for m in models:
                conf = row.get(m, 0.0)
                if conf < CONF_THRESH:
                    failed_models.append(f"{m} (Miss): {conf:.2f}")
            
            # If at least one model failed to detect it, draw the box
            if failed_models:
                needs_saving = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
                # Stack text labels above the bounding box
                text_y = y1 - 5 if y1 > 30 else y2 + 15
                for text in failed_models:
                    # White text on black background for missed detections
                    text_y = draw_label(img, text, x1, text_y, (255, 255, 255), (0, 0, 0))
                    
        # --- Task 2: False Positives (Red) ---
        for _, row in img_fp_data.iterrows():
            bbox = ast.literal_eval(row['fp_bbox'])
            x1, y1 = int(bbox * w), int(bbox[1] * h)
            x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
            
            active_fps =
            for m in models:
                conf = row.get(m, 0.0)
                # Only log it if the model's false positive was above your operational threshold
                if conf >= CONF_THRESH:
                    active_fps.append(f"{m} (FP): {conf:.2f}")
                    
            # If at least one model triggered a false positive here, draw the box
            if active_fps:
                needs_saving = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Stack text labels slightly below the top of the box to differentiate from misses
                text_y = y1 + 15 
                for text in active_fps:
                    # Red text on white background for high visibility
                    text_y = draw_label(img, text, x1, text_y, (0, 0, 255), (255, 255, 255))
                    
        # --- Save the Image ---
        # Only save if the image contains at least one error (Miss or FP)
        if needs_saving:
            out_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(out_path, img)
            saved_count += 1

    print(f"Done! Generated {saved_count} error visualizations in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    generate_visualizations()
