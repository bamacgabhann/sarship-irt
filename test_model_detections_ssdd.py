import csv
import os
import gc
import ast
import torch
import pandas as pd
from ultralytics import YOLO

# --- Configuration ---
architectures = [
    'yolo26x', 'yolo12x','yolo11x', 'yolov10x', 'yolov9e', 'yolov8x',
    'yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo12n', 'yolo26n',
    'yolov8s', 'yolov9s', 'yolov10s', 'yolo11s', 'yolo12s', 'yolo26s',
    'yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m', 'yolo26m',
    'yolov8l', 'yolov9c', 'yolov10b', 'yolov10l', 'yolo11l', 'yolo12l', 'yolo26l'
]

SSDD_MODEL_PATHS = {}
for a in architectures:
    for epoch in range(500):
        model_name = f'{a}_SSDD_epoch{epoch}'
        model_fname = f'epoch{epoch}.pt'
        model_froot = f'/media/breandan/Windows/YOLO_checkpoints/{a}_SSDD_weights'
        model_path = os.path.join(model_froot, model_fname)
        # Verify path exists before adding to prevent errors during the long run
        if os.path.exists(model_path):
            SSDD_MODEL_PATHS[model_name] = model_path

SSDD_IMAGES_DIR = "/home/breandan/sarship-yolo26/SSDD/images"
SSDD_LABELS_DIR = "/home/breandan/sarship-yolo26/SSDD/labels"
SSDD_GT_CSV = "ship_detection_ground_truths_ssdd.csv"
SSDD_FP_CSV = "ship_detection_false_positives_ssdd.csv"

HRSID_MODEL_PATHS = {}
for a in architectures:
    for epoch in range(500):
        model_name = f'{a}_HRSID_epoch{epoch}'
        model_fname = f'epoch{epoch}.pt'
        model_froot = f'/media/breandan/Windows/YOLO_checkpoints/{a}_HRSID_weights'
        model_path = os.path.join(model_froot, model_fname)
        # Verify path exists before adding to prevent errors during the long run
        if os.path.exists(model_path):
            HRSID_MODEL_PATHS[model_name] = model_path

HRSID_IMAGES_DIR = "/home/breandan/sarship-yolo26/HRSID/images"
HRSID_LABELS_DIR = "/home/breandan/sarship-yolo26/HRSID/labels"
HRSID_GT_CSV = "ship_detection_ground_truths_hrsid.csv"
HRSID_FP_CSV = "ship_detection_false_positives_hrsid.csv"

complete = [
'yolo26x_SSDD_epoch0',
'yolo26x_SSDD_epoch1',
'yolo26x_SSDD_epoch2',
'yolo26x_SSDD_epoch3',
'yolo26x_SSDD_epoch4',
'yolo26x_SSDD_epoch5',
'yolo26x_SSDD_epoch6',
'yolo26x_SSDD_epoch7',
'yolo26x_SSDD_epoch8',
'yolo26x_SSDD_epoch9',
'yolo26x_SSDD_epoch10',
'yolo26x_SSDD_epoch11',
'yolo26x_SSDD_epoch12',
'yolo26x_SSDD_epoch13',
'yolo26x_SSDD_epoch14',
'yolo26x_SSDD_epoch15',
'yolo26x_SSDD_epoch16',
'yolo26x_SSDD_epoch17',
'yolo26x_SSDD_epoch18',
'yolo26x_SSDD_epoch19',
'yolo26x_SSDD_epoch20',
'yolo26x_SSDD_epoch21',
'yolo26x_SSDD_epoch22',
'yolo26x_SSDD_epoch23',
'yolo26x_SSDD_epoch24',
'yolo26x_SSDD_epoch25',
'yolo26x_SSDD_epoch26',
'yolo26x_SSDD_epoch27',
'yolo26x_SSDD_epoch28',
'yolo26x_SSDD_epoch29',
'yolo26x_SSDD_epoch30',
'yolo26x_SSDD_epoch31',
'yolo26x_SSDD_epoch32',
'yolo26x_SSDD_epoch33',
'yolo26x_SSDD_epoch34',
'yolo26x_SSDD_epoch35',
'yolo26x_SSDD_epoch36',
'yolo26x_SSDD_epoch37',
'yolo26x_SSDD_epoch38',
'yolo26x_SSDD_epoch39',
'yolo26x_SSDD_epoch40',
'yolo26x_SSDD_epoch41',
'yolo26x_SSDD_epoch42',
'yolo26x_SSDD_epoch43',
'yolo26x_SSDD_epoch44',
'yolo26x_SSDD_epoch45',
'yolo26x_SSDD_epoch46',
'yolo26x_SSDD_epoch47',
'yolo26x_SSDD_epoch48',
'yolo26x_SSDD_epoch49',
'yolo26x_SSDD_epoch50',
'yolo26x_SSDD_epoch51',
'yolo26x_SSDD_epoch52',
'yolo26x_SSDD_epoch53',
'yolo26x_SSDD_epoch54',
'yolo26x_SSDD_epoch55',
'yolo26x_SSDD_epoch56',
'yolo26x_SSDD_epoch57',
'yolo26x_SSDD_epoch58',
'yolo26x_SSDD_epoch59',
'yolo26x_SSDD_epoch60',
'yolo26x_SSDD_epoch61',
'yolo26x_SSDD_epoch62',
'yolo26x_SSDD_epoch63',
'yolo26x_SSDD_epoch64',
'yolo26x_SSDD_epoch65',
'yolo26x_SSDD_epoch66',
'yolo26x_SSDD_epoch67',
'yolo26x_SSDD_epoch68',
'yolo26x_SSDD_epoch69',
'yolo26x_SSDD_epoch70',
'yolo26x_SSDD_epoch71',
'yolo26x_SSDD_epoch72',
'yolo26x_SSDD_epoch73',
'yolo26x_SSDD_epoch74',
'yolo26x_SSDD_epoch75',
'yolo26x_SSDD_epoch76',
'yolo26x_SSDD_epoch77',
'yolo26x_SSDD_epoch78',
'yolo26x_SSDD_epoch79',
'yolo26x_SSDD_epoch80',
'yolo26x_SSDD_epoch81',
'yolo26x_SSDD_epoch82',
'yolo26x_SSDD_epoch83',
'yolo26x_SSDD_epoch84',
'yolo26x_SSDD_epoch85',
'yolo26x_SSDD_epoch86',
'yolo26x_SSDD_epoch87',
'yolo26x_SSDD_epoch88',
'yolo26x_SSDD_epoch89',
'yolo26x_SSDD_epoch90',
'yolo26x_SSDD_epoch91',
'yolo26x_SSDD_epoch92',
'yolo26x_SSDD_epoch93',
'yolo26x_SSDD_epoch94',
'yolo26x_SSDD_epoch95',
'yolo26x_SSDD_epoch96',
'yolo26x_SSDD_epoch97',
'yolo26x_SSDD_epoch98',
'yolo26x_SSDD_epoch99',
'yolo26x_SSDD_epoch100',
'yolo26x_SSDD_epoch101',
'yolo26x_SSDD_epoch102',
'yolo26x_SSDD_epoch103',
'yolo26x_SSDD_epoch104',
'yolo26x_SSDD_epoch105',
'yolo26x_SSDD_epoch106',
'yolo26x_SSDD_epoch107',
'yolo26x_SSDD_epoch108',
'yolo26x_SSDD_epoch109',
'yolo26x_SSDD_epoch110',
'yolo26x_SSDD_epoch111',
'yolo26x_SSDD_epoch112',
'yolo26x_SSDD_epoch113',
'yolo26x_SSDD_epoch114',
'yolo26x_SSDD_epoch115',
'yolo26x_SSDD_epoch116',
'yolo26x_SSDD_epoch117',
'yolo26x_SSDD_epoch118',
'yolo26x_SSDD_epoch119',
'yolo26x_SSDD_epoch120',
'yolo26x_SSDD_epoch121',
'yolo26x_SSDD_epoch122',
'yolo26x_SSDD_epoch123',
'yolo26x_SSDD_epoch124',
'yolo26x_SSDD_epoch125',
'yolo26x_SSDD_epoch126',
'yolo26x_SSDD_epoch127',
'yolo26x_SSDD_epoch128',
'yolo26x_SSDD_epoch129',
'yolo26x_SSDD_epoch130',
'yolo26x_SSDD_epoch131',
'yolo26x_SSDD_epoch132',
'yolo26x_SSDD_epoch133',
'yolo26x_SSDD_epoch134',
'yolo26x_SSDD_epoch135',
'yolo26x_SSDD_epoch136',
'yolo26x_SSDD_epoch137',
'yolo26x_SSDD_epoch138',
'yolo26x_SSDD_epoch139',
'yolo26x_SSDD_epoch140',
'yolo26x_SSDD_epoch141',
'yolo26x_SSDD_epoch142',
'yolo26x_SSDD_epoch143',
'yolo26x_SSDD_epoch144',
'yolo26x_SSDD_epoch145',
'yolo26x_SSDD_epoch146',
'yolo26x_SSDD_epoch147',
'yolo26x_SSDD_epoch148',
'yolo26x_SSDD_epoch149',
'yolo26x_SSDD_epoch150',
'yolo26x_SSDD_epoch151',
'yolo26x_SSDD_epoch152',
'yolo26x_SSDD_epoch153',
'yolo26x_SSDD_epoch154',
'yolo26x_SSDD_epoch155',
'yolo26x_SSDD_epoch156',
'yolo26x_SSDD_epoch157',
'yolo26x_SSDD_epoch158',
'yolo26x_SSDD_epoch159',
'yolo26x_SSDD_epoch160',
'yolo26x_SSDD_epoch161',
'yolo26x_SSDD_epoch162',
'yolo26x_SSDD_epoch163',
'yolo26x_SSDD_epoch164',
'yolo26x_SSDD_epoch165',
'yolo26x_SSDD_epoch166',
'yolo26x_SSDD_epoch167',
'yolo26x_SSDD_epoch168',
'yolo26x_SSDD_epoch169',
'yolo26x_SSDD_epoch170',
'yolo26x_SSDD_epoch171',
'yolo26x_SSDD_epoch172',
'yolo26x_SSDD_epoch173',
'yolo26x_SSDD_epoch174',
'yolo26x_SSDD_epoch175',
'yolo26x_SSDD_epoch176',
'yolo26x_SSDD_epoch177'
]

IOU_THRESHOLD = 0.5

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def run_granular_evaluation(MODEL_PATHS, IMAGES_DIR, LABELS_DIR, GT_CSV, FP_CSV):
    # 1. Pre-load all Ground Truths into memory
    print("Pre-loading Ground Truths...")
    image_files =[f for f in os.listdir(IMAGES_DIR)]
    image_paths =[os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)]


    gt_records = []
    gt_from_csv = False
    for img_file in image_files:
        img_path = os.path.join(IMAGES_DIR, img_file)
        label_path = os.path.join(LABELS_DIR, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, x, y, w, h = map(float, parts)
                        x1, y1 = x - w/2, y - h/2
                        x2, y2 = x + w/2, y + h/2

                        record = {"image_path": img_file, "gt_bbox": [x1, y1, x2, y2]}
                        # Initialize all model columns to 0.0
                        for m_name in MODEL_PATHS.keys():
                            record[m_name] = 0.0
                        gt_records.append(record)

    fp_clusters = []

# 2. Iterate through models (Load each EXACTLY ONCE)
    for m_name, model_path in MODEL_PATHS.items():
        if m_name in complete:
            continue
        print(f"\nEvaluating Model: {m_name}")
        model = YOLO(model_path)

        # Load current state from disk into memory for this model's run
        if not gt_from_csv:
            if os.path.exists(GT_CSV):
                df_gt = pd.read_csv(GT_CSV)
                df_gt['gt_bbox'] = df_gt['gt_bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                gt_records = df_gt.to_dict('records')
                del df_gt
        if not fp_clusters:
            if os.path.exists(FP_CSV):
                df_fp = pd.read_csv(FP_CSV)
                if not df_fp.empty:
                    df_fp['fp_bbox'] = df_fp['fp_bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                fp_clusters = df_fp.to_dict('records')

        # 3. Inner loop: Iterate through images one by one
        # This prevents both the OS file limit error and the massive VRAM padding spikes
        for img_file in image_files:
            img_path = os.path.join(IMAGES_DIR, img_file)

            # Run inference on the single image
            results = model.predict(source=img_path, conf=0.001, iou=0.45, verbose=False)
            predictions = []
            for result in results:
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxyn.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        predictions.append({"box": box.tolist(), "conf": float(conf)})

            # Isolate the Ground Truths for this specific image
            img_gts = [gt for gt in gt_records if gt["image_path"] == img_file]

            # Match Predictions to Ground Truths
            matched_pred_indices = set()
            for gt_record in img_gts:
                best_iou = 0
                best_conf = 0.0
                best_pred_idx = -1

                for p_idx, pred in enumerate(predictions):
                    iou = compute_iou(gt_record["gt_bbox"], pred["box"])
                    if iou > IOU_THRESHOLD and iou > best_iou:
                        best_iou = iou
                        best_conf = pred["conf"]
                        best_pred_idx = p_idx

                if best_pred_idx!= -1:
                    gt_record[m_name] = best_conf
                    matched_pred_indices.add(best_pred_idx)
            gt_from_csv = True


            # Collect and Cluster False Positives
            img_fps = [p for j, p in enumerate(predictions) if j not in matched_pred_indices and p["conf"] >= 0.1]

            for fp in img_fps:
                matched_cluster = False
                for cluster in fp_clusters:
                    if cluster["image_path"] == img_file:
                        iou = compute_iou(cluster["fp_bbox"], fp["box"])
                        if iou > IOU_THRESHOLD:
                            # Use max() in case the model predicts two overlapping FPs
                            cluster[m_name] = max(cluster.get(m_name, 0.0), fp["conf"])
                            matched_cluster = True
                            break

                if not matched_cluster:
                    new_cluster = {"image_path": img_file, "fp_bbox": fp["box"]}
                    for model_col in MODEL_PATHS.keys():
                        new_cluster[model_col] = 0.0
                    new_cluster[m_name] = fp["conf"]
                    fp_clusters.append(new_cluster)

# 4. Save state back to disk
        print(f"Saving progress to CSV for {m_name}...")
        pd.DataFrame(gt_records).to_csv(GT_CSV, index=False)
        if fp_clusters:
            pd.DataFrame(fp_clusters).to_csv(FP_CSV, index=False)


        # 5. Aggressive memory cleanup before loading the next model
        complete.append(m_name)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # 6. Save final DataFrames to CSV
#    print("Writing results to CSV...")
#    pd.DataFrame(gt_records).to_csv(GT_CSV, index=False)

#    if fp_clusters:
#        pd.DataFrame(fp_clusters).to_csv(FP_CSV, index=False)
#    else:
#        pd.DataFrame(columns=["image_path", "fp_bbox"] + list(MODEL_PATHS.keys())).to_csv(FP_CSV, index=False)

    print("Analysis complete!")

if __name__ == "__main__":
    run_granular_evaluation(SSDD_MODEL_PATHS, SSDD_IMAGES_DIR, SSDD_LABELS_DIR, SSDD_GT_CSV, SSDD_FP_CSV)
    run_granular_evaluation(HRSID_MODEL_PATHS, HRSID_IMAGES_DIR, HRSID_LABELS_DIR, HRSID_GT_CSV, HRSID_FP_CSV)

