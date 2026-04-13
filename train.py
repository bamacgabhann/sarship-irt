import csv
import os
import gc
import ast

from torch.cuda import empty_cache
from ultralytics import YOLO

import pandas as pd

ARCHITECTURES = [
    'yolov8x', 'yolov9e', 'yolov10x', 'yolo11x', 'yolo12x', 'yolo26x',
    'yolov8l', 'yolov9c', 'yolov10b', 'yolov10l', 'yolo11l', 'yolo12l', 'yolo26l',
    'yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m', 'yolo26m',
    'yolov8s', 'yolov9s', 'yolov10s', 'yolo11s', 'yolo12s', 'yolo26s',
    'yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo12n', 'yolo26n'
]

TRAINING_DATASETS = ['SSDD']
TESTING_DATASETS = ['HRSID', 'SSDD', "LS-SSDD"]

PROJ = 'sarship-irt'

IOU_THRESHOLD = 0.5


def get_checkpoint_paths(
        architecture: str, 
        trained_on: str, 
        proj: str = PROJ,
        epochs: int = 500,
        checkpoint_dir: str = None
) -> dict:
    """
    Returns a dictionary of paths to checkpoint files. 
    The keys are formatted as '{architecture}_{trained_on}_epoch{epoch}' and 
    the values are the corresponding file paths.
     - architecture: The model architecture (e.g., 'yolov8x')
     - trained_on: The dataset the model was trained on (e.g., 'SSDD')
     - proj: The project name used in the directory structure
     - checkpoint_dir: Optional base directory for checkpoints; if None, defaults to '../../runs/detect/{proj}/{architecture}_{trained_on}/weights'
     - Returns: A dictionary mapping checkpoint identifiers to their file paths
     - Note: Only includes checkpoints that exist to prevent errors during testing
    """
    checkpoint_paths = {}
    for epoch in range(epochs):
        checkpoint_name = f'{architecture}_{trained_on}_epoch{epoch}'
        checkpoint_fname = f'epoch{epoch}.pt'
        if checkpoint_dir is not None:
            checkpoint_dir = f'../../runs/detect/{proj}/{architecture}_{trained_on}/weights'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_fname)
        # Verify path exists before adding to prevent later errors
        if os.path.exists(checkpoint_path):
            checkpoint_paths[checkpoint_name] = checkpoint_path
    return checkpoint_paths


def write_header(csv_file):
    """
    Write header for the results or validation CSV if the file does not already exist
    """
    if not os.path.isfile(csv_file):
        header = [
            "model",
            "trained on",
            "tested on",
            "preprocess",
            "inference",
            "loss",
            "postprocess",
            "ms/image",
            "mAP50",
            "mAP50-95",
            "Precision",
            "Recall",
            "F1",
        ]

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def run_tests(architecture, trained_on, test_on, results_csv, _proj=PROJ, test_pt=None, device=None, batch_size=10):
    """
    Run tests for a specified model checkpoint against specified test dataset, saving results to CSV
    """
    if device is None:
        device = [0, 1]
    test_model = YOLO(test_pt)
    metrics = test_model.val(
        data=f'./datasets/{test_on}.yaml',
        split='test',
        imgsz=800,
        device=device,
        batch=batch_size,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=True,
        visualize=True,
        save_txt=False,
        save_conf=False,
        project=f'{_proj}_tests',  # Organise all test runs in a dedicated folder
        name=f'{architecture}_{trained_on}_tested_on_{test_on}'  # Appropriate validation run name
    )

    # Write CSV header if file does not exist
    write_header(results_csv)

    # Collate metrics as a row for the CSV, ensuring precision, recall, and f1 are not empty before accessing
    row = [
        architecture,
        trained_on,
        test_on,
        metrics.speed['preprocess'],
        metrics.speed['inference'],
        metrics.speed['loss'],
        metrics.speed['postprocess'],
        sum(metrics.speed.values()),
        metrics.box.map50,
        metrics.box.map,
        metrics.box.p[0] if len(metrics.box.p) > 0 else 0,
        metrics.box.r[0] if len(metrics.box.r) > 0 else 0,
        metrics.box.f1[0] if len(metrics.box.f1) > 0 else 0
    ]

    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def test_checkpoints(
        architectures: list = None,
        training_datasets: list = None,
        testing_datasets: list = None,
        results_dir: str = None, 
        checkpoint_results: str = None,
        proj: str = PROJ,
        _device: int|list = None
):
    """
    Organise test runs for checkpoints of multiple architectures against multiple test sets
    """
    if architectures is None:
        architectures = ARCHITECTURES
    if training_datasets is None:
        training_datasets = TRAINING_DATASETS
    if testing_datasets is None:
        testing_datasets = TESTING_DATASETS
    if _device is None:
        _device = [0,1]
    for a in architectures:
        for t in training_datasets:
            if not checkpoint_results:
                checkpoint_results = f'tests_{a}_{t}_checkpoints.csv'
            if not results_dir:
                results_dir = '../../results/checkpoints/'

            checkpoint_results_file = os.path.join(results_dir, checkpoint_results)

            if os.path.isfile(checkpoint_results_file):
                df = pd.read_csv(checkpoint_results_file)
                complete = df['model'].tolist()
            else:
                complete = []

            checkpoints_dir = f'../../runs/detect/{proj}/{a}_{t}/weights'

            checkpoint_paths = get_checkpoint_paths(a, t, proj=proj, epochs=500, checkpoint_dir=checkpoints_dir)
            for checkpoint_name, checkpoint_path in checkpoint_paths.items():
                for test_dataset in testing_datasets:
                    if checkpoint_name not in complete:
                        run_tests(
                            checkpoint_name, 
                            t, test_dataset, 
                            results_csv=checkpoint_results_file, 
                            _proj=proj, 
                            test_pt=checkpoint_path, 
                            device=_device
                        )


def train_model(
        architecture: str,
        train_on: str,
        epochs: int = 500,
        patience: int = 0,
        imgsz: int = 800,
        plots: bool = True,
        val: bool = True,
        device: int | list = None,
        batch: int = 10,
        workers: int = 10,
        cache: bool = True,
        project: str = PROJ,
) -> None:
    
    # Use both GPUs by default
    if device is None:
        device = [0, 1]

    # Empty cache before next training
    empty_cache()

    # Set which architecture to train
    modelname = f'{architecture}.pt'
    model = YOLO(modelname)

    model.train(
        data=f'./datasets/{train_on}.yaml',
        epochs=epochs,
        imgsz=imgsz,
        plots=plots,
        val=val,
        optimizer='AdamW',
        close_mosaic=350,
        patience=patience,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        cos_lr=True,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        save_period=1,        # Save checkpoint weights after each epoch for dataset cartography / IRT
        device=device,        # Use both GPUs
        batch=batch,          # Batch size fixed to ensure all models fit in VRAM, and for consistency
        workers=workers,      # 10 workers fixed for consistency
        cache=cache,          # Cache the dataset in RAM to eliminate disk bottleneck
        project=project,      # Organise all runs into one folder
        name=f'{architecture}_{train_on}'  # Give this specific run a unique name
    )

def run_training(
        architectures: list = None,
        training_datasets: list = None,
        epochs: int = 500,
        patience: int = 0,
        imgsz: int = 800,
        plots: bool = True,
        val: bool = True,
        device: int | list = None,
        batch: int = 10,
        workers: int = 10,
        cache: bool = True,
        project: str = PROJ
):
    # Use both GPUs by default
    if device is None:
        device = [0, 1]
    if architectures is None:
        architectures = ARCHITECTURES
    if training_datasets is None:
        training_datasets = TRAINING_DATASETS
    
    for architecture in architectures:
        for train_on in training_datasets:
            train_model(
                architecture=architecture,
                train_on=train_on,
                epochs=epochs,
                patience=patience,
                imgsz=imgsz,
                plots=plots,
                val=val,
                device=device,
                batch=batch,
                workers=workers,
                cache=cache,
                project=project
            )

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def get_ground_truths(architectures: list, dataset: str) -> list:
    dataset_dir = f'../../datasets/{dataset}'
    gt_records = []
    splits = ["train", "val", "test"]
    for s in splits:
        split_images = os.path.join(dataset_dir, f'{s}/images')
        split_labels = os.path.join(dataset_dir, f'{s}/labels')
        image_files = [f for f in os.listdir(split_images)]
        for img in image_files:
            label_path = os.path.join(split_labels, img.replace('.jpg', '.txt').replace('.png', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            c, x, y, w, h = map(float, parts)
                            x1, y1 = x - w/2, y - h/2
                            x2, y2 = x + w/2, y + h/2

                            record = {
                                "image": img,
                                "image_path": os.path.join(split_images, img),
                                "label_path": os.path.join(split_labels, label_path),
                                "gt_bbox": [x1, y1, x2, y2]
                            }
                            # Initialize all model columns to 0.0
                            for a in architectures:
                                checkpoint_paths = get_checkpoint_paths(a, dataset)
                                for checkpoint_name in checkpoint_paths.keys():
                                    record[checkpoint_name] = 0.0
                            gt_records.append(record)

    return gt_records


def run_dataset_topography(
        test_dataset: str,
        results_dir: str = None,
        gt_file: str = None,
        architectures: list = None,
):
    if architectures is None:
        architectures = ARCHITECTURES

    if results_dir is None:
        results_dir = '../../results/topography'

    if gt_file is None:
        gt_file = os.path.join(results_dir, f'{test_dataset}_ground_truths.feather')

    checkpoints_complete = []
    fp_clusters = []

    # Pre-load all Ground Truths into memory from file if resuming, otherwise from labels
    if os.path.exists(gt_file):
        df_gt = pd.read_csv(gt_file)
        df_gt['gt_bbox'] = df_gt['gt_bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        gt_records = df_gt.to_dict('records')
        # TODO: line here to add checkpoints not all zero to 'done' list
        del df_gt
    else:
        gt_records = get_ground_truths(architectures, test_dataset)

    img_paths = {}
    for gt in gt_records:
        img_paths = {gt['image']: gt['image_path']}

    # Iterate through models (Load each EXACTLY ONCE)
    for a in architectures:
        checkpoint_paths = get_checkpoint_paths(a, test_dataset)
        for checkpoint_name, checkpoint_path in checkpoint_paths.items():
            if checkpoint_name in checkpoints_complete:
                continue

            model = YOLO(checkpoint_path)

            # Iterate through images one by one
            # This prevents both potential OS file limit error and VRAM padding spikes
            for img, img_path in img_paths.items():
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
                img_gts = [gt for gt in gt_records if gt["image_path"] == img_path]

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
                        gt_record[checkpoint_name] = best_conf
                        matched_pred_indices.add(best_pred_idx)



                # Collect and Cluster False Positives
                img_fps = [p for j, p in enumerate(predictions) if j not in matched_pred_indices and p["conf"] >= 0.1]

                for fp in img_fps:
                    matched_cluster = False
                    for cluster in fp_clusters:
                        if cluster["image_path"] == img_path:
                            iou = compute_iou(cluster["fp_bbox"], fp["box"])
                            if iou > IOU_THRESHOLD:
                                # Use max() in case the model predicts two overlapping FPs
                                cluster[checkpoint_name] = max(cluster.get(checkpoint_name, 0.0), fp["conf"])
                                matched_cluster = True
                                break

                    if not matched_cluster:
                        new_cluster = {
                            "image": img,
                            "image_path": img_path, 
                            "fp_bbox": fp["box"], 
                            checkpoint_name: fp["conf"]
                        }
                        fp_clusters.append(new_cluster)

# 4. Save state back to disk

            pd.DataFrame(gt_records).to_feather(gt_file)
            if fp_clusters:
                fp_fname = f'{checkpoint_name}_{test_dataset}_fp.feather'
                fp_file = os.path.join(results_dir, fp_fname)
                pd.DataFrame(fp_clusters).to_feather(fp_file)


            # 5. Aggressive memory cleanup before loading the next model
            checkpoints_complete.append(checkpoint_name)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



if __name__ == '__main__':
    run_training(ARCHITECTURES, TRAINING_DATASETS)
    test_checkpoints(ARCHITECTURES, TRAINING_DATASETS, TESTING_DATASETS)
    for test_dataset in TESTING_DATASETS:
        run_dataset_topography(test_dataset, architectures=ARCHITECTURES)
