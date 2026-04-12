import csv
import os

from torch.cuda import empty_cache
from ultralytics import YOLO


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

TEST_RESULTS = './output/tests.csv'
VALIDATION_RESULTS = './output/validation.csv'


def write_header(csv_file):
    '''
    Write header for the results or validation CSV if the file does not already exist
    '''
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

    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def run_tests(architecture, trained_on, test_on, results_csv, _proj=PROJ, test_pt=None, device=[0, 1], batch_size=10):
    """
    Run tests for a specified model checkpoint against specified test dataset, saving results to CSV
    """
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


def test_checkpoints(_architectures, _training_datasets, _testing_datasets, checkpoint_results=None, _device=[0,1]):
"""
Organise test runs for checkpoints of multiple architectures against multiple test sets
"""
    for a in _architectures:
        for t in _training_datasets:
            if not checkpoint_results:
                checkpoint_results = f'tests_{a}_{t}_checkpoints'

            checkpoint_results_file = f'/home/breandan/sarship/{checkpoint_results}.csv'

            if os.path.isfile(checkpoint_results_file):
                df = pd.read_csv(checkpoint_results_file)
                complete = df['model'].tolist()
            else:
                complete = []

            checkpoints_dir = f'/media/breandan/Windows/YOLO_checkpoints/{a}_{t}_weights'
            for f in os.listdir(checkpoints_dir):
                print(f"Next: {a} trained on {t}, {f}")
                if f.startswith('epoch'):
                    f_epoch = f.split('.')[0]
                    print(f"Ready to test {f_epoch}")
                    for test_dataset in _testing_datasets:
                        pt_to_test = os.path.join(checkpoints_dir, f)
                        arch_to_test = f'{a}_{t}_{f_epoch}'
                        if arch_to_test not in complete:
                            run_tests(arch_to_test, t, test_dataset, results_csv=checkpoint_results_file, _proj="sarship_checkpoints", test_pt=pt_to_test, device=_device)


def run_training(
        _architecture: str,
        train_on: str,
        _testing_datasets: str,
        epochs: int = 500,
        patience: int = 0,
        imgsz: int = 800,
        plots: bool = True,
        val: bool = True,
        device: int | list = [0, 1],
        batch: int = 10,
        workers: int = 10,
        cache: bool = True,
        project: str = PROJ,
        _test_results: str = TEST_RESULTS
) -> None:

    # Write CSV header if file does not exist
    write_header(_test_results)

    # Empty cache before next training
    empty_cache()

    # Set which architecture to train
    modelname = f'{_architecture}.pt'
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
        name=f'{_architecture}_{train_on}'  # Give this specific run a unique name
    )




if __name__ == '__main__':
    for training_dataset in TRAINING_DATASETS:
        for arch in ARCHITECTURES:
            run_training(arch, training_dataset, TESTING_DATASETS)
