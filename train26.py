import csv
import os

from datetime import datetime as dt
from shutil import move, copyfile

from torch.cuda import empty_cache
from ultralytics import YOLO


architectures = [
    'yolo26n', 'yolo26s','yolo26m', 'yolo26l', 'yolo26x'
]

training_datasets = ['SSDD']
testing_datasets = ['HRSID', 'SSDD', "LS-SSDD"]

proj = 'sar_ship_detection_dual'

test_results = f'/home/breandan/sarship/tests_dual.csv'

def copy_checkpoint_val(results_file=None, results_on_drive=None):
    if not results_file:
        results_file = f'/home/breandan/sarship/tests_checkpoints.csv'
    if not results_on_drive:
        results_on_drive = '/run/user/1000/gvfs/google-drive:host=gmail.com,user=b.macgabhann/0APHu4FCy59LUUk9PVA/1sO75zhWdpRY01UXP9--lxHQMskteUt1K/tests_checkpoints.csv'
    copyfile(results_file, results_on_drive)

def copy_val():
    results_file = '/home/breandan/sarship/tests_dual.csv'
    results_on_drive = '/run/user/1000/gvfs/google-drive:host=gmail.com,user=b.macgabhann/0APHu4FCy59LUUk9PVA/1sO75zhWdpRY01UXP9--lxHQMskteUt1K/tests_dual.csv'
    copyfile(results_file, results_on_drive)

def copy_results_csv(a, t, p):
    results_file = f'/home/breandan/sarship/runs/detect/{p}/{a}_{t}/results.csv'
    results_on_drive = f'/run/user/1000/gvfs/google-drive:host=gmail.com,user=b.macgabhann/0APHu4FCy59LUUk9PVA/1sO75zhWdpRY01UXP9--lxHQMskteUt1K/results_{a}_{t}.csv'
    copyfile(results_file, results_on_drive)

def copy_results_png(a, t, p):
    results_file = f'/home/breandan/sarship/runs/detect/{p}/{a}_{t}/results.png'
    results_on_drive = f'/run/user/1000/gvfs/google-drive:host=gmail.com,user=b.macgabhann/0APHu4FCy59LUUk9PVA/1sO75zhWdpRY01UXP9--lxHQMskteUt1K/results_{a}_{t}.png'
    copyfile(results_file, results_on_drive)

def move_checkpoints(a, t, p):
    models_dir = f'/home/breandan/sarship/runs/detect/{p}/{a}_{t}/weights/'
    checkpoints_dir = f'/media/breandan/Windows/YOLO_checkpoints/{a}_{t}_weights'
    os.makedirs(checkpoints_dir, exist_ok=True)
    for f in os.listdir(models_dir):
        if f.startswith('epoch'):
            move(os.path.join(models_dir, f), os.path.join(checkpoints_dir, f))
        elif f.startswith('best'):
            copyfile(os.path.join(models_dir, f), os.path.join(checkpoints_dir, f))
        elif f.startswith('last'):
            copyfile(os.path.join(models_dir, f), os.path.join(checkpoints_dir, f))
        else:
            move(os.path.join(models_dir, f), os.path.join(checkpoints_dir, f))

def write_status_update(a, t, p, current_epoch):
    status_file = f'/home/breandan/sarship/{p}/statusupdate.csv'

    header = ['datetime', 'architecture', 'training dataset', 'current epoch']
    if not os.path.isfile(status_file):
        with open(status_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    status_file_on_drive = f'/run/user/1000/gvfs/google-drive:host=gmail.com,user=b.macgabhann/0APHu4FCy59LUUk9PVA/1sO75zhWdpRY01UXP9--lxHQMskteUt1K/statusupdate.csv'
    current_time = dt.now().strftime('%Y-%m-%d %H:%M:%S')


    row = [current_time, a, t, current_epoch]
    with open(status_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    copyfile(status_file, status_file_on_drive)



def run_validation(architecture, trained_on, test_on, results_csv, _proj="sarship", test_pt=None, device=[0,1]):
    if test_pt is None:
        test_model = YOLO(f'/home/breandan/sarship/runs/detect/{_proj}/{architecture}_{trained_on}/weights/best.pt')
    else:
        test_model = YOLO(test_pt)
    metrics = test_model.val(
        data=f'/home/breandan/sarship/{test_on}.yaml',
        split='test',
        imgsz=800,
        device=device,
#        batch=batch,
        conf=0.001,
        iou=0.7,
        max_det=300,
#        plots=True,
#        visualize=True,
#        save_txt=True,
#        save_conf=True,
        plots=False,
        visualize=False,
        save_txt=False,
        save_conf=False,
        project=f'{_proj}_validation',  # Organise all validation runs in a dedicated folder
        name=f'{architecture}_{trained_on}_tested_on_{test_on}'  # Appropriate validation run name
    )

    write_header(results_csv)

    # Collate and save metrics as in your original script
    speed = sum(metrics.speed.values())
    mAP50 = metrics.box.map50
    mAP50_95 = metrics.box.map

    # Ensure precision, recall, and f1 arrays are not empty before accessing
    precis = metrics.box.p[0] if len(metrics.box.p) > 0 else 0
    recall = metrics.box.r[0] if len(metrics.box.r) > 0 else 0
    f1 = metrics.box.f1[0] if len(metrics.box.f1) > 0 else 0

#    df = metrics.confusion_matrix.to_df()
#    TP = df.loc[0, "ship"] if "ship" in df.columns and 0 in df.index else 0
#    FN = df.loc[1, "ship"] if "ship" in df.columns and 1 in df.index else 0
#    FP = df.loc[0, "background"] if "background" in df.columns and 0 in df.index else 0

    row = [
        architecture,
        trained_on,
        test_on,
        metrics.speed['preprocess'], metrics.speed['inference'], metrics.speed['loss'],
        metrics.speed['postprocess'],
        speed, mAP50, mAP50_95, precis, recall, f1
#, TP, FN, FP
    ]

    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def write_header(results_csv):
    # Define the header for the CSV file
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
        "TP",
        "FN",
        "FP"
    ]

    if not os.path.isfile(results_csv):
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def run_training(_architecture, train_on, _testing_datasets, epochs=500, patience=0, imgsz=800, plots=True, val=True, device=[0,1], batch=10, workers=10, cache=True, project=proj, _test_results=test_results):

    write_header(_test_results)

    empty_cache()
    modelname = f'{_architecture}.pt'
    model = YOLO(modelname)

    model.train(
        data=f'/home/breandan/sarship/{train_on}.yaml',
        epochs=epochs,
        imgsz=imgsz,
        plots=plots,
        val=val,
        optimizer='AdamW',
        close_mosaic=350,
        patience=patience,
#        augment=True,
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
        save_period=1,
        device=device,
        batch=batch,          # Batch size fixed to ensure all models fit in VRAM, and consistency
        workers=workers,      # Standard 10 workers
        cache=cache,          # Cache the dataset in RAM to eliminate disk bottleneck
        project=project,      # Organizes all runs into one folder
        name=f'{_architecture}_{train_on}'  # Give this specific run a unique name
    )



    for test_on in _testing_datasets:
        to_test = f'/home/breandan/sarship/runs/detect/{proj}/{_architecture}_{train_on}/weights/best.pt'
        run_validation(_architecture, train_on, test_on, results_csv=_test_results, _proj=project, test_pt=to_test)

    move_checkpoints(_architecture, train_on, project)
    print('Copying results CSV')
    copy_results_csv(_architecture, train_on, project)
    print('Copying results PNG')
    copy_results_png(_architecture, train_on, project)
    print('Copying validation results')
    copy_val()



if __name__ == '__main__':
    for training_dataset in training_datasets:
        for arch in architectures:
            run_training(arch, training_dataset, testing_datasets)
