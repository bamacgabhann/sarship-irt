import csv
import os
import pandas as pd
from datetime import datetime as dt
from shutil import move, copyfile

from torch.cuda import empty_cache
from ultralytics import YOLO
from train26 import run_validation, copy_checkpoint_val
import fnmatch
from time import sleep

architectures = [
#    'yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo12n',
#    'yolov8s', 'yolov9s', 'yolov10s', 'yolo11s', 'yolo12s',
#    'yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m',
#    'yolov8l', 'yolov9c', 'yolov10b', 'yolov10l', 'yolo11l', 'yolo12l',
#    'yolov8x', 'yolov9e', 'yolov10x', 'yolo11x', 'yolo12x'
    'yolo26n_dcn'
#, 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x'
]

#training_datasets = ['SSDD', 'HRSID']
training_datasets = ['SSDD']
testing_datasets = [
    'HRSID', 'SSDD', "LS-SSDD"
    #, "tests_ssdd_hrsid_ls-ssdd"
]

proj = 'sar_ship_detection_dcn'

test_results = f'/home/breandan/sarship-yolo26/tests_checkpoints_dcn.csv'

def count_pt_in_dir(directory):
    return len(fnmatch.filter(os.listdir(directory), "*.pt"))


def validate_checkpoints(_architectures, _training_datasets, _testing_datasets, checkpoint_results=None, _device=[0,1]):

    for a in _architectures:
        for t in _training_datasets:
            if not checkpoint_results:
                checkpoint_results = f'tests_{a}_{t}_checkpoints'

            checkpoint_results_file = f'/home/breandan/sarship-yolo26/{checkpoint_results}.csv'
            checkpoint_results_drive = f'/run/user/1000/gvfs/google-drive:host=gmail.com,user=b.macgabhann,prefix=%2F0APHu4FCy59LUUk9PVA/0APHu4FCy59LUUk9PVA/1sO75zhWdpRY01UXP9--lxHQMskteUt1K/{checkpoint_results}.csv'
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
                            run_validation(arch_to_test, t, test_dataset, results_csv=checkpoint_results_file, _proj=proj, test_pt=pt_to_test, device=_device)
                            copy_checkpoint_val(checkpoint_results_file, checkpoint_results_drive)

def cross_validate_26_dcn():
     validate_checkpoints(architectures, training_datasets, testing_datasets)


if __name__ == '__main__':
#     while count_pt_in_dir('/media/breandan/Windows/YOLO_checkpoints/yolo26m_SSDD_weights') != 502:
#            print('Training incomplete, waiting')
#            sleep(600)
     validate_checkpoints(architectures, training_datasets, testing_datasets)
