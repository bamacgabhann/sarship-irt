import subprocess as sp
import time
from train26 import run_training, run_validation
from torch.cuda import empty_cache
import csv
import os

from ultralytics import YOLO

architectures = [
    'yolo12x',
    'yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo12n',
    'yolov8s', 'yolov9s', 'yolov10s', 'yolo11s', 'yolo12s',
    'yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m',
    'yolov8l', 'yolov9c', 'yolov10b', 'yolov10l', 'yolo11l', 'yolo12l',
    'yolov8x', 'yolov9e', 'yolov10x', 'yolo11x',
    'yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x'
]
completed = [
    'yolo12x_SSDD',
    'yolov8n_SSDD', 'yolov9t_SSDD', 'yolov10n_SSDD', 'yolo11n_SSDD', 'yolo12n_SSDD', 'yolo26n_SSDD',
    'yolov8s_SSDD', 'yolov9s_SSDD', 'yolov10s_SSDD', 'yolo11s_SSDD', 'yolo12s_SSDD', 'yolo26s_SSDD',
    'yolov8m_SSDD', 'yolov9m_SSDD', 'yolov10m_SSDD', 'yolo11m_SSDD', 'yolo12m_SSDD', 'yolo26m_SSDD',
    'yolov8l_SSDD', 'yolov9c_SSDD', 'yolov10b_SSDD', 'yolov10l_SSDD', 'yolo11l_SSDD', 'yolo12l_SSDD', 'yolo26l_SSDD',
    'yolov8x_SSDD', 'yolov9e_SSDD', 'yolov10x_SSDD', 'yolo11x_SSDD',
    'yolo12x_HRSID',
    'yolov8n_HRSID', 'yolov9t_HRSID', 'yolov10n_HRSID', 'yolo11n_HRSID', 'yolo12n_HRSID', 'yolo26n_HRSID',
    'yolov8s_HRSID', 'yolov9s_HRSID', 'yolov10s_HRSID', 'yolo11s_HRSID', 'yolo12s_HRSID', 'yolo26s_HRSID',
    'yolov8m_HRSID', 'yolov9m_HRSID', 'yolov10m_HRSID', 'yolo11m_HRSID', 'yolo12m_HRSID', 'yolo26m_HRSID',
    'yolov8l_HRSID', 'yolov9c_HRSID', 'yolov10b_HRSID', 'yolov10l_HRSID', 'yolo11l_HRSID', 'yolo12l_HRSID', 'yolo26l_HRSID',
    'yolov8x_HRSID', 'yolov9e_HRSID', 'yolov10x_HRSID', 'yolo11x_HRSID'
]


training_datasets = ['HRSID', 'SSDD']
#testing_datasets = ['HRSID', 'SSDD', "LS-SSDD"]
testing_datasets = ['HRSID', 'SSDD', "LS-SSDD", "tests_ssdd_hrsid_ls-ssdd"]

proj = 'sar_ship_detection_dual'

test_results = f'/home/breandan/sarship/tests_dual.csv'

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

def is_training_ongoing():
    vram1, vram2 = get_gpu_memory()
    if vram1 > 3000 or vram2 > 3000:
        return True
    else:
        return False

def are_gpus_available():
    vram1, vram2 = get_gpu_memory()
    if vram1 < 3000 and vram2 < 3000:
        return True
    else:
        return False

if __name__ == '__main__':


#    while not are_gpus_available():
#        while is_training_ongoing():
#            vram1, vram2 = get_gpu_memory()
#            print(f'''Training process already in progress
#                  VRAM in use {vram1} MB, {vram2} MB
#                  Will try again in 5 minutes''')
#            time.sleep(300)
#        vram1, vram2 = get_gpu_memory()
#        print(f'''GPU memory may not be fully available
#              RAM in use {vram1} MB, {vram2} MB
#              Will try clearing cache''')
#        empty_cache()
#        time.sleep(5)
#

#    run_training('yolov8x', 'SSDD', testing_datasets, project=proj, _test_results=test_results)

#    run_validation("yolov10n", 'SSDD', "tests_ssdd_hrsid_ls-ssdd", results_csv=test_results, _proj="sarship")

    empty_cache()

    for arch in architectures:
        for training_dataset in training_datasets:
            while not are_gpus_available():
                vram1, vram2 = get_gpu_memory()
                print(f'''GPU memory may not be fully available
                      RAM in use {vram1} MB, {vram2} MB
                      Will try clearing cache''')
                empty_cache()
                time.sleep(5)
            task = f'{arch}_{training_dataset}'
            if task in completed:
                print(f'{arch} trained on {training_dataset} already trained')
                continue
            run_training(arch, training_dataset, testing_datasets, project=proj, _test_results=test_results)
            completed.append(task)

#    validate_checkpoints(architectures, training_datasets, testing_datasets)


