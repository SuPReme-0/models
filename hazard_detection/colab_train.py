from google.colab import drive
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import os
import yaml
from ultralytics import YOLO
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
import cv2

DRIVE_MOUNT_POINT = Path("/content/drive")
PREFERRED_DRIVE_PATHS = [
    DRIVE_MOUNT_POINT / "MyDrive" / "hazard.zip"
]
LOCAL_ZIP_COPY = Path("/content/hazard.zip")
LOCAL_EXTRACT_DIR = Path("/content/hazard-detection")
LOCAL_DATASET_BASE = LOCAL_EXTRACT_DIR / "ultimate_hazard_dataset"
DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/Security-Rover-Final")

LOCAL_YOLO_DATA_DIR = LOCAL_DATASET_BASE / "yolo"
LOCAL_DATA_YAML = LOCAL_YOLO_DATA_DIR / "data.yaml"

MODEL_TO_EVAL = DRIVE_PROJECT_ROOT / "stage3_finetune_320" / "weights" / "best.pt"
TFLITE_SAVE_DIR = DRIVE_PROJECT_ROOT / "stage4_final_output"
TFLITE_FINAL_PATH = TFLITE_SAVE_DIR / "best_native_float16.tflite"

VALIDATION_SPLIT_TO_USE = "test"
TEST_IMAGES_DIR = LOCAL_YOLO_DATA_DIR / "test" / "images"
TRAIN_IMAGES_DIR = LOCAL_YOLO_DATA_DIR / "train" / "images"
VALID_IMAGES_DIR = LOCAL_YOLO_DATA_DIR / "val" / "images"

IMG_SIZE = 320
CONF_THRESH = 0.25


def measure_detailed_speed(model, data_yaml, split, img_size, conf_thresh):
    """Measure detailed speed metrics for a model."""
    print(f"Measuring detailed speed metrics...")
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=img_size,
        conf=conf_thresh,
        plots=False,
        verbose=False,
        device='cpu'  
    )
    speed_data = {
        'preprocess_avg': metrics.speed['preprocess'],
        'inference_avg': metrics.speed['inference'],
        'postprocess_avg': metrics.speed['postprocess'],
        'total_avg': metrics.speed['preprocess'] + metrics.speed['inference'] + metrics.speed['postprocess']
    }
    speed_data['fps'] = 1000 / speed_data['total_avg'] if speed_data['total_avg'] > 0 else 0
    return speed_data

def build_comparison_df(pt_metrics, tflite_metrics, pt_speed_measurements, tflite_speed_measurements):
    """Builds the final comparison dataframe."""
    class_names = pt_metrics.names
    data = []

    for i, name in class_names.items():
        pt_p, pt_r, pt_ap50, pt_ap = pt_metrics.box.class_result(i)
        tf_p, tf_r, tf_ap50, tf_ap = tflite_metrics.box.class_result(i)
        data.append({'Class': name, 'Metric': 'Precision', 'PyTorch FP32': pt_p, 'TFLite FP16': tf_p})
        data.append({'Class': name, 'Metric': 'Recall', 'PyTorch FP32': pt_r, 'TFLite FP16': tf_r})
        data.append({'Class': name, 'Metric': 'mAP50', 'PyTorch FP32': pt_ap50, 'TFLite FP16': tf_ap50})
        data.append({'Class': name, 'Metric': 'mAP50-95', 'PyTorch FP32': pt_ap, 'TFLite FP16': tf_ap})

    # Overall Metrics
    data.append({'Class': 'ALL (Overall)', 'Metric': 'Precision', 'PyTorch FP32': pt_metrics.box.mp, 'TFLite FP16': tflite_metrics.box.mp})
    data.append({'Class': 'ALL (Overall)', 'Metric': 'Recall', 'PyTorch FP32': pt_metrics.box.mr, 'TFLite FP16': tflite_metrics.box.mr})
    data.append({'Class': 'ALL (Overall)', 'Metric': 'mAP50', 'PyTorch FP32': pt_metrics.box.map50, 'TFLite FP16': tflite_metrics.box.map50})
    data.append({'Class': 'ALL (Overall)', 'Metric': 'mAP50-95', 'PyTorch FP32': pt_metrics.box.map, 'TFLite FP16': tflite_metrics.box.map})

    # Speed Metrics
    data.append({'Class': 'SPEED', 'Metric': 'Avg. Latency (ms)', 'PyTorch FP32': pt_speed_measurements['total_avg'], 'TFLite FP16': tflite_speed_measurements['total_avg']})
    data.append({'Class': 'SPEED', 'Metric': 'FPS', 'PyTorch FP32': pt_speed_measurements['fps'], 'TFLite FP16': tflite_speed_measurements['fps']})
    data.append({'Class': 'SPEED', 'Metric': 'Inference (ms)', 'PyTorch FP32': pt_speed_measurements['inference_avg'], 'TFLite FP16': tflite_speed_measurements['inference_avg']})
    
    # Model Size Metrics
    pt_model_size = MODEL_TO_EVAL.stat().st_size / (1024 * 1024)
    tflite_model_size = TFLITE_FINAL_PATH.stat().st_size / (1024 * 1024)
    data.append({'Class': 'MODEL', 'Metric': 'Size (MB)', 'PyTorch FP32': pt_model_size, 'TFLite FP16': tflite_model_size})

    # Create and Format DataFrame
    df = pd.DataFrame(data)
    df['Delta'] = df['TFLite FP16'] - df['PyTorch FP32']
    pd.set_option('display.float_format', '{:,.4f}'.format)
    pd.set_option('display.width', 1000)
    df = df[['Class', 'Metric', 'PyTorch FP32', 'TFLite FP16', 'Delta']]
    return df


def run_pipeline():
  
  print(f"Attempting to clear mount point {DRIVE_MOUNT_POINT} to prevent errors...")
    try:
        drive.mount(str(DRIVE_MOUNT_POINT), force_remount=True)
        print("✅ Drive mounted at", DRIVE_MOUNT_POINT)
    except Exception as e:
        print(f"🔥 Drive mount failed: {e}")
        raise

    zip_path = None
    print("Searching for 'hazard.zip' in Google Drive...")
    for p in PREFERRED_DRIVE_PATHS:
        if p.exists():
            zip_path = p
            break
    if zip_path is None:
        print("Preferred paths not found — searching MyDrive for hazard*.zip...")
        for p in (DRIVE_MOUNT_POINT / "MyDrive").rglob("*.zip"):
            if "hazard" in p.name.lower():
                zip_path = p
                break
    if zip_path is None:
        raise FileNotFoundError(f"Could not find 'hazard.zip' in {DRIVE_MOUNT_POINT / 'MyDrive'}.")
    print("Found zip in Drive:", zip_path)

    print(f"Copying {zip_path} -> {LOCAL_ZIP_COPY} ...")
    shutil.copy2(str(zip_path), str(LOCAL_ZIP_COPY))
    print("Copy finished.")

    print(f"Extracting {LOCAL_ZIP_COPY} -> {LOCAL_EXTRACT_DIR} ...")
    shutil.rmtree(LOCAL_EXTRACT_DIR, ignore_errors=True)
    LOCAL_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(str(LOCAL_ZIP_COPY), 'r') as zf:
            zf.extractall(path=str(LOCAL_EXTRACT_DIR))
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}")
        raise RuntimeError("Failed to extract hazard.zip.") from e

    if not LOCAL_DATASET_BASE.exists() or not LOCAL_DATA_YAML.exists():
        print(f"Warning: Dataset base or data.yaml not found after extraction.")
        print(f"Expected YAML at: {LOCAL_DATA_YAML}")
        print("Please ensure 'hazard.zip' contains 'data.yaml' at the correct path.")
        raise FileNotFoundError(f"Missing {LOCAL_DATA_YAML}")
    else:
        print("Local dataset and data.yaml are ready.")
    
    STAGES = [
        {
            'name': 'stage1_backbone_640', 'model': 'yolov8n.pt',
            'imgsz': 640, 'epochs': 80, 'batch': 64,
            'lr0': 0.005, 'patience': 25, 'mosaic': 0.8, 'mixup': 0.05,
            'auto_augment': 'randaugment', 'hsv_h': 0.015, 'hsv_s': 0.6, 'hsv_v': 0.4
        },
        {
            'name': 'stage2_adaptation_320', 'model': None,
            'imgsz': 320, 'epochs': 30, 'batch': 64,
            'lr0': 0.001, 'patience': 20, 'mosaic': 0.8, 'mixup': 0.05,
            'auto_augment': 'randaugment', 'hsv_h': 0.015, 'hsv_s': 0.6, 'hsv_v': 0.4
        },
        {
            'name': 'stage3_finetune_320', 'model': None,
            'imgsz': 320, 'epochs': 20, 'batch': 64,
            'lr0': 0.0002, 'patience': 10, 'mosaic': 0.5, 'mixup': 0.03,
            'auto_augment': 'randaugment', 'hsv_h': 0.01, 'hsv_s': 0.4, 'hsv_v': 0.3
        },
    ]

    prev_weights = None
    for i, s in enumerate(STAGES):
        stage_dir = DRIVE_PROJECT_ROOT / s['name']
        last_checkpoint = stage_dir / "weights" / "last.pt"
        resume_training = last_checkpoint.exists()

        if resume_training:
            model_path = str(last_checkpoint)
            print(f"\n--- ♻️ RESUMING TRAINING for {s['name']} from {model_path} ---")
        else:
            if i == 0:
                model_path = s['model']
                print(f"\n--- 🚀 STARTING TRAINING: {s['name']} (from {model_path}) ---")
            else:
                if prev_weights is None:
                    prev_stage_name = STAGES[i-1]['name']
                    prev_weights = str(DRIVE_PROJECT_ROOT / prev_stage_name / "weights" / "best.pt")
                    if not Path(prev_weights).exists():
                        raise FileNotFoundError(f"Cannot start {s['name']}: previous weights {prev_weights} not found.")
                model_path = prev_weights
                print(f"\n--- 🚀 STARTING TRAINING: {s['name']} (from {model_path}) ---")

        model = YOLO(model_path)
        try:
            model.train(
                data=str(LOCAL_DATA_YAML),
                resume=resume_training, augment=True, cos_lr=True, workers=2,
                epochs=s['epochs'], imgsz=s['imgsz'], batch=s['batch'],
                lr0=s['lr0'], patience=s['patience'],
                project=str(DRIVE_PROJECT_ROOT), name=s['name'],
                exist_ok=True, cache=False, mosaic=s['mosaic'], mixup=s['mixup'],
                auto_augment=s['auto_augment'],
                hsv_h=s['hsv_h'], hsv_s=s['hsv_s'], hsv_v=s['hsv_v'],
                optimizer="SGD", device='0', val=True, save=True, verbose=True,
            )
        except Exception as e:
            print(f"TRAINING FAILED for {s['name']}: {e}")
            raise e

        best_weights = stage_dir / "weights" / "best.pt"
        last_weights = stage_dir / "weights" / "last.pt"
        if best_weights.exists():
            prev_weights = str(best_weights)
            print(f"{s['name']} finished. Best weights found at: {prev_weights}")
        elif last_weights.exists():
            prev_weights = str(last_weights)
            print(f"{s['name']} finished, but no best.pt. Using last.pt: {prev_weights}")
        else:
            raise RuntimeError(f"Training for {s['name']} failed. No weights found.")

    print(f"Final model path (on Drive): {prev_weights}")

    print("\n--- 🚀 PART 3: STARTING TFLITE EXPORT ---")
    TFLITE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not Path(prev_weights).exists():
        print(f"Final weights {prev_weights} not found. Using default path.")
        prev_weights = str(MODEL_TO_EVAL) 
        if not Path(prev_weights).exists():
             raise FileNotFoundError(f"Cannot find model to export at {prev_weights}")

    print(f"Loading model for export: {prev_weights}")
    model = YOLO(prev_weights)
    
    exported_path_str = model.export(
        format="tflite",
        half=True,  
        imgsz=IMG_SIZE
    )
    
    exported_path = Path(exported_path_str)
    print(f"TFLite model exported temporarily to: {exported_path}")
    
    shutil.move(str(exported_path), str(TFLITE_FINAL_PATH))
    print(f"Model moved to final destination: {TFLITE_FINAL_PATH}")

    
    if not MODEL_TO_EVAL.exists():
        raise FileNotFoundError(f"Base PyTorch model not found at: {MODEL_TO_EVAL}")
    
    print(f"Loading base PyTorch model: {MODEL_TO_EVAL}")
    base_model = YOLO(str(MODEL_TO_EVAL))
    class_names = base_model.names
    print(f"Found {len(class_names)} classes: {class_names}")

    pt_metrics = None
    tflite_metrics = None
    pt_speed_data = None
    tflite_speed_data = None

    try:
        pt_metrics = base_model.val(
            data=str(LOCAL_DATA_YAML), split=VALIDATION_SPLIT_TO_USE,
            imgsz=IMG_SIZE, conf=CONF_THRESH,
            plots=False, verbose=True, device='cpu'
        )
        pt_speed_data = measure_detailed_speed(
            base_model, LOCAL_DATA_YAML, VALIDATION_SPLIT_TO_USE,
            IMG_SIZE, CONF_THRESH
        )
        print("--- PyTorch FP32 Validation Complete ---")
    except Exception as e:
        print(f"PyTorch Validation FAILED: {e}")
        raise e

    print(f"\n\n--- 📊 VALIDATING TFLite FP16 (CPU Only) ---")
    if not TFLITE_FINAL_PATH.exists():
        print(f"TFLite file not found at {TFLITE_FINAL_PATH}, skipping validation.")
        return

    try:
        tflite_model = YOLO(TFLITE_FINAL_PATH)
        tflite_metrics = tflite_model.val(
            data=str(LOCAL_DATA_YAML), split=VALIDATION_SPLIT_TO_USE,
            imgsz=IMG_SIZE, conf=CONF_THRESH,
            plots=False, verbose=True, device='cpu'
        )
        tflite_speed_data = measure_detailed_speed(
            tflite_model, LOCAL_DATA_YAML, VALIDATION_SPLIT_TO_USE,
            IMG_SIZE, CONF_THRESH
        )
        print("--- TFLite FP16 Validation Complete ---")
    except Exception as e:
        print(f"TFLITE VALIDATION FAILED: {e}")
        raise e

    if pt_metrics and tflite_metrics and pt_speed_data and tflite_speed_data:
        comparison_df = build_comparison_df(pt_metrics, tflite_metrics, pt_speed_data, tflite_speed_data)
        else:
        print("Failed validation.")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print("PIPELINE FAILED")
        import traceback
        traceback.print_exc()
