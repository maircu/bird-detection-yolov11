import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TQDM_ASCII'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import torch
from ultralytics import YOLO
import time
import glob
import shutil

KEEP_FILES = {
    'weights', 'results.csv', 'results.png', 'args.yaml',
    'confusion_matrix.png', 'confusion_matrix_normalized.png',
    'BoxPR_curve.png', 'labels.jpg',
}

def cleanup_run_dir(run_dir):
    removed = 0
    for f in os.listdir(run_dir):
        fp = os.path.join(run_dir, f)
        if f in KEEP_FILES:
            continue
        if os.path.isdir(fp):
            shutil.rmtree(fp)
            removed += 1
        else:
            os.remove(fp)
            removed += 1
    return removed

def main():
    print("=" * 60)
    print("YOLOv11n Shanghai Birds Training (10 epochs)")
    print("=" * 60)

    print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    torch.cuda.empty_cache()

    for f in glob.glob('datasets/archive/Bird-Breeds--copy-2-2/**/*.cache', recursive=True):
        os.remove(f)

    model = YOLO('yolo11n.pt')

    print("\nDataset: Shanghai Birds (34 classes)")
    print("Config: epochs=10, batch=8, imgsz=640, workers=2, amp=True")
    print("Starting training...\n")

    start = time.time()

    results = model.train(
        data='datasets/shanghai_birds/data.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        device=0,
        project='runs/detect',
        name='shanghai_bird_10ep',
        exist_ok=True,
        workers=2,
        amp=True,
        verbose=True,
        cache=False,
        rect=False,
        patience=20,
        save=True,
        save_period=-1,
        lr0=0.001,
        optimizer='AdamW',
        cos_lr=True,
        freeze=0,
        close_mosaic=15,
        dropout=0.1,
        box=10.0,
        cls=0.3,
        dfl=2.0,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.15,
        scale=0.3,
        shear=3.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        auto_augment='randaugment',
        erasing=0.4,
        seed=42,
        deterministic=False,
        val=True,
        plots=True,
    )

    elapsed = time.time() - start

    run_dir = 'runs/detect/shanghai_bird_10ep'
    if os.path.exists(run_dir):
        n = cleanup_run_dir(run_dir)
        print(f"Cleaned up {n} unnecessary files from {run_dir}")

    print(f"\n{'=' * 60}")
    print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if hasattr(results, 'box'):
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
