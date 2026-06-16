import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent.resolve()
VIDEO_DIR = BASE_DIR / "video"
MODEL_CANDIDATES = [
    BASE_DIR / "runs" / "detect" / "bird_detect_v3" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "bird_detect_v2" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "bird_detect" / "weights" / "best.pt",
]
TRAIN_IMG_DIR = BASE_DIR / "datasets" / "bird_detect" / "train" / "images"
TRAIN_LBL_DIR = BASE_DIR / "datasets" / "bird_detect" / "train" / "labels"
VAL_IMG_DIR = BASE_DIR / "datasets" / "bird_detect" / "val" / "images"
VAL_LBL_DIR = BASE_DIR / "datasets" / "bird_detect" / "val" / "labels"
REVIEW_DIR = BASE_DIR / "hard_negatives_review"
CROPS_DIR = REVIEW_DIR / "crops"


def find_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


def list_videos():
    videos = []
    for ext in ("*.mp4", "*.avi", "*.mkv"):
        videos.extend(sorted(VIDEO_DIR.glob(ext)))
    return videos


def extract_crops(video_path, model, conf=0.10, max_crops=300, skip=5, imgsz=640, low_conf_only=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"  Video: {video_path.name} ({total} frames, {fps} fps)")

    crop_count = 0
    frame_idx = 0

    while crop_count < max_crops:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        results = model.predict(source=frame, conf=conf, verbose=False, imgsz=imgsz)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes
        h, w = frame.shape[:2]

        if low_conf_only and len(boxes) > 0:
            has_high_conf = any(float(c) > 0.5 for c in boxes.conf)
            if has_high_conf:
                low_mask = boxes.conf < 0.35
                if not low_mask.any():
                    frame_idx += 0
                    continue
                low_indices = low_mask.nonzero(as_tuple=True)[0]
            else:
                low_indices = range(len(boxes))
        else:
            low_indices = range(len(boxes))

        for i in low_indices:
            if crop_count >= max_crops:
                break

            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            conf_val = float(boxes.conf[i])

            bw = x2 - x1
            bh = y2 - y1
            pad = max(bw, bh) // 2
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            crop_x1 = max(0, cx - bw // 2 - pad)
            crop_y1 = max(0, cy - bh // 2 - pad)
            crop_x2 = min(w, cx + bw // 2 + pad)
            crop_y2 = min(h, cy + bh // 2 + pad)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue

            min_side = min(crop.shape[:2])
            if min_side < 64:
                scale = 64 / min_side
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            fname = f"c{crop_count:04d}_conf{conf_val:.2f}_{video_path.stem}_f{frame_idx}.jpg"
            cv2.imwrite(str(CROPS_DIR / fname), crop)
            crop_count += 1

        if frame_idx % 500 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"\r  Progress: {frame_idx}/{total} ({pct:.1f}%), crops: {crop_count}", end="", flush=True)

    cap.release()
    print(f"\r  Done: {frame_idx} frames scanned, {crop_count} crops extracted" + " " * 20)
    return crop_count


def main():
    print("=" * 60)
    print("  Hard Negative Mining (Crop Mode)")
    print("  Each detection box -> separate crop image")
    print("=" * 60)

    model_path = find_model()
    if not model_path:
        print("Error: No model found!")
        return

    print(f"  Model: {model_path.parent.parent.name}/{model_path.name}")

    videos = list_videos()
    if not videos:
        print(f"No videos found in: {VIDEO_DIR}")
        return

    print(f"\n  Available videos:")
    for i, v in enumerate(videos, 1):
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"    {i}. {v.name}  ({size_mb:.1f} MB)")

    print()
    choice = input("  Select video number (or 'a' for all): ").strip()
    if choice.lower() == 'a':
        selected = videos
    else:
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(videos):
                print(f"Invalid choice: {choice}")
                return
            selected = [videos[idx]]
        except ValueError:
            print(f"Invalid input: {choice}")
            return

    conf_input = input("  Min confidence (default=0.10): ").strip()
    conf = float(conf_input) if conf_input else 0.10

    max_input = input("  Max crops per video (default=300): ").strip()
    max_crops = int(max_input) if max_input else 300

    skip_input = input("  Frame skip (default=5): ").strip()
    skip = int(skip_input) if skip_input else 5

    print(f"\n  Filter mode:")
    print(f"    1. Low-confidence only (skip frames with high-conf birds)")
    print(f"    2. All detections")
    mode_input = input("  Select (default=1): ").strip()
    low_conf_only = mode_input != "2"

    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading model...")
    model = YOLO(str(model_path))

    total = 0
    for v in selected:
        print(f"\n  Processing: {v.name}")
        total += extract_crops(v, model, conf=conf, max_crops=max_crops, skip=skip, low_conf_only=low_conf_only)

    print(f"\n  Total: {total} crops saved to: {CROPS_DIR}")
    print(f"\n{'=' * 60}")
    print(f"  Review: open {CROPS_DIR}")
    print(f"  DELETE crops that show real birds")
    print(f"  KEEP crops that are false positives (buildings, leaves, etc.)")
    print(f"  Then run: python hard_negative_mining.py apply")
    print(f"{'=' * 60}")


def apply_negatives():
    print(f"\n{'=' * 60}")
    print(f"  Applying Negatives to Dataset")
    print(f"{'=' * 60}")

    images = list(CROPS_DIR.glob("c*.jpg"))
    if not images:
        print("  No crop images found!")
        return

    print(f"  Found {len(images)} crop images")

    import random
    random.seed(42)
    random.shuffle(images)
    n_val = max(1, len(images) // 5)
    val_set = set(f.stem for f in images[:n_val])

    train_count = 0
    val_count = 0

    for img_file in images:
        stem = img_file.stem
        if stem in val_set:
            shutil.copy2(str(img_file), str(VAL_IMG_DIR / f"{stem}.jpg"))
            with open(VAL_LBL_DIR / f"{stem}.txt", 'w') as f:
                pass
            val_count += 1
        else:
            shutil.copy2(str(img_file), str(TRAIN_IMG_DIR / f"{stem}.jpg"))
            with open(TRAIN_LBL_DIR / f"{stem}.txt", 'w') as f:
                pass
            train_count += 1

    for cache in BASE_DIR.glob("datasets/bird_detect/**/*.cache"):
        cache.unlink()

    print(f"  Train: {train_count} negatives added")
    print(f"  Val:   {val_count} negatives added")
    print(f"\n  Now finetune: python train_bird_detect.py finetune_v3")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'apply':
        apply_negatives()
    else:
        main()
