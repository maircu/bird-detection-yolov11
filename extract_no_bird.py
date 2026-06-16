import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import cv2
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent.resolve()
VIDEO_DIR = BASE_DIR / "video"
MODEL_CANDIDATES = [
    BASE_DIR / "runs" / "detect" / "bird_detect_v3" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "bird_detect" / "weights" / "best.pt",
]
TRAIN_IMG_DIR = BASE_DIR / "datasets" / "bird_detect" / "train" / "images"
TRAIN_LBL_DIR = BASE_DIR / "datasets" / "bird_detect" / "train" / "labels"
VAL_IMG_DIR = BASE_DIR / "datasets" / "bird_detect" / "val" / "images"
VAL_LBL_DIR = BASE_DIR / "datasets" / "bird_detect" / "val" / "labels"


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


def extract_no_bird_frames(video_path, model, max_frames=200, skip=10, conf=0.15, imgsz=640):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"  Video: {video_path.name} ({total} frames, {fps} fps)")

    saved = 0
    frame_idx = 0
    candidates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        results = model.predict(source=frame, conf=conf, verbose=False, imgsz=imgsz)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            candidates.append((frame_idx, frame))

        if frame_idx % 500 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"\r  Progress: {frame_idx}/{total} ({pct:.1f}%), no-bird frames: {len(candidates)}", end="", flush=True)

    cap.release()
    print(f"\r  Scanned {frame_idx} frames, found {len(candidates)} no-bird frames" + " " * 20)

    if len(candidates) > max_frames:
        random.seed(42)
        candidates = random.sample(candidates, max_frames)

    out_dir = BASE_DIR / "no_bird_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fidx, frame in candidates:
        fname = f"nobird_{video_path.stem}_f{fidx:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame)
        saved += 1

    print(f"  Saved {saved} no-bird frames to: {out_dir}")
    return saved


def apply_to_dataset():
    frames_dir = BASE_DIR / "no_bird_frames"
    images = list(frames_dir.glob("nobird_*.jpg"))
    if not images:
        print("  No no-bird frames found!")
        return

    print(f"  Found {len(images)} no-bird frames")

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


def main():
    print("=" * 60)
    print("  Extract No-Bird Frames from Videos")
    print("  These are REAL natural scenes without any birds")
    print("  (rocks, flowers, branches, other animals, etc.)")
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

    max_input = input("  Max no-bird frames per video (default=200): ").strip()
    max_frames = int(max_input) if max_input else 200

    skip_input = input("  Frame skip (default=10): ").strip()
    skip = int(skip_input) if skip_input else 10

    print(f"\n  Loading model...")
    model = YOLO(str(model_path))

    total = 0
    for v in selected:
        print(f"\n  Processing: {v.name}")
        total += extract_no_bird_frames(v, model, max_frames=max_frames, skip=skip)

    print(f"\n  Total: {total} no-bird frames extracted")
    print(f"\n{'=' * 60}")
    print(f"  Review: open no_bird_frames/ folder")
    print(f"  DELETE any frames that accidentally contain birds")
    print(f"  KEEP all frames with natural scenery (rocks, flowers, etc.)")
    print(f"  Then run: python extract_no_bird.py apply")
    print(f"{'=' * 60}")


def run_auto():
    print("=" * 60)
    print("  Auto-extract No-Bird Frames (all videos)")
    print("=" * 60)

    model_path = find_model()
    if not model_path:
        print("Error: No model found!")
        return

    print(f"  Model: {model_path.parent.parent.name}/{model_path.name}")

    videos = list_videos()
    if not videos:
        print("No videos found!")
        return

    print(f"  Found {len(videos)} videos")

    print(f"\n  Loading model...")
    model = YOLO(str(model_path))

    total = 0
    for v in videos:
        print(f"\n  Processing: {v.name}")
        total += extract_no_bird_frames(v, model, max_frames=200, skip=10)

    print(f"\n  Total: {total} no-bird frames extracted")
    print(f"\n  Auto-applying to dataset...")
    apply_to_dataset()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'apply':
        apply_to_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == 'auto':
        run_auto()
    else:
        main()
