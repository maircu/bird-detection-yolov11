import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from pathlib import Path
import random

DATASET_DIR = Path(r'C:\Users\HabichtZhang\Desktop\机器视觉\datasets\shanghai_33cls')
IMG_DIR = DATASET_DIR / 'train' / 'images'
LBL_DIR = DATASET_DIR / 'train' / 'labels'

NUM_AUG_PER_IMAGE = 1
SEED = 42


def apply_occlusion(img, rng):
    h, w = img.shape[:2]
    result = img.copy()
    num_occ = rng.randint(1, 3)

    for _ in range(num_occ):
        occ_type = rng.choice(['branch', 'leaf', 'rect'])
        color = rng.randint(30, 130)
        if occ_type == 'branch':
            color = (max(0, color - 15), color, min(255, color + 20))
        elif occ_type == 'leaf':
            color = (rng.randint(10, 50), rng.randint(60, 140), rng.randint(20, 80))
        else:
            color = (color, color, color)

        alpha = rng.uniform(0.6, 0.95)

        if occ_type == 'branch':
            for _ in range(rng.randint(1, 3)):
                x1 = rng.randint(0, w)
                y1 = rng.randint(0, h)
                angle = rng.uniform(0, np.pi)
                length = rng.randint(min(h, w) // 6, min(h, w) // 2)
                thickness = rng.randint(2, max(3, min(h, w) // 80))
                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))
                cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        elif occ_type == 'leaf':
            for _ in range(rng.randint(1, 3)):
                cx = rng.randint(0, w)
                cy = rng.randint(0, h)
                size = rng.randint(min(h, w) // 20, min(h, w) // 8)
                aspect = rng.uniform(0.4, 0.7)
                angle = rng.uniform(0, 180)
                cv2.ellipse(result, (cx, cy), (int(size * aspect), size), angle, 0, 360, color, -1)

        else:
            rw = rng.randint(w // 20, w // 5)
            rh = rng.randint(h // 20, h // 5)
            rx = rng.randint(0, max(1, w - rw))
            ry = rng.randint(0, max(1, h - rh))
            overlay = result.copy()
            cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), color, -1)
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    return result


def main():
    rng = random.Random(SEED)

    images = sorted([f for f in IMG_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png') and not f.name.startswith('occ')])
    print(f"Found {len(images)} original training images")
    print(f"Generating {NUM_AUG_PER_IMAGE} occluded version(s) per image")
    print()

    existing = set(f.name for f in IMG_DIR.iterdir() if f.name.startswith('occ'))
    print(f"Already existing augmented images: {len(existing)}")

    total = 0
    skipped = 0
    for idx, img_path in enumerate(images):
        stem = img_path.stem
        suffix = img_path.suffix
        lbl_path = LBL_DIR / (stem + '.txt')

        if not lbl_path.exists():
            continue

        lbl_content = lbl_path.read_text(encoding='utf-8')

        for aug_i in range(NUM_AUG_PER_IMAGE):
            aug_stem = f"occ{aug_i}_{stem}" if NUM_AUG_PER_IMAGE > 1 else f"occ_{stem}"
            aug_name = aug_stem + suffix

            if aug_name in existing:
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            aug_img = apply_occlusion(img, rng)
            cv2.imwrite(str(IMG_DIR / aug_name), aug_img)
            (LBL_DIR / (aug_stem + '.txt')).write_text(lbl_content, encoding='utf-8')
            total += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Progress: {idx + 1}/{len(images)}, generated: {total}, skipped: {skipped}")

    print(f"\nDone! Generated: {total}, Skipped (already exist): {skipped}")
    print(f"New training set size: {len(images) + total + skipped} images")


if __name__ == '__main__':
    main()
