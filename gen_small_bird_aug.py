"""
小目标鸟数据增强：将现有训练集中的大鸟裁剪缩小，粘贴到天空/水面/地面背景上，
模拟远处小鸟的剪影形态，同时生成飞机等形状相似的非鸟负样本。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import random
from pathlib import Path
from shutil import copy2

BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "datasets" / "bird_detect"
TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"
OUTPUT_PREFIX = "small_"  # 生成文件的前缀

# 目标尺寸范围（像素），模拟远处小鸟
SMALL_BIRD_SIZES = [(15, 12), (20, 16), (25, 20), (30, 24), (35, 28), (40, 32), (50, 40)]
# 每张背景图上粘贴的小鸟数量
BIRDS_PER_BG = [3, 5, 8, 10]
# 生成数量
NUM_SMALL_BIRD_IMAGES = 3000
# 飞机负样本数量
NUM_AIRPLANE_NEGATIVES = 500
# 其他形状负样本数量
NUM_SHAPE_NEGATIVES = 500


def load_bird_crops(max_crops=500):
    """从训练集中裁剪出鸟的区域"""
    crops = []
    label_files = sorted(TRAIN_LBL_DIR.glob("*.txt"))
    random.shuffle(label_files)

    for lbl_path in label_files:
        if len(crops) >= max_crops:
            break
        img_path = None
        for ext in (".jpg", ".png", ".jpeg", ".bmp"):
            candidate = TRAIN_IMG_DIR / (lbl_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            # 尝试匹配任意扩展名
            for candidate in TRAIN_IMG_DIR.glob(f"{lbl_path.stem}.*"):
                if candidate.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp"):
                    img_path = candidate
                    break
        if img_path is None:
            continue

        try:
            img = cv2.imread(str(img_path))
        except Exception:
            continue
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # 只取较大的鸟（面积>5%），缩小后才有意义
                if bw * bh < 0.03:
                    continue
                # 转换为像素坐标
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue
                crop = img[y1:y2, x1:x2]
                crops.append(crop)

    print(f"  Loaded {len(crops)} bird crops")
    return crops


def load_backgrounds(num=200):
    """从训练集中提取无鸟的背景区域，以及生成纯天空/水面背景"""
    backgrounds = []

    # 1. 从训练集图片中提取背景区域（无鸟的部分）
    label_files = sorted(TRAIN_LBL_DIR.glob("*.txt"))
    random.shuffle(label_files)

    for lbl_path in label_files:
        if len(backgrounds) >= num // 2:
            break
        img_path = None
        for ext in (".jpg", ".png", ".jpeg", ".bmp"):
            candidate = TRAIN_IMG_DIR / (lbl_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            for candidate in TRAIN_IMG_DIR.glob(f"{lbl_path.stem}.*"):
                if candidate.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp"):
                    img_path = candidate
                    break
        if img_path is None:
            continue

        try:
            img = cv2.imread(str(img_path))
        except Exception:
            continue
        if img is None:
            continue
        h, w = img.shape[:2]

        # 读取标注框
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    boxes.append((x1, y1, x2, y2))

        # 提取不与任何标注框重叠的区域
        bg_size = 640
        for _ in range(5):
            rx = random.randint(0, max(0, w - bg_size))
            ry = random.randint(0, max(0, h - bg_size))
            rx2, ry2 = min(rx + bg_size, w), min(ry + bg_size, h)
            # 检查是否与任何框重叠
            overlap = False
            for bx1, by1, bx2, by2 in boxes:
                if rx < bx2 and rx2 > bx1 and ry < by2 and ry2 > by1:
                    overlap = True
                    break
            if not overlap:
                bg = img[ry:ry2, rx:rx2]
                if bg.shape[0] > 100 and bg.shape[1] > 100:
                    backgrounds.append(bg)

    # 2. 生成纯天空/水面/地面背景
    sky_colors = [(135, 180, 230), (160, 200, 240), (180, 210, 245),
                  (200, 220, 240), (170, 190, 220), (140, 170, 210)]
    water_colors = [(100, 130, 160), (80, 120, 150), (60, 100, 140),
                    (90, 130, 160), (70, 110, 145)]
    ground_colors = [(120, 140, 100), (100, 130, 80), (140, 150, 110),
                     (90, 110, 70), (130, 140, 95)]

    for _ in range(num // 2):
        bg_type = random.choice(["sky", "sky", "water", "ground", "sky_water"])
        bg = np.zeros((random.randint(400, 800), random.randint(600, 1200), 3), dtype=np.uint8)
        if bg_type == "sky":
            color = random.choice(sky_colors)
            # 渐变天空
            for row in range(bg.shape[0]):
                factor = 1.0 - row / bg.shape[0] * 0.3
                bg[row, :] = [int(c * factor) for c in color]
        elif bg_type == "water":
            color = random.choice(water_colors)
            for row in range(bg.shape[0]):
                factor = 0.8 + row / bg.shape[0] * 0.2
                bg[row, :] = [int(min(255, c * factor)) for c in color]
        elif bg_type == "ground":
            color = random.choice(ground_colors)
            bg[:, :] = color
            # 加一些纹理
            noise = np.random.randint(-15, 15, bg.shape, dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif bg_type == "sky_water":
            sky_color = random.choice(sky_colors)
            water_color = random.choice(water_colors)
            horizon = bg.shape[0] // 2 + random.randint(-50, 50)
            for row in range(bg.shape[0]):
                if row < horizon:
                    factor = 1.0 - row / horizon * 0.2
                    bg[row, :] = [int(c * factor) for c in sky_color]
                else:
                    factor = 0.9 + (row - horizon) / (bg.shape[0] - horizon) * 0.1
                    bg[row, :] = [int(min(255, c * factor)) for c in water_color]
        backgrounds.append(bg)

    print(f"  Loaded {len(backgrounds)} backgrounds")
    return backgrounds


def shrink_bird(crop, target_size):
    """将鸟裁剪缩小到目标尺寸，模拟远处剪影"""
    tw, th = target_size
    # 先缩放
    small = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    # 随机决定是否转为剪影（远处鸟颜色信息丢失）
    silhouette_mode = random.random() < 0.6
    if silhouette_mode:
        # 转为暗色剪影
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Otsu阈值分割
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 如果前景占比太小，反转
        if np.sum(mask == 255) / mask.size < 0.2:
            mask = cv2.bitwise_not(mask)

        # 创建暗色剪影
        dark_color = random.choice([
            (20, 20, 30), (30, 25, 20), (15, 20, 25), (25, 30, 25),
            (40, 35, 30), (10, 15, 20), (35, 30, 35),
        ])
        silhouette = np.full_like(small, dark_color, dtype=np.uint8)
        # 用mask把前景替换为暗色
        result = small.copy()
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch == 255, silhouette, small)

        # 轻微模糊模拟远处大气散射
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        return result
    else:
        # 保留部分颜色但降低饱和度（远处鸟颜色变淡）
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * random.uniform(0.2, 0.6)).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        return result


def paste_bird_on_background(bg, bird_img, x, y):
    """将小鸟粘贴到背景上，使用简单的alpha混合"""
    bh, bw = bird_img.shape[:2]
    h, w = bg.shape[:2]
    if x + bw > w or y + bh > h:
        return bg, False

    # 使用GrabCut或简单的边缘融合
    gray = cv2.cvtColor(bird_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # 边缘柔化
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = mask.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask, mask, mask])

    roi = bg[y:y+bh, x:x+bw]
    blended = (roi.astype(np.float32) * (1 - mask_3ch) + bird_img.astype(np.float32) * mask_3ch)
    bg[y:y+bh, x:x+bw] = blended.astype(np.uint8)
    return bg, True


def generate_airplane_silhouette(size=(60, 20)):
    """生成飞机剪影作为负样本"""
    w, h = size
    img = np.full((h, w, 3), random.choice(sky_colors if 'sky_colors' in dir() else [(135,180,230)]), dtype=np.uint8)
    # 渐变天空背景
    for row in range(h):
        factor = 1.0 - row / h * 0.2
        img[row, :] = [int(c * factor) for c in (135, 180, 230)]

    # 画飞机剪影
    dark = (20, 20, 30)
    cx, cy = w // 2, h // 2
    # 机身
    cv2.ellipse(img, (cx, cy), (w//3, h//8), 0, 0, 360, dark, -1)
    # 机翼
    wing_pts = np.array([
        [cx - w//6, cy],
        [cx - w//3, cy - h//4],
        [cx + w//6, cy],
        [cx + w//3, cy - h//4],
    ])
    cv2.fillConvexPoly(img, wing_pts, dark)
    # 尾翼
    tail_pts = np.array([
        [cx + w//4, cy],
        [cx + w//3, cy - h//5],
        [cx + w//3 + 5, cy],
    ])
    cv2.fillConvexPoly(img, tail_pts, dark)
    return img


def generate_shape_negatives(num, output_img_dir, output_lbl_dir):
    """生成形状相似的非鸟负样本（飞机、风筝、无人机等）"""
    sky_colors = [(135, 180, 230), (160, 200, 240), (180, 210, 245), (200, 220, 240)]

    for i in range(num):
        bg_h = random.randint(300, 600)
        bg_w = random.randint(400, 900)
        bg = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
        sky_c = random.choice(sky_colors)
        for row in range(bg_h):
            factor = 1.0 - row / bg_h * 0.3
            bg[row, :] = [int(c * factor) for c in sky_c]

        # 随机画1-3个飞机/风筝形状
        num_objects = random.randint(1, 3)
        for _ in range(num_objects):
            obj_w = random.randint(30, 80)
            obj_h = random.randint(10, 30)
            ox = random.randint(20, bg_w - obj_w - 20)
            oy = random.randint(20, bg_h - obj_h - 20)
            dark = random.choice([(20, 20, 30), (30, 25, 20), (40, 35, 30)])
            cx, cy = ox + obj_w // 2, oy + obj_h // 2

            shape_type = random.choice(["airplane", "kite", "drone"])
            if shape_type == "airplane":
                cv2.ellipse(bg, (cx, cy), (obj_w//3, obj_h//3), 0, 0, 360, dark, -1)
                wing_pts = np.array([
                    [cx - obj_w//6, cy], [cx - obj_w//2, cy - obj_h//2],
                    [cx + obj_w//6, cy], [cx + obj_w//2, cy - obj_h//2],
                ])
                cv2.fillConvexPoly(bg, wing_pts, dark)
            elif shape_type == "kite":
                pts = np.array([
                    [cx, cy - obj_h//2], [cx + obj_w//3, cy],
                    [cx, cy + obj_h//2], [cx - obj_w//3, cy],
                ])
                cv2.fillConvexPoly(bg, pts, dark)
                # 风筝线
                cv2.line(bg, (cx, cy + obj_h//2), (cx + random.randint(-20, 20), cy + obj_h), dark, 1)
            elif shape_type == "drone":
                cv2.rectangle(bg, (cx - obj_w//4, cy - obj_h//4), (cx + obj_w//4, cy + obj_h//4), dark, -1)
                for dx in [-1, 1]:
                    cv2.circle(bg, (cx + dx * obj_w//3, cy - obj_h//3), obj_h//4, dark, -1)

        # 轻微模糊
        bg = cv2.GaussianBlur(bg, (3, 3), 0.5)

        fname = f"{OUTPUT_PREFIX}neg_shape_{i:05d}.jpg"
        cv2.imwrite(str(output_img_dir / fname), bg)
        # 负样本：空标签文件（无目标）
        (output_lbl_dir / fname.replace(".jpg", ".txt")).write_text("")

    print(f"  Generated {num} shape negatives")


def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 50)
    print("  Small Target Bird Data Augmentation")
    print("=" * 50)

    # 1. 加载鸟裁剪和背景
    print("\n[1/4] Loading bird crops...")
    bird_crops = load_bird_crops(max_crops=500)
    if not bird_crops:
        print("Error: No bird crops found!")
        return

    print("\n[2/4] Loading backgrounds...")
    backgrounds = load_backgrounds(num=300)
    if not backgrounds:
        print("Error: No backgrounds found!")
        return

    # 2. 生成小目标鸟图片
    print(f"\n[3/4] Generating {NUM_SMALL_BIRD_IMAGES} small bird images...")
    img_dir = TRAIN_IMG_DIR
    lbl_dir = TRAIN_LBL_DIR

    for i in range(NUM_SMALL_BIRD_IMAGES):
        bg = random.choice(backgrounds).copy()
        bg_h, bg_w = bg.shape[:2]

        # 随机调整背景大小
        if random.random() < 0.3:
            scale = random.uniform(0.5, 2.0)
            new_w = int(bg_w * scale)
            new_h = int(bg_h * scale)
            bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            bg_h, bg_w = bg.shape[:2]

        # 限制最大尺寸
        if bg_w > 1920:
            bg = cv2.resize(bg, (1920, int(1920 * bg_h / bg_w)))
            bg_h, bg_w = bg.shape[:2]

        num_birds = random.choice(BIRDS_PER_BG)
        labels = []

        for _ in range(num_birds):
            crop = random.choice(bird_crops)
            target_size = random.choice(SMALL_BIRD_SIZES)
            small_bird = shrink_bird(crop, target_size)

            bh, bw = small_bird.shape[:2]
            # 随机位置（偏向天空区域上半部分）
            if random.random() < 0.7:
                # 天空区域
                x = random.randint(10, max(11, bg_w - bw - 10))
                y = random.randint(10, max(11, bg_h // 2))
            else:
                x = random.randint(10, max(11, bg_w - bw - 10))
                y = random.randint(10, max(11, bg_h - bh - 10))

            bg, success = paste_bird_on_background(bg, small_bird, x, y)
            if success:
                # YOLO格式: class cx cy w h (归一化)
                cx = (x + bw / 2) / bg_w
                cy = (y + bh / 2) / bg_h
                nw = bw / bg_w
                nh = bh / bg_h
                labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # 保存
        fname = f"{OUTPUT_PREFIX}bird_{i:05d}.jpg"
        cv2.imwrite(str(img_dir / fname), bg)
        lbl_path = lbl_dir / fname.replace(".jpg", ".txt")
        lbl_path.write_text("\n".join(labels))

        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{NUM_SMALL_BIRD_IMAGES}")

    print(f"  Generated {NUM_SMALL_BIRD_IMAGES} small bird images")

    # 3. 生成形状负样本（飞机、风筝、无人机）
    print(f"\n[4/4] Generating {NUM_AIRPLANE_NEGATIVES} shape negatives...")
    generate_shape_negatives(NUM_AIRPLANE_NEGATIVES, img_dir, lbl_dir)

    # 统计
    total_imgs = len(list(img_dir.glob("*")))
    total_lbls = len(list(lbl_dir.glob("*.txt")))
    print(f"\n  Dataset now: {total_imgs} images, {total_lbls} labels")
    print("  Done! Ready for training.")


if __name__ == "__main__":
    main()
