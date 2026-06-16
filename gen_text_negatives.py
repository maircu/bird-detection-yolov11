import os
import sys
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent.resolve()
FONTS_DIR = Path(r"C:\Windows\Fonts")
TRAIN_IMG_DIR = BASE_DIR / "datasets" / "bird_detect" / "train" / "images"
TRAIN_LBL_DIR = BASE_DIR / "datasets" / "bird_detect" / "train" / "labels"
VAL_IMG_DIR = BASE_DIR / "datasets" / "bird_detect" / "val" / "images"
VAL_LBL_DIR = BASE_DIR / "datasets" / "bird_detect" / "val" / "labels"

FONT_FILES = [
    "msyh.ttc", "msyhbd.ttc", "msyhl.ttc",
    "simhei.ttf", "simsun.ttc", "simkai.ttf", "simfang.ttf",
    "STKAITI.TTF", "STSONG.TTF", "STXIHEI.TTF", "STFANGSO.TTF",
    "SIMYOU.TTF", "SIMLI.TTF",
]

CHINESE_TEXTS = [
    "鸟类", "白鹭", "翠鸟", "红隼", "喜鹊", "麻雀", "燕子", "鸽子",
    "天鹅", "鸳鸯", "鹦鹉", "画眉", "百灵", "杜鹃", "黄鹂", "白鹳",
    "东方白鹳", "黑脸琵鹭", "上海观鸟", "常见鸟类", "湿地公园",
    "自然保护区", "野生动物", "候鸟迁徙", "观鸟指南", "鸟类图鉴",
    "第1集", "第2集", "第3集", "上集", "下集", "预告", "精彩片段",
    "高清", "4K", "8K", "慢动作", "特写", "航拍", "纪录片",
    "上海", "北京", "广州", "深圳", "杭州", "南京", "成都",
    "春", "夏", "秋", "冬", "晨", "暮", "雨", "晴",
    "第一章", "第二章", "第三章", "简介", "目录", "总结",
    "拍摄于", "2024年", "2025年", "3月", "10月",
    "保护鸟类", "爱护自然", "生态保护", "人与自然",
]

WATERMARK_TEXTS = [
    "LOGO", "HD", "4K", "LIVE", "REC", "CAM1", "CAM2",
    "BBC", "CCTV", "NG", "DISCOVERY",
    "www.", ".com", ".cn", "@", "#",
    "VIP", "PRO", "PREMIUM", "OFFICIAL",
]

BG_COLORS = [
    (0, 0, 0), (20, 20, 30), (40, 40, 50), (10, 30, 10),
    (30, 10, 10), (50, 50, 60), (15, 15, 25), (25, 35, 20),
    (60, 60, 70), (80, 80, 90), (100, 100, 110), (35, 45, 55),
    (45, 35, 30), (20, 40, 40), (50, 30, 40), (30, 50, 35),
]

TEXT_COLORS = [
    (255, 255, 255), (255, 255, 200), (200, 200, 200),
    (255, 220, 100), (100, 200, 255), (255, 100, 100),
    (100, 255, 100), (220, 180, 255), (255, 180, 50),
    (200, 200, 255), (180, 255, 180), (255, 200, 200),
]


def load_fonts():
    fonts = []
    for fname in FONT_FILES:
        fpath = FONTS_DIR / fname
        if fpath.exists():
            fonts.append(fpath)
    return fonts


def make_text_image(font_path, text, font_size, text_color, bg_color, img_w, img_h, position):
    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    noise_intensity = random.randint(5, 25)
    for _ in range(random.randint(50, 300)):
        nx = random.randint(0, img_w - 1)
        ny = random.randint(0, img_h - 1)
        nr = random.randint(0, noise_intensity)
        nc = tuple(max(0, min(255, c + nr - noise_intensity // 2)) for c in bg_color)
        draw.point((nx, ny), fill=nc)

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        return img

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if position == "top":
        x, y = (img_w - tw) // 2, random.randint(10, 40)
    elif position == "bottom":
        x, y = (img_w - tw) // 2, img_h - th - random.randint(10, 40)
    elif position == "topleft":
        x, y = random.randint(10, 30), random.randint(10, 30)
    elif position == "topright":
        x, y = img_w - tw - random.randint(10, 30), random.randint(10, 30)
    elif position == "bottomleft":
        x, y = random.randint(10, 30), img_h - th - random.randint(10, 30)
    elif position == "bottomright":
        x, y = img_w - tw - random.randint(10, 30), img_h - th - random.randint(10, 30)
    elif position == "center":
        x, y = (img_w - tw) // 2, (img_h - th) // 2
    else:
        x, y = random.randint(10, max(11, img_w - tw - 10)), random.randint(10, max(11, img_h - th - 10))

    if random.random() < 0.4:
        shadow_color = (0, 0, 0)
        draw.text((x + 2, y + 2), text, font=font, fill=shadow_color)

    draw.text((x, y), text, font=font, fill=text_color)

    if random.random() < 0.3:
        outline_color = (0, 0, 0) if sum(text_color) > 400 else (255, 255, 255)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=text_color)

    return img


def main():
    print("=" * 60)
    print("  Text Negative Sample Generator")
    print("  Auto-generates Chinese text images as negatives")
    print("=" * 60)

    fonts = load_fonts()
    if not fonts:
        print("Error: No Chinese fonts found!")
        return

    print(f"  Found {len(fonts)} fonts")

    n_train = 500
    n_val = 100
    if len(sys.argv) > 1:
        n_train = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_val = int(sys.argv[2])

    positions = ["top", "bottom", "topleft", "topright", "bottomleft", "bottomright", "center", "random"]

    total = n_train + n_val
    generated = 0

    for i in range(n_train):
        font_path = random.choice(fonts)
        text = random.choice(CHINESE_TEXTS + WATERMARK_TEXTS)
        if random.random() < 0.3:
            text = random.choice(CHINESE_TEXTS) + " " + random.choice(WATERMARK_TEXTS)

        font_size = random.randint(16, 72)
        text_color = random.choice(TEXT_COLORS)
        bg_color = random.choice(BG_COLORS)
        img_w = random.randint(320, 960)
        img_h = random.randint(240, 720)
        position = random.choice(positions)

        img = make_text_image(font_path, text, font_size, text_color, bg_color, img_w, img_h, position)

        stem = f"txt_neg_{i:04d}"
        img.save(str(TRAIN_IMG_DIR / f"{stem}.jpg"), quality=90)
        with open(TRAIN_LBL_DIR / f"{stem}.txt", 'w') as f:
            pass

        generated += 1
        if generated % 100 == 0:
            print(f"  Generated: {generated}/{total}")

    for i in range(n_val):
        font_path = random.choice(fonts)
        text = random.choice(CHINESE_TEXTS + WATERMARK_TEXTS)
        if random.random() < 0.3:
            text = random.choice(CHINESE_TEXTS) + " " + random.choice(WATERMARK_TEXTS)

        font_size = random.randint(16, 72)
        text_color = random.choice(TEXT_COLORS)
        bg_color = random.choice(BG_COLORS)
        img_w = random.randint(320, 960)
        img_h = random.randint(240, 720)
        position = random.choice(positions)

        img = make_text_image(font_path, text, font_size, text_color, bg_color, img_w, img_h, position)

        stem = f"txt_neg_val_{i:04d}"
        img.save(str(VAL_IMG_DIR / f"{stem}.jpg"), quality=90)
        with open(VAL_LBL_DIR / f"{stem}.txt", 'w') as f:
            pass

        generated += 1
        if generated % 100 == 0:
            print(f"  Generated: {generated}/{total}")

    for cache in BASE_DIR.glob("datasets/bird_detect/**/*.cache"):
        cache.unlink()

    print(f"\n  Done! Generated {generated} text negatives")
    print(f"  Train: {n_train}, Val: {n_val}")
    print(f"\n  Now finetune: python train_bird_detect.py finetune")


if __name__ == '__main__':
    main()
