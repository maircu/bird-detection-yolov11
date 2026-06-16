import os
import ssl
import urllib.request
import zipfile
import json
import shutil
import random

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = r"D:\negative_dataset"
COCO_BIRD_CATEGORY_ID = 14

VAL_URL = "https://images.cocodataset.org/zips/val2017.zip"
TRAIN_URL = "https://images.cocodataset.org/zips/train2017.zip"
ANNO_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"

BIRD_DETECT_DIR = r"c:\Users\HabichtZhang\Desktop\机器视觉\datasets\bird_detect"


def download_file(url, save_path):
    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"[跳过] 已存在: {save_path} ({size_mb:.1f}MB)")
        return
    print(f"[下载] {url}")
    print(f"[保存] {save_path}")
    tmp_path = save_path + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=download_progress)
        os.rename(tmp_path, save_path)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n[完成] {save_path} ({size_mb:.1f}MB)")
    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    mb_down = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f"\r  进度: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f}MB)", end="", flush=True)


def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"[跳过] 文件不存在: {zip_path}")
        return
    print(f"[解压] {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"[完成] 解压完成")


def filter_non_bird_images(split="val2017", max_images=3000):
    anno_dir = os.path.join(BASE_DIR, "annotations")
    anno_file = os.path.join(anno_dir, f"instances_{split}.json")

    if not os.path.exists(anno_file):
        print(f"[错误] 标注文件不存在: {anno_file}")
        return 0

    print(f"[加载] {anno_file}")
    with open(anno_file, 'r') as f:
        coco = json.load(f)

    all_image_ids = set(img['id'] for img in coco['images'])
    bird_image_ids = set()
    for ann in coco['annotations']:
        if ann['category_id'] == COCO_BIRD_CATEGORY_ID:
            bird_image_ids.add(ann['image_id'])

    non_bird_ids = all_image_ids - bird_image_ids
    print(f"[统计] 总图像: {len(all_image_ids)}, 含鸟: {len(bird_image_ids)}, 不含鸟: {len(non_bird_ids)}")

    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

    non_bird_files = [id_to_filename[iid] for iid in non_bird_ids if iid in id_to_filename]
    random.shuffle(non_bird_files)
    selected = non_bird_files[:max_images]
    print(f"[选择] 取前{max_images}张非鸟类图像（共{len(non_bird_files)}张可选）")

    images_src = os.path.join(BASE_DIR, split)
    neg_train_dir = os.path.join(BIRD_DETECT_DIR, "train", "images")
    neg_val_dir = os.path.join(BIRD_DETECT_DIR, "val", "images")
    lbl_train_dir = os.path.join(BIRD_DETECT_DIR, "train", "labels")
    lbl_val_dir = os.path.join(BIRD_DETECT_DIR, "val", "labels")

    for d in [neg_train_dir, neg_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    train_count = 0
    val_count = 0
    prefix = "coco_neg_"

    for i, fname in enumerate(selected):
        src = os.path.join(images_src, fname)
        if not os.path.exists(src):
            continue

        stem = os.path.splitext(fname)[0]
        is_train = random.random() < 0.8
        img_dst_dir = neg_train_dir if is_train else neg_val_dir
        lbl_dst_dir = lbl_train_dir if is_train else lbl_val_dir

        img_dst = os.path.join(img_dst_dir, f"{prefix}{stem}.jpg")
        lbl_dst = os.path.join(lbl_dst_dir, f"{prefix}{stem}.txt")

        if os.path.exists(img_dst):
            continue

        shutil.copy2(src, img_dst)
        with open(lbl_dst, 'w') as f:
            pass

        if is_train:
            train_count += 1
        else:
            val_count += 1

    print(f"[完成] 复制负样本: train={train_count}, val={val_count}, 总计={train_count + val_count}")
    return train_count + val_count


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 60)
    print("COCO 2017 负样本下载与处理管道")
    print("=" * 60)

    print("\n[步骤1] 下载标注文件...")
    anno_zip = os.path.join(BASE_DIR, "annotations_trainval2017.zip")
    download_file(ANNO_URL, anno_zip)

    print("\n[步骤2] 下载val2017图片（1GB）...")
    val_zip = os.path.join(BASE_DIR, "val2017.zip")
    download_file(VAL_URL, val_zip)

    print("\n[步骤3] 解压标注文件...")
    extract_zip(anno_zip, BASE_DIR)

    print("\n[步骤4] 解压val2017图片...")
    extract_zip(val_zip, BASE_DIR)

    print("\n[步骤5] 过滤非鸟类图像并加入数据集...")
    total = filter_non_bird_images(split="val2017", max_images=3000)

    print("\n" + "=" * 60)
    print(f"[全部完成] 共添加 {total} 张COCO负样本到bird_detect数据集")
    print(f"数据集路径: {BIRD_DETECT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
