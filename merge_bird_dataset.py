import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

CUB_DIR = BASE_DIR / "datasets" / "birds"
SH_DIR = BASE_DIR / "datasets" / "shanghai_33cls"
OUT_DIR = BASE_DIR / "datasets" / "bird_detect"

SPLITS = {
    "train": {
        "cub_img": CUB_DIR / "images" / "train",
        "cub_lbl": CUB_DIR / "labels" / "train",
        "sh_img": SH_DIR / "train" / "images",
        "sh_lbl": SH_DIR / "train" / "labels",
    },
    "val": {
        "cub_img": CUB_DIR / "images" / "val",
        "cub_lbl": CUB_DIR / "labels" / "val",
        "sh_img": SH_DIR / "valid" / "images",
        "sh_lbl": SH_DIR / "valid" / "labels",
    },
}


def convert_labels(src_lbl_dir, dst_lbl_dir, prefix):
    count = 0
    empty_count = 0
    if not src_lbl_dir.exists():
        print(f"  [Skip] {src_lbl_dir} not found")
        return 0, 0

    for lbl_file in sorted(src_lbl_dir.glob("*.txt")):
        dst_file = dst_lbl_dir / f"{prefix}_{lbl_file.name}"

        lines_out = []
        with open(lbl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    parts[0] = "0"
                    lines_out.append(" ".join(parts))

        with open(dst_file, "w") as f:
            f.write("\n".join(lines_out))

        if lines_out:
            count += 1
        else:
            empty_count += 1

    return count, empty_count


def copy_images(src_img_dir, dst_img_dir, prefix):
    count = 0
    if not src_img_dir.exists():
        print(f"  [Skip] {src_img_dir} not found")
        return 0

    for img_file in sorted(src_img_dir.glob("*")):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        dst_file = dst_img_dir / f"{prefix}_{img_file.name}"
        shutil.copy2(img_file, dst_file)
        count += 1

    return count


def main():
    print("=" * 60)
    print("  Merging CUB-200 + Shanghai Birds -> Single-class Bird Detection")
    print("=" * 60)

    for split_name, paths in SPLITS.items():
        print(f"\n--- Processing {split_name} ---")

        dst_img_dir = OUT_DIR / split_name / "images"
        dst_lbl_dir = OUT_DIR / split_name / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        print("  Copying CUB-200 images...")
        cub_img_count = copy_images(paths["cub_img"], dst_img_dir, "cub")
        print(f"  -> {cub_img_count} images copied")

        print("  Converting CUB-200 labels...")
        cub_lbl_count, cub_empty = convert_labels(paths["cub_lbl"], dst_lbl_dir, "cub")
        print(f"  -> {cub_lbl_count} labels converted, {cub_empty} empty")

        print("  Copying Shanghai images...")
        sh_img_count = copy_images(paths["sh_img"], dst_img_dir, "sh")
        print(f"  -> {sh_img_count} images copied")

        print("  Converting Shanghai labels...")
        sh_lbl_count, sh_empty = convert_labels(paths["sh_lbl"], dst_lbl_dir, "sh")
        print(f"  -> {sh_lbl_count} labels converted, {sh_empty} empty")

        total_img = cub_img_count + sh_img_count
        total_lbl = cub_lbl_count + sh_lbl_count
        print(f"  {split_name} total: {total_img} images, {total_lbl} non-empty labels")

    data_yaml = OUT_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("\nnc: 1\n")
        f.write("names:\n")
        f.write("  0: bird\n")

    print(f"\n  data.yaml saved to: {data_yaml}")
    print("\n  Done! Dataset ready for single-class bird detection training.")


if __name__ == "__main__":
    main()
