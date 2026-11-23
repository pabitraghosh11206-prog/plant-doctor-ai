"""
Explore and inspect the PlantVillage dataset for Plant Doctor AI.
Analyzes class distribution, image dimensions, and dataset integrity.
"""
import os
import json
from collections import defaultdict
from PIL import Image

DATA_DIR = "/Users/pabitraghosh/Desktop/PlantVillage"
CONFIG_PATH = "/Users/pabitraghosh/Desktop/Plant Doctor AI/plantvillage_dataset.json"


def explore_dataset(data_dir=DATA_DIR):
    print("=" * 65)
    print("PLANT VILLAGE DATASET EXPLORATION")
    print("=" * 65)

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory not found at: {data_dir}")
        return

    print(f"Base Directory: {data_dir}\n")

    # Check top-level directories
    class_folders = []
    subsets = {}
    
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            if item in ['train', 'validation', 'test']:
                subsets[item] = {}
                for sub in sorted(os.listdir(item_path)):
                    sub_path = os.path.join(item_path, sub)
                    if os.path.isdir(sub_path):
                        files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        subsets[item][sub] = len(files)
            else:
                files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                class_folders.append((item, len(files)))

    if subsets:
        print("Pre-split Dataset Subsets:")
        print("-" * 50)
        for subset_name, classes in subsets.items():
            total = sum(classes.values())
            print(f"  [{subset_name.upper()}] Total: {total} images")
            for cls_name, count in classes.items():
                print(f"    - {cls_name:<30}: {count:5d} images ({count / total * 100:5.1f}%)")
        print()

    if class_folders:
        print("Root Class Folders:")
        print("-" * 50)
        total_root = sum(count for _, count in class_folders)
        for cls_name, count in class_folders:
            pct = count / total_root * 100 if total_root > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {cls_name:<30}: {count:5d} images ({pct:5.1f}%) [{bar:<20}]")

    # Sample image dimension check
    print("\n" + "=" * 65)
    print("IMAGE INTEGRITY & RESOLUTION CHECK:")
    print("=" * 65)
    
    check_dir = os.path.join(data_dir, 'train') if 'train' in subsets else data_dir
    checked_count = 0
    resolutions = defaultdict(int)
    corrupted = []

    for root, _, files in os.walk(check_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                checked_count += 1
                try:
                    with Image.open(full_path) as img:
                        resolutions[img.size] += 1
                except Exception as e:
                    corrupted.append((full_path, str(e)))
                if checked_count >= 200:
                    break
        if checked_count >= 200:
            break

    print(f"Sample checked: {checked_count} images")
    print("Resolution distribution:")
    for res, count in resolutions.items():
        print(f"  Dimensions (WxH): {res[0]}x{res[1]} -> {count} images")

    if corrupted:
        print(f"⚠️ Corrupted images found: {len(corrupted)}")
        for path, err in corrupted[:5]:
            print(f"  - {path}: {err}")
    else:
        print("✅ All sampled images verified intact!")
    print("=" * 65)


if __name__ == "__main__":
    explore_dataset()
