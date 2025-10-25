import os
import shutil
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Define original dataset path (edit this to your dataset path)
original_dataset_dir = Path(".")
output_base_dir = Path("dataset")

# Define classes
classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# Split ratios
train_split = 0.7
val_split = 0.15
test_split = 0.15

def create_dir_structure(base_dir):
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(base_dir / split / cls, exist_ok=True)

def split_and_copy():
    create_dir_structure(output_base_dir)
    
    for cls in classes:
        img_dir = original_dataset_dir / cls / "images"
        all_images = list(img_dir.glob("*.png"))
        random.shuffle(all_images)

        n_total = len(all_images)
        n_train = int(train_split * n_total)
        n_val = int(val_split * n_total)

        train_files = all_images[:n_train]
        val_files = all_images[n_train:n_train + n_val]
        test_files = all_images[n_train + n_val:]

        for f in train_files:
            shutil.copy(f, output_base_dir / "train" / cls / f.name)
        for f in val_files:
            shutil.copy(f, output_base_dir / "val" / cls / f.name)
        for f in test_files:
            shutil.copy(f, output_base_dir / "test" / cls / f.name)

        print(f"Class {cls}: {n_total} total - {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

if __name__ == "__main__":
    split_and_copy()
