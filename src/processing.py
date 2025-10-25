import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Settings
INPUT_DIR = Path("dataset")           
OUTPUT_DIR = Path("processed_dataset")  
IMG_SIZE = (224, 224)                   

# Dataset splits to process
SPLITS = ["train", "val", "test"]
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

def preprocess_image_with_mask(image_path, mask_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Fallback if either is missing
    if img is None or mask is None:
        return None

    # Get bounding box of the mask
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    cropped_img = img[y_min:y_max, x_min:x_max]
    resized_img = cv2.resize(cropped_img, IMG_SIZE)
    norm_img = resized_img / 255.0  # normalize to [0,1]

    return norm_img

def process_dataset_split(split):
    print(f"\nProcessing split: {split.upper()}")
    for cls in CLASSES:
        input_img_dir = INPUT_DIR / split / cls
        input_mask_dir = INPUT_DIR / split / cls
        output_dir = OUTPUT_DIR / split / cls
        output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        fail_count = 0

        for img_file in tqdm(list(input_img_dir.glob("*.png")), desc=f"{split}/{cls}"):
            mask_file = input_mask_dir / img_file.name
            processed_img = preprocess_image_with_mask(img_file, mask_file)

            if processed_img is not None:
                out_path = output_dir / img_file.name
                cv2.imwrite(str(out_path), (processed_img * 255).astype(np.uint8))
                success_count += 1
            else:
                fail_count += 1

        print(f"{split}/{cls} - Processed: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    for split in SPLITS:
        process_dataset_split(split)

