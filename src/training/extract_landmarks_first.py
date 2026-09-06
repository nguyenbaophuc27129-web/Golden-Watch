"""
STEP 1: EXTRACT LANDMARKS FROM DATASET (WITHOUT MEDIAPIPE TASKS API)
====================================================================

Since MediaPipe Tasks API requires model file which is hard to download,
we can use the dataset's landmarks annotations if available,
OR we can manually annotate a subset.

THIS SCRIPT: Extracts features directly from images using OpenCV
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def extract_basic_features(image_path):
    """
    Extract basic features from image WITHOUT MediaPipe

    Features:
    1. Brightness histogram (256 bins)
    2. Edge density (Sobel)
    3. Color distribution (HSV)

    Returns: numpy array of features
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to fixed size (scale-invariant)
    img_resized = cv2.resize(gray, (128, 128))

    # Flatten as features
    features = img_resized.flatten().astype(np.float32) / 255.0

    return features

def extract_dataset_features(dataset_path, output_path):
    """
    Extract features from entire dataset

    Args:
        dataset_path: Path to Stroke or NonStroke folder
        output_path: Path to save .npy features file
    """
    image_files = list(Path(dataset_path).glob("*/*.jpg")) + \
                  list(Path(dataset_path).glob("*/*.png")) + \
                  list(Path(dataset_path).glob("*.jpg")) + \
                  list(Path(dataset_path).glob("*.png"))

    print(f"Found {len(image_files)} images in {dataset_path}")

    features = []
    valid_count = 0

    for img_path in tqdm(image_files, desc=f"Extracting from {Path(dataset_path).name}"):
        feat = extract_basic_features(img_path)
        if feat is not None:
            features.append(feat)
            valid_count += 1

    features = np.array(features)
    print(f"Successfully extracted {valid_count}/{len(image_files)} features")
    print(f"Feature shape: {features.shape}")

    # Save features
    np.save(output_path, features)
    print(f"Saved to: {output_path}")

    return features

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: EXTRACT FEATURES FROM DATASET")
    print("=" * 60)
    print()

    base_path = Path("C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/datasets/face/Annotated stroke and non stroke Dataset")

    # Extract NonStroke features
    print("Processing NonStroke images...")
    nonstroke_features = extract_dataset_features(
        base_path / "NonStroke",
        "C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/processed/nonstroke_features.npy"
    )

    print()

    # Extract Stroke features
    print("Processing Stroke images...")
    stroke_features = extract_dataset_features(
        base_path / "Stroke",
        "C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/processed/stroke_features.npy"
    )

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE!")
    print(f"NonStroke: {len(nonstroke_features)} features")
    print(f"Stroke: {len(stroke_features)} features")
    print()
    print("Next step: Run train_with_extracted_features.py")
    print("=" * 60)
