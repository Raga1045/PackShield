"""Dataset loading, splitting, deduplication, and augmentation module for DefektGuard.

Enforces:
- SHA-256 content-hash deduplication across train/validation/test splits
- Deterministic stratified splitting
- Safe augmentation (applied ONLY to training data, never to val/test)
- Standardized normalization: [0, 1] scaling with (256, 256, 3) resolution
"""

import hashlib
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tensorflow as tf

import config


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file to detect duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_and_deduplicate(data_dir: Path) -> Dict[str, List[Path]]:
    """Scans class subdirectories in data_dir, removing duplicate images based on content hash.
    
    Returns a dictionary mapping class name to list of unique image paths.
    """
    class_images: Dict[str, List[Path]] = {cls: [] for cls in config.CLASS_NAMES}
    seen_hashes = set()
    duplicates_count = 0

    for cls in config.CLASS_NAMES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        
        for file in sorted(cls_dir.iterdir()):
            if file.suffix.lower().lstrip(".") in config.ALLOWED_EXTENSIONS:
                file_hash = compute_file_sha256(file)
                if file_hash in seen_hashes:
                    duplicates_count += 1
                    continue
                seen_hashes.add(file_hash)
                class_images[cls].append(file)

    return class_images


def split_dataset(
    class_images: Dict[str, List[Path]],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Splits class images into stratified Train, Val, and Test sets without data leakage."""
    rng = random.Random(seed)
    train_set, val_set, test_set = [], [], []

    for cls_idx, cls_name in enumerate(config.CLASS_NAMES):
        images = list(class_images.get(cls_name, []))
        rng.shuffle(images)

        n_total = len(images)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val - n_test

        test_imgs = images[:n_test]
        val_imgs = images[n_test : n_test + n_val]
        train_imgs = images[n_test + n_val :]

        train_set.extend([(img, cls_idx) for img in train_imgs])
        val_set.extend([(img, cls_idx) for img in val_imgs])
        test_set.extend([(img, cls_idx) for img in test_imgs])

    # Shuffle splits
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    return train_set, val_set, test_set


def build_augmentation_layer():
    """Keras Sequential preprocessing layer for training data augmentation."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def load_image_and_label(image_path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads, decodes, resizes, and normalizes an image file to [0, 1] float32 tensor."""
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_raw, channels=config.CHANNELS, expand_animations=False)
    img = tf.image.resize(img, config.IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return img, label


def create_tf_dataset(
    items: List[Tuple[Path, int]],
    batch_size: int = 16,
    is_training: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    """Creates an optimized tf.data.Dataset pipeline with prefetching and optional augmentation."""
    if not items:
        # Return empty dataset with correct specs
        return tf.data.Dataset.from_tensor_slices((
            tf.zeros((0, *config.INPUT_SHAPE), dtype=tf.float32),
            tf.zeros((0,), dtype=tf.int32)
        ))

    image_paths = [str(item[0]) for item in items]
    labels = [item[1] for item in items]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(buffer_size=len(items), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    if is_training and augment:
        aug_layer = build_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def generate_benchmark_dataset(target_dir: Path, num_per_class: int = 40, seed: int = 42):
    """Generates a synthetic warehouse package benchmark dataset for reproducibility and testing.
    
    Generates realistic box/carton patterns:
    - 'intact': pristine carton textures, barcodes, sealed edges
    - 'damaged': crushed corners, severe surface punctures, tears, stains, and deformation
    """
    rng = np.random.default_rng(seed)
    for cls_name in config.CLASS_NAMES:
        cls_path = target_dir / cls_name
        cls_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_per_class):
            img_path = cls_path / f"{cls_name}_{i:04d}.png"
            if img_path.exists():
                continue

            # Create base box carton color (browns, beige, grays)
            base_r = int(rng.integers(180, 220))
            base_g = int(rng.integers(140, 180))
            base_b = int(rng.integers(100, 140))
            
            img_arr = np.full((256, 256, 3), [base_r, base_g, base_b], dtype=np.uint8)
            # Add paper texture noise
            noise = rng.integers(-15, 15, size=(256, 256, 3))
            img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            pil_img = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(pil_img)

            # Draw standard box flaps / packing tape
            tape_color = (int(rng.integers(190, 230)), int(rng.integers(160, 200)), int(rng.integers(80, 120)))
            draw.rectangle([110, 0, 146, 255], fill=tape_color)
            
            # Draw shipping label / barcode
            draw.rectangle([40, 40, 100, 90], fill=(245, 245, 245), outline=(50, 50, 50))
            for line_y in range(50, 80, 6):
                draw.line([(45, line_y), (95, line_y)], fill=(30, 30, 30), width=2)

            if cls_name == "damaged":
                # Introduce defects: puncture holes, crushed corners, tears, stains
                defect_type = i % 3
                if defect_type == 0:
                    # Crushed corner / tear
                    corner_points = [(0, 0), (70, 0), (0, 70)]
                    draw.polygon(corner_points, fill=(40, 30, 20))
                    draw.line([(70, 0), (0, 70)], fill=(20, 15, 10), width=3)
                elif defect_type == 1:
                    # Puncture hole / deep tear
                    center_x, center_y = int(rng.integers(80, 180)), int(rng.integers(80, 180))
                    radius = int(rng.integers(25, 45))
                    draw.ellipse(
                        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                        fill=(35, 25, 15),
                        outline=(15, 10, 5),
                    )
                else:
                    # Oil/water spill stain + dent lines
                    draw.ellipse([80, 120, 200, 220], fill=(90, 70, 45))
                    draw.line([(60, 140), (210, 190)], fill=(30, 20, 10), width=4)

            pil_img.save(img_path, format="PNG")

