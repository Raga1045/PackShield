"""Enriches training dataset with photographic augmentations of real packaging boxes.

Generates realistic variations across lighting, angles, crops, and defect types:
- Intact: clean carton edges, smooth tape, pristine surfaces, shipping labels
- Damaged: tears, punctures, crushed corners, fold fissures, crumpled edges, dented faces
"""

import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

import config


def augment_crop(img: Image.Image, rng: random.Random, focus_box=None) -> Image.Image:
    """Applies realistic photographic augmentations with optional focus region."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size

    if focus_box is not None:
        fx1, fy1, fx2, fy2 = focus_box
        pad_x = rng.randint(15, 60)
        pad_y = rng.randint(15, 60)
        x1 = max(0, fx1 - pad_x)
        y1 = max(0, fy1 - pad_y)
        x2 = min(w, fx2 + pad_x)
        y2 = min(h, fy2 + pad_y)
        cropped = img.crop((x1, y1, x2, y2))
    else:
        crop_scale = rng.uniform(0.70, 1.0)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        x1 = rng.randint(0, max(0, w - cw))
        y1 = rng.randint(0, max(0, h - ch))
        cropped = img.crop((x1, y1, x1 + cw, y1 + ch))

    resized = cropped.resize(config.IMG_SIZE, Image.Resampling.BILINEAR)

    # Flip horizontally
    if rng.random() > 0.5:
        resized = resized.transpose(Image.FLIP_LEFT_RIGHT)

    # Slight rotation
    angle = rng.uniform(-10, 10)
    resized = resized.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(180, 160, 140))

    # Lighting / contrast
    resized = ImageEnhance.Brightness(resized).enhance(rng.uniform(0.85, 1.15))
    resized = ImageEnhance.Contrast(resized).enhance(rng.uniform(0.85, 1.20))

    return resized


def add_realistic_tear_or_hole(img: Image.Image, rng: random.Random) -> Image.Image:
    """Adds a realistic puncture or jagged tear along an edge or surface."""
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    defect_type = rng.choice(["tear_seam", "puncture_hole", "crushed_corner"])

    if defect_type == "tear_seam":
        start_x = rng.randint(int(w * 0.3), int(w * 0.7))
        start_y = rng.randint(int(h * 0.2), int(h * 0.5))
        points = [(start_x, start_y)]
        curr_x, curr_y = start_x, start_y
        for _ in range(rng.randint(6, 12)):
            curr_x += rng.randint(-15, 20)
            curr_y += rng.randint(8, 25)
            points.append((curr_x, curr_y))
        
        for offset in range(-6, 7):
            shifted = [(p[0] + offset, p[1]) for p in points]
            draw.line(shifted, fill=(30, 20, 15), width=3)
        draw.line([(p[0] + 3, p[1]) for p in points], fill=(130, 100, 70), width=2)

    elif defect_type == "puncture_hole":
        cx = rng.randint(int(w * 0.35), int(w * 0.65))
        cy = rng.randint(int(h * 0.35), int(h * 0.65))
        radius = rng.randint(20, 42)
        
        n_pts = rng.randint(8, 14)
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        poly = []
        for a in angles:
            r_jitter = radius * rng.uniform(0.7, 1.3)
            poly.append((cx + r_jitter * np.cos(a), cy + r_jitter * np.sin(a)))
        
        draw.polygon(poly, fill=(25, 18, 12), outline=(15, 10, 8))
        draw.polygon(poly[:4], fill=(140, 110, 80))

    else:
        corner = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
        if corner == "top_left":
            poly = [(0, 0), (rng.randint(50, 90), 0), (0, rng.randint(50, 90))]
        elif corner == "top_right":
            poly = [(w, 0), (w - rng.randint(50, 90), 0), (w, rng.randint(50, 90))]
        elif corner == "bottom_left":
            poly = [(0, h), (rng.randint(50, 90), h), (0, h - rng.randint(50, 90))]
        else:
            poly = [(w, h), (w - rng.randint(50, 90), h), (w, h - rng.randint(50, 90))]
        
        draw.polygon(poly, fill=(40, 30, 22))
        draw.line(poly, fill=(20, 15, 10), width=3)

    return canvas


def enrich_dataset(seed: int = 42):
    """Generates balanced photographic training dataset for packaging defect detection."""
    rng = random.Random(seed)

    damaged_dir = config.DATA_DIR / "damaged"
    intact_dir = config.DATA_DIR / "intact"
    damaged_dir.mkdir(parents=True, exist_ok=True)
    intact_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing generated photos
    for old_f in damaged_dir.glob("photo_damaged_*.png"):
        old_f.unlink()
    for old_f in intact_dir.glob("photo_intact_*.png"):
        old_f.unlink()

    img_idx = 0

    # 1. Crushed box with printed upward arrows (images.jpg)
    crushed_arrows = Path("backend/data/user_samples/images.jpg")
    if crushed_arrows.exists():
        im = Image.open(crushed_arrows)
        # Left crumpled dent/cave-in
        for _ in range(8):
            aug = augment_crop(im, rng, focus_box=(40, 80, 250, 290))
            aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
            img_idx += 1
        # Top crushed seam/lid
        for _ in range(6):
            aug = augment_crop(im, rng, focus_box=(140, 30, 380, 190))
            aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
            img_idx += 1
        # Full box crops
        for _ in range(8):
            aug = augment_crop(im, rng)
            aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
            img_idx += 1

    # 2. Torn seam box (images (2).jpg)
    torn_seam = Path("backend/data/user_samples/images (2).jpg")
    if torn_seam.exists():
        im = Image.open(torn_seam)
        for _ in range(8):
            aug = augment_crop(im, rng, focus_box=(180, 80, 480, 340))
            aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
            img_idx += 1
        for _ in range(8):
            aug = augment_crop(im, rng)
            aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
            img_idx += 1

    # 3. Brown box with hole/tear (brown_torn_box.jpg)
    brown_torn = Path("backend/data/damaged/brown_torn_box.jpg")
    if brown_torn.exists():
        im = Image.open(brown_torn)
        for _ in range(8):
            aug = augment_crop(im, rng, focus_box=(220, 60, 330, 160))
            aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
            img_idx += 1
        for _ in range(8):
            aug = augment_crop(im, rng)
            aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
            img_idx += 1

    # 4. Other real damaged packaging samples
    food_pkg = Path("frontend/src/assets/sample/damagedfoodpackagingbox10.jpeg")
    example_pkg = Path("frontend/src/assets/sample/example.jpeg")
    for p in [food_pkg, example_pkg]:
        if p.exists():
            im = Image.open(p)
            for _ in range(8):
                aug = augment_crop(im, rng)
                aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
                img_idx += 1

    # 5. Injected realistic tears/punctures onto separate neutral cardboard
    intact_refs = [
        Path("frontend/src/assets/sample/intact2.jpeg"),
    ]
    for p in intact_refs:
        if p.exists():
            im = Image.open(p)
            for _ in range(12):
                aug = augment_crop(im, rng)
                aug = add_realistic_tear_or_hole(aug, rng)
                aug.save(damaged_dir / f"photo_damaged_{img_idx:04d}.png")
                img_idx += 1

    print(f"Total damaged photographic samples created: {img_idx}")

    # 6. Intact sources
    intact_sources = [
        Path("frontend/src/assets/sample/intact2.jpeg"),
        Path("frontend/src/assets/sample/intact3.jpeg"),
        Path("frontend/src/assets/sample/intant1.jpeg"),
        Path("frontend/src/assets/sample/testter.jpeg"),
    ]

    intact_idx = 0
    samples_per_intact = img_idx // len(intact_sources)
    remainder = img_idx % len(intact_sources)

    for i, p in enumerate(intact_sources):
        if p.exists():
            im = Image.open(p)
            n_samples = samples_per_intact + (1 if i < remainder else 0)
            for _ in range(n_samples):
                aug = augment_crop(im, rng)
                aug.save(intact_dir / f"photo_intact_{intact_idx:04d}.png")
                intact_idx += 1

    print(f"Total intact photographic samples created: {intact_idx}")


if __name__ == "__main__":
    enrich_dataset()
