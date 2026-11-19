import numpy as np
from PIL import Image

def analyze_domain(im: Image.Image):
    im_rgb = im.convert("RGB").resize((128, 128))
    arr = np.array(im_rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # 1. Kraft / Brown Corrugated Cardboard:
    kraft = (r > b * 1.10) & (g > b * 1.01) & (r >= 0.18) & (r <= 0.95) & (b <= 0.75)
    
    # 2. Neutral White / Grey shipping box:
    color_diff = np.maximum(np.abs(r - g), np.abs(g - b))
    neutral_box = (color_diff < 0.14) & (r > 0.28) & (r < 0.98)
    
    pkg_mask = kraft | neutral_box
    pkg_coverage = float(np.mean(pkg_mask))
    
    # Non-packaging signatures:
    # A. Cool colors (blue, cyan, purple clothing/sky):
    cool_colors = (b > r * 1.15) & (b > 0.25)
    cool_ratio = float(np.mean(cool_colors))
    
    # B. Human skin tone cluster:
    skin = (r > 0.35) & (g > 0.20) & (b > 0.15) & (r > g + 0.08) & (g > b + 0.02) & (r < 0.92)
    skin_ratio = float(np.mean(skin))
    
    is_ood = bool((pkg_coverage < 0.25) or (cool_ratio > 0.18) or (skin_ratio > 0.15))
    return {
        "pkg_coverage": round(pkg_coverage, 3),
        "cool_ratio": round(cool_ratio, 3),
        "skin_ratio": round(skin_ratio, 3),
        "is_ood": is_ood,
    }

samples = [
    ("intact2.jpeg", "C:/Users/ragah/Downloads/PackShield/frontend/src/assets/sample/intact2.jpeg"),
    ("intact3.jpeg", "C:/Users/ragah/Downloads/PackShield/frontend/src/assets/sample/intact3.jpeg"),
    ("intant1.jpeg", "C:/Users/ragah/Downloads/PackShield/frontend/src/assets/sample/intant1.jpeg"),
    ("testter.jpeg", "C:/Users/ragah/Downloads/PackShield/frontend/src/assets/sample/testter.jpeg"),
    ("images.jpg", "C:/Users/ragah/Downloads/PackShield/backend/data/user_samples/images.jpg"),
    ("images (2).jpg", "C:/Users/ragah/Downloads/PackShield/backend/data/user_samples/images (2).jpg"),
    ("brown_torn.jpg", "C:/Users/ragah/Downloads/PackShield/backend/data/damaged/brown_torn_box.jpg"),
    ("passport_photo", "C:/Users/ragah/.gemini/antigravity/brain/bf51ce3b-1a91-4984-af00-725db143e0e1/.user_uploaded/media_1795099213268_31b588eb.jpg"),
]

for name, p in samples:
    try:
        im = Image.open(p)
        res = analyze_domain(im)
        print(f"{name:18} -> Pkg: {res['pkg_coverage']:5.1%}, Cool: {res['cool_ratio']:5.1%}, Skin: {res['skin_ratio']:5.1%}, IS_OOD: {res['is_ood']}")
    except Exception as e:
        print(f"{name:18} -> Error: {e}")
