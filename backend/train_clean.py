import sys
from pathlib import Path
import random
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model.model import preprocess_image, predict
from ml.resnet34 import build_transfer_model
from PIL import Image
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

dmg_files = list((config.DATA_DIR / "damaged").glob("*.png")) + list((config.DATA_DIR / "damaged").glob("*.jpg"))
intact_files = list((config.DATA_DIR / "intact").glob("*.png")) + list((config.DATA_DIR / "intact").glob("*.jpg"))

print(f"Total damaged: {len(dmg_files)}, Total intact: {len(intact_files)}")

items = [(p, 1) for p in dmg_files] + [(p, 0) for p in intact_files]
random.seed(42)
random.shuffle(items)

x_list = []
y_list = []
for p, label in items:
    try:
        t = preprocess_image(Image.open(p))
        x_list.append(t[0].numpy())
        y_list.append(label)
    except Exception as e:
        pass

X = np.array(x_list, dtype=np.float32)
Y = np.array(y_list, dtype=np.int32)
print(f"Training tensor shape: {X.shape}, labels: {Y.shape}")

model = build_transfer_model(input_shape=config.INPUT_SHAPE, num_classes=2, dropout_rate=0.2)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Balanced class weights with safety margin for defect detection
class_weights = {0: 1.0, 1: 1.5}

# Train 15 epochs
model.fit(X, Y, batch_size=16, epochs=15, class_weight=class_weights, verbose=1)

# Save to both .keras and fallback .h5
model.save(config.DEFAULT_MODEL_PATH)
model.save(config.FALLBACK_H5_MODEL_PATH)
print("Updated model saved successfully!")

# Quick check on user images
for p in ['backend/data/user_samples/images.jpg', 'backend/data/user_samples/images (2).jpg', 'backend/data/damaged/brown_torn_box.jpg', 'frontend/src/assets/sample/intact2.jpeg', 'frontend/src/assets/sample/testter.jpeg']:
    im = Image.open(p)
    t = preprocess_image(im)
    preds = model.predict(t, verbose=0)[0]
    lbl = "damaged" if preds[1] > 0.5 else "intact"
    print(f"{Path(p).name:30} -> {lbl:8} (intact: {preds[0]:.3f}, damaged: {preds[1]:.3f})")
