import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model.model import predict

for f in sorted(Path("backend/data/user_samples").glob("*.jpg")):
    im = Image.open(f)
    res = predict(im)
    print("=" * 45)
    print("File:", f.name)
    print("Label:", res["label"])
    print("Confidence:", f"{res['confidence']*100:.1f}%")
    print("Probabilities:", res["probabilities"])
    print("Uncertainty Score:", f"{res['uncertainty_score']*100:.1f}%")
    print("Triage Status:", res["triage_status"])
