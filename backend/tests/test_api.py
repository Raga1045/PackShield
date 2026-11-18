"""Integration tests for DefektGuard Flask API routes."""

import io
import unittest
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from __init__ import create_app


class TestDefektGuardAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True})
        cls.client = cls.app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("service", data)
        self.assertIn("classes", data)

    def test_stats_endpoint(self):
        res = self.client.get("/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("totalScanned", data)
        self.assertIn("damagedCount", data)
        self.assertIn("damageRate", data)

    def test_predict_missing_image_payload(self):
        res = self.client.post("/predict", data={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["code"], "MISSING_IMAGE_PAYLOAD")

    def test_predict_unsupported_file_extension(self):
        data = {
            "image": (io.BytesIO(b"dummy text file content"), "test_script.txt")
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 415)
        res_data = res.get_json()
        self.assertEqual(res_data["code"], "UNSUPPORTED_MEDIA_TYPE")

    def test_predict_corrupted_image_data(self):
        data = {
            "image": (io.BytesIO(b"invalid corrupt bytes header"), "damaged_box.jpg")
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 422)
        res_data = res.get_json()
        self.assertEqual(res_data["code"], "INVALID_IMAGE_DATA")

    def test_predict_empty_file_bytes(self):
        data = {
            "image": (io.BytesIO(b""), "empty.png")
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        res_data = res.get_json()
        self.assertEqual(res_data["code"], "EMPTY_IMAGE_BYTES")

    def test_predict_valid_image_success(self):
        # Generate dummy 100x100 RGB image in memory
        img = Image.new("RGB", (100, 100), color=(200, 150, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        data = {
            "image": (img_bytes, "valid_box.jpg"),
            "sku": "SKU-TEST-99",
            "location": "Aisle 4",
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        res_data = res.get_json()
        self.assertIn("label", res_data)
        self.assertIn("confidence", res_data)
        self.assertIn("probabilities", res_data)
        self.assertIn("scan_id", res_data)
        self.assertIn(res_data["label"], ["intact", "damaged"])

        # Trustworthy AI response assertions
        self.assertIn("uncertainty_score", res_data)
        self.assertIn("triage_status", res_data)
        self.assertIn("gradcam_image", res_data)
        self.assertTrue(res_data["gradcam_image"].startswith("data:image/jpeg;base64,"))

        # Test human-in-the-loop review endpoint
        scan_id = res_data["scan_id"]
        review_res = self.client.post(
            f"/scans/{scan_id}/review",
            json={"human_decision": "intact", "notes": "Audited by QA Lead"},
        )
        self.assertEqual(review_res.status_code, 200)
        review_data = review_res.get_json()
        self.assertEqual(review_data["record"]["human_decision"], "intact")
        self.assertEqual(review_data["record"]["review_status"], "REVIEWED")


if __name__ == "__main__":
    unittest.main()

