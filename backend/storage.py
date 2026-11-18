"""Storage abstraction module for DefektGuard.

Provides persistent scan logging and statistics retrieval with:
- Local JSON storage fallback
- Optional MongoDB client integration if pymongo and MONGODB_URI are available
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("defektguard.storage")

STORAGE_FILE = config.REPORTS_DIR / "scan_records.json"


class StorageManager:
    def __init__(self, mongo_uri: Optional[str] = None):
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
        self.mongo_client = None
        self.db = None
        self.use_mongo = False

        if self.mongo_uri:
            try:
                import pymongo
                self.mongo_client = pymongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=2000)
                # Test connection
                self.mongo_client.server_info()
                self.db = self.mongo_client["defektguard"]
                self.use_mongo = True
                logger.info("Connected to MongoDB storage backend.")
            except Exception as e:
                logger.warning(f"MongoDB connection failed ({e}). Falling back to local storage.")
                self.use_mongo = False

        if not self.use_mongo:
            self._init_local_store()

    def _init_local_store(self):
        if not STORAGE_FILE.exists():
            STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STORAGE_FILE, "w") as f:
                json.dump({"records": []}, f, indent=2)

    def log_scan(
        self,
        label: str,
        confidence: float,
        filename: str,
        probabilities: Optional[Dict[str, float]] = None,
        sku: Optional[str] = None,
        location: Optional[str] = None,
        uncertainty_score: Optional[float] = None,
        triage_status: Optional[str] = None,
        requires_human_review: Optional[bool] = None,
        gradcam_image: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = {
            "id": f"SCAN-{int(time.time() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "label": label,
            "condition": label.capitalize(),
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities or {},
            "sku": sku or f"SKU-{int(time.time()) % 10000:04d}",
            "location": location or "Main Conveyor A",
            # Trustworthy AI Fields
            "uncertainty": round((uncertainty_score or 0.0) * 100, 2),
            "triage_status": triage_status or ("AUTO_REJECTED" if label.lower() == "damaged" else "AUTO_APPROVED"),
            "requires_human_review": bool(requires_human_review),
            "review_status": "PENDING_REVIEW" if requires_human_review else "COMPLETED",
            "gradcam_image": gradcam_image,
            "human_decision": None,
            "reviewer_notes": None,
        }

        if self.use_mongo and self.db is not None:
            try:
                self.db["scans"].insert_one(dict(record))
                return record
            except Exception as e:
                logger.error(f"MongoDB insert error ({e}). Falling back to local store.")

        # Local storage fallback
        try:
            records = self.get_records()
            records.insert(0, record)
            # Keep last 500 records
            records = records[:500]
            with open(STORAGE_FILE, "w") as f:
                json.dump({"records": records}, f, indent=2)
        except Exception as e:
            logger.error(f"Local storage write error: {e}")

        return record

    def update_review(
        self,
        scan_id: str,
        human_decision: str,
        notes: Optional[str] = None,
        reviewer_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Applies human-in-the-loop audit decision to a flagged scan."""
        normalized_decision = human_decision.lower().strip()
        if normalized_decision not in ("intact", "damaged"):
            raise ValueError("human_decision must be 'intact' or 'damaged'")

        if self.use_mongo and self.db is not None:
            try:
                update_fields = {
                    "human_decision": normalized_decision,
                    "condition": normalized_decision.capitalize(),
                    "review_status": "REVIEWED",
                    "requires_human_review": False,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "reviewer_name": reviewer_name or "Human Auditor",
                    "reviewer_notes": notes,
                }
                self.db["scans"].update_one({"id": scan_id}, {"$set": update_fields})
                return self.db["scans"].find_one({"id": scan_id}, {"_id": 0})
            except Exception as e:
                logger.error(f"MongoDB update review error: {e}")

        records = self.get_records(limit=500)
        updated_record = None
        for r in records:
            if r.get("id") == scan_id:
                r["human_decision"] = normalized_decision
                r["condition"] = normalized_decision.capitalize()
                r["review_status"] = "REVIEWED"
                r["requires_human_review"] = False
                r["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                r["reviewer_name"] = reviewer_name or "Human Auditor"
                r["reviewer_notes"] = notes
                updated_record = r
                break

        if updated_record:
            try:
                with open(STORAGE_FILE, "w") as f:
                    json.dump({"records": records}, f, indent=2)
            except Exception as e:
                logger.error(f"Local storage update error: {e}")

        return updated_record

    def get_flagged_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves scans currently pending human verification."""
        records = self.get_records(limit=limit)
        return [r for r in records if r.get("requires_human_review", False)]

    def get_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self.use_mongo and self.db is not None:
            try:
                cursor = self.db["scans"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                return list(cursor)
            except Exception as e:
                logger.error(f"MongoDB fetch error: {e}")

        if STORAGE_FILE.exists():
            try:
                with open(STORAGE_FILE, "r") as f:
                    data = json.load(f)
                    return data.get("records", [])[:limit]
            except Exception:
                return []
        return []

    def get_dashboard_stats(self) -> Dict[str, Any]:
        records = self.get_records(limit=1000)
        total_scanned = len(records)
        damaged_count = sum(1 for r in records if r.get("label", "").lower() == "damaged")
        intact_count = total_scanned - damaged_count
        damage_rate = round((damaged_count / total_scanned * 100), 2) if total_scanned > 0 else 0.0

        # Trustworthy AI Metrics
        flagged_count = sum(1 for r in records if r.get("requires_human_review", False))
        reviewed_count = sum(1 for r in records if r.get("review_status") == "REVIEWED")
        auto_approved = sum(1 for r in records if r.get("triage_status") == "AUTO_APPROVED")
        auto_rejected = sum(1 for r in records if r.get("triage_status") == "AUTO_REJECTED")

        uncertainties = [r.get("uncertainty", 0.0) for r in records if "uncertainty" in r]
        avg_uncertainty = round(sum(uncertainties) / len(uncertainties), 2) if uncertainties else 0.0

        return {
            "totalScanned": total_scanned,
            "damagedCount": damaged_count,
            "intactCount": intact_count,
            "damageRate": damage_rate,
            "storageBackend": "MongoDB" if self.use_mongo else "LocalJSON",
            # Trustworthy AI extensions
            "flaggedForReviewCount": flagged_count,
            "humanReviewedCount": reviewed_count,
            "autoApprovedCount": auto_approved,
            "autoRejectedCount": auto_rejected,
            "avgUncertainty": avg_uncertainty,
        }


# Global singleton instance
storage = StorageManager()

