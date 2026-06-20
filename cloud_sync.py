"""
cloud_sync.py  -  BrailleAI Module 5: SD Logging + Google Sheets Sync
====================================================================
Implements the remote-monitoring path from Section V:

    interaction event -> JSON line on SD card  -> HTTP POST to a Google
    Apps Script webhook -> row appended to a Google Sheet

Design goals for a battery / Wi-Fi-flaky wearable:
  * Every event is durably written to the SD card FIRST (source of truth).
  * Unsynced events are flushed to the webhook in batches when online.
  * Failed pushes stay queued and retry later (offline-tolerant).

The HTTP transport is injected, so tests use a fake sink and the device
uses urequests/requests against the real Apps Script URL.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Dict, Any, Optional


@dataclass
class InteractionEvent:
    timestamp: float
    mode: str           # "face_to_face" | "tutor" | "assistant"
    direction: str      # "in" (from user) | "out" (to user)
    text: str
    emotion: str = "neutral"
    meta: Dict[str, Any] = field(default_factory=dict)


# Transport: returns True on a successful push of a batch
HttpPoster = Callable[[str, List[Dict[str, Any]]], bool]


def fake_poster_factory(sink: List[Dict[str, Any]], fail_times: int = 0) -> HttpPoster:
    """Make a fake webhook that fails the first `fail_times` calls."""
    state = {"fails": fail_times}
    def _post(url: str, batch: List[Dict[str, Any]]) -> bool:
        if state["fails"] > 0:
            state["fails"] -= 1
            return False
        sink.extend(batch)
        return True
    return _post


@dataclass
class CloudLogger:
    sd_path: str                       # JSON-lines log file on the SD card
    webhook_url: str
    poster: HttpPoster
    batch_size: int = 20

    # -- durable local write --------------------------------------------
    def log(self, event: InteractionEvent) -> None:
        """Append one event as a JSON line; mark it unsynced."""
        rec = asdict(event)
        rec["_synced"] = False
        with open(self.sd_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- read helpers ----------------------------------------------------
    def _read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.sd_path):
            return []
        with open(self.sd_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _rewrite(self, records: List[Dict[str, Any]]) -> None:
        tmp = self.sd_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.sd_path)

    # -- sync ------------------------------------------------------------
    def pending_count(self) -> int:
        return sum(1 for r in self._read_all() if not r.get("_synced"))

    def sync(self) -> int:
        """Push unsynced events in batches. Returns number synced this call."""
        records = self._read_all()
        unsynced_idx = [i for i, r in enumerate(records) if not r.get("_synced")]
        pushed = 0
        for start in range(0, len(unsynced_idx), self.batch_size):
            idxs = unsynced_idx[start:start + self.batch_size]
            batch = [records[i] for i in idxs]
            if self.poster(self.webhook_url, batch):
                for i in idxs:
                    records[i]["_synced"] = True
                pushed += len(idxs)
            else:
                break  # stop on first failure; retry next sync()
        if pushed:
            self._rewrite(records)
        return pushed


if __name__ == "__main__":
    sink: List[Dict[str, Any]] = []
    log_path = os.path.join(os.path.dirname(__file__), "_session_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    # First webhook attempt fails once -> tests offline tolerance
    logger = CloudLogger(sd_path=log_path,
                         webhook_url="https://script.google.com/macros/s/MOCK/exec",
                         poster=fake_poster_factory(sink, fail_times=1),
                         batch_size=2)

    t = time.time()
    logger.log(InteractionEvent(t, "face_to_face", "in", "Hello!", "happy"))
    logger.log(InteractionEvent(t+1, "face_to_face", "out", "Hi back"))
    logger.log(InteractionEvent(t+2, "tutor", "out", "Spell APPLE"))

    print("pending before sync:", logger.pending_count())
    print("sync #1 (1st batch fails):", logger.sync(), "synced")
    print("pending after  sync1:", logger.pending_count())
    print("sync #2 (retry)       :", logger.sync(), "synced")
    print("pending after  sync2:", logger.pending_count())
    print("rows in Google Sheet sink:", len(sink))