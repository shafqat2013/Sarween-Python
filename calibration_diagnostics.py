"""Small local capture summaries; never stores camera images."""

from collections import Counter, deque
from datetime import datetime, timezone
import math

from app_paths import atomic_write_json, data_path


class CalibrationDiagnostics:
    def __init__(self, profile_id, profile_name, *, total=10):
        self.profile_id, self.profile_name = profile_id, profile_name
        self.total = int(total)
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.reasons = Counter()
        self.events = deque(maxlen=240)
        self.accepted = self.frames = self.maximum_corners = 0
        self.maximum_hold = 0.0
        self.last_message = "No camera views were examined."
        self.error = None
        self.frame_size = None

    def observe(self, state, *, detected_corners, frame_size, automatic, duplicate_markers=False):
        self.frames += 1
        self.accepted += int(bool(state.get("capture")))
        self.maximum_corners = max(self.maximum_corners, int(detected_corners))
        self.maximum_hold = max(self.maximum_hold, float(state.get("stableProgress", 0)))
        self.frame_size = list(frame_size)
        reason = "duplicate_markers" if duplicate_markers else str(state.get("reason", "unknown"))
        self.reasons[reason] += 1
        self.last_message = str(state.get("message", ""))
        event = {"frame": self.frames, "reason": reason, "corners": int(detected_corners),
                 "automatic": bool(automatic)}
        for key in ("stableProgress", "sharpness", "motionRatio", "motionPixels", "motionLimitPixels", "holdResets",
                    "lastHoldResetReason", "motionBasis", "sharedCorners", "validCorners", "poseCorners",
                    "grossCornerOutliers", "captureMode"):
            value = state.get(key)
            if isinstance(value, str) or isinstance(value, (int, float)) and math.isfinite(value):
                event[key] = value
        self.events.append(event)

    def summary(self):
        if self.accepted >= self.total and not self.error:
            return f"{self.accepted} views accepted; camera capture completed."
        transient = {"starting", "cooldown", "captured", "complete", "manual", "timed_wait"}
        blocking = [(reason, count) for reason, count in self.reasons.most_common() if reason not in transient]
        reason = (blocking or self.reasons.most_common(1) or [("no_board", 0)])[0][0]
        explanations = {
            "no_board": "The calibration board was not detected.",
            "corners": "Too few clear board corners were detected.",
            "coverage": "The board was too small or too edge-on.",
            "blur": "The board was too blurry.",
            "corner_outlier": "A board corner was inconsistent; wait for a clear frame.",
            "holding": "The steady-view check kept waiting; check motion and visible corners.",
            "moving": "The board moved too much between frames.",
            "duplicate_markers": "Duplicate marker IDs were visible; show only the calibration board.",
            "duplicate": "The saved view was repeated; move to another angle.",
            "angle": "More varied camera tilts were needed.",
            "starting": "The startup countdown had not finished.",
            "timed_wait": "The five-second capture timer was still counting down.",
            "cooldown": "The capture pause was still running.",
        }
        explanation = self.error or explanations.get(reason, self.last_message)
        return f"{self.accepted} views accepted; at most {self.maximum_corners} corners detected. {explanation}"

    def snapshot(self, outcome):
        return {"schemaVersion": 1, "createdAt": self.created_at,
                "profileId": self.profile_id, "profileName": self.profile_name,
                "outcome": outcome, "acceptedViews": self.accepted, "framesExamined": self.frames,
                "targetViews": self.total,
                "maximumDetectedCorners": self.maximum_corners, "maximumHoldProgress": self.maximum_hold,
                "frameSize": self.frame_size, "reasonCounts": dict(self.reasons),
                "summary": self.summary(), "error": self.error, "events": list(self.events)}

    def save(self, outcome, *, path=None):
        path = data_path("Calibration/last_attempt.json") if path is None else path
        atomic_write_json(path, self.snapshot(outcome))
        return path
