"""Printed replay consumes recorded layout/view state, never user hardware."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import cv_core as core
import printed_markers as pm
from recording_state import CORE_SCALARS, restore_checkpoint, save_checkpoint
from tests.test_printed_markers import detected, layout, view


def registration():
    saved = layout()
    return {"mode": "printed", "profileId": "recorded-tv", "profileName": "Recorded TV",
            "printedMarkerLayout": saved, "layoutFingerprint": pm.layout_fingerprint(saved),
            "revision": 3, "controllerRevision": 5, "pending": False,
            "calibrationMethod": "co-visible", "viewport": {
                "viewportWidth": 1920, "viewportHeight": 1080, "registration": view()["registration"]}}


def recorded_session():
    session = core.CVCoreSession.__new__(core.CVCoreSession)
    for name in CORE_SCALARS:
        setattr(session, name, 0)
    session.__dict__.update(_source_is_file=True, marker_mode="viewport", _printed_marker_controller=None,
        _screen_corner_ids=dict(pm.SCREEN_IDS), corner_ids=dict(pm.SCREEN_IDS), required_ids=set(pm.SCREEN_IDS.values()),
        warp_w=1280, warp_h=720, grid_w=48, grid_h=27, frame_idx=17, _frame_phase_offset=0,
        aruco_every_n=1, aruco_every_n_fast=1, last_marker_count=4, last_seen_ids=[20, 21, 22, 23],
        last_missing_ids=[], _recorder=None, _registration_token=None, _registration_revision=3,
        BG_cam={"bgr": np.zeros((1080, 1920, 3), np.uint8), "blur": np.zeros((1080, 1920), np.uint8)},
        BG_warp_f32=np.full((720, 1280), 17, np.float32), _bg_seeded=True)
    c, ids = detected()
    session.H_saved = pm.solve_homography(c, ids, layout(), view(), (1280, 720))
    session._H_inv = np.linalg.inv(session.H_saved)
    session.last_mask_cam, session.last_mask_warp = pm.roi_masks((1080, 1920), session.H_saved, (1280, 720))
    session.registration_metadata = registration()
    return session


class PrintedRecordingTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_background_then_two_cards_invalidate_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = recorded_session()
            path = root / "state.npz"
            checksum = save_checkpoint(path, source)
            entry = {"path": path.name, "sha256": checksum, "frame": 0, "kind": "initial"}
            replay = recorded_session()
            with patch("hardware_profiles.HardwareProfiles", side_effect=AssertionError("Replay cannot read user profiles")), \
                 patch("foundryoutput.get_vtt_status", side_effect=AssertionError("Replay cannot read live connection")), \
                 patch("foundryoutput.get_view_transform_payload", return_value=view()):
                restore_checkpoint(root / "video.tracking.json", entry, replay, Mock())
                np.testing.assert_array_equal(replay.H_saved, source.H_saved)
                np.testing.assert_array_equal(replay.BG_warp_f32, source.BG_warp_f32)
                c, ids = detected()
                with patch.object(core, "detect_markers", return_value=(c, ids, [])):
                    replay._update_lock_if_due(source.BG_cam["bgr"])
                self.assertIsNotNone(replay.last_mask_cam)
                np.testing.assert_array_equal(replay.BG_warp_f32, source.BG_warp_f32)
                c, ids = detected(included=(20, 21))
                with patch.object(core, "detect_markers", return_value=(c, ids, [])):
                    replay._update_lock_if_due(source.BG_cam["bgr"])
                self.assertIsNone(replay.last_mask_cam)
            self.assertEqual(replay.registration_metadata["layoutFingerprint"], registration()["layoutFingerprint"])

    def test_historical_checkpoint_without_registration_keeps_old_file_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "old.npz"
            save_checkpoint(path, recorded_session())
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            metadata = json.loads(arrays["metadata"].tobytes())
            metadata.pop("markerRegistration")
            arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode(), np.uint8)
            np.savez_compressed(path, **arrays)
            replay = recorded_session()
            replay._printed_marker_controller = None
            entry = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frame": 0}
            restore_checkpoint(root / "old.tracking.json", entry, replay, Mock())
            self.assertIsNone(replay._printed_marker_controller)

    def test_recorded_registration_rejects_tampered_layout_or_live_session(self):
        session = recorded_session()
        metadata = registration()
        metadata["layoutFingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            session.configure_recorded_registration(metadata)
        session._source_is_file = False
        with self.assertRaisesRegex(ValueError, "file sources"):
            session.configure_recorded_registration(registration())

    def test_header_only_replay_selects_paper_ids_without_current_profiles(self):
        import tracking_regression as runner
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "paper.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((64, 96, 3), np.uint8))
            writer.release()
            fake = SimpleNamespace(data={"markerMode": "viewport", "markerRegistration": registration(),
                "gridCols": 48, "gridRows": 27, "warpWidth": 1280, "warpHeight": 720,
                "framesUndistorted": True}, apply_through=Mock(), has_checkpoint=False,
                has_controls=True, selected_mini=None, prediction_paused=False, path=video.with_suffix(".tracking.json"))
            diagnostics = {}
            with patch.object(runner, "FoundryTimelineReplay", return_value=fake), \
                 patch("hardware_profiles.HardwareProfiles", side_effect=AssertionError("No user profiles in replay")), \
                 patch("foundryoutput.get_vtt_status", side_effect=AssertionError("No live connection in replay")), \
                 patch("foundryoutput.get_view_transform_payload", return_value=view()):
                runner._run_video(video, timeline_path=fake.path,
                    profiles_path=Path(__file__).resolve().parents[1] / "tests/fixtures/red10_baseline.profiles.json",
                    diagnostics=diagnostics)
            self.assertEqual(set(diagnostics["marker_missing_frames"]), {"20", "21", "22", "23"})
            self.assertEqual(diagnostics["processed_frames"], 4)


if __name__ == "__main__":
    unittest.main()
