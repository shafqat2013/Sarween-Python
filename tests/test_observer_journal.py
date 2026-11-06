"""Exact recorded-input replay, corruption checks and bounded intervals."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from observer_engine import BoardGeometry, ObserverEngine
from observer_journal import (ObserverJournal, replay_journal, fingerprints,
                              ENGINE_SOURCES, ENGINE_ASSETS, _resolve_detector)
from poc_ring_detector import RingDetector


def make_engine(key=None, **kwargs):
    view = {"registration": {"left": 0, "top": 0, "right": 400, "bottom": 240},
            "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
            "gridSize": 40}
    return ObserverEngine(BoardGeometry(view, (400, 240)),
                          {"piece": (100, 100)}, {"piece": "C3"}, geometry_key=key, **kwargs)


def image(center=(100, 100)):
    pixels = np.full((240, 400, 3), (150, 90, 40), np.uint8)
    cv2.circle(pixels, center, 25, (35, 75, 180), -1)
    cv2.line(pixels, center, (center[0] - 43, center[1]), (35, 75, 180), 8)
    return pixels


def read_json(path):
    return json.loads(Path(path).read_text())


def edit_json(path, function):
    data = read_json(path)
    function(data)
    Path(path).write_text(json.dumps(data))


class JournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def capture(self, folder, *, loss=False, geometry_change=False, close=True, offset=0):
        key = {"scene": "A", "view": 1}
        engine = make_engine(key)
        journal = ObserverJournal(folder, engine.initialization())
        for index in range(12):
            pixels = image((100, 100) if index < 4 else (260, 100))
            mask = np.zeros(pixels.shape[:2], np.uint8)
            if index == 2:
                mask[40:160, 40:160] = 255
                pixels[mask > 0] = 255
            locked = not (loss and index == 5)
            context = {"scene": "A", "view": 2} if geometry_change and index >= 5 else key
            row = engine.process(pixels, mask, offset + index / 8, locked=locked, geometry_key=context)
            self.assertTrue(journal.append(pixels, mask, offset + index / 8, row,
                                           locked=locked, geometry_key=context))
        if close:
            journal.close()
        return engine, journal

    def test_lossless_recording_replays_identical_events_cells_and_every_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            live, journal = self.capture(folder)
            report = replay_journal(folder)
            self.assertTrue(report["verification_passed"])
            self.assertTrue(report["identical_decisions"])
            self.assertEqual(report["frames"], 12)
            self.assertEqual(live.events, report["events"])
            self.assertEqual(len(report["events"]), 1)
            self.assertEqual(report["events"][0]["to_cell"], "G3")
            self.assertEqual(journal.frames, 12)
            # PNG really retains exact source bytes, including the exclusion mask.
            pixels = cv2.imread(str(folder / "inputs/000002-board.png"))
            mask = cv2.imread(str(folder / "inputs/000002-mask.png"), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.all(pixels[mask > 0] == 255))
            self.assertEqual(int(np.count_nonzero(mask)), 120 * 120)

    def test_lock_loss_and_geometry_change_replay_the_same_latched_holds(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, options, reason in (
                    ("loss", {"loss": True}, "camera_lock_lost"),
                    ("view", {"geometry_change": True}, "geometry_changed")):
                folder = Path(directory) / name
                live, _journal = self.capture(folder, **options)
                report = replay_journal(folder)
                self.assertTrue(report["verification_passed"])
                self.assertEqual(report["events"], [])
                self.assertEqual(live.last_record["state"]["reason"], reason)
                rows = [json.loads(line) for line in (folder / "decisions.jsonl").read_text().splitlines()]
                self.assertEqual(rows[-1]["held"], ["piece"])
                self.assertTrue(rows[-1]["needs_reset"])

    def test_modified_pixels_fail_fingerprint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            path = folder / "inputs/000004-board.png"
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "input fingerprint differs"):
                replay_journal(folder)

    def test_rehashed_undecodable_image_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            rows = [json.loads(line) for line in (folder / "inputs.jsonl").read_text().splitlines()]
            data = b"not an image"
            (folder / rows[0]["mask"]).write_bytes(data)
            rows[0]["mask_sha256"] = hashlib.sha256(data).hexdigest()
            (folder / "inputs.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "image cannot be decoded"):
                replay_journal(folder)

    def test_dropped_tail_decision_or_wrong_manifest_count_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("tail", "count"):
                folder = Path(directory) / name
                self.capture(folder)
                if name == "tail":
                    path = folder / "decisions.jsonl"
                    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
                else:
                    edit_json(folder / "manifest.json", lambda data: data.update(frames=13))
                report = replay_journal(folder)
                self.assertFalse(report["frame_counts_match"])
                self.assertFalse(report["identical_decisions"])
                self.assertFalse(report["verification_passed"])

    def test_modified_decision_is_reported_at_its_original_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            path = folder / "decisions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[8]["state"]["reason"] = "fabricated"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            report = replay_journal(folder)
            self.assertEqual(report["discrepant_frames"], [8])
            self.assertFalse(report["verification_passed"])

    def test_source_fingerprint_mismatch_never_claims_same_version_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            edit_json(folder / "manifest.json", lambda data: data["code_sha256"].update({"observer_engine.py": "wrong"}))
            with self.assertRaisesRegex(ValueError, "source fingerprints differ"):
                replay_journal(folder)
            report = replay_journal(folder, require_same_code=False)
            self.assertFalse(report["code_matches"])
            self.assertTrue(report["identical_decisions"])
            self.assertFalse(report["verification_passed"])

    def test_sources_changed_while_recording_invalidates_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            edit_json(folder / "manifest.json", lambda data: data.update(sources_unchanged=False))
            with self.assertRaisesRegex(ValueError, "source fingerprints differ"):
                replay_journal(folder)
            self.assertFalse(replay_journal(folder, require_same_code=False)["verification_passed"])

    def test_close_detects_source_changes_since_the_interval_started(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            _engine, journal = self.capture(folder, close=False)
            changed = dict(journal.manifest["code_sha256"], **{"observer_engine.py": "changed-on-disk"})
            with patch("observer_journal.fingerprints", return_value=changed):
                journal.close()
            self.assertFalse(read_json(folder / "manifest.json")["sources_unchanged"])
            with self.assertRaisesRegex(ValueError, "source fingerprints differ"):
                replay_journal(folder)

    def test_zero_frame_and_incomplete_intervals_do_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty"
            journal = ObserverJournal(empty, make_engine().initialization())
            journal.close()
            report = replay_journal(empty)
            self.assertFalse(report["identical_decisions"])
            self.assertFalse(report["recording_complete"])
            self.assertFalse(report["verification_passed"])
            partial = Path(directory) / "partial"
            _engine, journal = self.capture(partial, close=False)
            journal.close("inputError", error="camera stopped")
            report = replay_journal(partial)
            self.assertTrue(report["identical_decisions"])
            self.assertFalse(report["recording_complete"])
            self.assertFalse(report["verification_passed"])

    def test_input_paths_cannot_escape_the_recording_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            path = folder / "inputs.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["board"] = "../../outside.png"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "must stay in its recording folder"):
                replay_journal(folder)

    def test_recording_interval_stops_before_exceeding_capture_time_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = make_engine()
            journal = ObserverJournal(Path(directory) / "capture", engine.initialization(), max_seconds=.25)
            for timestamp in (0, .125, .25):
                pixels = image()
                self.assertTrue(journal.append(pixels, None, timestamp, engine.process(pixels, None, timestamp)))
            self.assertFalse(journal.append(image(), None, .375, engine.process(image(), None, .375)))
            self.assertTrue(journal.closed)
            self.assertEqual(journal.frames, 3)
            self.assertEqual(journal.manifest["limit_reached"], "seconds")
            self.assertTrue(replay_journal(journal.folder)["verification_passed"])

    def test_byte_limit_rejects_entire_png_pair_before_any_excess_write(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = make_engine()
            pixels = image()
            row = engine.process(pixels, None, 0)
            probe = ObserverJournal(Path(directory) / "probe", engine.initialization())
            probe.append(pixels, None, 0, row)
            first_packet_size = probe.bytes
            probe.close()
            bounded = ObserverJournal(Path(directory) / "bounded", engine.initialization(),
                                       max_bytes=first_packet_size + 32)
            self.assertTrue(bounded.append(pixels, None, 0, row))
            self.assertFalse(bounded.append(pixels, None, .125, engine.process(pixels, None, .125)))
            self.assertTrue(bounded.closed)
            self.assertEqual(bounded.frames, 1)
            self.assertLessEqual(bounded.bytes, bounded.max_bytes)
            self.assertEqual(len(list((bounded.folder / "inputs").glob("*.png"))), 2)
            self.assertEqual(bounded.manifest["limit_reached"], "bytes")
            self.assertTrue(replay_journal(bounded.folder)["verification_passed"])

    def test_byte_limit_too_small_for_one_frame_cannot_produce_success(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = make_engine()
            journal = ObserverJournal(Path(directory) / "capture", engine.initialization(), max_bytes=1)
            self.assertFalse(journal.append(image(), None, 0, engine.process(image(), None, 0)))
            self.assertEqual(list((journal.folder / "inputs").iterdir()), [])
            self.assertFalse(replay_journal(journal.folder)["verification_passed"])

    def test_invalid_clocks_pixels_and_limits_do_not_create_valid_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, limits in enumerate(({"max_seconds": 0}, {"max_seconds": -1},
                                             {"max_seconds": float("nan")}, {"max_bytes": 0})):
                with self.assertRaises(ValueError):
                    ObserverJournal(root / str(index), make_engine().initialization(), **limits)
                self.assertFalse((root / str(index)).exists())
            engine = make_engine()
            journal = ObserverJournal(root / "clock", engine.initialization())
            row = engine.process(image(), None, 0)
            with self.assertRaises(ValueError):
                journal.append(np.zeros((240, 400, 4), np.uint8), None, 0, row)
            for timestamp in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    journal.append(image(), None, timestamp, row)
            self.assertTrue(journal.append(image(), None, 0, row))
            with self.assertRaises(ValueError):
                journal.append(image(), None, 0, row)
            journal.close()

    def test_preview_uses_capture_clock_and_does_not_change_decisions_or_events(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            live, _journal = self.capture(folder, offset=1_000_000)
            # An unrelated nominal rate must not stretch the capture-time video.
            edit_json(folder / "manifest.json", lambda data: data.update(fps=2))
            report = replay_journal(folder, write_preview=True)
            self.assertTrue(report["verification_passed"])
            self.assertEqual(live.events, report["events"])
            self.assertAlmostEqual(report["capture_duration_seconds"], 11 / 8)
            self.assertEqual(report["source_clock_origin"], 1_000_000)
            cap = cv2.VideoCapture(str(folder / "annotated.mp4"))
            self.assertTrue(cap.isOpened())
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)
            frames = []
            while True:
                ok, pixels = cap.read()
                if not ok:
                    break
                frames.append(pixels)
            cap.release()
            self.assertEqual(len(frames), math.ceil((11 / 8) * 10) + 1)
            self.assertEqual(frames[0].shape, (304, 400, 3))
            # At t=.4 the image is still from the source frame at .375. At
            # t=.5 the moved base is available; no future image appears early.
            self.assertGreater(int(frames[4][100, 100, 2]), int(frames[4][100, 260, 2]))
            self.assertGreater(int(frames[5][100, 260, 2]), int(frames[5][100, 100, 2]))
            transcript = (folder / "transcript.txt").read_text()
            self.assertIn("C3 -> G3", transcript)
            self.assertNotIn("1000000", transcript)
            import csv
            with (folder / "movements.csv").open() as stream:
                movements = list(csv.DictReader(stream))
            self.assertEqual(len(movements), 1)
            self.assertLess(float(movements[0]["capture_seconds"]), 2)
            self.assertGreater(float(movements[0]["source_capture_timestamp"]), 1_000_000)

    def test_invalid_clock_interval_is_rejected_before_preview_export(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            path = folder / "inputs.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1]["timestamp"] = 1_000_000
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "exceed the declared recording interval"):
                replay_journal(folder, write_preview=True)
            self.assertFalse((folder / "annotated.mp4").exists())

    def test_missing_packaged_sources_use_explicit_loader_bytecode_fingerprints(self):
        loader = SimpleNamespace(get_code=lambda name: compile("value = 1", name, "exec"))
        with patch("observer_journal.Path.read_bytes", side_effect=FileNotFoundError), \
                patch("observer_journal.importlib.util.find_spec", return_value=SimpleNamespace(loader=loader)):
            first, second = fingerprints(), fingerprints()
        self.assertEqual(first, second)
        self.assertEqual(set(first), {name + ":bytecode" for name in ENGINE_SOURCES} | set(ENGINE_ASSETS))
        self.assertTrue(all(len(value) == 64 for value in first.values()))

    def test_missing_source_and_unavailable_loader_fail_explicitly(self):
        with patch("observer_journal.Path.read_bytes", side_effect=FileNotFoundError), \
                patch("observer_journal.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Cannot fingerprint observer module"):
                fingerprints()

    def test_missing_caption_asset_cannot_claim_reproducible_inference(self):
        with patch("observer_journal.open", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Cannot fingerprint observer watermark asset"):
                fingerprints()

    def test_unsupported_watermark_metadata_is_rejected_before_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            for metadata in ({"mode": "arbitrary-module"}, "not-metadata", {}, {"mode": 3}):
                with self.subTest(metadata=metadata):
                    edit_json(folder / "manifest.json", lambda data: data["initialization"].update(
                        watermark=metadata))
                    with self.assertRaisesRegex(ValueError, "Unsupported observer watermark mode"):
                        replay_journal(folder)

    def test_omitted_watermark_mode_preserves_provided_mask_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            edit_json(folder / "manifest.json", lambda data: data["initialization"].pop("watermark"))
            self.assertTrue(replay_journal(folder)["verification_passed"])

    def test_recorded_board_caption_mode_replays_the_same_processing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = make_engine(watermark_mode="board-template-v1")
            folder = Path(directory) / "capture"
            journal = ObserverJournal(folder, engine.initialization())
            for index in range(12):
                pixels = image((100, 100) if index < 4 else (260, 100))
                row = engine.process(pixels, None, index / 8, frame_index=index)
                journal.append(pixels, None, index / 8, row, frame_index=index)
            journal.close()
            report = replay_journal(folder)
            self.assertTrue(report["verification_passed"])
            self.assertEqual(report["events"], engine.events)
            self.assertEqual(len(engine.events), 1)
            self.assertEqual(read_json(folder / "manifest.json")["initialization"]["watermark"],
                             {"mode": "board-template-v1"})

    def test_legacy_journals_select_legacy_detector_without_a_migration(self):
        self.assertIs(_resolve_detector({}), RingDetector)
        self.assertIs(_resolve_detector({"detector": {"module": "poc_ring_detector", "class": "RingDetector"}}), RingDetector)

    def test_unknown_detector_metadata_cannot_trigger_arbitrary_imports(self):
        examples = ({"module": "os", "class": "system"},
                    {"module": "annular_ring_detector", "class": "NotAllowed"},
                    {"module": "hybrid_ring_detector", "class": "NotAllowed"},
                    {"module": "poc_ring_detector", "class": "NotAllowed"},
                    {"module": "other_package", "class": "RingDetector"}, "not-a-description")
        for description in examples:
            with self.subTest(description=description), patch("builtins.__import__") as importer:
                with self.assertRaisesRegex(ValueError, "detector"):
                    _resolve_detector({"detector": description})
                importer.assert_not_called()

    def test_recorded_detector_metadata_is_required_for_replay_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            edit_json(folder / "manifest.json", lambda data: data["initialization"].update(
                detector={"module": "arbitrary", "class": "Detector"}))
            with self.assertRaisesRegex(ValueError, "unsupported detector"):
                replay_journal(folder)

    def test_annular_factory_replays_actual_ring_measurements_and_is_fingerprinted(self):
        from annular_ring_detector import AnnularRingDetector
        self.assertIs(_resolve_detector({"detector": {"module": "annular_ring_detector",
                                                      "class": "AnnularRingDetector"}}), AnnularRingDetector)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "annular"
            geometry = make_engine().geometry
            engine = ObserverEngine(geometry, {"piece": (100, 100)}, {"piece": "C3"},
                                     detector_factory=AnnularRingDetector)
            journal = ObserverJournal(folder, engine.initialization())
            for index in range(8):
                center = (100, 100) if index < 4 else (260, 100)
                pixels = np.full((240, 400, 3), (150, 90, 40), np.uint8)
                cv2.circle(pixels, center, 30, (40, 80, 210), -1)
                cv2.circle(pixels, center, 21, (150, 90, 40), -1)
                cv2.circle(pixels, center, 15, (130, 130, 130), -1)
                row = engine.process(pixels, None, index / 8)
                self.assertTrue(journal.append(pixels, None, index / 8, row))
            journal.close()
            manifest = read_json(folder / "manifest.json")
            self.assertEqual(manifest["initialization"]["detector"],
                             {"module": "annular_ring_detector", "class": "AnnularRingDetector"})
            self.assertIn("annular_ring_detector.py", manifest["code_sha256"])
            report = replay_journal(folder)
            self.assertTrue(report["verification_passed"])
            self.assertEqual(engine.events, report["events"])
            self.assertEqual(len(report["events"]), 1)
            self.assertEqual(report["events"][0]["to_cell"], "G3")
            # Removing metadata explicitly falls back to the historical solid
            # detector; it must not silently migrate old recordings to annular.
            edit_json(folder / "manifest.json", lambda data: data["initialization"].pop("detector"))
            legacy_comparison = replay_journal(folder)
            self.assertFalse(legacy_comparison["verification_passed"])

    def test_new_recording_refuses_new_disk_code_while_old_engine_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "must-not-be-created"
            original_read_bytes = Path.read_bytes

            def changed_source(path):
                data = original_read_bytes(path)
                return data + b"\n# modified after this process imported the engine\n" if path.name == "observer_engine.py" else data

            with patch("observer_journal.Path.read_bytes", new=changed_source):
                with self.assertRaisesRegex(RuntimeError, "Restart Sarween after tracker code changes"):
                    ObserverJournal(folder, make_engine().initialization())
            self.assertFalse(folder.exists())

    def test_replay_cannot_claim_new_disk_code_is_the_old_loaded_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            self.capture(folder)
            from observer_journal import LOADED_ENGINE_FINGERPRINTS
            changed = dict(LOADED_ENGINE_FINGERPRINTS, **{"observer_engine.py": "new-file-version"})
            edit_json(folder / "manifest.json", lambda data: data.update(code_sha256=changed))
            with patch("observer_journal.fingerprints", return_value=changed):
                with self.assertRaisesRegex(ValueError, "source fingerprints differ"):
                    replay_journal(folder)
                comparison = replay_journal(folder, require_same_code=False)
            self.assertFalse(comparison["loaded_code_matches_disk"])
            self.assertFalse(comparison["code_matches"])
            self.assertFalse(comparison["verification_passed"])

    def test_hybrid_factory_replays_annular_and_previously_valid_warm_targets(self):
        from hybrid_ring_detector import HybridRingDetector
        metadata = {"module": "hybrid_ring_detector", "class": "HybridRingDetector"}
        self.assertIs(_resolve_detector({"detector": metadata}), HybridRingDetector)
        with tempfile.TemporaryDirectory() as directory:
            for fixture in ("annular", "warm-partial-boundary"):
                with self.subTest(fixture=fixture):
                    folder = Path(directory) / fixture
                    engine = ObserverEngine(make_engine().geometry, {"piece": (100, 100)},
                                             {"piece": "C3"}, detector_factory=HybridRingDetector)
                    journal = ObserverJournal(folder, engine.initialization())
                    try:
                        for index in range(8):
                            center = (100, 100) if index < 4 else (260, 100)
                            if fixture == "annular":
                                pixels = np.full((240, 400, 3), (150, 90, 40), np.uint8)
                                cv2.circle(pixels, center, 30, (40, 80, 210), -1)
                                cv2.circle(pixels, center, 21, (150, 90, 40), -1)
                                cv2.circle(pixels, center, 15, (130, 130, 130), -1)
                            else:
                                # The projecting warm body interrupts part of
                                # the outer circular boundary; no separate
                                # concentric inner-ring edge is visible.
                                pixels = image(center)
                            row = engine.process(pixels, None, index / 8)
                            self.assertTrue(journal.append(pixels, None, index / 8, row))
                    finally:
                        journal.close()
                    manifest = read_json(folder / "manifest.json")
                    self.assertEqual(manifest["initialization"]["detector"], metadata)
                    self.assertIn("hybrid_ring_detector.py", manifest["code_sha256"])
                    report = replay_journal(folder)
                    self.assertTrue(report["verification_passed"])
                    self.assertEqual(engine.events, report["events"])
                    self.assertEqual(len(report["events"]), 1)
                    self.assertEqual(report["events"][0]["to_cell"], "G3")


if __name__ == "__main__":
    unittest.main()
