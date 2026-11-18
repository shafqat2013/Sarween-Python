"""Artwork inputs are self-contained, tamper checked and bounded on replay."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2

from avatar_catalog import MAX_CATALOG_BYTES, validate_avatar_catalog
from observer_engine import BoardGeometry, ObserverEngine
from observer_journal import ObserverJournal, _read_avatar_catalog, canonical, replay_journal
from tests import test_avatar_artwork as artwork_fixture
from tests import test_avatar_catalog_transport as transport_fixture
from tests import test_observer_journal as journal_fixture


def catalog(revision=1, *, normalized=True):
    result = transport_fixture.payload(revision=revision)
    view = result["viewTransform"]
    view.update(viewportWidth=400, viewportHeight=240, gridSize=40)
    view["registration"] = {"left": 0, "top": 0, "right": 400, "bottom": 240}
    ok, encoded = cv2.imencode(".png", artwork_fixture.icon())
    assert ok
    result["templates"][0].update(image="data:image/png;base64," + base64.b64encode(encoded).decode(),
                                  width=64, height=64, logicalWidth=32, logicalHeight=32)
    return validate_avatar_catalog(result) if normalized else result


def engine_for(art):
    return ObserverEngine(BoardGeometry(art["viewTransform"], (400, 240)),
                          {"piece": (100, 100)}, {"piece": "C3"})


def write_asset(folder, value=None, *, data=None):
    data = canonical(value).encode() if data is None else data
    digest = hashlib.sha256(data).hexdigest()
    relative = f"avatars/{digest}.json"
    path = folder / relative
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return {"avatar_catalog": {"kind": "avatar-catalog-asset", "path": relative, "sha256": digest}}, path


class AvatarCatalogJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def capture(self, folder, frames=12):
        art = catalog()
        live = engine_for(art)
        journal = ObserverJournal(folder, live.initialization())
        for index in range(frames):
            pixels = journal_fixture.image((100, 100) if index < 4 else (260, 100))
            row = live.process(pixels, None, index / 8, avatar_catalog=art)
            self.assertTrue(journal.append(pixels, None, index / 8, row,
                                           metadata={"avatar_catalog": art}))
        journal.close()
        return live, journal, art

    def test_snapshot_saved_once_and_self_contained_replay_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            live, journal, art = self.capture(folder)
            self.assertTrue({"avatar_artwork.py", "avatar_catalog.py", "reference_health.py"}
                            <= journal.manifest["code_sha256"].keys())
            self.assertEqual(len(list((folder / "avatars").glob("*.json"))), 1)
            rows = [json.loads(line) for line in (folder / "inputs.jsonl").read_text().splitlines()]
            descriptors = [row["metadata"]["avatar_catalog"] for row in rows]
            self.assertTrue(all(item == descriptors[0] for item in descriptors))
            self.assertNotIn("data:image", (folder / "inputs.jsonl").read_text())
            self.assertEqual(_read_avatar_catalog(folder, rows[0]["metadata"], {}), art)
            self.assertTrue(live.last_record["avatar_artwork"]["ready"])
            with patch("foundryoutput.get_latest_avatar_catalog", side_effect=AssertionError("live Foundry lookup")):
                report = replay_journal(folder)
            self.assertTrue(report["verification_passed"])
            self.assertTrue(report["identical_decisions"])
            self.assertEqual(report["events"], live.events)
            counted = sum(path.stat().st_size for path in folder.rglob("*")
                          if path.is_file() and (path.parent.name in {"inputs", "avatars"}
                                                or path.name in {"inputs.jsonl", "decisions.jsonl"}))
            self.assertEqual(journal.bytes, counted)

    def test_changed_raw_snapshot_with_same_ids_cannot_reuse_old_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            art = catalog(normalized=False)
            live = engine_for(art)
            journal = ObserverJournal(Path(directory) / "capture", live.initialization())
            for index in range(2):
                changed = copy.deepcopy(art)
                changed["templates"][0]["logicalWidth"] += index
                pixels = journal_fixture.image()
                row = live.process(pixels, None, index / 8, avatar_catalog=changed)
                journal.append(pixels, None, index / 8, row, metadata={"avatar_catalog": changed})
            journal.close()
            rows = [json.loads(line) for line in (journal.folder / "inputs.jsonl").read_text().splitlines()]
            self.assertNotEqual(rows[0]["metadata"]["avatar_catalog"], rows[1]["metadata"]["avatar_catalog"])
            self.assertEqual(len(list((journal.folder / "avatars").glob("*.json"))), 2)
            self.assertTrue(replay_journal(journal.folder)["identical_decisions"])

    def test_catalog_bytes_participate_in_atomic_recording_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            _live, probe, art = self.capture(Path(directory) / "probe", frames=1)
            asset_bytes = next((probe.folder / "avatars").glob("*.json")).stat().st_size
            live = engine_for(art)
            journal = ObserverJournal(Path(directory) / "bounded", live.initialization(),
                                      max_bytes=probe.bytes - asset_bytes)
            pixels = journal_fixture.image()
            row = live.process(pixels, None, 0, avatar_catalog=art)
            self.assertFalse(journal.append(pixels, None, 0, row, metadata={"avatar_catalog": art}))
            self.assertEqual(journal.frames, 0)
            self.assertEqual(journal.bytes, 0)
            self.assertEqual(list((journal.folder / "inputs").iterdir()), [])
            self.assertFalse((journal.folder / "avatars").exists())
            self.assertEqual(journal.manifest["limit_reached"], "bytes")

    def test_oversized_snapshot_cannot_write_any_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            art = catalog()
            live = engine_for(art)
            journal = ObserverJournal(Path(directory) / "capture", live.initialization())
            art["extra"] = "x" * MAX_CATALOG_BYTES
            with self.assertRaisesRegex(ValueError, "size limit"):
                journal.append(journal_fixture.image(), None, 0, {}, metadata={"avatar_catalog": art})
            self.assertEqual(journal.frames, 0)
            self.assertFalse((journal.folder / "avatars").exists())
            journal.close()

    def test_missing_tampered_and_changed_cached_assets_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            metadata, path = write_asset(folder, catalog())
            cache = {}
            self.assertIsInstance(_read_avatar_catalog(folder, metadata, cache), dict)
            original = path.read_bytes()
            path.write_bytes(b"!" + original[1:])
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                _read_avatar_catalog(folder, metadata, cache)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing or unreadable"):
                _read_avatar_catalog(folder, metadata, cache)

    def test_malformed_descriptors_paths_and_symlinks_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "capture"
            folder.mkdir()
            metadata, path = write_asset(folder, catalog())
            for changed in ({"path": None}, {"path": []}, {"path": "../outside.json"},
                            {"path": str(path)}, {"sha256": "x" * 64}, {"extra": 1}):
                candidate = copy.deepcopy(metadata)
                candidate["avatar_catalog"].update(changed)
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    _read_avatar_catalog(folder, candidate, {})
            outside = Path(directory) / "outside.json"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "inside"):
                _read_avatar_catalog(folder, metadata, {})

    def test_oversized_and_malformed_asset_files_fail_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for data, reason in ((b"x" * (MAX_CATALOG_BYTES + 1), "size limit"),
                                 (b"not JSON", "unreadable"), (b"[]", "JSON object")):
                metadata, _path = write_asset(folder, data=data)
                with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                    _read_avatar_catalog(folder, metadata, {})

    def test_replay_cache_retains_at_most_four_small_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, cache = Path(directory), {}
            for index in range(8):
                metadata, _path = write_asset(folder, catalog(revision=index + 1))
                self.assertEqual(_read_avatar_catalog(folder, metadata, cache)["revision"], index + 1)
                self.assertLessEqual(len(cache), 4)
            self.assertEqual(len(cache), 4)
            self.assertEqual(sorted(entry["catalog"]["revision"] for entry in cache.values()), [5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
