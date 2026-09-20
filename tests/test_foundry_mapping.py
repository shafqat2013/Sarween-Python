import json
import tempfile
import unittest
from pathlib import Path

import foundryoutput as foundry


class FoundryMappingTest(unittest.TestCase):
    def test_named_players_are_reassigned_on_new_scene(self):
        original_mapping = foundry.MINI_TO_TOKEN
        original_path = foundry.MAP_PATH
        try:
            with tempfile.TemporaryDirectory() as directory:
                foundry.MINI_TO_TOKEN = {"red10": "old-red", "blue": "old-blue"}
                foundry.MAP_PATH = Path(directory) / "mini_token_map.json"
                foundry.reconcile_scene_bindings({
                    "tokens": [{"id": "new-red"}, {"id": "new-blue"}],
                    "miniBindings": {"red10": "new-red", "blue": "new-blue"},
                })
                self.assertEqual(foundry.MINI_TO_TOKEN, {"red10": "new-red", "blue": "new-blue"})
        finally:
            foundry.MINI_TO_TOKEN = original_mapping
            foundry.MAP_PATH = original_path

    def test_remove_stale_token_mapping_persists_remaining_mappings(self):
        original_mapping = foundry.MINI_TO_TOKEN
        original_path = foundry.MAP_PATH
        try:
            with tempfile.TemporaryDirectory() as directory:
                foundry.MINI_TO_TOKEN = {
                    "red10": "deleted-token",
                    "blue": "live-token",
                }
                foundry.MAP_PATH = Path(directory) / "mini_token_map.json"

                removed = foundry._remove_stale_token_mapping("deleted-token")

                self.assertEqual(removed, ["red10"])
                self.assertEqual(foundry.MINI_TO_TOKEN, {"blue": "live-token"})
                self.assertEqual(
                    json.loads(foundry.MAP_PATH.read_text(encoding="utf-8")),
                    {"blue": "live-token"},
                )
        finally:
            foundry.MINI_TO_TOKEN = original_mapping
            foundry.MAP_PATH = original_path


if __name__ == "__main__":
    unittest.main()
