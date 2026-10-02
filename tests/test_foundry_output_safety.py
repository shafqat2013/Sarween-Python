import asyncio
import unittest

import foundryoutput as foundry
from foundry_delivery import MoveOutbox


class FoundryOutputSafetyTest(unittest.IsolatedAsyncioTestCase):
    async def check_output(self, paused):
        original = {name: getattr(foundry, name) for name in (
            "SCENE_ID", "MINI_TO_TOKEN", "GRID_PX", "_delivery", "_connection_ready", "_tracking_output_paused",
        )}
        sent = []

        class Socket:
            async def send(self, payload):
                sent.append(payload)

        task = None
        try:
            foundry.SCENE_ID = "new-scene"
            foundry.MINI_TO_TOKEN = {"red10": "token"}
            foundry.GRID_PX = 50
            foundry._tracking_output_paused = paused
            foundry._delivery = MoveOutbox()
            foundry._connection_ready = True
            old_context = ("old-scene", *foundry._scene_context()[1:])
            foundry._delivery.set_context(old_context)
            foundry._delivery.offer("red10", "A1", old_context)
            foundry._delivery.set_context(foundry._scene_context())
            foundry.queue_cell_move("red10", "B2")
            task = asyncio.create_task(foundry.send_loop(Socket()))
            await asyncio.sleep(0.01)
            return sent
        finally:
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for name, value in original.items():
                setattr(foundry, name, value)

    async def test_old_scene_moves_are_discarded(self):
        self.assertEqual(len(await self.check_output(False)), 1)

    async def test_capture_blocks_all_tracker_moves(self):
        self.assertEqual(await self.check_output(True), [])


if __name__ == "__main__":
    unittest.main()
