import asyncio
import unittest

import foundryoutput as foundry


class FoundryOutputSafetyTest(unittest.IsolatedAsyncioTestCase):
    async def check_output(self, paused):
        original = {name: getattr(foundry, name) for name in (
            "SCENE_ID", "MINI_TO_TOKEN", "GRID_PX", "_move_queue", "_tracking_output_paused",
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
            foundry._move_queue = asyncio.Queue()
            foundry._move_queue.put_nowait(("red10", "A1", "old-scene"))
            foundry._move_queue.put_nowait(("red10", "B2", "new-scene"))
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
