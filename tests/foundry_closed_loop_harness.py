"""Deterministic camera/Foundry feedback, driven by actual queued commands.

Physical positions are supplied only to the image renderer and initial user
bindings. New command IDs, token positions, ACKs and camera artwork are produced
from the real outbox; they are never copied from a recorded successful run.
This is a synthetic feedback test, not validation of hardware or real artwork.
"""
from collections import deque
from copy import deepcopy
from pathlib import Path
import tempfile

import cv2
import numpy as np

from core_token_bridge import CoreTokenBridge
from foundry_delivery import MoveOutbox
from observer_session import LiveObserverController
from tests.test_observer_session import FakeWindow


POSITIONS = {f"mini{i + 1}": point for i, point in enumerate(
    ((80, 80), (200, 80), (320, 80), (80, 280), (200, 280), (320, 280)))}
VIEW = {"sceneId": "feedback-scene", "registration": {"left": 0, "top": 0, "right": 400, "bottom": 360},
        "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
        "gridSize": 40, "gridOriginX": 0, "gridOriginY": 0}


class Window(FakeWindow):
    def set_output_status(self, enabled, message):
        self.output_status = (bool(enabled), str(message))


class Assignments:
    def __init__(self):
        self.context, self.confirmed = None, {}

    def set_tracking_minis(self, names, generation, scene, **_kwargs):
        self.context = (tuple(names), generation, scene)
        self.confirmed = {}

    def confirm(self):
        self.confirmed = {mini: "token" + mini[4:] for mini in self.context[0]}

    def get_confirmed_tracking_assignments(self):
        return dict(self.confirmed)


class MockFoundry:
    """Apply actual outbox payloads, with independent ACK/render delivery clocks."""
    def __init__(self, *, ack_delay=0., render_delay=0., artwork="hidden", ack_timeout=.5):
        self.box = MoveOutbox(ack_timeout=ack_timeout, max_attempts=8)
        self.context = (VIEW["sceneId"], 400, 360, 40, 0, 0, 1)
        self.box.set_context(self.context)
        self.paused = False
        self.fail_mini = None
        self.ack_delay, self.render_delay, self.artwork = ack_delay, render_delay, artwork
        self.bindings = {mini: "token" + mini[4:] for mini in POSITIONS}
        self.tokens = {self.bindings[mini]: tuple(point) for mini, point in POSITIONS.items()}
        self.rendered_tokens = deepcopy(self.tokens)
        self.offers, self.sent, self.applied, self.ack_results = [], [], [], []
        self._cached = {}
        self._acks, self._renders = [], []
        self._history = deque(maxlen=32)
        self.visual = {"revision": 0, "sceneId": VIEW["sceneId"], "changes": []}

    def set_observation_only(self, paused):
        self.paused = bool(paused)
        self.box.clear()

    def observation_only(self):
        return self.paused

    def clear_pending_moves(self):
        self.box.clear()

    def get_delivery_status(self):
        moves = self.box.snapshot()
        return {"moves": moves, "connected": True, "message": "Mock Foundry delivery",
                "retryAvailable": any(row["state"] == "failed" for row in moves.values())}

    def get_scene_visual_change(self):
        return deepcopy(self.visual)

    def queue_position_move(self, mini, x, y, **kwargs):
        if (self.paused or mini == self.fail_mini or kwargs.get("expected_scene_id") != self.context[0]
                or kwargs.get("expected_token_id") != self.bindings.get(mini)):
            return False
        accepted = self.box.offer_position(mini, (x, y), self.context, source=kwargs["source"],
            generation=kwargs["generation"], expected_token_id=kwargs["expected_token_id"],
            expected_scene_id=kwargs["expected_scene_id"])
        if accepted:
            self.offers.append((mini, (x, y), deepcopy(kwargs)))
        return accepted

    def notice(self, reason, *, commands=(), tokens=()):
        row = {"revision": self.visual["revision"] + 1, "sceneId": self.context[0],
               "reason": reason, "reasons": [reason], "commandIds": list(commands), "tokenIds": list(tokens)}
        self._history.append(row)
        self.visual = {**row, "changes": deepcopy(list(self._history))}

    def deliver(self, now):
        pending = []
        for due, payload in self._acks:
            if due <= now:
                self.ack_results.append((deepcopy(payload), self.box.reply(payload, now)))
            else:
                pending.append((due, payload))
        self._acks = pending
        pending = []
        for due, token, center in self._renders:
            if due <= now:
                self.rendered_tokens[token] = center
            else:
                pending.append((due, token, center))
        self._renders = pending

    def dispatch(self, now):
        if self.paused:
            return
        changed_commands, changed_tokens = [], []
        for mini, state in self.box.snapshot().items():
            if state["state"] == "confirmed":
                continue
            token = self.bindings.get(mini)
            if state.get("expected_token_id") != token:
                self.box.reject_position_binding(mini, state["command_id"])
                continue
            # All test tokens are one 40-pixel grid square. The production
            # dimension/bounds conversion is tested in position-delivery tests.
            point = tuple(round(value - 20) for value in state["center"])
            payload = self.box.prepare_position(mini, state["center"], token, point, now,
                                                generation=state["generation"])
            if payload is None:
                continue
            self.sent.append(deepcopy(payload))
            identity = payload["commandId"]
            if identity not in self._cached:
                center = (payload["x"] + 20, payload["y"] + 20)
                changed = self.tokens[token] != center
                self.tokens[token] = center
                reply = {**payload, "type": "tokenMoveApplied", "changed": changed}
                self._cached[identity] = reply
                self.applied.append(deepcopy(reply))
                if changed:
                    self._renders.append((now + self.render_delay, token, center))
                    changed_commands.append(identity)
                    changed_tokens.append(token)
            self._acks.append((now + self.ack_delay, deepcopy(self._cached[identity])))
        if changed_commands:
            self.notice("sarweenTokenMove", commands=changed_commands, tokens=changed_tokens)

    def camera(self, physical, *, glare=0):
        board = np.full((360, 400, 3), (150, 90, 40), np.uint8)
        for token, center in self.rendered_tokens.items():
            x, y = map(round, center)
            if self.artwork == "hidden":
                # A legitimately unobservable render: e.g. transparent artwork
                # or a camera that cannot resolve the displayed token pixels.
                # Logical apply/ACK/render clocks still advance normally.
                continue
            elif self.artwork == "minimal":
                cv2.rectangle(board, (x - 1, y - 1), (x + 1, y + 1), (85, 130, 125), -1)
            elif self.artwork == "disk":
                # Actual displayed source/destination artwork. It is smaller
                # than the physical base and is drawn before physical pieces.
                cv2.circle(board, (x, y), 16, (120, 120, 120), -1)
        if glare and glare != 10:
            # A reflection-only change, deliberately separate from base color.
            x = 130 + 4 * glare
            cv2.rectangle(board, (x, 120), (x + 24, 210), (185, 185, 185), -1)
        for center in physical.values():
            x, y = map(round, center)
            cv2.circle(board, (x, y), 25, (35, 75, 180), -1)
            cv2.line(board, (x, y), (x - 30, y), (35, 75, 180), 7)
        if glare == 10:
            # A broad bright reflection temporarily washes out two stationary
            # bases. It is not a physical movement or an exclusion annotation.
            cv2.rectangle(board, (180, 40), (222, 320), (200, 200, 200), -1)
        return board


class ClosedLoopHarness:
    def __init__(self, **backend_options):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = MockFoundry(**backend_options)
        self.assignments = Assignments()
        self.controller = LiveObserverController(None, Path(self.temporary.name),
            window_factory=Window, observation_only=False)
        self.bridge = CoreTokenBridge(self.output, self.assignments)
        self.physical = deepcopy(POSITIONS)
        self.time, self.index, self.rows = 0., 0, []

    def close(self):
        self.bridge.close()
        self.controller.close()
        self.temporary.cleanup()

    def frame(self, *, glare=0, seconds=.125):
        self.time += seconds
        self.index += 1
        self.output.deliver(self.time)
        self.output.dispatch(self.time)
        evidence = self.bridge.before_frame(self.controller, VIEW)
        row = self.controller.update(self.output.camera(self.physical, glare=glare), None, self.time,
            view=VIEW, locked=True, frame_index=self.index,
            metadata={"scene_visual_revision": evidence["scene_visual_change"]["revision"],
                      # These generated camera images contain no watermark,
                      # matching the owned USB camera app's encoder contract.
                      "camera_source": {"source": "sarween-phone-usb", "transport": "usb", "freshFrame": True},
                      **evidence})
        self.bridge.after_frame(self.controller, row)
        self.output.dispatch(self.time)
        self.rows.append(deepcopy(row))
        return row

    def frames(self, count=8, **kwargs):
        row = None
        for _ in range(count):
            row = self.frame(**kwargs)
        return row

    def initialize(self):
        self.frame()
        self.controller.window.choose(POSITIONS, *self.controller.current)
        self.frame()
        self.assignments.confirm()
        self.frames(10)
        if not self.bridge.aligned or self.controller.engine is None:
            raise AssertionError("Synthetic starting layout did not acquire and align")
        # Pixel-journal persistence has its own tests. The feedback cases below
        # exercise the same live pipeline without repeatedly writing PNGs.
        self.controller._stop_recording("synthetic_feedback_harness")

    def move(self, mini, destination):
        self.physical[mini] = tuple(destination)

    @property
    def events(self):
        return list(self.controller.engine.events) if self.controller.engine is not None else []
