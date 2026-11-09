"""Connect shared base tracking to the existing scene-scoped Foundry outbox.

The detector never sends commands. This adapter requires current-generation
assignments, records command provenance and converts base centers into world
coordinates. Delivery progress never resets physical observations.
"""
from __future__ import annotations

import math


class CoreTokenBridge:
    def __init__(self, output, assignments, *, observation_only=False):
        self.output, self.assignments = output, assignments
        self.observation_only = bool(observation_only)
        self.context = None
        self.mapping = {}
        self.aligned = False
        self._commands = {}
        self._latest_commands = {}
        self._failure = None
        self._failed_generation = None
        self._last_result_key = None
        self._revoked_mapping = None
        self.message = "Choose the six starting positions."
        self.output.set_observation_only(True)

    def _pause(self):
        if not self.output.observation_only():
            self.output.set_observation_only(True)

    def _resume(self):
        if self.output.observation_only():
            self.output.set_observation_only(False)

    def _discard_delivery(self, controller):
        """Revoke output authority without discarding measured physical IDs."""
        self._pause()
        self.output.clear_pending_moves()
        for command in self._commands:
            controller.cancel_expected_display_change(command)
        self._commands.clear()
        self._latest_commands.clear()
        self._last_result_key = None

    def _remember_command(self, command, mini):
        self._commands[command] = mini
        protected = set(self._latest_commands.values())
        while len(self._commands) > 32:
            old = next((identity for identity in self._commands if identity not in protected), None)
            if old is None:
                break
            self._commands.pop(old)

    def _binding_context(self, controller, view):
        scene = (view or {}).get("sceneId")
        generation = controller.binding_generation
        engine = controller.engine
        names = tuple(engine.mover.positions) if engine is not None else tuple(f"mini{i}" for i in range(1, 7))
        token = f"{controller.folder.name}:{generation}" if generation is not None else None
        return scene, token, names

    def before_frame(self, controller, view):
        """Refresh bindings and collect exact command evidence before detection."""
        context = self._binding_context(controller, view)
        scene, generation, names = context
        if self._failure and generation is not None and generation != self._failed_generation:
            self._failure = None
            self._failed_generation = None
        if context != self.context:
            self._discard_delivery(controller)
            self.context, self.mapping, self.aligned = context, {}, False
            self._revoked_mapping = None
            self.assignments.set_tracking_minis(names if generation is not None else [], generation, scene,
                                               read_only=self.observation_only)
        mapping = self.assignments.get_confirmed_tracking_assignments()
        if self._revoked_mapping is not None and mapping != self._revoked_mapping:
            self._revoked_mapping = None
        valid = (not self._failure and generation is not None and bool(scene) and set(mapping) == set(names)
                 and self._revoked_mapping is None
                 and len(set(mapping.values())) == len(names) and all(mapping.values()))
        accepted = dict(mapping) if valid else {}
        if accepted != self.mapping:
            self._discard_delivery(controller)
            self.aligned = False
            self.mapping = accepted
        confirmed, unchanged = [], []
        for mini, move in self.output.get_delivery_status().get("moves", {}).items():
            if (move.get("target_kind") != "position" or move.get("generation") != generation
                    or mini not in self.mapping):
                continue
            if move.get("binding_invalid"):
                self._revoked_mapping = dict(self.mapping)
                self._discard_delivery(controller)
                self.mapping, self.aligned = {}, False
                self.message = "Token assignment changed. Confirm the current assignments before resuming output."
                break
            command = move.get("command_id")
            if not isinstance(command, str) or not command:
                continue
            previous = self._latest_commands.get(mini)
            self._latest_commands[mini] = command
            if command not in self._commands:
                if controller.expect_display_change(command, scene, self.mapping[mini]):
                    self._remember_command(command, mini)
            if previous and previous != command:
                controller.cancel_expected_display_change(previous, replacement_id=command)
            if move.get("state") == "confirmed":
                confirmed.append(command)
                if move.get("applied_changed") is False:
                    unchanged.append(command)
        return {"scene_visual_change": self.output.get_scene_visual_change(),
                "confirmed_command_ids": confirmed, "unchanged_command_ids": unchanged,
                }

    @staticmethod
    def _scene_center(engine, xy):
        gx, gy = engine.geometry.grid_point(xy)
        view = engine.geometry.view
        grid = float(view["gridSize"])
        point = (gx * grid + float(view.get("gridOriginX", 0)),
                 gy * grid + float(view.get("gridOriginY", 0)))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("The tracked base position has no finite Foundry position")
        return point

    def _queue(self, controller, mini, xy):
        scene, generation, _names = self.context
        center = self._scene_center(controller.engine, xy)
        if not self.output.queue_position_move(mini, *center, source="base-tracker", generation=generation,
                                               expected_token_id=self.mapping[mini], expected_scene_id=scene):
            raise ValueError("Foundry did not accept the tracked position")
        move = self.output.get_delivery_status().get("moves", {}).get(mini, {})
        command = move.get("command_id")
        if not command or (command not in self._commands and
                not controller.expect_display_change(command, scene, self.mapping[mini])):
            raise ValueError("The token move could not be registered for display recovery")
        previous = self._latest_commands.get(mini)
        self._latest_commands[mini] = command
        self._remember_command(command, mini)
        if previous and previous != command:
            controller.cancel_expected_display_change(previous, replacement_id=command)

    def fail(self, controller, message):
        """Latch output errors while retaining incoming physical observations."""
        self._discard_delivery(controller)
        self._failed_generation = self.context[1] if self.context else None
        self._failure = (f"Foundry output stopped; physical observations continue. {message}. "
                         "Choose starting positions again to retry output.")
        self.aligned = False
        self.mapping = {}
        controller.window.invalidate_initialization(self._failure)
        self.message = self._failure
        controller.window.set_output_status(False, self.message)
        return self.message

    def after_frame(self, controller, result):
        if self._failure:
            self._pause()
            self.message = self._failure
        elif self.observation_only:
            self._pause()
            self.message = "Observation only · Foundry movement off"
        elif controller.engine is None or self.context[1] is None:
            self._pause()
            self.message = "Choose the six starting positions."
        elif result is None or result.get("needs_reset"):
            self._pause()
            self.message = "Tracking held · choose starting positions again when ready."
        elif not self.mapping:
            self._pause()
            self.message = "Assign Mini 1–6 in VTT Controls → Assignment and confirm the assignments."
        elif not result.get("acquired"):
            self._pause()
            self.message = "Waiting for all six bases to be acquired."
        elif not self.aligned:
            if result.get("held") or (result.get("acquisition") or {}).get("visible_count", 0) != len(self.mapping):
                self.message = "Keep all six bases visible and still before aligning Foundry tokens."
            else:
                self._resume()
                try:
                    for mini, xy in result["positions"].items():
                        self._queue(controller, mini, xy)
                    self.aligned = True
                    self.message = "Aligning the assigned Foundry tokens with their mini bases…"
                except (ValueError, TypeError, KeyError) as exc:
                    return self.fail(controller, str(exc))
        else:
            self._resume()
            try:
                result_key = (result.get("frame"), result.get("capture_seconds"))
                events = [] if result.get("held") or result_key == self._last_result_key else result.get("events", [])
                for event in events:
                    self._queue(controller, event["mini"], event["to_xy"])
                self._last_result_key = result_key
                delivery = self.output.get_delivery_status()
                self.message = ("Tracking held · " + str(result.get("state", {}).get("reason", "rechecking bases")).replace("_", " ")
                                if result.get("held") and not delivery.get("retryAvailable")
                                else "Foundry token following · " + delivery["message"])
            except (ValueError, TypeError, KeyError) as exc:
                return self.fail(controller, str(exc))
        controller.window.set_output_status(not self.observation_only and self.aligned
            and not self.output.observation_only() and not (result or {}).get("held"), self.message)
        return self.message

    def close(self):
        self.output.clear_pending_moves()
        self.assignments.set_tracking_minis([], None, None)
        self.output.set_observation_only(False)
