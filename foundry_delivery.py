"""Thread-safe, latest-position outbox for acknowledged Foundry moves."""

from dataclasses import dataclass
import math
import threading
import uuid


@dataclass
class DesiredMove:
    mini_id: str
    cell: str | None
    source: str
    target_kind: str = "cell"
    center: tuple | None = None
    generation: str | None = None
    expected_token_id: str | None = None
    binding_invalid: bool = False
    command_id: str = ""
    token_id: str | None = None
    coordinates: tuple | None = None
    attempts: int = 0
    retry_at: float = 0.0
    state: str = "pending"
    error: str = ""
    assignment_requested: bool = False
    applied_changed: bool | None = None


class MoveOutbox:
    def __init__(self, *, ack_timeout=2.0, max_attempts=3):
        self.ack_timeout = ack_timeout
        self.max_attempts = max_attempts
        self.lock = threading.RLock()
        self.context = None
        self.moves = {}

    def set_context(self, context):
        with self.lock:
            if context != self.context:
                self.context = context
                self.moves.clear()

    def clear(self):
        with self.lock:
            self.moves.clear()

    def offer(self, mini_id, cell, context, source="tracking"):
        with self.lock:
            if context != self.context or not context[0]:
                return False
            previous = self.moves.get(mini_id)
            if previous is None or previous.target_kind != "cell" or previous.cell != cell:
                self.moves[mini_id] = DesiredMove(
                    mini_id, cell, source, command_id=uuid.uuid4().hex,
                    assignment_requested=bool(previous and previous.token_id is None and previous.assignment_requested),
                )
            return True

    def offer_position(self, mini_id, center, context, source="base-tracker", *, generation=None,
                       expected_token_id=None, expected_scene_id=None):
        """Offer an absolute scene-space base center; token dimensions resolve later."""
        if (not isinstance(center, (tuple, list)) or len(center) != 2
                or not all(type(value) in (int, float) and math.isfinite(value) for value in center)):
            raise ValueError("A base center needs two finite coordinates")
        if generation is not None and (not isinstance(generation, str) or not generation or len(generation) > 256):
            raise ValueError("Provide a valid tracking generation")
        if expected_token_id is not None and (not isinstance(expected_token_id, str)
                or not expected_token_id or len(expected_token_id) > 256):
            raise ValueError("Provide the exact assigned Foundry token ID")
        if expected_scene_id is not None and (not isinstance(expected_scene_id, str)
                or not expected_scene_id or len(expected_scene_id) > 256):
            raise ValueError("Provide the scene used to measure the base position")
        center = tuple(map(float, center))
        with self.lock:
            if context != self.context or not context[0]:
                return False
            if expected_scene_id is not None and context[0] != expected_scene_id:
                return False
            previous = self.moves.get(mini_id)
            if (previous is None or previous.target_kind != "position"
                    or previous.center != center or previous.generation != generation
                    or previous.expected_token_id != expected_token_id):
                self.moves[mini_id] = DesiredMove(mini_id, None, source,
                    target_kind="position", center=center, generation=generation, expected_token_id=expected_token_id,
                    command_id=uuid.uuid4().hex,
                    assignment_requested=bool(previous and previous.token_id is None and previous.assignment_requested))
            return True

    def snapshot(self):
        with self.lock:
            return {name: vars(move).copy() for name, move in self.moves.items()}

    def waiting_for_assignment(self, mini_id):
        with self.lock:
            move = self.moves.get(mini_id)
            if move is None:
                return False
            move.state = "assignment"
            move.token_id = None
            request = not move.assignment_requested
            move.assignment_requested = True
            return request

    def prepare(self, mini_id, cell, token_id, coordinates, now):
        return self._prepare(mini_id, token_id, coordinates, now,
                             matches=lambda move: move.target_kind == "cell" and move.cell == cell)

    def prepare_position(self, mini_id, center, token_id, coordinates, now, *, generation=None):
        center = tuple(center)
        return self._prepare(mini_id, token_id, coordinates, now,
            matches=lambda move: move.target_kind == "position" and move.center == center and move.generation == generation)

    def _prepare(self, mini_id, token_id, coordinates, now, *, matches):
        with self.lock:
            move = self.moves.get(mini_id)
            if move is None or not matches(move):
                return None
            if move.binding_invalid:
                return None
            if move.expected_token_id is not None and token_id != move.expected_token_id:
                self.reject_position_binding(mini_id, move.command_id)
                return None
            if (move.token_id, move.coordinates) != (token_id, coordinates):
                # The initial offer ID is visible to the tracking controller so
                # it can register the expected redraw before the command sends.
                if move.token_id is not None or move.coordinates is not None:
                    move.command_id = uuid.uuid4().hex
                move.token_id, move.coordinates = token_id, coordinates
                move.attempts, move.retry_at = 0, 0.0
                move.state, move.error = "pending", ""
                move.assignment_requested = False
                move.applied_changed = None
            elif move.state == "geometry":
                move.state, move.error, move.retry_at = "pending", "", 0.0
            if move.state in {"confirmed", "failed"} or now < move.retry_at:
                return None
            if move.attempts >= self.max_attempts:
                move.state = "failed"
                move.error = move.error or "No acknowledgement from Foundry"
                return None
            move.attempts += 1
            move.retry_at = now + self.ack_timeout
            move.state = "waiting_ack"
            payload = {
                "type": "moveToken", "commandId": move.command_id,
                "sceneId": self.context[0], "miniId": mini_id,
                "tokenId": token_id, "cell": move.cell, "source": move.source,
                "x": coordinates[0], "y": coordinates[1],
            }
            if move.target_kind == "position":
                payload.update(targetKind="position", center=list(move.center))
                if move.generation is not None:
                    payload["bindingGeneration"] = move.generation
            return payload

    def reject_position_binding(self, mini_id, command_id):
        """An assignment change revokes an existing physical move permanently."""
        with self.lock:
            move = self.moves.get(mini_id)
            if (move is None or move.target_kind != "position" or move.command_id != command_id
                    or move.expected_token_id is None):
                return False
            move.binding_invalid = True
            move.state = "failed"
            move.error = "Token assignment changed. Choose starting positions and confirm assignments again."
            return True

    def wait_for_geometry(self, mini_id, center, message, *, generation=None, failed=False):
        with self.lock:
            move = self.moves.get(mini_id)
            if (move is None or move.target_kind != "position" or move.center != tuple(center)
                    or move.generation != generation or move.state == "confirmed" or move.binding_invalid):
                return False
            move.state, move.error = ("failed" if failed else "geometry"), str(message)
            return True

    def reply(self, payload, now):
        with self.lock:
            move = self.moves.get(payload.get("miniId"))
            if (move is None or not move.token_id or not self.context
                    or payload.get("sceneId") != self.context[0]
                    or payload.get("commandId") != move.command_id
                    or payload.get("tokenId") != move.token_id
                    or move.state != "waiting_ack"):
                return False
            if (payload.get("type") == "tokenMoveApplied"
                    and (payload.get("x"), payload.get("y")) == move.coordinates):
                move.state, move.error = "confirmed", ""
                move.applied_changed = payload.get("changed") if type(payload.get("changed")) is bool else None
            else:
                move.error = str(payload.get("error") or "Foundry position did not match the requested cell")
                move.state = "failed" if move.attempts >= self.max_attempts else "retry"
                move.retry_at = now + min(2.0, 0.5 * move.attempts)
            return True

    def retry(self, *, reconnect=False):
        with self.lock:
            for move in self.moves.values():
                if move.binding_invalid:
                    continue
                if reconnect or move.state in {"failed", "assignment"}:
                    move.command_id = uuid.uuid4().hex
                    move.attempts, move.retry_at = 0, 0.0
                    move.state, move.error = "pending", ""
                    move.assignment_requested = False
                    move.applied_changed = None
