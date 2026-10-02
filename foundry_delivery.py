"""Thread-safe, latest-position outbox for acknowledged Foundry moves."""

from dataclasses import dataclass
import threading
import uuid


@dataclass
class DesiredMove:
    mini_id: str
    cell: str
    source: str
    command_id: str = ""
    token_id: str | None = None
    coordinates: tuple | None = None
    attempts: int = 0
    retry_at: float = 0.0
    state: str = "pending"
    error: str = ""
    assignment_requested: bool = False


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
            if previous is None or previous.cell != cell:
                self.moves[mini_id] = DesiredMove(
                    mini_id, cell, source, command_id=uuid.uuid4().hex,
                    assignment_requested=bool(previous and previous.token_id is None and previous.assignment_requested),
                )
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
        with self.lock:
            move = self.moves.get(mini_id)
            if move is None or move.cell != cell:
                return None
            if (move.token_id, move.coordinates) != (token_id, coordinates):
                move.command_id = uuid.uuid4().hex
                move.token_id, move.coordinates = token_id, coordinates
                move.attempts, move.retry_at = 0, 0.0
                move.state, move.error = "pending", ""
                move.assignment_requested = False
            if move.state in {"confirmed", "failed"} or now < move.retry_at:
                return None
            if move.attempts >= self.max_attempts:
                move.state = "failed"
                move.error = move.error or "No acknowledgement from Foundry"
                return None
            move.attempts += 1
            move.retry_at = now + self.ack_timeout
            move.state = "waiting_ack"
            return {
                "type": "moveToken", "commandId": move.command_id,
                "sceneId": self.context[0], "miniId": mini_id,
                "tokenId": token_id, "cell": cell, "source": move.source,
                "x": coordinates[0], "y": coordinates[1],
            }

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
            else:
                move.error = str(payload.get("error") or "Foundry position did not match the requested cell")
                move.state = "failed" if move.attempts >= self.max_attempts else "retry"
                move.retry_at = now + min(2.0, 0.5 * move.attempts)
            return True

    def retry(self, *, reconnect=False):
        with self.lock:
            for move in self.moves.values():
                if reconnect or move.state in {"failed", "assignment"}:
                    move.command_id = uuid.uuid4().hex
                    move.attempts, move.retry_at = 0, 0.0
                    move.state, move.error = "pending", ""
                    move.assignment_requested = False
