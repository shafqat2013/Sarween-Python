"""Explicit scene-scoped token creation requests; no Foundry or GUI side effects."""
from copy import deepcopy
import json
import re
from uuid import uuid4


def parse_cell(value):
    if not isinstance(value, str):
        raise ValueError("Enter a grid cell such as A1.")
    match = re.fullmatch(r"([A-Za-z]{1,4})([1-9][0-9]{0,5})", value.strip())
    if not match:
        raise ValueError("Enter a grid cell such as A1.")
    col = 0
    for letter in match[1].upper():
        col = col * 26 + ord(letter) - ord("A") + 1
    row, col = int(match[2]) - 1, col - 1
    if max(row, col) > 100000:
        raise ValueError("That grid cell is outside the supported range.")
    return row, col


def validate_token_request(values):
    allowed = {"actorId", "name", "placement", "row", "col", "creationId"}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("Invalid token creation fields.")
    if (not isinstance(values.get("actorId"), str) or not 1 <= len(values["actorId"]) <= 128
            or not isinstance(values.get("name"), str) or len(values["name"]) > 100
            or any(ord(ch) < 32 for ch in values["name"])
            or not isinstance(values.get("creationId"), str)
            or re.fullmatch(r"[A-Za-z0-9_-]{16,64}", values["creationId"]) is None):
        raise ValueError("Choose an actor and a valid token name.")
    if values.get("placement") == "cell":
        if any(type(values.get(field)) is not int or not 0 <= values[field] <= 100000 for field in ("row", "col")):
            raise ValueError("Choose a valid grid cell.")
    elif values.get("placement") == "viewport":
        if "row" in values or "col" in values:
            raise ValueError("Viewport placement has no grid cell.")
    else:
        raise ValueError("Choose a token placement.")
    return deepcopy(values)


def blocked_reason(status):
    if status.get("outputPaused"):
        return "Stop the mini observer or finish capture before creating tokens."
    if not status.get("connected") or not status.get("supported"):
        return "Connect Foundry with the updated Sarween module to create tokens."
    if "token.create" not in status.get("capabilities", []):
        return "Reload Foundry with the updated Sarween module to create tokens."
    capture = status.get("capture") or {}
    if (capture.get("active") or capture.get("recording")
            or capture.get("phase") in {"Starting", "Saving", "Restoring"}):
        return "Finish recording before creating tokens."
    if status.get("blocked") or status.get("pending"):
        return "Wait for the current Foundry action to finish."
    if not status.get("sceneId"):
        return "Open a Foundry scene first."
    return ""


class TokenCreationController:
    """Keep one creation ID across unconfirmed retries, never retry automatically."""
    def __init__(self, request, *, new_id=None):
        self.request = request
        self.new_id = new_id or (lambda: uuid4().hex)
        self.operation = None
        self.pending_request = None
        self.message = "Choose a world actor and placement, then create its prototype token."
        self.last_created = None
        self.context = None

    def refresh(self, status):
        context = ((status.get("world") or {}).get("id"), status.get("sceneId"))
        known_context = bool(status.get("connected") and context[1])
        if known_context and self.context is not None and context != self.context:
            self.operation = self.pending_request = None
            self.last_created = None
            self.message = "Scene changed. Check the placement before creating a token."
        if known_context:
            self.context = context
        if self.pending_request and not status.get("connected"):
            self.pending_request = None
            self.message = "Creation was not confirmed. Reconnect, then retry the same choices to check it."
        result = status.get("lastResult") or {}
        if self.pending_request and result.get("requestId") == self.pending_request:
            self.pending_request = None
            self.message = str(result.get("message") or "Foundry did not confirm creation.")
            if result.get("success") is True:
                self.last_created = deepcopy(result.get("result"))
                self.operation = None
        return self.message

    def create(self, draft, status):
        self.refresh(status)
        reason = blocked_reason(status)
        if reason:
            self.message = reason
            return False
        if self.pending_request:
            self.message = "Waiting for Foundry to confirm token creation."
            return False
        if draft.get("actorId") not in {actor.get("actorId") for actor in status.get("actors", [])}:
            self.message = "Choose an available world actor."
            return False
        values = dict(draft)
        signature = json.dumps({"context": self.context, "values": values}, sort_keys=True, allow_nan=False)
        if self.operation is None or self.operation["signature"] != signature:
            self.operation = {"signature": signature, "creationId": self.new_id()}
        values["creationId"] = self.operation["creationId"]
        try:
            values = validate_token_request(values)
            request_id = self.request("token.create", values)
            if not isinstance(request_id, str) or not request_id:
                self.message = "Foundry did not accept creation. Check the connection, then retry."
                return False
            self.pending_request = request_id
            self.message = "Creating token…"
            return True
        except (ValueError, OSError, RuntimeError) as exc:
            self.message = str(exc)
            return False
