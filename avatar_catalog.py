"""Bounded, scene-scoped artwork assets for optional negative observations.

These isolated sprites are static artwork, not a synchronized rendered-map
reference. Geometry supplies expected size only; neither their capture clock
nor their availability establishes camera/display alignment.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
import math
import re

import cv2
import numpy as np
from PIL import Image

from reference_health import matching_view, validated_view


CATALOG_METHOD = "isolated-token-sprite-v1"
MAX_CATALOG_BYTES = 1_048_576
MAX_TEMPLATES = 32
MAX_IMAGE_SIDE = 96
MAX_IMAGE_CHARACTERS = 65_536
_PNG_PREFIX = "data:image/png;base64,"
_SAFE_REASON = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")


def _identifier(value, field):
    if (not isinstance(value, str) or not 1 <= len(value) <= 256
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"Invalid avatar catalog {field}")
    return value


def _integer(value, field, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid avatar catalog {field}")
    return value


def _positive_number(value, field, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 < value <= maximum or not math.isfinite(value)):
        raise ValueError(f"Invalid avatar catalog {field}")
    return value


def _png_bytes(item):
    encoded = item.get("image")
    if (not isinstance(encoded, str) or not encoded.startswith(_PNG_PREFIX)
            or len(encoded) > MAX_IMAGE_CHARACTERS):
        raise ValueError("Invalid or oversized avatar PNG encoding")
    try:
        raw = base64.b64decode(encoded[len(_PNG_PREFIX):], validate=True)
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Avatar image is not PNG")
        # Inspect headers before pixel decoding, including tiny compressed files
        # which declare an enormous raster. verify checks packaging, not pixels.
        with Image.open(io.BytesIO(raw)) as header:
            if (header.format != "PNG" or header.size != (item["width"], item["height"])
                    or not all(1 <= side <= MAX_IMAGE_SIDE for side in header.size)):
                raise ValueError("Avatar PNG dimensions do not match its metadata")
            header.verify()
    except (binascii.Error, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid avatar PNG data") from exc
    return raw


def validate_avatar_catalog(payload, scene_id=None) -> dict:
    """Copy and validate a complete wire snapshot, without decoding pixels.

    The digest covers the normalized snapshot except contentSha256 itself.
    Unsupported/unavailable snapshots are valid empty reset messages.
    """
    if not isinstance(payload, dict):
        raise ValueError("Avatar catalog must be a JSON object")
    try:
        serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        if len(serialized.encode("utf-8")) > MAX_CATALOG_BYTES:
            raise ValueError("Avatar catalog exceeds its size limit")
        source = json.loads(serialized)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Avatar catalog must contain bounded JSON data") from exc
    if source.get("type", "avatarCatalog") != "avatarCatalog":
        raise ValueError("Invalid avatar catalog message type")
    _integer(source.get("schemaVersion"), "schemaVersion", 1, 1)
    scene = _identifier(source.get("sceneId"), "sceneId")
    if scene_id is not None and scene != str(scene_id):
        raise ValueError("Avatar catalog belongs to another scene")
    catalog_id = _identifier(source.get("catalogId"), "catalogId")
    revision = _integer(source.get("revision"), "revision", 1, 2**53 - 1)
    if source.get("method") != CATALOG_METHOD:
        raise ValueError("Unsupported avatar capture method")
    status = source.get("status")
    if not isinstance(status, str) or status not in {"available", "unsupported", "unavailable"}:
        raise ValueError("Invalid avatar catalog status")
    timestamp = _positive_number(source.get("capturedAtEpochMs"), "capture clock", 10**15)
    items = source.get("templates")
    if not isinstance(items, list) or len(items) > MAX_TEMPLATES:
        raise ValueError("Invalid avatar catalog template count")
    if status != "available" and items:
        raise ValueError("Unavailable avatar catalogs must be empty")
    captured_view = source.get("viewTransform")
    if captured_view is not None or status == "available":
        captured_view = validated_view(captured_view, scene)
    templates, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid avatar template")
        template_id = _identifier(item.get("templateId"), "templateId")
        if template_id in seen:
            raise ValueError("Duplicate avatar template ID")
        seen.add(template_id)
        token_ids = item.get("tokenIds")
        if not isinstance(token_ids, list) or len(token_ids) > 64:
            raise ValueError("Invalid avatar template token IDs")
        normalized = {
            "templateId": template_id, "image": item.get("image"),
            "width": _integer(item.get("width"), "image width", 1, MAX_IMAGE_SIDE),
            "height": _integer(item.get("height"), "image height", 1, MAX_IMAGE_SIDE),
            "logicalWidth": _positive_number(item.get("logicalWidth"), "logical width", 8192),
            "logicalHeight": _positive_number(item.get("logicalHeight"), "logical height", 8192),
            "tokenIds": [_identifier(value, "tokenId") for value in token_ids],
        }
        if len(set(normalized["tokenIds"])) != len(token_ids):
            raise ValueError("Duplicate avatar token ID")
        _png_bytes(normalized)
        templates.append(normalized)
    unsupported = source.get("unsupportedTokens", [])
    if not isinstance(unsupported, list) or len(unsupported) > 64:
        raise ValueError("Invalid unsupported-token list")
    unsupported_tokens = []
    for item in unsupported:
        if (not isinstance(item, dict) or not isinstance(item.get("reason"), str)
                or not _SAFE_REASON.fullmatch(item["reason"])):
            raise ValueError("Invalid unsupported-token reason")
        unsupported_tokens.append({"tokenId": _identifier(item.get("tokenId"), "tokenId"),
                                   "reason": item["reason"]})
    result = {"type": "avatarCatalog", "schemaVersion": 1, "sceneId": scene,
              "catalogId": catalog_id, "revision": revision, "method": CATALOG_METHOD,
              "viewTransform": captured_view, "capturedAtEpochMs": timestamp, "status": status,
              "templates": templates, "unsupportedTokens": unsupported_tokens}
    if "reason" in source:
        if not isinstance(source["reason"], str) or not _SAFE_REASON.fullmatch(source["reason"]):
            raise ValueError("Invalid avatar catalog reason")
        result["reason"] = source["reason"]
    digest = hashlib.sha256(json.dumps(result, allow_nan=False, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    if source.get("contentSha256", digest) != digest:
        raise ValueError("Avatar catalog content digest changed")
    result["contentSha256"] = digest
    return result


def decode_avatar_catalog(payload, view, warp_size):
    """Decode matching static assets and map expected CSS size to board pixels.

    Invalid or mismatched metadata disables this optional filter. Old artwork
    can remain valid indefinitely while its appearance and geometry are stable.
    Callers may cache by fingerprint; returned pixel arrays are read-only.
    """
    try:
        catalog = validate_avatar_catalog(payload)
        if catalog["status"] != "available":
            return None
        current_view = validated_view(view, catalog["sceneId"])
        if not matching_view(catalog["viewTransform"], current_view):
            return None
        if not isinstance(warp_size, (tuple, list)) or len(warp_size) != 2:
            return None
        width, height = (_integer(value, "board dimension", 1, 16384) for value in warp_size)
        registration = current_view["registration"]
        scale_x = width / (float(registration["right"]) - float(registration["left"]))
        scale_y = height / (float(registration["bottom"]) - float(registration["top"]))
        if not all(math.isfinite(scale) for scale in (scale_x, scale_y)):
            return None
        templates = []
        for item in catalog["templates"]:
            raw = _png_bytes(item)
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if (decoded is None or decoded.dtype != np.uint8
                    or decoded.shape[:2] != (item["height"], item["width"])):
                return None
            if decoded.ndim == 2:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGRA)
            elif decoded.ndim == 3 and decoded.shape[2] == 3:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2BGRA)
            elif decoded.ndim != 3 or decoded.shape[2] != 4:
                return None
            decoded.flags.writeable = False
            board_width, board_height = item["logicalWidth"] * scale_x, item["logicalHeight"] * scale_y
            if not all(math.isfinite(value) and value > 0 for value in (board_width, board_height)):
                return None
            templates.append({"templateId": item["templateId"], "image_bgra": decoded,
                              "board_width": board_width, "board_height": board_height,
                              "tokenIds": tuple(item["tokenIds"])})
        geometry = {key: copy.deepcopy(current_view[key]) for key in (
            "sceneId", "viewportWidth", "viewportHeight", "registration", "canvasTransform",
            "clientToCanvasTransform", "gridSize", "gridOriginX", "gridOriginY")}
        fingerprint = hashlib.sha256(json.dumps([catalog["contentSha256"], geometry, [width, height]],
                                                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"sceneId": catalog["sceneId"], "catalogId": catalog["catalogId"],
                "revision": catalog["revision"], "contentSha256": catalog["contentSha256"],
                "fingerprint": fingerprint, "templates": templates}
    except (KeyError, TypeError, ValueError, OverflowError, cv2.error):
        return None
