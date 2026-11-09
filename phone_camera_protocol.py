"""Bounded v1 Sarween Camera packets, independent of USB and camera devices.

Each packet is a four-byte big-endian JSON length, a UTF-8 JSON object, and
exactly ``payloadBytes`` bytes. Phone timestamps belong to the phone clock;
they must not be subtracted from the Mac monotonic clock.
"""
from __future__ import annotations

import json
import math
import socket
import struct
import time
import uuid

PROTOCOL_VERSION = 1
PHONE_PORT = 52781
MAX_HEADER_BYTES = 64 * 1024
MAX_JPEG_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 32 * 1024 * 1024
MAX_IMAGE_SIDE = 8192
PHONE_REQUEST_ACTIONS = frozenset({"calibration.capture", "calibration.finish", "calibration.cancel",
                                   "cpl.reference", "cpl.target", "cpl.reset"})


class ProtocolError(ValueError):
    """Peer supplied an invalid or unsupported camera packet."""


class CameraCancelled(OSError):
    """Capture was cancelled while waiting for transport data."""


def integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProtocolError(f"Invalid {name}")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"Duplicate header field: {key}")
        result[key] = value
    return result


def _header(header):
    if not isinstance(header, dict):
        raise ProtocolError("Camera header must be a JSON object")
    if header.get("protocolVersion") != PROTOCOL_VERSION or type(header.get("protocolVersion")) is not int:
        raise ProtocolError("Unsupported camera protocol version")
    kind = header.get("type")
    if not isinstance(kind, str) or kind not in {"hello", "frame", "start", "stop", "ping", "pong", "ack", "status", "error", "feedback", "request"}:
        raise ProtocolError("Unknown camera packet type")
    size = integer(header.get("payloadBytes"), "payloadBytes", 0, MAX_JPEG_BYTES)
    if kind != "frame" and size:
        raise ProtocolError("Only frame packets may contain a payload")
    if kind == "frame" and not size:
        raise ProtocolError("Frame payload is empty")
    session = header.get("sessionId")
    if not isinstance(session, str) or not session or len(session) > 128:
        raise ProtocolError("Missing or invalid sessionId")
    return size


def encode_packet(header, payload=b""):
    """Validate and encode one packet; never silently fix its byte count."""
    payload = bytes(payload)
    if _header(header) != len(payload):
        raise ProtocolError("Payload byte count differs from header")
    try:
        raw = json.dumps(header, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ProtocolError("Invalid JSON camera header") from exc
    if not 0 < len(raw) <= MAX_HEADER_BYTES:
        raise ProtocolError("Camera header exceeds limit")
    return struct.pack(">I", len(raw)) + raw + payload


def read_exact(sock, count, *, cancel_event=None, deadline=None):
    """Preserve partial reads across socket timeouts, with cancellation/deadline.

    Callers set a short socket timeout so cancellation is checked during quiet
    connections. A timeout after partial data fails the connection rather than
    treating the remaining bytes as a new packet.
    """
    chunks = bytearray()
    while len(chunks) < count:
        if cancel_event is not None and cancel_event.is_set():
            raise CameraCancelled("Camera capture cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for phone camera data")
        try:
            part = sock.recv(min(count - len(chunks), 1024 * 1024))
        except socket.timeout:
            continue
        if not part:
            raise EOFError("Phone camera connection closed")
        chunks.extend(part)
    return bytes(chunks)


def read_packet(sock, *, cancel_event=None, deadline=None):
    raw_size = read_exact(sock, 4, cancel_event=cancel_event, deadline=deadline)
    size = struct.unpack(">I", raw_size)[0]
    if not 0 < size <= MAX_HEADER_BYTES:
        raise ProtocolError("Camera header exceeds limit")
    raw = read_exact(sock, size, cancel_event=cancel_event, deadline=deadline)
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_float=_finite_float,
                            parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("Non-finite JSON value")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("Invalid JSON camera header") from exc
    payload_size = _header(header)
    return header, read_exact(sock, payload_size, cancel_event=cancel_event, deadline=deadline)


def _finite_float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ProtocolError("Non-finite JSON value")
    return value


def validate_dimensions(width, height):
    width = integer(width, "width", 1, MAX_IMAGE_SIDE)
    height = integer(height, "height", 1, MAX_IMAGE_SIDE)
    if width * height > MAX_IMAGE_PIXELS:
        raise ProtocolError("Camera image exceeds pixel limit")
    return width, height


def validate_hello(header):
    _header(header)
    if header["type"] != "hello":
        raise ProtocolError("Phone must send hello before frames")
    name = header.get("deviceName")
    if not isinstance(name, str) or not name or len(name) > 256:
        raise ProtocolError("Invalid deviceName")
    if "hostFeedback" in header and type(header["hostFeedback"]) is not bool:
        raise ProtocolError("Invalid hostFeedback capability")
    if "sourceId" in header and (not isinstance(header["sourceId"], str)
                                  or not header["sourceId"] or len(header["sourceId"]) > 512):
        raise ProtocolError("Invalid sourceId")
    fmt = header.get("captureFormat")
    if not isinstance(fmt, dict) or fmt.get("pixelFormat") != "jpeg":
        raise ProtocolError("Phone must offer JPEG capture")
    validate_dimensions(fmt.get("width"), fmt.get("height"))
    fps = fmt.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not 0 < fps <= 120:
        raise ProtocolError("Invalid capture frame rate")
    if "lens" in fmt and (not isinstance(fmt["lens"], str) or fmt["lens"] not in {"wide", "ultra-wide", "telephoto"}):
        raise ProtocolError("Unsupported phone camera lens")
    if "nativeFieldOfViewDegrees" in fmt:
        fov = fmt["nativeFieldOfViewDegrees"]
        if type(fov) not in (int, float) or not math.isfinite(fov) or not 0 < fov < 180:
            raise ProtocolError("Invalid phone camera field of view")
    return dict(fmt)


def validate_request(header):
    if (header.get("type") != "request" or not isinstance(header.get("action"), str)
            or header["action"] not in PHONE_REQUEST_ACTIONS):
        raise ProtocolError("Unsupported phone request")
    request_id = header.get("requestId")
    try:
        if not isinstance(request_id, str) or len(request_id) != 36:
            raise ValueError()
        uuid.UUID(request_id)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("Invalid phone requestId") from exc
    return {"requestId": request_id, "action": header["action"]}


def validate_feedback(value):
    """Allow only small documented UI values, never arbitrary remote commands."""
    if not isinstance(value, dict) or value.get("kind") not in ("preview", "tracking", "calibration"):
        raise ProtocolError("Invalid phone feedback kind")
    if set(value) - {"kind", "sourceSeq", "markers", "calibration", "cpl"}:
        raise ProtocolError("Unknown phone feedback field")
    result = {"kind": value["kind"]}
    if "sourceSeq" in value:
        result["sourceSeq"] = integer(value["sourceSeq"], "feedback sourceSeq")
    for section in ("markers", "calibration", "cpl"):
        if section not in value:
            continue
        item = value[section]
        if not isinstance(item, dict):
            raise ProtocolError(f"Invalid {section} feedback")
        allowed = ({"ids", "requiredIds", "count", "required", "locked"} if section == "markers" else
                   {"active", "captured", "required", "title", "message", "reason", "mode", "canFinish", "countdownSeconds"}
                   if section == "calibration" else {"state", "index", "target", "message"})
        if set(item) - allowed:
            raise ProtocolError(f"Unknown {section} feedback field")
        output = {}
        for key, field in item.items():
            if key in ("ids", "requiredIds"):
                if not isinstance(field, list) or len(field) > 4:
                    raise ProtocolError("Invalid feedback marker IDs")
                output[key] = [integer(i, "marker ID", 0, 249) for i in field]
                if len(set(output[key])) != len(output[key]):
                    raise ProtocolError("Duplicate feedback marker IDs")
            elif key in ("locked", "active", "canFinish"):
                if type(field) is not bool:
                    raise ProtocolError(f"Invalid feedback {key}")
                output[key] = field
            elif key in ("count", "captured", "required"):
                output[key] = integer(field, f"feedback {key}", 0, 500)
            elif key in ("index", "target", "countdownSeconds"):
                if type(field) not in (int, float) or not math.isfinite(field) or not 0 <= field <= 10000:
                    raise ProtocolError(f"Invalid feedback {key}")
                output[key] = field
            else:
                if not isinstance(field, str) or len(field) > (300 if key == "message" else 120):
                    raise ProtocolError(f"Invalid feedback {key}")
                if key == "mode" and field not in ("sweep", "manual", "timed", "steady"):
                    raise ProtocolError("Invalid feedback calibration mode")
                output[key] = field
        if section == "markers":
            if (not {"ids", "requiredIds", "count", "required"}.issubset(output)
                    or len(output["requiredIds"]) != 4 or output["required"] != 4
                    or not set(output["ids"]).issubset(output["requiredIds"])
                    or output["count"] != len(output["ids"])):
                raise ProtocolError("Inconsistent feedback marker count")
        result[section] = output
    if len(json.dumps(result, allow_nan=False).encode()) > 4096:
        raise ProtocolError("Phone feedback exceeds limit")
    return result


def validate_frame_optics(header, capture_format):
    """Optical changes require a new hello, even at the same image size.

    Older v1 frames omitted these redundant fields; compare them when sent.
    The negotiated hello remains the authoritative calibration identity.
    """
    for key in ("cameraPosition", "lens", "zoom", "stabilization",
                "geometricDistortionCorrection", "hdr", "nativeFieldOfViewDegrees"):
        if key in header and (key not in capture_format or header[key] != capture_format[key]):
            raise ProtocolError(f"Phone camera {key} changed without a new handshake")


def validate_frame(header, *, session_id, previous_seq=-1, previous_capture_ns=-1):
    _header(header)
    if header["type"] != "frame" or header["sessionId"] != session_id:
        raise ProtocolError("Camera session changed without a fresh handshake")
    seq = integer(header.get("seq"), "seq")
    stamp = integer(header.get("captureTimeNs"), "captureTimeNs")
    if seq <= previous_seq or stamp <= previous_capture_ns:
        raise ProtocolError("Camera frame sequence or timestamp moved backwards")
    if type(header.get("rotationDegrees")) is not int or header["rotationDegrees"] != 0:
        raise ProtocolError("Phone must send correctly oriented pixels (rotationDegrees=0)")
    if header.get("mirrored") is not False:
        raise ProtocolError("Mirrored camera pixels are unsupported")
    validate_dimensions(header.get("width"), header.get("height"))
    if "encodeDurationNs" in header:
        integer(header["encodeDurationNs"], "encodeDurationNs")
    return seq, stamp


def jpeg_dimensions(payload):
    """Read JPEG SOF dimensions before allocating a decoded image."""
    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise ProtocolError("Frame payload is not JPEG")
    offset = 2
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset < len(payload):
        if payload[offset] != 0xFF:
            raise ProtocolError("Invalid JPEG segment")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if marker in {0xDA, 0xD9} or offset + 2 > len(payload):
            break
        length = struct.unpack(">H", payload[offset:offset + 2])[0]
        if length < 2 or offset + length > len(payload):
            raise ProtocolError("Invalid JPEG segment length")
        if marker in sof:
            if length < 8:
                raise ProtocolError("Invalid JPEG dimensions")
            height, width = struct.unpack(">HH", payload[offset + 3:offset + 7])
            return validate_dimensions(width, height)
        offset += length
    raise ProtocolError("JPEG contains no supported image dimensions")
