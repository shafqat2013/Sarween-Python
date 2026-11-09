"""Direct USB iPhone capture with bounded latency and OpenCV-like reads.

Discovery never starts capture. Constructing PhoneCameraCapture is the explicit
camera-open action: its one reader connects, accepts hello, and requests start.
Only the newest decoded frame is retained. ``read`` delivers a frame once and
returns False on timeout/loss; callers must invalidate tracking while disconnected.
"""
from __future__ import annotations

import copy
from collections import deque
import socket
import threading
import time

import cv2
import numpy as np

from phone_camera_protocol import (
    PHONE_PORT, PROTOCOL_VERSION, CameraCancelled, ProtocolError, encode_packet,
    jpeg_dimensions, read_packet, validate_frame, validate_frame_optics, validate_hello,
    validate_feedback, validate_request,
)
from phone_usb import PhoneUSBError, USBMuxClient, list_phone_cameras


class PhoneCameraCapture:
    """Asynchronous, USB-only source; no queue or helper process is created.

    ``isOpened`` means the source is requested and its worker may reconnect;
    inspect ``status['connected']``/``status['fresh']`` for live availability.
    Receive age uses only the Mac clock, not the unrelated phone clock, and is
    not an estimate of sensor-to-Mac latency. Reconnect never changes devices.
    """
    def __init__(self, device_id=None, *, transport=None, read_timeout=1.0,
                 stale_timeout=1.0, reconnect_delays=(.25, .5, 1., 2.),
                 packet_timeout=3.0, clock=time.monotonic):
        if device_id is not None and (not isinstance(device_id, str) or not device_id):
            raise ValueError("Select a valid USB phone device ID")
        self.device_id = device_id
        self.transport = transport or USBMuxClient()
        self.read_timeout = max(.01, float(read_timeout))
        self.stale_timeout = max(.01, float(stale_timeout))
        self.packet_timeout = max(.1, float(packet_timeout))
        self._delays = tuple(max(.01, float(value)) for value in reconnect_delays) or (.25,)
        self._clock = clock
        self._condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._socket = None
        self._latest = None
        self._last_received = None
        self._last_frame_metadata = None
        self._terminal = False
        self._generation = 0
        self._session_id = None
        self._previous_session = None
        self._device_name = None
        self._source_id = None
        self._format = {"width": 1080, "height": 1920, "fps": 15, "pixelFormat": "jpeg"}
        self._state = "connecting"
        self._message = "Connecting to Sarween Camera over USB…"
        self._code = None
        self._connected = False
        self._dropped = 0
        self._host_feedback = False
        self._pending_feedback = None
        self._feedback_sent_at = 0.
        self._requests = deque(maxlen=16)
        self._request_ids = deque(maxlen=64)
        self._thread = threading.Thread(target=self._run, name="SarweenPhoneCamera", daemon=True)
        self._thread.start()

    def _update(self, state, message, *, code=None, connected=False, clear=True):
        with self._condition:
            self._state, self._message, self._code = state, message, code
            self._connected = connected
            if clear:
                self._latest = None
                self._last_received = None
            self._condition.notify_all()

    def _command(self, kind, **fields):
        with self._send_lock:
            sock = self._socket
            if sock is None or self._session_id is None:
                raise OSError("Phone camera transport is not connected")
            sock.sendall(encode_packet({"type": kind, "protocolVersion": PROTOCOL_VERSION,
                                       "sessionId": self._session_id, "payloadBytes": 0, **fields}))

    def _choose_device(self):
        if self.device_id is not None:
            return
        devices = self.transport.list_devices(cancel_event=self._stop)
        # Even injected transports may list network devices: never accept them.
        devices = [device for device in devices if device.get("transport") == "usb"]
        if not devices:
            raise PhoneUSBError("phone_disconnected", "No USB iPhone is connected. Connect a data cable, unlock the iPhone, and open Sarween Camera.")
        if len(devices) != 1:
            self._terminal = True
            raise PhoneUSBError("selection_required", "Several USB phones are connected. Select the iPhone to use in Sarween.")
        self.device_id = devices[0]["id"]

    def _run_connection(self):
        self._choose_device()
        sock = self.transport.connect(self.device_id, PHONE_PORT, cancel_event=self._stop)
        with self._condition:
            if self._stop.is_set():
                sock.close()
                raise CameraCancelled("Camera capture cancelled")
            self._socket = sock
        sock.settimeout(.2)
        hello, _ = read_packet(sock, cancel_event=self._stop, deadline=time.monotonic() + self.packet_timeout)
        fmt = validate_hello(hello)
        session = hello["sessionId"]
        if session == self._previous_session:
            raise ProtocolError("Reconnected phone reused its old sessionId")
        with self._condition:
            self._generation += 1
            self._session_id = session
            self._previous_session = session
            self._format = fmt
            self._device_name = hello["deviceName"]
            self._source_id = hello.get("sourceId")
            self._latest = None
            self._last_received = None
            self._last_frame_metadata = None
            self._host_feedback = hello.get("hostFeedback") is True
            self._pending_feedback = None
            self._requests.clear()
            self._request_ids.clear()
            self._feedback_sent_at = 0.
        self._command("start", ackFrames=True, **({"hostFeedback": True} if self._host_feedback else {}))
        self._update("waiting", "USB connected. Waiting for the iPhone camera…", connected=True)
        previous_seq = previous_capture = -1
        while not self._stop.is_set():
            self._flush_feedback()
            header, payload = read_packet(sock, cancel_event=self._stop, deadline=time.monotonic() + self.packet_timeout)
            received = self._clock()
            if header["sessionId"] != session:
                raise ProtocolError("Camera session changed without reconnecting")
            if header["type"] == "request":
                if not self._host_feedback:
                    raise ProtocolError("Phone requests were not negotiated")
                request = validate_request(header)
                with self._condition:
                    if request["requestId"] not in self._request_ids:
                        self._request_ids.append(request["requestId"])
                        self._requests.append({**request, "sessionId": session,
                            "connectionGeneration": self._generation, "hostReceivedMonotonic": received})
                continue
            if header["type"] in {"status", "pong"}:
                message = header.get("message", "Waiting for the iPhone camera…")
                if not isinstance(message, str):
                    raise ProtocolError("Invalid camera status message")
                # Status/pong cannot refresh a frame's age or make it visible.
                with self._condition:
                    self._message = message[:1024]
                    self._condition.notify_all()
                continue
            if header["type"] == "error":
                message = header.get("message", "The iPhone camera is unavailable. Check camera permission and keep Sarween Camera open.")
                raise PhoneUSBError("phone_camera_error", str(message)[:1024])
            seq, capture_ns = validate_frame(header, session_id=session, previous_seq=previous_seq,
                                              previous_capture_ns=previous_capture)
            validate_frame_optics(header, fmt)
            dimensions = (header["width"], header["height"])
            if dimensions != (fmt["width"], fmt["height"]):
                raise ProtocolError("Phone capture dimensions changed without a new handshake")
            if jpeg_dimensions(payload) != dimensions:
                raise ProtocolError("JPEG dimensions differ from the declared frame size")
            try:
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            except cv2.error as exc:
                raise ProtocolError("Phone JPEG could not be decoded") from exc
            if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape != (dimensions[1], dimensions[0], 3):
                raise ProtocolError("Decoded phone image has invalid dimensions or pixel type")
            previous_seq, previous_capture = seq, capture_ns
            metadata = {"source": "sarween-phone-usb", "transport": "usb", "protocolVersion": PROTOCOL_VERSION,
                        "deviceId": self.device_id, "deviceName": self._device_name, "sessionId": session,
                        "sourceId": self._source_id,
                        "connectionGeneration": self._generation, "seq": seq,
                        "width": dimensions[0], "height": dimensions[1], "rotationDegrees": 0, "mirrored": False,
                        "captureTimeNs": capture_ns, "hostReceivedMonotonic": received,
                        "hostDecodedMonotonic": self._clock(), "format": dict(fmt)}
            if "encodeDurationNs" in header:
                metadata["encodeDurationNs"] = header["encodeDurationNs"]
            metadata.update(_diagnostic_telemetry(header))
            with self._condition:
                if self._stop.is_set():
                    raise CameraCancelled("Camera capture cancelled")
                if self._latest is not None:
                    self._dropped += 1
                metadata["droppedFrames"] = self._dropped
                self._latest = image, metadata
                self._last_received = received
                self._state, self._message, self._code = "streaming", "Receiving iPhone camera over USB", None
                self._connected = True
                self._condition.notify_all()
            # Acknowledge after validation/decode/publication, never after UI read.
            # The phone retains at most one encoded in-flight frame until this.
            self._command("ack", seq=seq)

    def _close_socket(self):
        with self._condition:
            sock, self._socket = self._socket, None
            self._session_id = None
            self._host_feedback = False
            self._pending_feedback = None
            self._requests.clear()
            self._request_ids.clear()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _run(self):
        attempts = 0
        try:
            while not self._stop.is_set():
                try:
                    self._run_connection()
                except CameraCancelled:
                    break
                except (OSError, EOFError, ProtocolError, ValueError) as exc:
                    if self._stop.is_set():
                        break
                    code = getattr(exc, "code", "invalid_stream" if isinstance(exc, ProtocolError) else "connection_lost")
                    self._update("error" if self._terminal else "reconnecting", str(exc), code=code)
                finally:
                    self._close_socket()
                if self._terminal:
                    break
                # Delay is capped and cancellable; there is no background device
                # selection change and no helper process to outlive release().
                delay = self._delays[min(attempts, len(self._delays) - 1)]
                attempts += 1
                if self._stop.wait(delay):
                    break
        finally:
            self._close_socket()
            if self._stop.is_set():
                self._update("closed", "iPhone camera stopped", code="released")

    def read(self):
        """Return one fresh new BGR frame, or (False,None) within read_timeout."""
        deadline = time.monotonic() + self.read_timeout
        with self._condition:
            while not self._stop.is_set() and not self._terminal:
                if self._latest is not None:
                    image, metadata = self._latest
                    self._latest = None
                    age = self._clock() - metadata["hostReceivedMonotonic"]
                    if self._connected and 0 <= age <= self.stale_timeout:
                        self._last_frame_metadata = {**metadata, "hostReceiveAgeSeconds": age}
                        return True, image
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
        return False, None

    @property
    def last_frame_metadata(self):
        with self._condition:
            return copy.deepcopy(self._last_frame_metadata)

    @property
    def status(self):
        with self._condition:
            age = None if self._last_received is None else max(0., self._clock() - self._last_received)
            return {"state": self._state, "message": self._message, "code": self._code,
                    "source": "sarween-phone-usb", "transport": "usb", "protocolVersion": PROTOCOL_VERSION,
                    "connected": self._connected, "fresh": bool(self._connected and age is not None and age <= self.stale_timeout),
                    "deviceId": self.device_id, "deviceName": self._device_name, "sessionId": self._session_id,
                    "sourceId": self._source_id,
                    "connectionGeneration": self._generation, "format": copy.deepcopy(self._format),
                    "hostFeedback": self._host_feedback,
                    "hostReceiveAgeSeconds": age, "droppedFrames": self._dropped}

    def snapshot_metadata(self):
        return self.status

    def send_feedback(self, feedback):
        """Queue one latest UI update; never wait for USB or perform a read."""
        value = validate_feedback(feedback)
        with self._condition:
            if not self._host_feedback or not self._connected or self._stop.is_set():
                return False
            if "sourceSeq" in value:
                last = self._last_frame_metadata
                if (not last or last.get("sessionId") != self._session_id
                        or last.get("seq") != value["sourceSeq"]
                        or self._clock() - last["hostReceivedMonotonic"] > self.stale_timeout):
                    return False
            self._pending_feedback = self._generation, value
        return True

    def _flush_feedback(self, *, force=False):
        with self._condition:
            pending = self._pending_feedback
            if (pending is None or not self._host_feedback or self._socket is None
                    or (not force and time.monotonic() - self._feedback_sent_at < .5)):
                return
            self._pending_feedback = None
            if pending[0] != self._generation:
                return
            if "sourceSeq" in pending[1]:
                last = self._last_frame_metadata
                if (not last or last.get("sessionId") != self._session_id
                        or self._clock() - last["hostReceivedMonotonic"] > self.stale_timeout):
                    return
            self._feedback_sent_at = time.monotonic()
        self._command("feedback", feedback=pending[1])

    def take_requests(self):
        with self._condition:
            requests = list(self._requests)
            self._requests.clear()
            return copy.deepcopy(requests)

    def isOpened(self):
        return not self._stop.is_set() and not self._terminal

    def get(self, prop):
        with self._condition:
            return float({cv2.CAP_PROP_FRAME_WIDTH: self._format["width"],
                          cv2.CAP_PROP_FRAME_HEIGHT: self._format["height"],
                          cv2.CAP_PROP_FPS: self._format["fps"],
                          cv2.CAP_PROP_BUFFERSIZE: 1}.get(prop, 0.))

    def set(self, prop, value):
        # Capture format is chosen on the phone, never silently altered by Mac.
        return False

    def release(self):
        """Stop, close the socket to unblock recv, and cancel pending backoff."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._flush_feedback(force=True)
            self._command("stop")
        except (OSError, ProtocolError):
            pass
        self._close_socket()
        self._update("closed", "iPhone camera stopped", code="released")
        if threading.current_thread() is not self._thread:
            # Native operations have bounded timeouts, including socket connect.
            self._thread.join(timeout=2.5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


def _diagnostic_telemetry(header):
    """Carry small validated measurements; none participates in stream identity."""
    result = {}
    for key in ("torchOn", "exposureLocked", "whiteBalanceLocked"):
        if type(header.get(key)) is bool:
            result[key] = header[key]
    for key, low, high in (("torchLevel", 0, 1), ("lensPosition", 0, 1), ("iso", 0, 1_000_000)):
        value = header.get(key)
        if type(value) in (int, float) and low <= value <= high:
            result[key] = value
    exposure = header.get("exposureDurationNs")
    if type(exposure) is int and 0 <= exposure <= 2**63 - 1:
        result["exposureDurationNs"] = exposure
    gains = header.get("whiteBalanceGains")
    if isinstance(gains, list) and len(gains) == 3 and all(type(value) in (int, float) and 0 < value <= 1000 for value in gains):
        result["whiteBalanceGains"] = list(gains)
    return result
