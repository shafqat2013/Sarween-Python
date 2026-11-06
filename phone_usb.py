"""Minimal read-only USB discovery and app-port transport through macOS usbmuxd.

No drivers, helper processes, pairing changes or network-device fallback. Wire
format follows libusbmuxd's plist ListDevices/Connect interface:
https://github.com/libimobiledevice/libusbmuxd/blob/master/src/libusbmuxd.c
"""
from __future__ import annotations

import itertools
import plistlib
import socket
import struct
import time

from phone_camera_protocol import PHONE_PORT, CameraCancelled, read_exact

USBMUX_SOCKET = "/var/run/usbmuxd"
MAX_PLIST_BYTES = 1024 * 1024
_HEADER = struct.Struct("<IIII")


class PhoneUSBError(OSError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class USBMuxClient:
    """One short-lived daemon socket per operation; Connect returns raw stream."""
    def __init__(self, *, socket_factory=None, socket_path=USBMUX_SOCKET, timeout=2.0):
        self.socket_factory = socket_factory or socket.socket
        self.socket_path = socket_path
        self.timeout = max(.1, float(timeout))
        self._tags = itertools.count(1)

    def _open(self):
        sock = self.socket_factory(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
            sock.settimeout(.2)
            return sock
        except OSError as exc:
            sock.close()
            raise PhoneUSBError("usbmux_unavailable", "The macOS USB phone service is unavailable. Connect the iPhone by USB and unlock it.") from exc

    def _request(self, sock, message, *, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise CameraCancelled("Camera capture cancelled")
        request = {"ClientVersionString": "Sarween Camera 1", "ProgName": "Sarween",
                   "kLibUSBMuxVersion": 3, **message}
        payload = plistlib.dumps(request, fmt=plistlib.FMT_XML, sort_keys=False)
        tag = next(self._tags)
        sock.sendall(_HEADER.pack(_HEADER.size + len(payload), 1, 8, tag) + payload)
        deadline = time.monotonic() + self.timeout
        raw = read_exact(sock, _HEADER.size, cancel_event=cancel_event, deadline=deadline)
        size, version, kind, actual_tag = _HEADER.unpack(raw)
        if not _HEADER.size <= size <= MAX_PLIST_BYTES + _HEADER.size or version != 1 or kind != 8 or actual_tag != tag:
            raise PhoneUSBError("invalid_usbmux_reply", "The macOS USB service returned an invalid response.")
        raw = read_exact(sock, size - _HEADER.size, cancel_event=cancel_event, deadline=deadline)
        try:
            response = plistlib.loads(raw)
        except Exception as exc:
            raise PhoneUSBError("invalid_usbmux_reply", "The macOS USB service returned an invalid plist.") from exc
        if not isinstance(response, dict):
            raise PhoneUSBError("invalid_usbmux_reply", "The macOS USB response was not a dictionary.")
        return response

    @staticmethod
    def _result(response):
        code = response.get("Number")
        if response.get("MessageType") != "Result" or type(code) is not int:
            raise PhoneUSBError("invalid_usbmux_reply", "Missing USB connection result.")
        if code == 0:
            return
        if code == 2:
            raise PhoneUSBError("phone_disconnected", "The selected iPhone is no longer connected by USB.")
        if code == 3:
            raise PhoneUSBError("app_unavailable", "The iPhone is connected, but Sarween Camera is not accepting connections. Unlock the phone, open Sarween Camera, resume if paused, and keep it in the foreground. Accept Trust This Computer if asked.")
        raise PhoneUSBError("usbmux_rejected", f"The macOS USB service rejected the connection (code {code}). Unlock the iPhone and accept Trust This Computer if asked.")

    def list_devices(self, *, cancel_event=None):
        """List USB attachments only. Does not connect to an app or camera."""
        sock = self._open()
        try:
            result = self._request(sock, {"MessageType": "ListDevices"}, cancel_event=cancel_event)
        finally:
            sock.close()
        if "DeviceList" not in result:
            self._result(result)
            raise PhoneUSBError("invalid_usbmux_reply", "The USB service did not return a device list.")
        if not isinstance(result["DeviceList"], list):
            raise PhoneUSBError("invalid_usbmux_reply", "Invalid USB device list.")
        devices = {}
        for entry in result["DeviceList"]:
            if not isinstance(entry, dict):
                continue
            props = entry.get("Properties", {})
            if not isinstance(props, dict) or props.get("ConnectionType") != "USB":
                continue
            udid = props.get("SerialNumber")
            handle = entry.get("DeviceID", props.get("DeviceID"))
            if not isinstance(udid, str) or not udid or len(udid) > 256 or type(handle) is not int or handle <= 0:
                continue
            devices[udid] = {"id": udid, "deviceId": udid, "label": f"iPhone / iPad USB · {udid[-6:]}",
                             "transport": "usb", "usbDeviceId": handle, "productId": props.get("ProductID")}
        return sorted(devices.values(), key=lambda item: item["id"])

    def pairing_status(self, device_id, *, cancel_event=None):
        """Read pairing-record presence without returning certificates or keys.

        A saved record is only a hint: the phone can still be locked, untrusted
        after reset, or have the camera app closed. This never initiates pairing.
        """
        sock = self._open()
        try:
            response = self._request(sock, {"MessageType": "ReadPairRecord", "PairRecordID": device_id}, cancel_event=cancel_event)
        finally:
            sock.close()
        record = response.get("PairRecordData")
        if isinstance(record, bytes) and record:
            return {"pairedRecordPresent": True, "message": "A Mac pairing record exists; keep the iPhone unlocked and the camera app open."}
        return {"pairedRecordPresent": False, "message": "No saved pairing record was reported. Unlock the iPhone and accept Trust This Computer if asked."}

    def connect(self, device_id, port=PHONE_PORT, *, cancel_event=None):
        """Re-resolve stable USB serial before every connection/reconnection."""
        devices = self.list_devices(cancel_event=cancel_event)
        matches = [device for device in devices if device["id"] == device_id]
        if not matches:
            raise PhoneUSBError("phone_disconnected", "The selected iPhone is not connected by USB. Reconnect its cable and unlock it.")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid phone app port")
        sock = self._open()
        try:
            response = self._request(sock, {"MessageType": "Connect", "DeviceID": matches[0]["usbDeviceId"],
                                           "PortNumber": socket.htons(port)}, cancel_event=cancel_event)
            self._result(response)
            return sock
        except BaseException:
            sock.close()
            raise


def list_phone_cameras(*, client=None):
    """Explicitly invoked, USB-only discovery; raises actionable PhoneUSBError."""
    return (client or USBMuxClient()).list_devices()
