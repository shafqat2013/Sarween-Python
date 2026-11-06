"""Minimal usbmux protocol tests; every socket is an in-memory fake."""
import plistlib
import socket
import struct
import unittest

from phone_camera_protocol import PHONE_PORT
from phone_usb import MAX_PLIST_BYTES, PhoneUSBError, USBMuxClient, list_phone_cameras


def response(value, *, tag=1, version=1, kind=8):
    raw = plistlib.dumps(value)
    return struct.pack("<IIII", 16 + len(raw), version, kind, tag) + raw


def attached(identifier="phone-a", handle=7, connection="USB"):
    return {"DeviceID": handle, "Properties": {"SerialNumber": identifier,
            "DeviceID": handle, "ConnectionType": connection, "ProductID": 123}}


class FakeSocket:
    def __init__(self, reply):
        self.reply = bytearray(reply)
        self.sent = []
        self.closed = False
        self.path = None

    def settimeout(self, value):
        self.timeout = value

    def connect(self, path):
        self.path = path

    def recv(self, size):
        size = min(size, 5, len(self.reply))
        value = bytes(self.reply[:size])
        del self.reply[:size]
        return value

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True

    def request(self):
        packet = self.sent[0]
        size, version, kind, tag = struct.unpack("<IIII", packet[:16])
        assert size == len(packet) and (version, kind) == (1, 8)
        return tag, plistlib.loads(packet[16:])


class USBMuxTests(unittest.TestCase):
    def client(self, *responses):
        sockets = [FakeSocket(value) for value in responses]
        pending = list(sockets)
        client = USBMuxClient(socket_factory=lambda *_: pending.pop(0))
        return client, sockets

    def test_discovery_filters_wifi_and_malformed_entries_and_never_connects(self):
        client, sockets = self.client(response({"DeviceList": [attached(), attached("wifi", 8, "Network"),
            {"Properties": {"ConnectionType": "USB"}}, "invalid-entry"]}))
        cameras = list_phone_cameras(client=client)
        self.assertEqual([item["id"] for item in cameras], ["phone-a"])
        self.assertEqual(cameras[0]["transport"], "usb")
        self.assertEqual(sockets[0].request()[1]["MessageType"], "ListDevices")
        self.assertTrue(sockets[0].closed)
        self.assertEqual(sockets[0].path, "/var/run/usbmuxd")

    def test_connect_uses_latest_handle_and_network_order_port_and_keeps_stream(self):
        client, sockets = self.client(response({"DeviceList": [attached(handle=99)]}),
                                      response({"MessageType": "Result", "Number": 0}, tag=2) + b"APP")
        connected = client.connect("phone-a")
        _, request = sockets[1].request()
        self.assertEqual(request["MessageType"], "Connect")
        self.assertEqual(request["DeviceID"], 99)
        self.assertEqual(request["PortNumber"], socket.htons(PHONE_PORT))
        self.assertFalse(connected.closed)
        self.assertEqual(connected.recv(3), b"APP")

    def test_device_absence_and_app_closed_have_distinct_status(self):
        client, _ = self.client(response({"DeviceList": [attached("other")]}))
        with self.assertRaises(PhoneUSBError) as caught:
            client.connect("phone-a")
        self.assertEqual(caught.exception.code, "phone_disconnected")
        client, sockets = self.client(response({"DeviceList": [attached()]}),
                                      response({"MessageType": "Result", "Number": 3}, tag=2))
        with self.assertRaises(PhoneUSBError) as caught:
            client.connect("phone-a")
        self.assertEqual(caught.exception.code, "app_unavailable")
        self.assertTrue(sockets[1].closed)

    def test_bad_tags_versions_sizes_and_plists_rejected(self):
        replies = [response({}, tag=2), response({}, version=0), response({}, kind=1),
                   struct.pack("<IIII", MAX_PLIST_BYTES + 17, 1, 8, 1),
                   struct.pack("<IIII", 17, 1, 8, 1) + b"x"]
        for reply in replies:
            client, sockets = self.client(reply)
            with self.subTest(reply=reply[:16]), self.assertRaises(PhoneUSBError):
                client.list_devices()
            self.assertTrue(sockets[0].closed)

    def test_pairing_hint_contains_no_record_secrets_and_never_pairs(self):
        client, sockets = self.client(response({"PairRecordData": b"private certificate bytes"}))
        result = client.pairing_status("phone-a")
        self.assertTrue(result["pairedRecordPresent"])
        self.assertNotIn("private", repr(result))
        self.assertEqual(sockets[0].request()[1]["MessageType"], "ReadPairRecord")


if __name__ == "__main__":
    unittest.main()
