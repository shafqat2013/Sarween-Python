"""Exercise owner commands through the real SDK without credentials or network."""

import base64
import contextlib
import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from scripts import manage_alpha


class OwnerCommandsTest(unittest.TestCase):
    def run_command(self, action, *, opaque=False):
        body = base64.urlsafe_b64encode(b'{"role":"service_role"}').decode().rstrip("=")
        secret = "sb_secret_fake_test_only" if opaque else "test." + body + ".signature"
        uid = "00000000-0000-4000-8000-000000000001"
        user = {"id": uid, "email": "tester@example.invalid", "aud": "authenticated",
                "app_metadata": {}, "user_metadata": {}, "created_at": "2026-10-01T00:00:00Z"}
        calls, approval = [], {"approved": True}

        def respond(request):
            calls.append((request.method, request.url.path))
            self.assertEqual(request.headers["apikey"], secret)
            # The hosted Auth service rejected legacy owner requests without this.
            if not opaque and request.headers.get("Authorization") != "Bearer " + secret:
                return httpx.Response(401, json={"message": "Missing owner authorization"})
            if opaque:
                self.assertNotIn("Authorization", request.headers)
            if request.url.path == "/auth/v1/admin/users":
                if request.method == "GET":
                    return httpx.Response(200, json={"users": [] if action == "approve" else [user],
                                                    "aud": "authenticated"})
                self.assertEqual(action, "approve")
                payload = json.loads(request.content)
                self.assertEqual(payload["email"], user["email"])
                self.assertTrue(payload["email_confirm"])
                self.assertNotIn("password", payload)
                return httpx.Response(200, json=user)
            if request.url.path == "/rest/v1/alpha_testers":
                if request.method == "GET":
                    self.assertEqual(request.url.params["user_id"], "eq." + uid)
                    return httpx.Response(200, json=[approval])
                payload = json.loads(request.content)
                self.assertEqual(payload["user_id"], uid)
                self.assertEqual(payload["approved"], action == "approve")
                approval.update(payload)
                return httpx.Response(201, json=payload)
            self.fail("Unexpected endpoint; owner commands must not send email")

        with contextlib.closing(httpx.Client(transport=httpx.MockTransport(respond))) as client:
            output = io.StringIO()
            with patch("httpx.Client", return_value=client), \
                    patch("auth_config.AuthConfig.load", return_value=SimpleNamespace(url="https://alpha.invalid")), \
                    patch("getpass.getpass", return_value=secret), \
                    patch.object(sys, "argv", ["manage_alpha.py", action, user["email"]]), \
                    contextlib.redirect_stdout(output):
                manage_alpha.main()
            self.assertNotIn(secret, output.getvalue())
        self.assertEqual(calls[0], ("GET", "/auth/v1/admin/users"))
        self.assertEqual(calls[-1], ("GET" if action == "status" else "POST", "/rest/v1/alpha_testers"))
        return output.getvalue()

    def test_approve_creates_passwordless_account_and_separate_approval(self):
        for opaque in (False, True):
            with self.subTest(opaque=opaque):
                self.assertIn("Tester approved", self.run_command("approve", opaque=opaque))

    def test_revoke_updates_approval_without_deleting_account(self):
        for opaque in (False, True):
            with self.subTest(opaque=opaque):
                self.assertIn("Access revoked", self.run_command("revoke", opaque=opaque))

    def test_status_reads_approval(self):
        for opaque in (False, True):
            with self.subTest(opaque=opaque):
                self.assertEqual("Approved\n", self.run_command("status", opaque=opaque))


if __name__ == "__main__":
    unittest.main()
