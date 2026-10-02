import copy
import json
import ssl
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from alpha_access import AlphaAccess, OFFLINE_SECONDS, CHECK_SECONDS
from auth_config import AuthConfig
from auth_provider import SupabaseProvider, AccessDenied, LoginRejected, Unavailable, UnsafeResponse, translate_error


USER = "11111111-1111-4111-8111-111111111111"
SESSION = "22222222-2222-4222-8222-222222222222"


class MemoryStore:
    def __init__(self):
        self.value = None

    def load(self):
        return copy.deepcopy(self.value)

    def save(self, value):
        self.value = copy.deepcopy(value)

    def clear(self):
        self.value = None


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.now, self.mono = 1800000000, 500
        self.private = Ed25519PrivateKey.generate()
        public = self.private.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
        self.config = AuthConfig("https://example.supabase.co", "sb_publishable_example", public, "alpha-v1")
        self.store = MemoryStore()
        self.provider = Mock()
        self.session = {"user_id": USER, "email": "tester@example.com", "refresh_token": "refresh-1",
            "expires_at": self.now + 3600, "access_token": jwt.encode(
                {"sub": USER, "session_id": SESSION}, "test-only-long-secret-not-a-real-key", algorithm="HS256")}
        self.provider.verify_code.return_value = dict(self.session)
        self.provider.permission.side_effect = lambda _token, device: self.permission(device)
        self.access = self.manager()

    def manager(self):
        return AlphaAccess(self.config, self.provider, self.store,
                           wall=lambda: self.now, monotonic=lambda: self.mono)

    def permission(self, device, **overrides):
        claims = {"sub": USER, "sid": SESSION, "device_id": device, "iat": self.now,
                  "nbf": self.now, "exp": self.now + OFFLINE_SECONDS,
                  "iss": self.config.issuer, "aud": "sarween-desktop-alpha", **overrides}
        return jwt.encode(claims, self.private, algorithm="EdDSA", headers={"kid": "alpha-v1"})

    def login(self):
        self.access.verify_code("tester@example.com", "12345678")

    def advance(self, seconds):
        self.now += seconds
        self.mono += seconds

    def test_approved_login_is_remembered_and_rechecked_at_startup(self):
        self.login()
        self.assertTrue(self.access.view().allowed)
        restored = self.manager()
        restored.restore()
        self.assertTrue(restored.view().allowed)
        self.assertEqual(self.provider.permission.call_count, 2)
        self.assertEqual(self.store.value["session"]["refresh_token"], "refresh-1")

    def test_first_login_requires_online_approval(self):
        self.provider.permission.side_effect = Unavailable()
        self.login()
        self.assertFalse(self.access.view().allowed)

    def test_usage_failures_do_not_interrupt_access_or_logout(self):
        self.access.metrics = Mock()
        self.access.metrics.snapshot.return_value = [{"app_seconds": 1}]
        self.provider.report_usage.side_effect = RuntimeError("metrics service unavailable")
        self.login()
        self.assertTrue(self.access.view().allowed)
        self.provider.report_usage.assert_called_once()
        self.access.metrics.acknowledge.assert_not_called()
        self.access.metrics.persist.side_effect = OSError("disk full")
        self.access.check()
        self.assertTrue(self.access.view().allowed)
        self.access.logout()
        self.assertFalse(self.access.view().allowed)
        self.assertIsNone(self.store.value)
        self.access.metrics.bind.assert_called_with(None)

    def test_usage_upload_stops_offline_and_after_revocation(self):
        self.access.metrics = Mock()
        self.access.metrics.snapshot.return_value = [{"app_seconds": 1}]
        self.login()
        self.provider.report_usage.reset_mock()
        self.provider.permission.side_effect = Unavailable()
        self.access.check()
        self.assertTrue(self.access.view().allowed)
        self.provider.report_usage.assert_not_called()
        self.provider.permission.side_effect = AccessDenied("Revoked")
        self.access.check()
        self.assertFalse(self.access.view().allowed)
        self.provider.report_usage.assert_not_called()
        self.access.metrics.bind.assert_called_with(None)

    def test_authenticated_but_unapproved_user_is_denied(self):
        self.provider.permission.side_effect = AccessDenied("Not approved")
        self.login()
        self.assertFalse(self.access.view().allowed)
        self.assertIsNone(self.store.value)

    def test_revocation_wipes_offline_permission_and_cannot_restore_during_outage(self):
        self.login()
        self.provider.permission.side_effect = AccessDenied("Revoked")
        self.access.check()
        self.assertFalse(self.access.view().allowed)
        self.assertIsNone(self.store.value)
        self.provider.permission.side_effect = Unavailable()
        restored = self.manager()
        restored.restore()
        self.assertFalse(restored.view().allowed)

    def test_offline_restore_survives_expired_auth_token_but_not_permission_deadline(self):
        self.login()
        self.advance(4000)
        self.provider.refresh.side_effect = Unavailable()
        restored = self.manager()
        restored.restore()
        self.assertTrue(restored.view().allowed)
        self.assertTrue(restored.view().offline)
        self.advance(OFFLINE_SECONDS - 4000)
        self.assertFalse(restored.view().allowed)
        with self.assertRaises(AccessDenied):
            restored.require()

    def test_outage_does_not_extend_offline_deadline(self):
        self.login()
        deadline = self.access.claims["exp"]
        self.provider.permission.side_effect = Unavailable()
        self.advance(300)
        self.access.check()
        self.assertEqual(self.access.claims["exp"], deadline)
        self.assertTrue(self.access.view().allowed)
        self.assertEqual(self.access.view().remaining, OFFLINE_SECONDS - 300)

    def test_online_checks_due_every_five_minutes_and_recovery_renews_permission(self):
        self.login()
        self.assertFalse(self.access.due())
        self.advance(CHECK_SECONDS)
        self.assertTrue(self.access.due())
        self.access.check()
        self.assertEqual(self.access.view().remaining, OFFLINE_SECONDS)

    def test_expired_session_refreshes_and_persists_rotation_before_access_outage(self):
        self.login()
        self.advance(3601)
        self.provider.refresh.return_value = {**self.session, "refresh_token": "refresh-2", "expires_at": self.now + 3600}
        self.provider.permission.side_effect = Unavailable()
        self.access.check()
        self.assertTrue(self.access.view().allowed)
        self.assertEqual(self.store.value["session"]["refresh_token"], "refresh-2")

    def test_invalid_refresh_is_not_treated_as_an_outage(self):
        self.login()
        self.advance(3601)
        self.provider.refresh.side_effect = LoginRejected("Expired login")
        self.access.check()
        self.assertFalse(self.access.view().allowed)
        self.assertIsNone(self.store.value)

    def test_logout_is_local_even_without_internet(self):
        self.login()
        self.provider.logout.side_effect = Unavailable()
        message = self.access.logout()
        self.assertIn("Signed out on this Mac", message)
        self.assertFalse(self.access.view().allowed)
        self.assertIsNone(self.store.value)
        restored = self.manager()
        restored.restore()
        self.assertFalse(restored.view().allowed)

    def test_logout_discards_inflight_success(self):
        self.login()
        started, released = threading.Event(), threading.Event()
        def delayed(_token, device):
            started.set()
            self.assertTrue(released.wait(2))
            return self.permission(device)
        self.provider.permission.side_effect = delayed
        thread = threading.Thread(target=self.access.check)
        thread.start()
        self.assertTrue(started.wait(2))
        self.access.logout()
        released.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.access.view().allowed)
        self.assertIsNone(self.store.value)

    def test_signed_permissions_reject_wrong_identity_device_audience_key_and_duration(self):
        self.login()
        for patch in ({"sub": SESSION}, {"sid": USER}, {"device_id": USER},
                      {"aud": "another-app"}, {"iss": "https://other.example"},
                      {"exp": self.now + OFFLINE_SECONDS + 1}, {"exp": self.now},
                      {"iat": self.now + 3600}, {"nbf": self.now + 3600}):
            with self.subTest(patch=patch):
                token = self.permission(self.access.record["device_id"], **patch)
                with self.assertRaises((ValueError, jwt.InvalidTokenError)):
                    self.access._verify(token, self.access.record)
        tampered = jwt.encode({"sub": USER}, "not-the-real-key", algorithm="HS256", headers={"kid": "alpha-v1"})
        with self.assertRaises(jwt.InvalidTokenError):
            self.access._verify(tampered, self.access.record)

    def test_rollback_of_system_clock_does_not_extend_active_permission(self):
        self.login()
        self.now -= 3600
        self.mono += OFFLINE_SECONDS
        self.assertFalse(self.access.view().allowed)

    def test_restart_with_backward_clock_requires_online_validation(self):
        self.login()
        self.now -= 3600
        self.provider.permission.side_effect = Unavailable()
        restored = self.manager()
        restored.restore()
        self.assertFalse(restored.view().allowed)

    def test_storage_failure_never_grants_access(self):
        self.store.save = Mock(side_effect=RuntimeError("Keychain denied"))
        with self.assertRaises(RuntimeError):
            self.login()
        self.assertFalse(self.access.view().allowed)

    def test_bad_tls_or_invalid_response_never_uses_cached_permission(self):
        self.login()
        self.provider.permission.side_effect = UnsafeResponse("Bad TLS")
        self.access.check()
        self.assertFalse(self.access.view().allowed)


class ProviderTests(unittest.TestCase):
    def config(self):
        return AuthConfig("https://example.supabase.co", "sb_publishable_example", "public", "alpha-v1")

    def test_otp_request_uses_official_client_and_disables_account_creation(self):
        captured = []
        def respond(request):
            captured.append(request)
            return httpx.Response(200, json={})
        provider = SupabaseProvider(self.config(), httpx.MockTransport(respond))
        try:
            provider.send_code("tester@example.com")
            request = captured[0]
            self.assertEqual(str(request.url), self.config().url + "/auth/v1/otp")
            body = json.loads(request.content)
            self.assertFalse(body["create_user"])
            self.assertEqual(body["email"], "tester@example.com")
        finally:
            provider.close()

    def test_permission_statuses_distinguish_denial_outage_and_malformed_reply(self):
        for status, exception in ((401, LoginRejected), (403, AccessDenied), (429, Unavailable),
                                  (503, Unavailable), (302, UnsafeResponse), (200, UnsafeResponse)):
            with self.subTest(status=status):
                provider = SupabaseProvider(self.config(), httpx.MockTransport(lambda _r: httpx.Response(status, json={})))
                try:
                    with self.assertRaises(exception):
                        provider.permission("user-token", USER)
                finally:
                    provider.close()

    def test_official_sdk_verifies_code_refreshes_rotates_and_logs_out_this_session(self):
        import time
        now = int(time.time())
        token = jwt.encode({"sub": USER, "session_id": SESSION, "exp": now + 3600},
                           "test-only-long-secret-not-a-real-key", algorithm="HS256")
        captured = []
        def respond(request):
            captured.append(request)
            if request.url.path.endswith("/logout"):
                return httpx.Response(204)
            return httpx.Response(200, json={"access_token": token, "token_type": "bearer",
                "expires_in": 3600, "expires_at": now + 3600,
                "refresh_token": "rotated" if "token" in request.url.path else "original",
                "user": {"id": USER, "email": "tester@example.com", "aud": "authenticated",
                         "app_metadata": {}, "user_metadata": {}, "created_at": "2026-10-01T00:00:00Z"}})
        provider = SupabaseProvider(self.config(), httpx.MockTransport(respond))
        try:
            session = provider.verify_code("tester@example.com", "12345678")
            self.assertEqual(session["user_id"], USER)
            self.assertEqual(json.loads(captured[0].content)["type"], "email")
            self.assertEqual(json.loads(captured[0].content)["token"], "12345678")
            refreshed = provider.refresh(session["refresh_token"])
            self.assertEqual(refreshed["refresh_token"], "rotated")
            self.assertEqual(json.loads(captured[1].content)["refresh_token"], "original")
            provider.logout(refreshed)
            self.assertEqual(captured[-1].url.params["scope"], "local")
            self.assertEqual(captured[-1].headers["Authorization"], "Bearer " + token)
            self.assertIsNone(provider.auth.get_session())
        finally:
            provider.close()

    def test_tls_failure_is_not_an_offline_condition(self):
        exception = httpx.ConnectError("cannot connect")
        exception.__cause__ = ssl.SSLCertVerificationError("certificate failed")
        self.assertIsInstance(translate_error(exception), UnsafeResponse)
        self.assertIsInstance(translate_error(httpx.ConnectError("DNS unavailable")), Unavailable)

    def test_bundle_configuration_rejects_secret_keys_and_private_signing_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            base = {"url": "https://example.supabase.co", "publishable_key": "sb_publishable_example",
                    "verification_key": "-----BEGIN PUBLIC KEY-----\nexample", "key_id": "alpha-v1"}
            for update in ({"publishable_key": "sb_secret_example"},
                           {"publishable_key": jwt.encode({"role": "service_role"}, "test")},
                           {"url": "http://example.supabase.co"},
                           {"verification_key": "-----BEGIN PRIVATE KEY-----"}, {"secret": "never bundle"}):
                path.write_text(json.dumps({**base, **update}))
                with self.assertRaises(ValueError):
                    AuthConfig.load(path)
