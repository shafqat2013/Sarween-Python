"""Approval/session state, independent of Tk and camera processing."""

from dataclasses import dataclass
import math
import threading
import time
import uuid

import jwt
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from auth_provider import AccessDenied, LoginRejected, Unavailable, UnsafeResponse
from auth_storage import SessionStorageError


OFFLINE_SECONDS = 24 * 60 * 60
CHECK_SECONDS = 5 * 60


@dataclass(frozen=True)
class AccessView:
    allowed: bool
    message: str
    email: str = ""
    offline: bool = False
    remaining: int = 0


class AlphaAccess:
    def __init__(self, config, provider, store, *, wall=time.time, monotonic=time.monotonic, metrics=None):
        self.config, self.provider, self.store = config, provider, store
        self.wall, self.monotonic = wall, monotonic
        self.lock = threading.RLock()
        self.metrics = metrics
        self.generation = 0
        self.record = None
        self.claims = None
        self.offline = False
        self.message = "Sign in with your invited email address."
        self.next_check = 0
        self._anchor_wall, self._anchor_mono = wall(), monotonic()
        self._public_key = load_pem_public_key(config.verification_key.encode())
        if not isinstance(self._public_key, Ed25519PublicKey):
            raise ValueError("Offline permissions require an Ed25519 public key.")

    def _now(self):
        return max(self.wall(), self._anchor_wall + self.monotonic() - self._anchor_mono)

    def _verify(self, token, record):
        session = record["session"]
        header = jwt.get_unverified_header(token)
        if header.get("kid") != self.config.key_id or header.get("typ") != "JWT":
            raise ValueError("Unknown signing key")
        claims = jwt.decode(token, self._public_key, algorithms=["EdDSA"],
            audience="sarween-desktop-alpha", issuer=self.config.issuer,
            options={"require": ["exp", "iat", "nbf", "sub", "sid", "device_id"],
                     "verify_exp": False, "verify_iat": False, "verify_nbf": False})
        # Auth JWT is only decoded to bind two server-issued tokens. The permission's
        # signature is verified above; this never authenticates an unverified JWT.
        access_claims = jwt.decode(session["access_token"], options={"verify_signature": False})
        now = self._now()
        for name in ("iat", "nbf", "exp"):
            if type(claims[name]) not in (int, float) or not math.isfinite(claims[name]):
                raise ValueError("Invalid permission time")
        if (claims["sub"] != session["user_id"] or claims["sub"] != access_claims.get("sub")
                or claims["sid"] != access_claims.get("session_id")
                or claims["device_id"] != record["device_id"]
                or not claims["iat"] <= now + 120 or claims["nbf"] > now + 120
                or claims["exp"] <= now or claims["exp"] <= claims["iat"]
                or claims["exp"] - claims["iat"] > OFFLINE_SECONDS
                or claims["nbf"] != claims["iat"]):
            raise ValueError("Permission expired or belongs to another login")
        return claims

    def view(self):
        with self.lock:
            now = self._now()
            allowed = bool(self.record and self.claims and now < self.claims["exp"])
            email = self.record["session"]["email"] if self.record else ""
            remaining = max(0, int(self.claims["exp"] - now)) if self.claims else 0
            message = self.message
            if self.claims and not allowed:
                message = "Offline permission expired. Reconnect to verify your access."
            elif allowed and self.offline:
                hours, minutes = divmod(remaining // 60, 60)
                message = f"Offline access: {hours}h {minutes}m remaining. Reconnect before it expires."
            return AccessView(allowed, message, email, self.offline, remaining)

    def require(self):
        if not self.view().allowed:
            raise AccessDenied(self.view().message)

    def due(self):
        with self.lock:
            return bool(self.record) and self.monotonic() >= self.next_check

    def restore(self):
        try:
            record = self.store.load()
            if record is not None:
                session = record["session"]
                for name in ("access_token", "refresh_token", "user_id", "email"):
                    if not isinstance(session[name], str) or not session[name]:
                        raise ValueError("Invalid saved session")
                uuid.UUID(record["device_id"])
                if not math.isfinite(session["expires_at"]) or not math.isfinite(record["last_seen"]):
                    raise ValueError("Invalid session timestamp")
        except Exception:
            with self.lock:
                self.message = "Saved login could not be read. Unlock Keychain, then log out and sign in again."
            return
        if record is None or not record.get("session"):
            return
        with self.lock:
            self.record = record
            self.message = "Checking your saved login…"
        self.check()

    def send_code(self, email):
        if not email or len(email) > 254 or "@" not in email or any(c.isspace() for c in email):
            raise LoginRejected("Enter your invited email address.")
        # Generic response avoids exposing whether an address was invited.
        try:
            self.provider.send_code(email)
        except LoginRejected:
            pass
        return "If this address is invited, a code is on its way. Check your inbox."

    def verify_code(self, email, code):
        if not code.isdigit() or not 6 <= len(code) <= 10:
            raise LoginRejected("Enter the code from your email.")
        with self.lock:
            epoch = self.generation
        session = self.provider.verify_code(email, code)
        with self.lock:
            if epoch != self.generation:
                return
            self.claims = None
            self.record = {"version": 1, "session": session, "device_id": str(uuid.uuid4()),
                           "permission": None, "last_seen": self._now()}
            self.store.save(self.record)
        self.check()

    def check(self):
        with self.lock:
            if not self.record:
                return
            epoch = self.generation
            record = {**self.record, "session": dict(self.record["session"])}
            self.next_check = self.monotonic() + CHECK_SECONDS
        try:
            if record["session"]["expires_at"] <= self._now() + 60:
                record["session"] = self.provider.refresh(record["session"]["refresh_token"])
                # Persist rotated refresh tokens before the separate permission request.
                with self.lock:
                    if epoch != self.generation:
                        return
                    self.store.save(record)
                    self.record = record
            try:
                permission = self.provider.permission(record["session"]["access_token"], record["device_id"])
            except LoginRejected:
                record["session"] = self.provider.refresh(record["session"]["refresh_token"])
                with self.lock:
                    if epoch != self.generation:
                        return
                    self.store.save(record)
                    self.record = record
                permission = self.provider.permission(record["session"]["access_token"], record["device_id"])
            claims = self._verify(permission, record)
            record.update(permission=permission, last_seen=self._now())
            with self.lock:
                if epoch != self.generation:
                    return
                self.store.save(record)
                self.record, self.claims = record, claims
                self.offline = False
                self.message = "Alpha access approved"
        except Unavailable:
            with self.lock:
                if epoch != self.generation:
                    return
                self.offline = True
                self.next_check = self.monotonic() + 30
                try:
                    if self.wall() < record.get("last_seen", self.wall()) - 120:
                        raise ValueError("Clock moved backwards")
                    claims = self._verify(record["permission"], record)
                    record["last_seen"] = self._now()
                    self.store.save(record)
                    self.record, self.claims = record, claims
                except Exception:
                    self.claims = None
                    self.message = "Connect to the internet to verify access. Check your Mac's date and time."
        except (AccessDenied, LoginRejected) as exc:
            with self.lock:
                if epoch != self.generation:
                    return
                self.claims = None
                self.record = None
                self.message = str(exc)
                try:
                    self.store.clear()
                except Exception:
                    self.message += " Unlock Keychain and use Log out to remove the saved login."
        except Exception:
            # Bad TLS, forged replies, malformed data and storage failures are not outages.
            with self.lock:
                if epoch != self.generation:
                    return
                self.claims = None
                if self.record:
                    self.record = {**self.record, "permission": None}
                    try:
                        self.store.save(self.record)
                    except Exception:
                        pass
                self.message = "Access could not be securely verified. Check Keychain, date/time and connection, then retry."
        self.flush_usage()

    def usage_tick(self):
        if self.metrics:
            try:
                with self.lock:
                    allowed = self.view().allowed
                    user_id = self.record["session"]["user_id"] if self.record else None
                self.metrics.bind(user_id)
                self.metrics.tick(allowed)
            except Exception:
                pass

    def usage_state(self, mode, active):
        if self.metrics:
            try:
                self.metrics.state(mode, bool(active and self.view().allowed))
            except Exception:
                pass

    def usage_event(self, name):
        if self.metrics and self.view().allowed:
            try:
                self.metrics.event(name)
            except Exception:
                pass

    def flush_usage(self):
        if not self.metrics:
            return
        try:
            self.usage_tick()
            self.metrics.persist()
            with self.lock:
                if not self.record or not self.view().allowed or self.offline:
                    return
                session = dict(self.record["session"])
                epoch = self.generation
            reports = self.metrics.snapshot()
            if reports and self.provider.report_usage(session["access_token"], reports):
                with self.lock:
                    if epoch == self.generation:
                        self.metrics.acknowledge(reports)
                        self.metrics.persist()
        except Exception:
            # Metrics, storage and network failures never lock or interrupt play.
            pass

    def persist_usage(self):
        if self.metrics:
            try:
                self.usage_tick()
                self.metrics.persist()
            except Exception:
                pass

    def invalidate(self):
        """Immediate UI-thread logout; makes in-flight replies harmless."""
        with self.lock:
            self.generation += 1
            session = self.record["session"] if self.record else None
            self.record = self.claims = None
            self.offline = False
            self.message = "Signed out"
            if self.metrics:
                self.metrics.bind(None)
            return session

    def finish_logout(self, session):
        if self.metrics:
            try:
                self.metrics.persist()
            except Exception:
                pass
        try:
            self.store.clear()
        except Exception:
            raise SessionStorageError("Access is locked, but Keychain could not be cleared. Unlock it and log out again.") from None
        if session:
            try:
                self.provider.logout(session)
            except Exception:
                return "Signed out on this Mac. The server could not be reached to end the remote session."
        return "Signed out"

    def logout(self):
        return self.finish_logout(self.invalidate())
