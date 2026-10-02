"""Session persistence in macOS Keychain only; no plaintext fallback."""

import hashlib
import json
import sys


class SessionStorageError(Exception):
    pass


class KeychainStore:
    def __init__(self, project_url):
        if sys.platform != "darwin":
            raise RuntimeError("Alpha login currently supports macOS Keychain only.")
        # Explicit backend: ignore environment-selected or third-party plaintext backends.
        from keyring.backends.macOS import Keyring
        self.backend = Keyring()
        self.service = "com.sarween.app.auth." + hashlib.sha256(project_url.encode()).hexdigest()[:20]
        self.account = "session-v1"

    def load(self):
        raw = self.backend.get_password(self.service, self.account)
        if raw is None:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("Saved login is unreadable. Log out and sign in again.")
        return value

    def save(self, value):
        self.backend.set_password(self.service, self.account, json.dumps(value, allow_nan=False))

    def clear(self):
        from keyring.errors import PasswordDeleteError
        try:
            self.backend.delete_password(self.service, self.account)
        except PasswordDeleteError:
            # Absence is fine; denial/locked Keychain must remain a visible error.
            if self.backend.get_password(self.service, self.account) is not None:
                raise
