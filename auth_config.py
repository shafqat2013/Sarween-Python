"""Public, build-time authentication configuration. Never accepts server secrets."""

from dataclasses import dataclass
import base64
import json
from pathlib import Path
import re
import sys


@dataclass(frozen=True)
class AuthConfig:
    url: str
    publishable_key: str
    verification_key: str
    key_id: str

    @classmethod
    def load(cls, path=None):
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        value = json.loads(Path(path or root / "auth_config.json").read_text())
        expected = {"url", "publishable_key", "verification_key", "key_id"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("Alpha login is not configured. Contact the app owner for a configured build.")
        if not all(isinstance(v, str) and v for v in value.values()):
            raise ValueError("Incomplete public authentication configuration.")
        if not re.fullmatch(r"https://[a-z0-9-]+\.supabase\.co", value["url"]):
            raise ValueError("Authentication must use the project's HTTPS Supabase URL.")
        key = value["publishable_key"]
        if not re.fullmatch(r"sb_publishable_[A-Za-z0-9_-]+", key):
            # Older projects have a public JWT-shaped anon key. Never accept service_role.
            try:
                payload = key.split(".")[1]
                claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
                if claims.get("role") != "anon":
                    raise ValueError()
            except Exception:
                raise ValueError("Only a public publishable/anon key may be bundled.") from None
        if (not value["verification_key"].startswith("-----BEGIN PUBLIC KEY-----")
                or "PRIVATE" in value["verification_key"]):
            raise ValueError("Only the offline verification PUBLIC key may be bundled.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value["key_id"]):
            raise ValueError("Invalid public signing-key identifier.")
        return cls(**value)

    @property
    def issuer(self):
        return self.url + "/functions/v1/alpha-access"
