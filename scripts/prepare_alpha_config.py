"""Create public app configuration and a separate, owner-only server secret file.

Never upload the secret file as an app asset. Import it with the Supabase CLI's
secrets command. Neither secret values nor private key material are printed.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from auth_config import AuthConfig
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-url", required=True)
    parser.add_argument("--publishable-key", required=True, help="Public key only, never a service-role/secret key")
    parser.add_argument("--secret-file", type=Path, required=True, help="New file OUTSIDE the checkout")
    args = parser.parse_args()
    secret_path = args.secret_file.expanduser().resolve()
    if secret_path.is_relative_to(ROOT) or secret_path.exists():
        parser.error("Use a new secret-file path outside the checkout.")
    config_path = ROOT / "auth_config.json"
    if json.loads(config_path.read_text()) != {}:
        parser.error("App configuration already exists. Key rotation requires a separately reviewed build.")
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
    value = {"url": args.project_url, "publishable_key": args.publishable_key,
             "verification_key": public, "key_id": "alpha-" + uuid.uuid4().hex[:12]}
    # Validate via a temporary PUBLIC file before writing any secret or config.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        check = Path(folder) / "public.json"
        check.write_text(json.dumps(value))
        AuthConfig.load(check)
    pem = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    env = "ALPHA_PERMISSION_PRIVATE_KEY_B64=" + base64.b64encode(pem).decode() + "\n"
    env += "ALPHA_PERMISSION_KEY_ID=" + value["key_id"] + "\n"
    fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(env)
    config_path.write_text(json.dumps(value, indent=2) + "\n")
    print("Public app configuration prepared. Server secret written to the requested private file.")
    print("Keep the secret in a password manager after importing it into Supabase; it is not an app asset.")


if __name__ == "__main__":
    main()
