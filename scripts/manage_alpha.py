"""Owner-only tester management. This script and its credentials never ship in the app."""

import argparse
import getpass
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import httpx
    from supabase_auth import SyncGoTrueClient
    from auth_config import AuthConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("approve", "revoke", "status"))
    parser.add_argument("email", help="Exact tester email selected by the owner")
    args = parser.parse_args()
    email = args.email.strip().lower()
    if "@" not in email or len(email) > 254 or any(c.isspace() for c in email):
        parser.error("Enter the tester's email address.")
    config = AuthConfig.load()
    # Prompt without echo; privileged credentials never enter shell history or files.
    secret = getpass.getpass("Supabase secret/service-role key (hidden): ")
    valid = secret.startswith("sb_secret_")
    if not valid:
        import base64
        try:
            body = secret.split(".")[1]
            valid = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))).get("role") == "service_role"
        except Exception:
            pass
    if not valid:
        parser.error("An owner-only secret/service-role key is required.")
    headers = {"apikey": secret}
    # Legacy service-role JWTs also need a bearer header for Auth admin and
    # PostgREST. Opaque secret keys are authenticated by the API gateway.
    if not secret.startswith("sb_secret_"):
        headers["Authorization"] = "Bearer " + secret
    with httpx.Client(timeout=15, trust_env=False, follow_redirects=False) as http:
        client = SyncGoTrueClient(url=config.url + "/auth/v1", headers=headers,
            persist_session=False, auto_refresh_token=False, http_client=http)
        user = None
        for page in range(1, 101):
            users = client.admin.list_users(page=page, per_page=100)
            user = next((u for u in users if (u.email or "").lower() == email), None)
            if user or len(users) < 100:
                break
        else:
            raise RuntimeError("User directory exceeds this alpha tool's limit. Use the dashboard.")
        if user is None and args.action == "approve":
            # No password, public signup, or automatic recruitment import. The tester
            # must still prove email ownership with an OTP when using the app.
            user = client.admin.create_user({"email": email, "email_confirm": True}).user
        if user is None:
            print("No account exists for that email; app access is not approved.")
            return
        headers = {**headers, "Content-Type": "application/json"}
        endpoint = config.url + "/rest/v1/alpha_testers"
        if args.action == "status":
            response = http.get(endpoint, headers=headers, params={"user_id": "eq." + str(user.id), "select": "approved"})
            response.raise_for_status()
            records = response.json()
            print("Approved" if records and records[0]["approved"] else "Not approved")
        else:
            from datetime import datetime, timezone
            response = http.post(endpoint, headers={**headers, "Prefer": "resolution=merge-duplicates"},
                json={"user_id": str(user.id), "approved": args.action == "approve",
                      "updated_at": datetime.now(timezone.utc).isoformat()})
            response.raise_for_status()
            print("Tester approved. They can request a login code in Sarween." if args.action == "approve" else
                  "Access revoked. Online apps detect this within five minutes; offline permission lasts at most 24 hours.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        # Provider exception bodies may contain credentials or identity data.
        print("Operation failed. Check the project/key and dashboard before retrying; no secret details were logged.", file=sys.stderr)
        raise SystemExit(1)
