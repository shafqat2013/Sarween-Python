"""Small adapter around the official Supabase Auth client."""

import ssl

import httpx
from supabase_auth import SyncGoTrueClient
from supabase_auth.errors import AuthApiError, AuthRetryableError


class Unavailable(Exception):
    """Network/service outage; an existing signed offline permission may be used."""


class LoginRejected(Exception):
    pass


class AccessDenied(Exception):
    pass


class UnsafeResponse(Exception):
    pass


def translate_error(exc):
    current = exc
    while current is not None:
        if isinstance(current, ssl.SSLError):
            return UnsafeResponse("Cannot verify the secure connection. Check your network and retry.")
        current = current.__cause__ or current.__context__
    if isinstance(exc, AuthApiError):
        if exc.status == 429 or exc.status >= 500:
            return Unavailable("Login service is temporarily unavailable. Please retry shortly.")
        return LoginRejected("Login expired or the code was not accepted. Request a new code.")
    if isinstance(exc, (AuthRetryableError, httpx.TimeoutException, httpx.NetworkError)):
        return Unavailable("Cannot reach the login service. Check your internet connection.")
    return UnsafeResponse("Login could not be verified. Please retry.")


class SupabaseProvider:
    def __init__(self, config, transport=None):
        self.config = config
        self.http = httpx.Client(timeout=8, follow_redirects=False, trust_env=False, transport=transport)
        self.auth = self._client()

    def _client(self):
        return SyncGoTrueClient(url=self.config.url + "/auth/v1",
                    headers={"apikey": self.config.publishable_key}, http_client=self.http,
                    persist_session=False, auto_refresh_token=False)

    def _call(self, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            raise translate_error(exc) from None

    def send_code(self, email):
        self._call(self.auth.sign_in_with_otp,
                   {"email": email, "options": {"should_create_user": False}})

    @staticmethod
    def _session(response):
        session = response.session
        if session is None or session.user is None:
            raise LoginRejected("Login could not be completed. Request another code.")
        return {"access_token": session.access_token, "refresh_token": session.refresh_token,
                "expires_at": session.expires_at, "user_id": str(session.user.id),
                "email": session.user.email or ""}

    def verify_code(self, email, code):
        return self._session(self._call(self.auth.verify_otp,
                                      {"email": email, "token": code, "type": "email"}))

    def refresh(self, refresh_token):
        return self._session(self._call(self.auth.refresh_session, refresh_token))

    def permission(self, access_token, device_id):
        try:
            response = self.http.post(self.config.issuer,
                headers={"apikey": self.config.publishable_key, "Authorization": "Bearer " + access_token},
                json={"device_id": device_id})
        except Exception as exc:
            raise translate_error(exc) from None
        if response.status_code == 401:
            raise LoginRejected("Your login has expired. Sign in again.")
        if response.status_code == 403:
            raise AccessDenied("Alpha access has not been approved or has been revoked. Contact the app owner.")
        if response.status_code == 429 or response.status_code >= 500:
            raise Unavailable("Access service is temporarily unavailable.")
        if response.status_code != 200 or len(response.content) > 16384:
            raise UnsafeResponse("Access service returned an invalid response.")
        try:
            token = response.json()["permission"]
            if not isinstance(token, str) or len(token) > 8192:
                raise ValueError()
            return token
        except (ValueError, KeyError, TypeError):
            raise UnsafeResponse("Access service returned an invalid permission.") from None

    def logout(self, session):
        # Use the captured token directly; never refresh a session during logout.
        try:
            self._call(self.auth.admin.sign_out, session["access_token"], "local")
        finally:
            self.auth = self._client()

    def report_usage(self, access_token, reports):
        # Best-effort analytics has a short timeout and never refreshes sessions.
        response = self.http.post(self.config.url + "/rest/v1/rpc/record_alpha_usage",
            headers={"apikey": self.config.publishable_key, "Authorization": "Bearer " + access_token},
            json={"reports": reports}, timeout=2)
        return response.status_code == 200 and response.json() is True

    def close(self):
        self.http.close()
