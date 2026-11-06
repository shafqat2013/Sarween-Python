"""Explicit login bypass for local source runs; excluded from app bundles."""

import sys
from tkinter import ttk


def _packaged():
    return bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))


class DeveloperAuthPanel:
    """The normal access-gate interface without a remote login or saved session."""

    def __init__(self, root, parent, *, account_parent=None, changed=lambda: None):
        if _packaged():
            raise ValueError("Developer login bypass is only available when running from source.")
        self._active = True
        self.account = ttk.Frame(account_parent if account_parent is not None else parent,
                                 padding=10)
        self.account.pack(fill="x", pady=(0, 12))
        self.banner = ttk.Label(self.account, text=self.status(),
                                font=("Helvetica", 12, "bold"), foreground="#a15c00")
        self.banner.pack(anchor="w")
        ttk.Label(self.account,
                  text="Restart without --dev-no-login to use normal sign-in.").pack(anchor="w")

    @property
    def allowed(self):
        return self._active and not _packaged()

    def status(self):
        if self.allowed:
            return "Developer mode — login bypassed"
        return "Developer testing access is unavailable."

    def require(self):
        if not self.allowed:
            raise PermissionError(self.status())

    def usage_state(self, mode, active):
        pass

    def usage_event(self, name):
        pass

    def close(self):
        self._active = False
