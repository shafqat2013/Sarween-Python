"""Native login controls; all network and Keychain work runs outside Tk callbacks."""

import queue
import threading
import tkinter as tk
from tkinter import ttk


def create_access():
    from auth_config import AuthConfig
    config = AuthConfig.load()
    from auth_storage import KeychainStore
    from auth_provider import SupabaseProvider
    from alpha_access import AlphaAccess
    from app_paths import data_path
    from usage_metrics import UsageMetrics
    return AlphaAccess(config, SupabaseProvider(config), KeychainStore(config.url),
                       metrics=UsageMetrics(data_path("usage-metrics.json")))


class AuthPanel:
    def __init__(self, root, parent, *, account_parent=None, factory=create_access, changed=lambda: None):
        self.root, self.changed = root, changed
        self.access = None
        self.closed = False
        self.pending = 0
        self.notice = ""
        self.results = queue.Queue()
        self.jobs = queue.Queue()
        self._poll_id = None
        self._last_allowed = False
        self.message = tk.StringVar(root, value="Checking login…")
        self.identity = tk.StringVar(root, value="Invitation-only alpha")
        self.frame = ttk.Frame(parent, padding=24)
        self.frame.pack(fill="both", expand=True)
        self.frame.columnconfigure((0, 2), weight=1)
        self.frame.rowconfigure((0, 2), weight=1)
        card = ttk.Frame(self.frame, width=400)
        card.grid(row=1, column=1, sticky="ew")
        ttk.Label(card, text="Sarween", font=("Helvetica", 28, "bold")).pack(anchor="w")
        ttk.Label(card, text="Sign in", font=("Helvetica", 16)).pack(anchor="w", pady=(6, 4))
        ttk.Label(card, text="Invitation-only alpha").pack(anchor="w", pady=(0, 20))
        self.fields = ttk.Frame(card)
        self.fields.pack(fill="x")
        self.email = tk.StringVar(root)
        self.code = tk.StringVar(root)
        ttk.Label(self.fields, text="Email address").grid(row=0, column=0, columnspan=2, sticky="w")
        self.email_entry = ttk.Entry(self.fields, textvariable=self.email, width=32)
        self.email_entry.grid(row=1, column=0, sticky="ew", pady=(5, 16), padx=(0, 8))
        self.email_entry.bind("<Return>", lambda _event: self.send())
        self.send_button = ttk.Button(self.fields, text="Send code", command=self.send)
        self.send_button.grid(row=1, column=1, sticky="ew", pady=(5, 16))
        ttk.Label(self.fields, text="Code from your email").grid(row=2, column=0, columnspan=2, sticky="w")
        self.code_entry = ttk.Entry(self.fields, textvariable=self.code, width=14)
        self.code_entry.grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(5, 0))
        self.code_entry.bind("<Return>", lambda _event: self.verify())
        self.verify_button = ttk.Button(self.fields, text="Sign in", command=self.verify)
        self.verify_button.grid(row=3, column=1, sticky="ew", pady=(5, 0))
        self.fields.columnconfigure(0, weight=1)
        self.login_message = label = ttk.Label(card, textvariable=self.message, wraplength=400)
        label.pack(fill="x", pady=(16, 0))
        label.bind("<Configure>", lambda e: label.configure(wraplength=max(100, e.width)))
        self.recovery = recovery = ttk.Frame(card)
        self.retry_button = ttk.Button(recovery, text="Check access", command=self.retry)
        self.retry_button.pack(side="left")
        self.reset_button = ttk.Button(recovery, text="Log out", command=self.logout)
        self.reset_button.pack(side="right")
        self.disclosure = ttk.Label(card, text="Alpha feedback includes app-use counts and durations. "
                  "No video or gameplay content is sent.", wraplength=400, font=("Helvetica", 11))
        self.disclosure.pack(fill="x", pady=(16, 0))

        # A compact account row in the signed-in home keeps logout and the
        # offline deadline visible without repeating the sign-in form.
        self.account = ttk.Frame(account_parent or parent)
        if account_parent is not None:
            self.account.pack(fill="x", pady=(0, 12))
        top = ttk.Frame(self.account)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.identity).pack(side="left")
        self.logout_button = ttk.Button(top, text="Log out", command=self.logout)
        self.logout_button.pack(side="right")
        self.account_retry = ttk.Button(top, text="Check access", command=self.retry)
        self.account_retry.pack(side="right", padx=6)
        account_status = ttk.Label(self.account, textvariable=self.message, wraplength=700)
        account_status.pack(fill="x", pady=(4, 0))
        account_status.bind("<Configure>", lambda e: account_status.configure(wraplength=max(100, e.width)))

        def worker():
            while True:
                task = self.jobs.get()
                if task is None:
                    if self.access:
                        self.access.flush_usage()
                        self.access.provider.close()
                    return
                try:
                    self.results.put((True, task()))
                except Exception as exc:
                    # Never display arbitrary provider exceptions (which can include secrets).
                    safe = type(exc).__module__ in {"auth_provider", "auth_storage"}
                    self.results.put((False, str(exc) if safe else "Login setup or secure storage is unavailable. Contact the app owner."))

        self.thread = threading.Thread(target=worker, name="Sarween-Login", daemon=True)
        self.thread.start()

        def initialize():
            try:
                access = factory()
            except (ValueError, FileNotFoundError):
                return "Alpha login is not configured. Contact the app owner for a configured build."
            self.access = access
            access.restore()

        self._submit(initialize)
        self._poll()

    @property
    def allowed(self):
        return bool(self.access and self.access.view().allowed)

    def require(self):
        from auth_provider import AccessDenied
        if not self.allowed:
            raise AccessDenied(self.access.view().message if self.access else self.message.get())

    def status(self):
        return self.access.view().message if self.access else self.message.get()

    def usage_state(self, mode, active):
        if self.access:
            self.access.usage_state(mode, active)

    def usage_event(self, name):
        if self.access:
            self.access.usage_event(name)

    def _submit(self, job):
        if self.pending or self.closed:
            return
        self.pending += 1
        self.notice = ""
        epoch = self.access.generation if self.access else None
        def current_job():
            if epoch is None or self.access.generation == epoch:
                return job()
        self.jobs.put(current_job)

    def send(self):
        if self.access:
            email = self.email.get().strip()
            self._submit(lambda: self.access.send_code(email))

    def verify(self):
        if self.access:
            email, code = self.email.get().strip(), self.code.get().strip()
            self.code.set("")
            self._submit(lambda: self.access.verify_code(email, code))

    def retry(self):
        if self.access:
            self._submit(self.access.check)

    def logout(self):
        if not self.access:
            return
        session = self.access.invalidate()
        self.code.set("")
        self.email.set("")
        self.message.set("Signing out…")
        self.changed()
        # Queue even if a request is in flight. Generation invalidation above is immediate.
        self.pending += 1
        self.jobs.put(lambda: self.access.finish_logout(session))

    def _poll(self):
        if self.closed:
            return
        result_message = None
        while not self.results.empty():
            ok, result = self.results.get_nowait()
            self.pending -= 1
            if result:
                result_message = result
            if not ok:
                result_message = result
        if result_message:
            self.notice = result_message
        if self.access:
            self.access.usage_tick()
            view = self.access.view()
            self.identity.set(view.email or "Invitation-only alpha")
            self.message.set(view.message if view.allowed else self.notice or view.message)
            if view.allowed:
                self.fields.pack_forget()
            elif not self.fields.winfo_manager():
                self.fields.pack(fill="x", before=self.login_message)
            if self.access.due() and not self.pending:
                self._submit(self.access.check)
        elif result_message:
            self.message.set(result_message)
        needs_recovery = self.access and not self.allowed and (
            self.access.view().email or "Keychain" in self.message.get())
        if needs_recovery and not self.recovery.winfo_manager():
            self.recovery.pack(fill="x", pady=(8, 0), before=self.disclosure)
        elif not needs_recovery:
            self.recovery.pack_forget()
        for button in (self.send_button, self.verify_button, self.retry_button, self.account_retry):
            button.state(["disabled"] if self.pending or not self.access else ["!disabled"])
        self.logout_button.state(["!disabled"] if self.access else ["disabled"])
        self.reset_button.state(["!disabled"] if self.access else ["disabled"])
        if self._last_allowed != self.allowed:
            self._last_allowed = self.allowed
            self.changed()
        self._poll_id = self.root.after(200, self._poll)

    def close(self):
        if self.access:
            # Preserve the final offline counters before the daemon exits. Network
            # upload remains on the worker and may also resume next launch.
            self.access.persist_usage()
        self.closed = True
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
        self.jobs.put(None)
