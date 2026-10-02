# Sarween invitation-only alpha access

## Current status

The desktop integration, Supabase migration/function, owner tools and local tests
are implemented. The dedicated Free project `sarween-alpha`
(`ugwcxftwgkkpsbzdacjz`) has the approval database and access endpoint deployed.
Public signup is disabled; anonymous sign-in is off; email OTP length is 8 and
expiry is 600 seconds. Live backend checks with disposable synthetic accounts
pass. These checks generate OTPs administratively without sending email.

Resend has verified `auth.laserfit.ai`. Its three public DNS records were added
with owner approval; the original website CNAME remains unchanged. The scoped
SMTP key was entered by the owner and custom SMTP is saved. The hosted OTP email
template now shows the login code; its source is retained in
`supabase/templates/magic_link.html`. On October 2, the first designated owner
account was approved and a code request through the app's provider was accepted.
Resend reported the email delivered. The owner confirmed inbox receipt of the
eight-digit code and successful login in the packaged review app. The owner also
confirmed remembered login after restarting, offline restart within the valid
permission window, offline logout with access locked after restarting, and fresh
sign-in after reconnecting. The implementation is ready for final change review;
physical camera/TV and Foundry play validation remains a separate hardware check.
No new app has been published or distributed.

The October 2 engagement extension adds owner-only Supabase reports and a separate
local metrics review build. Sign-in and online presence collection is deployed;
detailed app/tracking/Foundry counters require the new build. See
[engagement metrics and signup notifications](ENGAGEMENT_METRICS.md) for report
links, definitions, the immediate Airtable notification and verification status.

`auth_config.json` now contains the real public project configuration and Ed25519
verification key. The server signing secret is provisioned in Supabase and backed
up in the owner-only file `/Users/shafqat/.supabase/sarween-alpha-signing.env`,
outside the checkout. Keep a secure recovery copy; never distribute that file.
The local CLI login is named `sarween-alpha-setup` in Supabase Access Tokens.

The website and its Cloudflare Pages → Airtable recruitment form are independent.
Nothing imports prospects or creates accounts from interest submissions. No payment
or subscription checks are implemented.

## Tester experience and agreed offline policy

1. The owner explicitly approves an email with the owner tool below.
2. The tester enters that email in Sarween, requests a code, and enters the code.
   The official Supabase Auth client handles verification and session refresh.
3. A separate server check confirms the current account/session is approved. A
   successful login alone does not grant app access.
4. Tokens and the signed offline permission are saved only in macOS Keychain.
   There is no plaintext fallback. A locked/unavailable Keychain prevents access.
5. Startup checks approval online when possible. While open, the app checks every
   five minutes. A successful check permits up to **24 hours** of offline use from
   that check, including restarting the app. First login always requires internet.
6. An internet/service outage preserves a valid permission, even if the short-lived
   login token expires. Outages never extend its deadline. Offline status and time
   remaining are visible in the home window and live control panel. Recovery is
   retried every 30 seconds while offline.
7. Confirmed denial/revocation or an invalid refresh session immediately removes
   local permission; it cannot fall back to offline access. Invalid signatures,
   TLS failures, storage errors and malformed responses do not grant access.
8. At offline expiry or confirmed denial, live tracking exits through its normal
   recording/camera cleanup, the Foundry relay stops, and replay closes. Saved
   profiles, recordings and calibration are retained. The account controls remain
   available to reconnect and sign in.
9. Log out immediately locks the app and clears Keychain, even offline. Online it
   also asks Supabase to end this login session. If the server is unreachable, the
   UI explicitly says remote session termination could not be confirmed. If
   Keychain cannot be cleared, logout reports the problem and must be retried after
   unlocking Keychain; do not assume persistent logout succeeded in that case.

An offline app cannot receive revocation immediately: the maximum delay is the
remaining 24-hour permission. The server signature prevents editing an approval
flag to extend access. Monotonic time protects the running app against clock
rollback; saved last-seen time rejects obvious rollback across restarts. Client
gating is not tamper-proof DRM on a tester-controlled machine. Older builds that
predate this change do not gain authentication retroactively.

## Account setup (owner)

Use Supabase Free and Resend Free as agreed. The steps below also document how to
reproduce this setup; do not recreate the existing project or signing key.
Supabase Free can pause after low
activity; a paused service cannot issue/renew permission. Supabase Pro is an
optional later reliability upgrade, not enabled by this work.

1. Create/sign into both service accounts and create a **Free** Supabase project
   named `sarween-alpha`. Keep its database password in your password manager.
2. Verify a dedicated sending subdomain, for example `auth.laserfit.ai`, in Resend.
   Add exactly the DNS records Resend provides in the existing Cloudflare account;
   preserve the website and existing recruitment configuration. Use a sending-only
   API key. Configure Supabase Custom SMTP with Resend's documented server settings
   and sender `Sarween <login@auth.laserfit.ai>`. Put the SMTP credential in Supabase,
   never the desktop configuration or repository.
3. In Supabase Authentication settings, disable **Allow new users to sign up** and
   anonymous sign-in. Keep only the email provider needed for this alpha. Set email
   OTP length to 8, expiry to 600 seconds, and retain service rate limits. Change
   the Magic Link template to show `{{ .Token }}` as the login code; no callback
   webpage is needed. `supabase/config.toml` records corresponding local settings;
   deploying a function does **not** apply hosted Auth/SMTP dashboard settings.
4. Apply `supabase/migrations/202610010001_alpha_access.sql` to this new project
   with the Supabase CLI or SQL Editor. It creates a separate approval table and
   a restricted read-only function. Do not apply it to an unrelated project.
5. Install `requirements-auth.txt` alongside the existing Python/OpenCV/Tkinter
   environment. Prepare public configuration with the project URL and its
   **publishable** key:

   ```sh
   python scripts/prepare_alpha_config.py \
     --project-url https://YOUR_PROJECT.supabase.co \
     --publishable-key YOUR_PUBLIC_PUBLISHABLE_KEY \
     --secret-file /absolute/private/path/sarween-alpha-secrets.env
   ```

   The secret file must be new and outside the checkout. It has owner-only mode
   and contains an Ed25519 private key in base64 plus its key ID. No secret is
   printed. `auth_config.json` contains only the URL, public API key, public
   verification key and key ID. The tool refuses to overwrite an existing setup.
6. Authenticate/link the Supabase CLI to this project and import that private
   environment file using `supabase secrets set --env-file PATH`. Deploy
   `supabase functions deploy alpha-access`. `verify_jwt=false` is deliberate:
   the handler forwards the bearer token to Supabase PostgREST, which verifies it
   before the SQL function checks the actual server session, ban and approval.
   Missing or invalid bearer tokens never receive a permission. No service-role
   key is needed by the function. Store the signing secret securely for recovery;
   it must never be included in a build, upload folder, log, or public asset.

Only `auth_config.json` is a new runtime data asset. The root Python modules are
included by the existing staging builder; backend files, tests and owner scripts
are not desktop build inputs. A signing-key rotation requires updating the
trusted public key in a reviewed app build and coordinating the server change.

## Approve and revoke testers

The owner-only script prompts for a Supabase secret/service-role key without
echoing it. It does not store the key or place it in command history. Run it from
a private terminal; never give this key to testers.

```sh
python scripts/manage_alpha.py approve tester@example.com
python scripts/manage_alpha.py status tester@example.com
python scripts/manage_alpha.py revoke tester@example.com
```

`approve` creates a passwordless account if needed, then sets its approval row.
It sends no invitation email itself. Tell the selected tester which email to use;
they must prove ownership through the app's emailed code. If account creation
succeeds but approval fails, the account remains denied; retrying is safe.
`revoke` changes approval without deleting account history. Existing accounts can
also be managed by user UUID in the Supabase `alpha_testers` table. Keep its RLS
and grants intact; app users must never receive table-write privileges.

The owner tool sends legacy service-role JWTs in both the API-key and bearer
headers required by Auth admin and PostgREST. Opaque secret keys use the API-key
header. Neither credential type is saved or shipped with the app.

## Verification and release checklist

Local automated checks use generated test keys, fake accounts and memory storage.
The database tests execute the real migration in isolated PostgreSQL (PGlite),
with minimal stand-ins for Supabase's Auth schema/claim helpers. They do not prove
the hosted project's settings or email delivery.

```sh
python -m unittest discover -s tests -p 'test_*.py'
SARWEEN_GUI_TESTS=1 python -m unittest discover -s tests -p 'test_auth_ui.py'
SARWEEN_GUI_TESTS=1 python -m unittest discover -s tests -p 'test_app_window.py'
node --experimental-vm-modules --test tests/test_module_capture.mjs tests/test_movement_logic.mjs
cd supabase
npm ci --ignore-scripts
npm test
```

Before distributing a configured build, complete live checks with designated test
accounts: code delivery, approved/unapproved access, restart, logout, expired
refresh session, revoke during use, offline restart, expiry and reconnect. Verify
the signed permission on the actual backend and test macOS Keychain in the signed
app. Run the existing recorded-tracking suite and Foundry tests; physical
camera/TV/Foundry acceptance remains a separate hardware check. Review the final
diff and reports with the owner **before publishing or distributing**. Existing
Developer ID signing/notarization work is not changed by this task.

References: [Supabase passwordless login](https://supabase.com/docs/guides/auth/auth-email-passwordless),
[custom SMTP](https://supabase.com/docs/guides/auth/auth-smtp),
[Supabase pricing](https://supabase.com/pricing),
[Resend pricing](https://resend.com/pricing).
