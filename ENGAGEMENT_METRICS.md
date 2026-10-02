# Alpha notifications and engagement metrics

## Immediate signup email

The existing website still submits through Cloudflare Pages into
[Alpha / Signups in Airtable](https://airtable.com/appn0QGP8HHX7uQl6/tbl388KNkJbfTduuV/viw8vo9YVMmmA67ZL).
All seven website files match the existing deployment manifest; no website
deployment or new Resend API key is needed for these notifications.

[Immediate Sarween signup notification](https://airtable.com/appn0QGP8HHX7uQl6/wfl9WdrXvBnrobBQa)
is a validated Airtable automation. On creation of a Signups record it emails
`shafqat2013@gmail.com` with the applicant's details, follow-up preference and
record link, then sets `Owner notified`. This checkbox means Airtable's email
action succeeded; it is not an inbox-delivery receipt. It never grants access or
emails the applicant. The owner chose the current Free automation allowance and
will revisit it if volume exceeds 100 applicants per month; other automations
share that allowance. This is an immediate notification, not a daily digest.

On October 2 its live test completed both actions using the existing, clearly
labelled website QA record. The owner then switched it ON; a connector read
confirmed `deploymentStatus: deployed`, a valid configuration, no deployment
error and no unpublished draft differences. Airtable's connector requires the
owner to activate new automations. Existing records are not backfilled by the
record-created trigger. Use Automation run history to investigate failed emails.

## Owner reports

- [Per-tester metrics](https://supabase.com/dashboard/project/ugwcxftwgkkpsbzdacjz/editor/17675?schema=public)
  (`public.alpha_tester_metrics`): email, approval, last sign-in, sign-in counts,
  last seen, activity frequency and detailed app-use counters.
- [Engagement overview](https://supabase.com/dashboard/project/ugwcxftwgkkpsbzdacjz/editor/17680?schema=public)
  (`public.alpha_metrics_overview`): approved testers, recent and returning users,
  app opens, tracking sessions and tracking hours.
- [Authentication users](https://supabase.com/dashboard/project/ugwcxftwgkkpsbzdacjz/auth/users):
  the account list. Accounts and approval remain separate; use
  [the owner approval/revocation tool](AUTHENTICATION.md#approve-and-revoke-testers).

These are read-only views for the owner in the existing Supabase dashboard.
Filter the per-tester view by `approved = true` for the approved tester list.
Recruitment prospects remain in Airtable until the owner explicitly creates and
approves an account. No automatic synchronization or applicant contact is added.

| Metric | Meaning and limits |
| --- | --- |
| `last_sign_in_at` | Supabase's latest authentication time. Remembered app launches need not create a new sign-in. |
| `recorded_sign_ins`, `sign_ins_30d` | Distinct Auth sessions recorded since collection began, plus sessions still retained at setup. Token refreshes and periodic approval checks do not count. Earlier deleted sessions cannot be reconstructed. |
| `last_seen_online_at`, `approval_checks` | Successful online approval checks, normally at startup and every five minutes while the app is open. Older authenticated builds also contribute. |
| `active_days`, `active_days_7d`, `active_days_30d` | Distinct UTC dates with an approval check or a usage report; recent windows include today. Uploaded offline activity can contribute after reconnection. |
| `estimated_online_hours` | Time between nearby approval checks, omitting gaps longer than six minutes. A coarse online-presence estimate, not play time. |
| `app_opens`, `app_runs_active_30d` | Distinct app runs with approved activity reported; the latter counts runs active in the last 30 UTC calendar days. Signing out and into another account starts another run. |
| `app_open_hours` | Approved app runtime sampled by its UI timer. An idle open app counts; sleeping or blocked UI gaps over five seconds do not. |
| `tracking_sessions`, `tracking_hours` | Starts after camera frames reach the tracking loop, plus time the tracking engine remains active. Paused tracking is included; this does not measure physical mini movement or player attention. |
| `foundry_connections`, `foundry_connected_hours` | Observed transitions to a connected Foundry client and duration connected. Reconnects count separately. |
| `replay_opens`, `demo_opens` | Successful replay/demo window openings. |
| `live_errors` | Exceptions caught by the live-mode launcher; not a complete crash or error-reporting service. |
| Overview `returning_users_30d` | Users active on at least two UTC dates in the last 30 calendar days. |

Detailed client values are null until the metrics-enabled build reports data.
This is not a complete lifetime history. `collection_started_at` and
`last_usage_upload_at` show when collection began and how fresh client data is.
Client counters are bounded and deduplicated, but remain estimates from a
tester-controlled app and must not be used for billing or access enforcement.

## Collection, offline use and security

Migration `supabase/migrations/202610020001_alpha_metrics.sql` is deployed in the
dedicated alpha project. It captures new Auth sessions, retains daily online
presence, adds the restricted usage RPC and exposes owner-only report views.
RLS and explicit grants prevent app users and anonymous clients from reading
other testers' metrics or changing approval. The RPC derives identity from the
verified session and requires current approval; revoked and expired sessions
cannot submit. Usage failures never deny access or interrupt tracking.

The new client sends cumulative counters during existing approval checks using
its normal user token and public project key. No privileged credential is added.
Retries use the same run/day identity and merge with the greatest totals, avoiding
double counting. Metrics do not cause extra email sends or token refreshes.

A visible login-panel notice explains the collection. Payloads contain a random
run ID, UTC date, app version and fixed counts/durations. They contain no video,
audio, screenshots, movement coordinates, game content, filenames or free text.
The owner dashboard joins those counters to the existing account email.

Offline counters stay in an atomic, owner-readable-only `usage-metrics.json` in
the app data directory, with no credentials. They upload after reconnection and
valid approval. The queue holds at most 128 run/day reports, retained for up to
30 days. Closing saves locally; a crash can lose counters since the last check.
Logout, account switching or confirmed revocation discards pending counters.
Older queued reports and unuploaded final activity can therefore be missing.
None of this changes the agreed 24-hour offline access policy.

Deleting an Auth account cascades its metrics. Revoking approval preserves history.
Only aggregate counts and dates are retained, not raw gameplay events.

## Review build and verification

Local review artifact (not published or installed over the existing app):
`/private/tmp/sarween-metrics-review-build/Sarween-k08rlv0l/Sarween.app`.
Open that copy directly rather than using Spotlight to find an older installation.
The previous authentication review artifact remains available separately.

The backend and client tests cover identity checks, denied/revoked/expired access,
counter validation, duplicate retries, offline persistence, logout/account
separation and metrics failures. Real hosted backend checks passed with temporary
synthetic accounts, then removed them. The packaged arm64 startup check passed
with authentication locked, native OpenCV working and the Foundry relay stopped.
See [verification results](AUTHENTICATION_VERIFICATION.md) for counts and limits.

Before distribution, review this build and confirm a normal tracking/Foundry
session appears in the owner report after an online check. Physical camera/TV
play remains a manual hardware check; no new app has been distributed.
