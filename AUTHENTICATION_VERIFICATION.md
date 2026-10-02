# Authentication verification — 2026-10-01

This is an implementation and backend checkpoint, not a release sign-off.
No new app build has been published or distributed.

| Check | Result |
| --- | --- |
| Full Python suite | 215 tests run: 205 passed, 10 intentionally skipped |
| Foundry setup cancellation guard | Included in the final full run; tracking/auth suite: 4 passed |
| Authentication unit/SDK contract tests | 21 passed, included in the full suite |
| Native login UI | 2 passed with fake accounts and memory storage |
| Existing native home/Foundry relay tests | 10 passed, including actual local WebSocket connection/cleanup |
| Backend signing and PostgreSQL permissions | 6 passed; real migration executed in isolated PGlite |
| Hosted Supabase backend | 12 live checks passed using temporary synthetic accounts; cleanup passed |
| Hosted access controls | RLS enabled; app cannot approve itself; anonymous RPC denied; missing/invalid endpoint login returns 401 |
| Hosted signup | Public signup disabled; email codes configured for 8 digits and 600 seconds |
| Resend domain | auth.laserfit.ai verified; three approved DNS records added; website CNAME retained |
| Foundry module JavaScript suites | Both passed with Node's VM-modules flag |
| Five-mini synthetic replay | 11/11 expected events; zero extras |
| Reviewed single-mini real recording | 6/6 expected events; zero extras |
| macOS Keychain | Unique dummy item written, read and deleted successfully |
| Bundle input check | Passed: 47 files; existing bundle overrides retained |
| Configured local macOS build | Created successfully in an isolated output directory; public-only configuration verified |
| Packaged startup | Passed: frozen arm64 app, 860x620 native window, signed-out lock, native OpenCV check, relay off and clean exit |
| Packaged owner acceptance (October 2) | Owner confirmed email-code login, remembered login after restart, offline restart, offline logout persisting across restart with tracking locked, and fresh login after reconnecting |
| Missing public configuration | Build refused before invoking PyInstaller |
| Secret/backend bundle exclusions | Environment files, private-key files and backend directory rejected |
| Website preservation | All seven site files still match the existing deployment manifest |
| Whitespace/diff validation | Passed |

Tests cover approved and unapproved users, remembered login, expired access-token
refresh and token rotation, invalid refresh sessions, revocation, offline startup,
24-hour expiry, recovery without extending an old grant, signature/identity/device
binding, clock rollback, logout while offline and during an in-flight request,
Keychain failure, malformed responses and TLS failures. The real SDK HTTP contract
is exercised with an isolated HTTP transport; no tester emails are sent by these
tests. Live tracking tests verify denial before camera startup and cleanup before
any further movement when access ends. A denied Foundry setup wait also closes.

The native tests first encountered macOS GUI/local-socket restrictions inside the
sandbox. They passed when run with the required desktop access. Test dependencies
were installed in `/private/tmp/sarween-auth-venv`, without modifying the existing
Conda environment. Server test dependencies are local to `supabase/node_modules`.

PyInstaller's first attempt from the overlay venv encountered its Conda NumPy hook
limitation. Building with the original Conda interpreter plus the isolated auth
dependency overlay succeeded without changing the original environment. The local
review artifact is `/private/tmp/sarween-auth-review-build/Sarween-vkd4kua3/Sarween.app`.
Its startup report is `/private/tmp/sarween-auth-packaged-smoke.json`.

The recorded-video tests establish consistency with existing fixture labels;
they do not establish physical camera/TV accuracy or real five-mini Foundry play.
The real recording has the pre-existing replay limitations documented in README.

## Live backend verification

The deployed endpoint was exercised with real Supabase sessions and its actual
Ed25519 signing key. Checks passed for approved and authenticated-but-unapproved
users, remembered login, real token refresh, online revocation, owner reapproval,
local logout, rejection of the logged-out server session, and a server-expired
session. A saved real permission also passed simulated offline restart with an
expired login token, fixed expiry during an outage, and locking at its 24-hour
deadline. Synthetic accounts used reserved `.invalid` addresses; no test email
was sent and all temporary accounts were removed.

## Pending verification

- Review the final changes and verification results with the owner before
  distribution. Owner acceptance of initial login, restart persistence, offline
  restart, offline logout and subsequent sign-in in the configured packaged build
  is complete.
- Physical camera/TV and real Foundry play validation remains a separate hardware
  check. Automated tracking and Foundry regression results are recorded above.

See [setup and policy](AUTHENTICATION.md) for exact owner steps. The existing
website, recruitment form, previous app binaries and unrelated uncommitted work
are preserved.

## October 2 email setup and owner-tool follow-up

- Custom SMTP remained enabled after reloading the hosted dashboard, with the
  Resend host, TLS port 465 and Sarween sender name. The owner entered the secret.
- Saved the hosted Magic Link / OTP template with `{{ .Token }}` and the subject
  `Your Sarween sign-in code`; retained its HTML and local configuration in source.
- Fixed a live owner-tool failure: a legacy service-role key without an
  Authorization bearer header received HTTP 401 from Auth admin; the same key
  with the bearer header received HTTP 200. Both Auth admin and approval-table
  requests now use the appropriate owner headers.
- Three owner-command regression tests passed (six subcases covering legacy JWT
  and opaque secret-key headers). They exercise the actual SDK with an isolated
  HTTP transport, passwordless creation, separate approval writes, revocation and
  status reads; no email is sent and no real credentials are used by these tests.
- The combined owner, authentication and tracking-access regression run passed
  all 28 tests. The local TOML template path and code placeholder were validated.
- The corrected owner tool approved the designated real test account and its
  subsequent status read confirmed approval.
- The app's actual `SupabaseProvider.send_code` accepted the real email request.
  Resend reported **Delivered** with the expected subject. The owner subsequently
  confirmed inbox receipt of the eight-digit code.
- Native Computer Use permission was unavailable. The owner has the review-app
  path and instructions to enter the code directly. After initially opening the
  older installed app through Spotlight, the owner opened the separate review
  build and reported successful login. This is owner-reported acceptance, not an
  automated observation of the native UI.
- The owner then confirmed all four packaged acceptance steps: restart while
  online retained login; restart with Wi-Fi off retained access; offline logout
  remained signed out and locked tracking after restart; reconnecting and using
  a fresh email code signed in successfully. These checks used the real app and
  macOS Keychain. The full 24-hour deadline, revocation, denied accounts and expired
  sessions were covered by the automated and live-backend checks above, rather
  than by this short manual acceptance session.

These changes affect the owner script, hosted email template and documentation.
The existing local desktop review artifact remains the current acceptance build.
No app publication or distribution occurred.

## October 2 engagement metrics and immediate signup notification

| Check | Result |
| --- | --- |
| Final full Python suite | 226 tests run: 216 passed, 10 intentionally skipped |
| Native auth/home/Foundry GUI tests | 12 passed with desktop access |
| Backend tests | 10 passed, including actual SQL migration, owner-only permissions, session counting, retry deduplication, revoked/expired access and malformed input |
| Live Supabase checks | 21 passed with disposable synthetic accounts; cleanup passed |
| Metrics failure isolation | Upload/storage failures do not interrupt approved access or logout; offline and revoked clients do not upload |
| Client usage counters | Time/mode transitions, suspension gaps, offline persistence, account separation, acknowledgement, daily expiry and corrupt-cache checks passed |
| Foundry module JavaScript | Both existing suites passed |
| Website | All six existing tests passed; all seven production site files match the deployment manifest |
| Bundle inputs | All 48 inputs match the newly built review artifact |
| Packaged startup | Passed: frozen arm64 app, native OpenCV check, 860x620 window, signed-out lock, relay off and clean exit |
| Immediate Airtable notification | Live test sent the owner QA email and updated Owner notified; owner activated it; connector confirmed deployed and valid, without draft differences |
| Diff validation | Passed |

The hosted migration adds owner-only `alpha_tester_metrics` and
`alpha_metrics_overview` views. The live tests verified that approved clients can
submit only their own bounded counters, repeated submissions do not double count,
private metrics cannot be read by app users, anonymous writes are denied and
revoked accounts cannot submit. Existing online permission checks and new Auth
sessions contribute immediately, including from older authenticated builds.

Detailed app, tracking and Foundry counters require the new local review build:
`/private/tmp/sarween-metrics-review-build/Sarween-k08rlv0l/Sarween.app`.
The startup report is `/private/tmp/sarween-metrics-smoke.json`. The new build has
not replaced the installed app or been published/distributed. Its packaged smoke
test was signed out; a real normal-use session in this metrics-enabled build is
still the owner's manual acceptance check. Automated client and hosted tests
cover the collection/upload path, but are not a physical camera/TV play test.

The metrics extension does not upload gameplay content or add privileged desktop
credentials. Collection is disclosed in the login panel. Offline counters are
bounded and uploaded after approval resumes; they do not extend offline access.
See [metric definitions and limitations](ENGAGEMENT_METRICS.md).

Per the owner's choice, signup notifications use Airtable's immediate automation.
The unused Resend website notification changes were removed; no website deployment
or new notification secret was needed. Resend continues serving existing login
emails. No applicant is automatically approved or contacted by the notification.

## October 2: login screen and demo flow

The owner chose to keep email/code login and to supply their own website/form
wording. This change is limited to the desktop presentation and its checks:

- Signed out: the full home window shows the login form; library/session controls
  are hidden, including during session restoration. Logout, confirmed revocation
  and offline expiry restore this screen. Recovery controls remain available for
  an expired saved login or a Keychain error.
- Signed in: a compact account row retains logout, access checks and offline time
  remaining. The existing approval, Keychain and 24-hour offline policies remain.
- Demo: the bundled synthetic footage and measured tracking state are side by
  side. Analysis completes before automatic playback; seeking uses the same video
  time for both views. Review tools are available on demand. Ordinary recording
  review retains its tabs and annotation controls.

Verification:

| Check | Result |
| --- | --- |
| Python discovery suite | 227 run: 216 passed, 11 opt-in UI checks skipped |
| Targeted native suites, run separately | Login 3/3, replay 7/7, home/Foundry 10/10 passed; these totals include non-UI helper tests repeated from discovery |
| Login transitions | Email/code, restored login, logout, valid offline access, offline expiry, recovery and revocation verified with isolated accounts |
| Demo playback | 11/11 expected detections; forward/backward seeking shows matching measured map positions; worker cleanup passed |
| Visual review | Login at 860x620 and 700x460; demo at 1100x640 and 780x460; corrected caption clipping at the smaller size |
| Bundle | 48 validated inputs; four changed runtime modules match final source hashes; no backend or test fixtures included |
| Packaged startup | Passed on arm64: signed-out screen, native OpenCV check, relay off, clean exit |

The separate local review build is
`/Users/shafqat/Documents/New project/sarween-ui-review-20261002/build/Sarween-vi9jp9xq/Sarween.app`.
The same review folder contains screenshots and `packaged-smoke.json`. Packaged
startup was signed out; authenticated demo playback was verified in the native
source tests, not bypassed in the packaged app. This does not establish physical
camera/TV or live Foundry play accuracy. No installed app was replaced, no build
was published/distributed, and the website/recruitment form were not edited.

## October 2: real-video correction

The owner clarified that the product example must show real camera footage.
The synthetic test clip was unsuitable for that purpose. **Watch real example**
now opens about 15 seconds of the owner's approved clips 10 and 11, retaining
reflection blur and hiding the room around the table. The camera angle is
restored from those approved exports, with only the original physical marker
borders copied from the source so registration remains possible. No markers or
mini movements are generated. Provenance and source hashes are in
`demo/tabletop.provenance.json`.

The two views are labeled **What the camera sees** and **What the software sees**.
The second view is recomputed by the normal tracking worker from the bundled
video. Its sidecar contains scene geometry but no tracking events, synthetic
positions or scripted movement. The synthetic five-mini fixture remains in the
repository for developer tests and is excluded from both application specs.

**Open your video** uses the same layout and processes files locally. Recordings
with metadata and profiles start analysis automatically if they have no saved
analysis. Existing analyses, review decisions and completion status survive
reopening without a new worker run. Other files open for
playback with explicit setup instructions and an empty tracking view. Users must
provide the supported marker layout, grid and mini profiles; this does not claim
automatic tracking of arbitrary footage.

- Full discovery: 229 tests, 216 passed, 13 opt-in UI tests skipped.
- Native replay suite: 10/10 passed, including the real example, synchronized
  seeking forward/backward, unconfigured own-video import, saved-review
  preservation and worker cleanup. This final run includes the added regression
  test after the full discovery run above.
- Real recording case: all five recorded placement labels matched, zero extras.
  Labels derive from the approved recording and its log, not independent ground
  truth. The first detection is acquisition; four later detections change cells.
- Packaged worker: the bundled MP4 produced J7, Y14, AL20, AK7 and K21; 203 frames
  processed, 199 locked. Source hashes matched, worker exited cleanly, and no
  synthetic video was included. `packaged-real-video.json` records this check.
- Native screenshots reviewed at 1100x640 and 780x460. The MP4 is silent H.264,
  204 frames, 1164x984, at the original nominal 13.544 fps.

The corrected review build and evidence are under
`/Users/shafqat/Documents/New project/sarween-real-video-review-20261002/`.
The final build is `build/Sarween-noaokns2/Sarween.app`. Website and recruitment form
content remain unchanged; the owner is preparing their own wording.

After the owner closed Sarween, this final build replaced `/Applications/Sarween.app`
using a reversible directory swap. The previous installed bundle remains at
`previous-installation/Sarween-before-final-review-fix.app.backup` in the review
folder. The installer verified the bundle signature and identifier and did not
change Application Support files or Keychain entries. This was a local installation;
no release was published or distributed.

## Handoff for October 3

The installed real-video build is the stopping point for October 2. Next session:

1. Revise the website and recruitment form from the owner's rough wording.
2. Prepare the Reddit announcement in the owner's voice, using the four approved
   clips (07, 08, 10 and 11) in the October 6 announcement folder's `v4/exports`.
3. Review the finished website/form and post before publishing. The launch goal
   remains announcing the alpha test by October 6.

No website or form edits, Reddit post, or public app release were made during
this demo correction. Unrelated working-tree changes remain untouched.
