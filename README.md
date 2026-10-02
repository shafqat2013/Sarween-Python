# Sarween-Python
Camera Vision App for TTRPGs

## Invitation-only Alpha Login

Sarween now gates live tracking, Foundry connection and recording review behind
an individually approved Supabase login. Sessions are stored in macOS Keychain;
an approved tester can continue offline for up to 24 hours after the last online
approval check. The recruitment website does not grant app access.

The dedicated Supabase backend and public app configuration are installed. Live
backend checks pass; SMTP credential setup, real email delivery and final packaged
app acceptance are still pending. Builds reject missing or secret configuration.
See [authentication setup and policy](AUTHENTICATION.md)
for owner approval/revocation tools, the agreed offline rules and verification.

## Foundry Viewport Tracking

Sarween's Foundry module displays ArUco IDs 10-13 as fixed overlays in the four
screen corners. They remain visible above fog while the GM pans or zooms even on
maps much larger than the TV. The module sends Foundry's exact canvas transform
to Python, so camera positions are converted to scene grid cells without moving
or nudging the map into alignment.

After the module connects, live Foundry sessions use the viewport markers.
Older recordings use the legacy scene-tile markers 0-3. Newer recordings with
viewport timeline sidecars replay using fixed markers 10-13.
All four viewport markers are required to acquire the initial lock. After that,
Sarween keeps the saved homography while any three visible markers still agree
with it, so intermittent glare on one corner does not stop tracking.
When the map is panned, Sarween briefly reseeds the visual background and moves
each already-tracked Foundry token to the cell now underneath its stationary
physical mini.

Player minis are identified by persistent, solid-color base rings. Each tracked
player must use a different ring color; untracked enemies may use anything.
Runtime tracking looks for the ring on every settled frame, so fog and Foundry
token updates cannot erase it from a captured background. Hand/arm motion is
used only to delay a move until the new placement has settled. The minimum ring
footprint rejects smaller same-color Foundry token artwork, preventing a moved
digital token from being mistaken for its own physical mini.

Planned player test identities and Foundry token names are A=Red, B=Blue,
C=Yellow, D=Green, and E=White. The existing trained `red10` profile remains
the tracking identity for the Red token until profiles are migrated into the
future Mini Library.

When recording is enabled, Sarween creates a video and timeline sidecar in
`~/Library/Application Support/Sarween/Recordings/` on macOS:

```text
sarween_rec_YYYYMMDD_HHMMSS.mp4
sarween_rec_YYYYMMDD_HHMMSS.tracking.json
```

The JSON sidecar records scene geometry, viewport transforms, pans, zooms, and
Foundry visual changes against exact video frame numbers. Keep the two files
together. The regression runner discovers the sidecar automatically, so a
fixed-marker recording can be replayed later without Foundry, the camera, or
the TV running.

In viewport-marker mode, the Foundry module also saves one low-resolution,
mini-free canvas snapshot roughly every two seconds in a sibling
`sarween_rec_YYYYMMDD_HHMMSS_references/` directory. This is an optional
offline diagnostic, never a live tracking input. Keep that directory with the
video and sidecar to compare what the camera saw with the actual rendered fog,
lighting, and digital tokens:

```bash
python rendered_reference.py sarween_rec_YYYYMMDD_HHMMSS.mp4
```

The output reports changed-pixel percentages for frame pairs with matching
scene and viewport, plus how much of each current mini candidate differs from
the clean display. It does not yet score detection precision or improve live
tracking. Older recordings have no rendered frames; the original map image
cannot reconstruct their fog, so this comparison requires a new recording.

Relevant client settings are `Fixed viewport ArUco markers`, `Viewport marker
size`, `Viewport marker inset`, `Grid square size (inches)`, and `Animate tracked
token movement`. Animation is on by default; Foundry controls the slide timing.
The physical grid setting already supports 1.25-inch squares when a larger mini
setup is desired. The default 120-pixel marker plates are a good starting point;
increase the size if the far pair is intermittent.

The people button opens guided dataset capture for one to five minis. It starts
the camera recording automatically, pauses tracker predictions, selects the
chosen tokens for combined token vision, and shows one magenta destination at a
time. Confirm each physical placement with `Placed`; only that confirmation
moves the digital token and records ground truth. No successful mini scans are
required. Routes are A=Red (`red10`), B=Blue, C=Yellow, D=Green, E=White.

Each route includes initial placement 0 and moves 1-5. The guide captures an
empty-board baseline, 15 seconds with all minis stationary after initial
placement, and a stationary tail. The resulting `.tracking.json` contains
`groundTruth` independently from detector `trackingEvents`. It also saves a
`.profiles.json` snapshot and a `.case.json` regression definition. The generated
case uses prompted-to-confirmed time windows plus settling time and references
the captured `.profiles.json` snapshot. Use the regression runner's `--profiles`
option to evaluate an improved scan against the same footage and labels.

Capture saves partial results on disconnect, map/viewport changes, or app exit,
and has a 15-minute limit. The original digital token positions and selection
are restored when the guide receives the saved acknowledgement. Tracking stays
paused until `Resume tracking` is clicked. See [the recording checklist](CAPTURE_CHECKLIST.md).

The QR-style button toggles all four viewport markers when they obstruct a
Foundry dialog. Tracking requires the markers to be restored afterward. Drag
the Sarween bar with its grip handle; its saved default is at the bottom-left,
immediately to the right of the bottom-left marker.

Token assignments are validated before every move. If a mapped Foundry token
has been deleted or replaced, the module tells Python to discard the stale ID
and request assignment to the replacement token instead of silently dropping
future movements.

### Confirmed Foundry Delivery

Sarween 1.5 retains the latest intended cell for each mini until Foundry replies
with the applied position. Missing acknowledgements and rejected moves receive
up to three attempts. After that, the control panel shows a failed move and
enables `Retry moves`. Assignment and reconnection recover the latest position
without requiring another physical move. Old paths are not replayed, and
scene/grid changes discard obsolete intentions.

Reload the Foundry world after updating this module: the Python relay requires
the version-2 acknowledgement handshake and reports an outdated module instead
of silently assuming delivery. Only one Foundry display can own the relay at a
time. Repeated commands for a token already at the target are acknowledged
without repeating its animation or spending movement allowance again.

## Mini Library

The control panel's `Mini Library` button opens the player-mini roster. It shows
all five planned ring colors, whether each mini has a usable tracker profile,
how many verified appearance samples are saved, its token assignment in the
active Foundry scene, live recognition confidence, and its last known cell.

`Scan selected` opens a frozen, registered camera image. Click a clean patch of
the colored ring, inspect the magnified sample, then choose `Use ring sample`.
Mixed boundary pixels are rejected. A gray or wrong-hue sample is also rejected
for named Red, Blue, Green, and Yellow minis. The `Calibrate` and `Auto` buttons
use this same explicit picker; they no longer learn the largest moving blob.
No miniature movement is required for a quick scan.

Confirmed scans add a color sample while preserving existing brightness curves
and thresholds. `Replace active colors` explicitly replaces a bad color model;
other profile settings and other minis remain unchanged. Saves are atomic and
retain the previous complete profile file as `combo_profiles.json.bak`.

`Full brightness scan` uses a dedicated legacy-marker calibration screen with
matching camera geometry. Click the ring itself when labeling a mini. Brightness
samples use that chosen patch, not the blob center. Unlocked frames are not
reused from a previous brightness step. Scanning a subset preserves other minis.
Imported historical curve points retain their brightness labels but are not
automatically marked user-verified; explicit ring confirmations can promote
matching samples. The existing saved scan data is not rewritten merely by this
code update.

The durable metadata lives in `mini_library.json`; the detector-compatible
profile remains in `combo_profiles.json`. Keeping these separate lets offline
experiments evaluate larger portfolios before they are allowed to change live
matching behavior.

## Tap Selection And Movement Budget

Selecting a known Red, Blue, Yellow, Green, or White token in Foundry starts a
movement session. Sarween uses the actor's walking speed when available and the
module's `Fallback movement speed` setting otherwise. Each confirmed grid move
adds Foundry's measured path distance, so diagonal behavior follows the active
scene rather than a separate Sarween rule.

The selected mini receives a white reachable-range ring, path, and remaining
movement label. The white range shrinks around its current position. Once the
cumulative path exceeds the budget, the token ring, path, and counter turn red.
The Sarween bar provides icon buttons to undo the last confirmed segment or
reset the budget from the mini's current square.

With module 1.5.1, viewport remaps carry their cause all the way from Python to
Foundry. Repositioning a stationary mini after a map pan shifts the saved path
and Undo destinations without changing movement spent or remaining. The next
physical move is measured from the repositioned cell. Failed token updates and
repeated acknowledgements do not add a movement segment.

`Physical tap selects a mini` is enabled by default. Python watches the existing
foreground/obstruction mask only near already-known mini positions. A brief
touch followed by the same mini reappearing in place toggles selection; picking
the mini up and placing it elsewhere is treated as movement instead of a tap.
The selected mini enters search mode sooner after pickup, helping it reacquire
after a long move. Tap the stationary mini again, or deselect its Foundry token,
to end the movement session. Double-tap actions, attack ranges, and aura ranges
are deliberately reserved for later versions.

Scene activation and grid/background edits now resend scene geometry before the
viewport transform. Python derives row/column counts from the Foundry grid,
clears old anchors, and discards moves queued for a different scene. Player
tokens uniquely named Red, Blue, Yellow, Green, and White are automatically
bound to the corresponding mini identities in each scene. Other token naming
schemes still use the assignment dialog. Scene padding is included in movement
coordinates. A printed image grid must still be aligned with Foundry's grid;
Sarween does not infer the intended scale of arbitrary artwork.

## Tracking Regression Tests

You can run the combo tracker against a recorded video without opening the
camera/control-panel UI:

```bash
python3 tracking_regression.py run sarween_rec_20260420_163421.mp4
```

For a fixed-marker recording, the command is identical:

```bash
python3 tracking_regression.py run sarween_rec_YYYYMMDD_HHMMSS.mp4
```

Use `--timeline path/to/file.tracking.json` only when the sidecar has been moved
or renamed separately from its MP4.

To turn video behavior into assertions, edit the starter
`tests/fixtures/tracking_cases.json` using
`tests/fixtures/tracking_cases.example.json` as the shape, then run:

```bash
bash scripts/run_tracking_regression.sh
```

If your default `python3` does not have OpenCV installed, activate the project
Conda environment or point the helper at its Python executable:

```bash
conda activate py_arm
bash scripts/run_tracking_regression.sh

# Equivalent without activating Conda first:
PYTHON_BIN=/opt/homebrew/Caskroom/miniforge/base/envs/py_arm/bin/python \
  bash scripts/run_tracking_regression.sh
```

Each expectation can pin a mini, source cell, destination cell, expected video
timestamp, and tolerance. By default, unexpected emitted moves fail the case so
false positives are caught as well as missed real movements.

### Offline Development

No camera, TV, or running Foundry instance is needed. Replays run sequentially
in disposable processes, use one OpenCV thread by default, print progress every
five seconds, and stop after a 180-second wall-clock deadline per video. Ctrl-C
stops and reaps the active worker. `--timeout-seconds` changes the deadline;
`--max-seconds` on `run` limits how much **video time** is processed instead.

The camera app and replay runner both use `TrackingEngine` for scene/pause
resets, selected-mini recovery, movement consensus, viewport remapping, and tap
recognition. Each frame uses a frozen mapping snapshot. Multi-mini overlap
filtering retains the strongest surviving candidates with deterministic ties;
a rejected candidate cannot suppress another mini.

New timeline sidecars include the selected mini and prediction-pause state.
Normal replay honors those inputs; guided-capture evaluation intentionally
ignores the live prediction pause used while collecting independent labels.
Replay only reports recognized taps; it does not send them to Foundry or invent
selection responses. Recorded selection responses are used when available.

Sharing the engine does **not** make older recordings exact session restores.
For schema-1 recordings, timeouts use nominal video FPS, not original frame wall-clock times.
Tracking/background state starts fresh (apart from explicit initial cells), and
mid-recording rescans/background recaptures are not restored. Missing historical
selection/pause inputs default to no selection and tracking enabled. These
limitations appear in the terminal and saved reports. New schema-2 recordings
restore capture clocks and checkpoints; see **Recording Review And Demo** below.

```bash
# Inventory the available footage without processing frames.
python tracking_regression.py inventory

# Save a baseline with the frozen scan used by the five existing scenarios.
python tracking_regression.py check --report-dir tracking_reports/baseline

# Quickly rerun one scenario.
python tracking_regression.py check \
  --case fog_on_red10_numbered_targets_48x27 \
  --report-dir tracking_reports/experiment \
  --baseline tracking_reports/baseline/report.json

# Compare an improved scan against the frozen baseline.
python tracking_regression.py check \
  --profiles "$HOME/Library/Application Support/Sarween/combo_profiles.json" \
  --report-dir tracking_reports/new-scan \
  --baseline tracking_reports/baseline/report.json

# A future guided capture is directly runnable too.
python tracking_regression.py check recording.case.json

# Fast code checks; local video regressions are explicitly opt-in.
python -m unittest discover -s tests -p 'test_*.py'
```

Each check writes `report.md` and `report.json` (default:
`tracking_reports/latest`). Reports include matched/missed/extra events,
per-mini results, timing differences, marker-lock coverage, runtime, and SHA-256
fingerprints for footage, profiles, timelines, and local camera calibration.
Reports also fingerprint the relevant tracking code. Baseline comparisons flag
changed inputs and code. Missing local videos are reported as
skipped; `--require-videos` makes any skip fail. A run that evaluates no videos
always fails. Missing mini profiles, timeouts, and unusable marker lock fail
explicitly instead of producing a misleading no-movement pass.

The existing five fixtures are marked `legacy-unverified`: their exact physical
placement times have not been independently confirmed. The frozen
`tests/fixtures/red10_baseline.profiles.json` is a copy of the scan available when
this baseline was established, not a recovered capture-time scan. Passing these
cases demonstrates consistency with the historical labels, not proven accuracy
on five minis. Do not generate expected moves from detector predictions.

For new labels, use `label_source: "manually-reviewed"` after checking the video,
or the guided capture's `"user-confirmed"` labels. A destination may use an
`at` timestamp with `tolerance_seconds`, or a `between: [start, end]` window.
Moves are matched once each, in order per mini. Other minis may move in any
interleaved order. An explicit `stationary: true` case with no expectations
checks for unwanted movements throughout its scoring window. `ignore_before`
and `ignore_after` define that window without skipping tracker warmup frames.
For minis already on the board, set `initial_positions` to their known starting
cells so their first detection is not counted as a new movement.

Timing error against `at` is not a measurement of reaction speed. Add
`settled_at` only after manually identifying when physical placement finished;
the report then includes signed `response_seconds` for matched moves. Guided
`confirmed_at` records when `Placed` was clicked, which can be later than the
actual placement, so it is reported separately. A viewport remap only matches
an expectation with `source: "viewportTransform"`.

These tests exercise camera processing, detection, and movement consensus.
They do not verify delivery/animation in a live Foundry instance. Older footage
also lacks recorded tap/selection state and rendered display references; those
features need corresponding capture data before they can be validated offline.

### Independent Footage Review

`tests/fixtures/tracking_cases.reviewed.json` is separate from the historical
fixtures. Its first case covers the physical Red mini's starting square and five
moves in `sarween_rec_20260911_135922.mp4`, including stationary gaps. Destinations
were checked against visible base positions, not inferred from tracker output.
Labels are explicitly `assistant-visually-reviewed`, not user-confirmed or
blinded/held-out data. The evidence notes record sampling coverage, base-center
points, input hashes, and hand-clear timing intervals. The initial replay passed
all six position expectations with no extra moves across the whole clip.

```bash
python tracking_regression.py check tests/fixtures/tracking_cases.reviewed.json \
  --timeout-seconds 60 --threads 1 --require-videos \
  --report-dir tracking_reports/reviewed

# Make camera-free review images using marker/grid geometry only.
# Output must be a new directory. At most 120 frames are extracted per invocation.
python scripts/review_tracking_video.py sarween_rec_20260911_135922.mp4 \
  --output tracking_reports/review-overview --step 1

# Inspect specific zero-based frames; map a manually identified base center.
python scripts/review_tracking_video.py sarween_rec_20260911_135922.mp4 \
  --output tracking_reports/review-detail --frames 56 --point 233 164
```

The review utility does not run a mini detector, start a server, or read active
profiles. It produces clean and cell-labeled frames, contact sheets, and an
evidence manifest. Each rectified frame requires all four markers; missing
registration is shown explicitly, never reconstructed from tracker guesses.
It currently supports already-undistorted viewport recordings only. Processing
uses no OpenCV parallel workers, a cooperative 45-second extraction deadline,
and always releases the video handle. It does not automatically assign labels.

Review timestamps use the encoded video's nominal FPS. First hand-clear samples
only bound placement time; response measurements are estimates, not exact live
latency. This single-mini, static-fog clip does not validate five minis, changing
fog, or Foundry delivery. See `tests/fixtures/reviewed_red10_135922.labels.json`
for the review's limitations and reproducible frame selections.

## Opening Sarween While Travelling

The real app now opens without a camera, TV, Foundry, or ArUco markers:

```bash
conda activate py_arm
cd /Users/shafqat/Batcave/Sarween-Python
python main.py
```

The first screen is **Mini Library**, using your saved profiles and portfolios.
Selecting a mini shows verified versus unverified sample counts and the last
capture date. Browsing does not change scans or assignments. Saved token IDs
are shown as saved data, not proof of a current Foundry connection.

- **Connections** shows the relay, Foundry, and scene status. **Connect Foundry**
  starts the local relay only; it does not open the camera.
- **Start live session** opens hardware setup. Camera detection runs only after
  this explicit action. With devices connected, choose the camera, display, and
  mode as before. Cancelling setup returns to the home window.
- Live controls are divided into **Session**, **Scans**, and **Diagnostics**.
  **Stop session** closes tracking and returns to the library. Missing profiles
  no longer force an automatic brightness scan; scanning remains explicit.
- Closing the home window stops and joins its Foundry relay thread. An idle
  home window runs no video processing or regression workers.

## Hardware-Free UI Preview

You can open the actual control panel and Mini Library while travelling:

```bash
conda activate py_arm
cd /Users/shafqat/Batcave/Sarween-Python
python ui_preview.py
```

No camera, TV, Foundry connection, ArUco lock, or saved profiles are required.
The title and status lines identify sample data. The **Preview** menu switches
between ready, missing markers, pending/failed delivery, and disconnected states.
**Mini Library** opens the real roster panel with five sample minis. In the failed
state, **Retry moves** changes only the simulated status.

Hardware/recording/scan controls are disabled; no Foundry server or background
tracking worker starts, and no settings or scans are saved. Closing the control
panel or clicking **Stop session** ends the process. This is a sample-state
preview, not a successful tracking session. Use `python main.py` for the real
app and saved library. For a bounded native smoke test, use
`python ui_preview.py --close-after 15`.

## Building Safely

All four legacy build entry points now use `scripts/build_bundle.py`. Nothing
is copied over the checkout and no Git restore/checkout command is run. The
builder copies current root Python/JS code and the spec's data files into a
temporary directory, applies the existing `app_bundle_overrides` there, and
runs PyInstaller with separate work, cache, and output directories. Inputs must
be real files inside the project; escaping paths and symlinks are rejected.

```bash
conda activate py_arm

# Read-only input check. This does not import the app or start PyInstaller.
bash build_bundle.sh --check

# Inspect the prepared inputs without compiling an app.
bash build_bundle.sh --stage-only

# Build with the selected environment's Python/PyInstaller.
bash build_bundle.sh --timeout-seconds 600
```

`PYTHON_BIN` can select an explicit interpreter. Successful builds go into a
new `dist/Sarween-<unique-id>/Sarween.app`; previous builds remain untouched.
Each output includes `build-inputs.json` with file hashes, original input paths,
and the Python version. Failure, timeout, or Ctrl-C removes temporary staging;
timeout/cancellation terminates and reaps the builder's process group.

This makes the **build workflow** non-destructive; it is not release validation.
Both specs exclude personal profiles, settings, token mappings, calibration, and
captures. The staging builder rejects those known user-data files even if they
are accidentally added to a spec. The legacy frozen-app setup still differs
from development setup. Both specs now ship only the demo map, not the user's
full maps folder. Successful builds also retain PyInstaller's warning report.

### Personal Mac Installer

The current personal build is available at
`dist/Sarween-0.2.0-Apple-Silicon.dmg` (about 123 MB), including the recording
reviewer, synthetic demo, and map-state view. Open the DMG, drag
**Sarween** onto **Applications**, then open Sarween from Applications. No Conda
environment or terminal command is needed to run the packaged app. Existing
data stays in `~/Library/Application Support/Sarween/`.

This is an **arm64 / Apple Silicon** build, locally ad-hoc signed, not a Universal
binary or an Apple-notarized release. It was tested on this Mac running macOS
26.3. Intel support, Developer ID signing/notarization, older macOS versions,
and physical camera/TV acceptance remain separate release checks.

The earlier 0.1.0 packaged home window passed auto-closing checks with clean data, existing
user data, and a read-only mounted DMG. These checks also exercised lazy tracker
imports, synthetic ArUco detection, and NumPy linear algebra. All eight saved
state files were byte-identical afterward. Native signatures and the disk image
checksum were verified; no app was installed or existing installation replaced.
Reports and source fingerprints for that older build are archived in
`deprecated/builds/dist/Sarween-j9k7r73g/`.

The 0.2.0 app and its reports are in `dist/Sarween-hn007ne9/`. Its smoke check
also analyzes the synthetic video in a bundled child process, verifies all
11 events and five map positions, and reaps the worker. Clean data and temporary
copies of existing saved data both passed; the eight original state files
remained unchanged. The installer is still locally ad-hoc signed, not notarized.

To package another already-built app, choose a new output filename:

```bash
python scripts/package_dmg.py "/path/to/Sarween.app" "dist/Sarween-personal.dmg"
```

The packager preserves bundle symlinks, includes an Applications shortcut, verifies
the compressed image, and refuses to overwrite an existing output. Disk-image
commands have a 180-second deadline and process-group cleanup. For automated
app checks, the executable accepts `--smoke-test-report /absolute/report.json`;
this opens the home window, checks dependencies, writes a report, and exits.
Always run such checks under an external process deadline as well.

## Saved User Data

Development runs and packaged apps now share
`~/Library/Application Support/Sarween/` on macOS. Program resources such as the
demo map and module scripts stay separate and read-only. The data directory holds:

- `combo_profiles.json` and `band_profiles.json`: active color profiles.
- `mini_library.json`: miniature roster and scan portfolio.
- `hardware_config.json` and `mini_token_map.json`: setup and token assignments.
- `camera_matrix.npy` and `dist_coeffs.npy`: camera calibration.
- `mini_database.csv` and `mini_captures/`: legacy capture data, when present.
- `Recordings/`: new videos, timelines, profile snapshots, cases, and references.
- `Cache/`: regenerated marker and blended-map images.

The first live startup copies recognized legacy files from the source checkout
(or an older bundle's resource/executable directories) without deleting or
modifying them. Existing destination files always win. Copies are verified with
SHA-256 and a `migration-v1.json` receipt records their origins. Migration runs
once: subsequent launches or app upgrades do not overwrite newer profiles or
resurrect deliberately deleted files. An interrupted copy can be retried; a
malformed JSON state file stops initialization instead of being silently reset.
Symlinked legacy files are rejected; an explicit import must name the actual
data directory rather than a directory of links.

On this Mac, the eight existing files were migrated and verified on 2026-09-21.
The original checkout files remain as the migration-time backup; they are no
longer the active settings. Future JSON saves and camera-calibration saves are
atomic and keep the previous file as a sibling `.bak`. Git does not back up the
new data folder; include it in normal Mac backups. Older code versions still
use their old paths, so they will not see scans made by the migrated app.

Old videos are not copied or moved, and their existing fixture paths still work.
Custom maps are not copied either: saved relative paths resolve against their
original project when available. Keep those map assets in place. Legacy CSV
references into `mini_captures` resolve to the copied captures, so those scans do
not depend on the original checkout remaining present.

Imports and the offline UI preview do not initialize or modify user data. These
camera-free commands are available for inspecting or explicitly migrating a
legacy installation; they are not required again on this Mac:

```bash
python app_paths.py check
python app_paths.py migrate --legacy-dir /path/to/old/Sarween-Python
```

Set `SARWEEN_DATA_DIR` to an absolute directory **before starting Python** for an
isolated development/test profile. An override does not automatically import
the checkout's data; use an explicit `--legacy-dir` only when wanted. Explicit
regression `--profiles` and fixture profile paths remain supported. Default live
and replay profiles now come from the user-data folder. Stop Sarween before
manually restoring a backup or changing that folder.

## Recording Review And Demo

Open Sarween and choose **Try demo**, **Open recording**, or **Start live session**.
The demo is a bundled 24-second synthetic five-mini video generated by
`scripts/make_demo.py`. Its pixels run through the same detector and tracking
engine as a live session. It is not a claim about real-camera accuracy and
contains no personal footage. After tracking analysis finishes, demo playback
starts automatically with the video and tracked map side by side. Playback,
seeking and speed changes share one video clock. **Show review tools** reveals
the analysis settings and movement review controls when needed.

While signed out, the whole home window is an email/code sign-in screen. The
library and session controls appear after approval. Logout, confirmed revocation
or offline-permission expiry returns to that screen; the signed-in account row
keeps logout and offline time remaining visible.

The recording window works without a camera, TV, or Foundry. Replay never sends
token moves to Foundry and does not replace active mini scans. A matching
`.tracking.json` supplies registration/grid information; a matching
`.profiles.json` supplies the recorded profiles. For older or external videos,
check Columns, Rows, marker mode, and **Choose profiles** before **Analyze**.
Keep a Sarween recording together with its sidecars and checkpoint directory.

- Play/pause, frame stepping, the time slider, and playback speed support review.
  Selecting a transcript row seeks to its video timestamp.
- **Detected** contains detector output (or the original live log before analysis).
  **Foundry** contains verified acknowledgements/failures when recorded, not just
  commands that Python attempted to send. Older footage has no acknowledgement
  history; an empty tab does not establish success or failure.
- **Reviewed** contains separate human labels. The plus button adds a missed
  movement; the pencil edits a label; check confirms a prediction; cross rejects
  a prediction or removes a human label. Tooltips name each icon. Initial
  acquisitions have no known From cell. Viewport remaps are distinguished from
  physical detection with the Cause field.
- **Review complete** is a human assertion that the whole clip was inspected.
  **Score** compares raw predictions against reviewed labels; rejected predictions
  remain in the raw track and count as extras when appropriate. Confirmed
  predictions are labeled as such, never called independent ground truth.
- **Export transcript** writes CSV, text, and JSON. **Export regression case**
  creates a runnable case after review is marked complete. The case points to
  the local video/profiles/timeline; it is not a portable video bundle.
- **Map state** is a lightweight faceted top-down grid with colored minis,
  coordinates, registration status, and dimmed lost positions. It follows the
  analyzed timeline when seeking. The live control panel also has a **Map** tab.
  This first version is 2D, not a 3D reconstruction or Foundry fog renderer.

Reviews, automatic transcript copies, and immutable analysis-run JSON files live
under `~/Library/Application Support/Sarween/Reviews/`. Reanalysis preserves human
labels, resets run-specific decisions, and clears Review complete. Changing the
source video's path, size, or modification time starts a new review identity.
CSV exports escape spreadsheet-formula prefixes in user-provided names.

Only one recording worker is active per app window. The default wall-clock limit
is 180 seconds (adjustable up to 900). Cancel, window close, starting a live
session, or app exit kills and reaps the owned worker. OpenCV parallel workers
are disabled; the child additionally enforces its own deadline and checks that
its parent still exists. Analysis stops at EOF, never loops in the background.

New V3 recordings write schema-2 timelines with per-frame capture/tracking
clocks, recorded controls, motion thresholds, initial tracking/background state,
profile changes, and background-recapture checkpoints. Numeric arrays are NPZ
with JSON metadata, no pickle, relative-path/hash checks, and size limits. Replay
restores the initial state and recapture actions rather than starting with a
blank tracker. Times shown to reviewers remain encoded-video positions; original
monotonic clocks are used only for tracking decisions. MP4 is lossy, so this is
not a guarantee of bit-identical pixel processing. Calibration phases lacking
complete clocks are explicitly reported as partial fidelity.

Stopping a new recording also writes sibling `.movements.csv`, `.movements.txt`,
and `.movements.json` transcripts. These are detected/acknowledged events, not
an exhaustive list of physical movements: missed moves must be added during
human review. Old footage stays compatible, but its missing timing, recaptures,
profiles, and starting state cannot be reconstructed retroactively.

Useful checks:

```bash
python tracking_regression.py check demo/five_minis.case.json --timeout-seconds 60
python tracking_regression.py check tests/fixtures/tracking_cases.holdout.json --timeout-seconds 45
SARWEEN_GUI_TESTS=1 python -m unittest tests.test_replay_workflow
```

The synthetic demo passes 11/11 scripted events. The separately visually labeled
first 8.5 seconds of `sarween_rec_20260911_141952.mp4` pass 2/2 events with no extras.
The latter is explicitly a sampled, single-mini prefix; later poorly registered
footage is not silently labeled or included in that score. Neither substitutes
for a real five-mini, fog-on acceptance run after returning home.

## Deprecated Files

Retired trackers, old debug tools, superseded builds and obsolete generated
caches were moved into `deprecated/` without deleting their contents. See
[the archive index](deprecated/README.md) for original paths and recovery notes.
The current 0.2.0 app and installer remain in `dist/`. Recordings, saved scans,
calibration data, map/marker assets and the Auto-wall source were left in place.
The current app still uses `mini_tracking.py`, `ui_preview.py` (UI test helpers),
and the remaining `app_bundle_overrides/`; those are not retired.

Future builds exclude the archive and no longer bundle the old band/blob
trackers. The already-verified 0.2.0 installer is unchanged; this cleanup does
not require reinstalling it. Historical build paths in the reliability review
now resolve under `deprecated/builds/` as documented in the archive index.

## Backlog

- Polish the Mini Library portfolio editor and device-setup experience. The
  real hardware-optional home window, connection status, setup cancellation,
  and separate live diagnostics tab are implemented. Hardware setup still uses
  the existing device scan and selection dialog.
- Validate the personal DMG with the real camera/TV setup after returning home.
  The Apple Silicon app, writable-data migration, and drag-to-Applications DMG
  are built and startup-tested. Developer ID signing, notarization, Intel or
  Universal distribution, and older macOS testing remain open.
- Add a pre-gridded image setup mode that derives the Foundry grid dimensions,
  scale, and alignment from grid lines already baked into a map image.
- Evaluate how verified portfolio samples should update live color thresholds
  without reintroducing fog-of-war false positives. Add explicit room-lighting,
  map-brightness, and fog labels plus conservative outlier review.
- Add import/export and archive controls for minis that are not part of the
  default five-player roster.
