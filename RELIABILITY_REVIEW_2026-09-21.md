# Sarween Reliability Review

Date: 2026-09-21
Branch reviewed: `app-bundle`, HEAD `3cda77a`, including the uncommitted offline-regression improvements.

## Findings

Priority P1 means repair before relying on the affected workflow. P2 means an important correctness or validation gap. These findings are based on the current code, not a claim that every historical live failure has been diagnosed.

### 1. [P1] Detected movements can be lost before Foundry applies them

The tracker updates `last_emitted` before calling the output callback. The output loop removes a command from its queue even if the mini is not assigned yet; assignment does not resend its latest position. A failed socket send also loses that dequeued command. Foundry sends `tokenMoveError` when its update fails, but Python ignores that message. There is no successful-move acknowledgement.

Consequently, a correctly detected stationary placement can remain absent from Foundry until the mini moves to a different cell. The log saying "Sending move command" does not establish delivery.

Evidence: [tracker emission](/Users/shafqat/Batcave/Sarween-Python/v3_tracking.py:1476), [queue consumption](/Users/shafqat/Batcave/Sarween-Python/foundryoutput.py:1042), [assignment handling](/Users/shafqat/Batcave/Sarween-Python/foundryoutput.py:1146), [ignored messages](/Users/shafqat/Batcave/Sarween-Python/foundryoutput.py:1217), [Foundry error reply](/Users/shafqat/Batcave/Sarween-Python/module.js:1838).

Reproduction: queue Red at B2 without an assignment, let the assignment request occur, then provide the binding. Result: zero moves sent, zero moves retained. A separate module probe confirmed the `tokenMoveError` response to a rejected update.

Repair: separate detected/desired position from confirmed Foundry position. Use command IDs, acknowledgement/error handling, bounded retry, and latest-position reconciliation after assignment or reconnect. Scope commands to a scene/session; do not replay obsolete queued paths. Show pending/failed delivery in the UI.

### 2. [P1] Quick scan can learn the miniature body instead of its ring

Runtime tracking identifies colored ring regions. Quick scan instead selects the largest motion blob and samples a small circle at its center. It neither restricts sampling to a known target square nor isolates the ring. The selected mini's name provides only a broad color sanity check. A scan that passes immediately replaces the live profile and is stored as a verified "known-position-scan", although no position was supplied.

Evidence: [sampling](/Users/shafqat/Batcave/Sarween-Python/v3_tracking.py:824), [profile replacement and verification](/Users/shafqat/Batcave/Sarween-Python/v3_tracking.py:1090).

Reproduction: a synthetic bright-red ring surrounding a muted-red miniature body passes the scan's color check. The saved Lab value exactly matches the body; its distance from the ring is 75.26, versus the resulting ring-match threshold of 20. A gray body would be rejected by the existing guard, but that does not make the sampler ring-aware.

Profile preservation is also unsafe: quick scan discards an existing brightness curve for that mini, while full calibration constructs a new empty profile dictionary and replaces the entire file. Scanning a subset can remove other minis. See [full-calibration save](/Users/shafqat/Batcave/Sarween-Python/mini_calibration.py:548).

Repair: explicitly target one mini in a known square, identify its visible ring pixels over several settled frames, preview the sampled region, and reject ambiguous captures. Store new scans as candidates before promoting them; retain known-good profiles, unrelated minis, brightness curves, and a rollback copy. Do not describe automatic imports or unreviewed samples as user-verified.

### 3. [P1] Full brightness calibration uses incompatible marker IDs

The calibration screen draws marker images 0-3. Its camera session receives no explicit marker mode. With the saved setup in Foundry mode, that session expects viewport markers 10-13. When the calibration image fills the TV and covers Foundry, it cannot acquire the required lock.

Evidence: [drawn markers](/Users/shafqat/Batcave/Sarween-Python/mini_calibration.py:131), [session creation](/Users/shafqat/Batcave/Sarween-Python/mini_calibration.py:411), [automatic marker mode](/Users/shafqat/Batcave/Sarween-Python/cv_core.py:455).

Reproduction: generate the real calibration image and feed it into the actual marker solver. Legacy mode detects four required markers and computes a homography; viewport mode detects zero required markers and cannot compute one.

Repair: make the calibration display and camera use one explicit, consistent registration scheme. Prefer preserving the Foundry viewport overlay and transform while controlling calibration brightness there. A minimal legacy-mode repair must also use matching calibration geometry; changing IDs alone is insufficient.

### 4. [P1] The bundle helper can overwrite or discard source changes

`build_bundle.sh` copies override Python files and build configuration into the working tree. After a successful build it runs `git checkout` on four source files, discarding local changes to those files. If the build fails, `set -e` prevents that cleanup, leaving copied overrides in place. The specification and builder copies are not restored even on success.

Evidence: [build helper](/Users/shafqat/Batcave/Sarween-Python/build_bundle.sh:3).

This was verified by inspection only. The destructive helper was not run.

Repair: build from a temporary staging directory without modifying the checkout. Avoid checkout-based cleanup. Then separate packaged resources from writable user data: profiles and library currently use paths beside their Python modules, and the spec bundles the developer's current profiles, hardware settings, and token mapping. See [bundle data](/Users/shafqat/Batcave/Sarween-Python/sarween.spec:18) and [library storage](/Users/shafqat/Batcave/Sarween-Python/mini_library.py:14). An application upgrade must not replace a player's scans or assignments.

### 5. [P2] Multi-mini collision filtering can discard a valid third mini

When candidate A loses to B, the inner loop continues comparing the now-rejected A against later candidates. A can therefore eliminate C even when surviving candidate B and C do not collide. Results depend on profile iteration order.

Evidence: [collision filtering](/Users/shafqat/Batcave/Sarween-Python/v3_tracking.py:650).

Reproduction: at a 40-pixel grid size, put A at x=90 (score .8), B at x=106 (.9), and C at x=74 (.7). Collision radius is 20. B and C are 32 pixels apart and should survive after A loses. Current result: only B survives.

Repair: choose survivors consistently from strongest to weakest, comparing only against surviving candidates. Add order-permutation and three-to-five-mini tests before claiming multi-mini readiness.

### 6. [P2] Replay and live tracking do not run the same state machine

They share image processing and `detect_minis`, but separately implement scene resets, timeouts, anchors, consensus, and remapping. Live tracking clears state when scene geometry or capture pause changes and uses a 0.75-second lost timeout for the selected mini. Replay lacks the corresponding scene reset and uses its default 2-second timeout. A scene change in a replay can remap an old scene's physical anchors instead of clearing them.

Evidence: [live scene reset](/Users/shafqat/Batcave/Sarween-Python/v3_tracking.py:1013), [live selection timeout](/Users/shafqat/Batcave/Sarween-Python/v3_tracking.py:1278), [replay remapping](/Users/shafqat/Batcave/Sarween-Python/tracking_regression.py:297), [replay timeout](/Users/shafqat/Batcave/Sarween-Python/tracking_regression.py:336).

Recording also omits selection and background-recapture events. Video is written at a fixed nominal FPS, whereas live timeouts use elapsed wall time. Variable live frame rate therefore cannot be reconstructed exactly from frame/FPS. See [recorded state](/Users/shafqat/Batcave/Sarween-Python/foundryoutput.py:402) and [video writer](/Users/shafqat/Batcave/Sarween-Python/cv_core.py:645).

Repair: extract one headless per-frame tracking state machine used by the live UI and replay adapters. Inject its clock and inputs. Record frame timestamps, selection, recapture, scene transitions, and relevant session initialization for future footage. Add live-adapter/replay-adapter equivalence tests. Mark missing historical inputs explicitly instead of pretending old recordings contain them.

### 7. [P2] Map panning can spend movement allowance

The tracker labels a stationary mini's coordinate change as `viewportTransform` in the recording, but passes only mini ID and cell to the live output callback. The resulting move command has no cause field. Foundry's `updateToken` hook then charges that coordinate adjustment as player movement.

Evidence: [stationary remap](/Users/shafqat/Batcave/Sarween-Python/v3_tracking.py:1263), [move payload](/Users/shafqat/Batcave/Sarween-Python/foundryoutput.py:1064), [budget accounting](/Users/shafqat/Batcave/Sarween-Python/module.js:1136).

Reproduction: deliver the current remap-shaped payload through the real module message handler and token-update hook. A selected stationary mini acquires 5 feet of used movement after a one-cell map shift.

Repair: carry movement cause through the protocol. Rebase the selected token's movement state when the map moves without consuming its allowance. Distinguish physical movement, viewport remapping, undo, capture placement, and resynchronization.

### 8. [P2] Multiple Foundry GM windows have no display ownership

Every connecting client gets send/receive loops consuming shared queues, and each may overwrite the global scene and viewport transform. The module restricts connection to GMs but does not designate one window as the camera-visible display. Two GM windows on the same local server can compete for output and supply different registrations.

Evidence: [server handler](/Users/shafqat/Batcave/Sarween-Python/foundryoutput.py:1221), [module connection](/Users/shafqat/Batcave/Sarween-Python/module.js:1861).

This is a conditional code-path finding, not a claim that two windows caused the previous failures. No multi-window live test was run.

Repair: designate an authoritative display client with a connection/session ID and explicitly reject or separate other clients. This should precede a laptop-GM/TV-player/OBS workflow.

## Evidence And Limits

- Current Python test discovery: 72 tests, 71 passed, one intentional opt-in video skip; 1.43 seconds reported by the test runner.
- Both `tests/test_movement_logic.mjs` and `tests/test_module_capture.mjs` passed.
- Four temporary Python probes reproduced the marker mismatch, wrong scan target, collision error, and assignment-time lost move. A temporary JavaScript probe exercised the real module handler/hooks for budget charging and a rejected token update. These are failure reproductions, not fixes or new passing regression coverage.
- No real camera, TV, Foundry connection, application build, or full browser UI smoke test was used. Review subprocesses completed and their asynchronous tasks were cancelled and awaited. Long video analyses were not rerun during this review.
- The existing [baseline report](/Users/shafqat/Batcave/Sarween-Python/tracking_reports/baseline-20260921/report.md) matches 23 of 23 historical expected moves across five cases. Its labels are explicitly `legacy-unverified`; this is consistency with supplied expectations, not independently measured tracking accuracy. Initial detections outside scoring windows are not included in that pass count.
- Inventory found 17 recordings, eight with timeline sidecars, but none with guided-confirmation labels, rendered display references, or capture-time profile snapshots. Current live profiles contain only `red10`. The frozen regression profile is a baseline copy, not a recovered historical scan.
- The new timeout-limited replay worker, data/code fingerprints, independent per-mini event matching, and honest label provenance are useful improvements. They should be kept. They do not fix live/replay divergence or test Foundry delivery.

## What I Would Keep

- Fixed viewport markers plus Foundry's explicit coordinate transform. This is the right separation between camera registration and movable maps.
- Distinct solid-color rings for a small player roster. There is no evidence yet that replacing the entire detector with a larger learned model is the most efficient next step.
- Guided capture that records user-confirmed placements independently from detector predictions, including empty-board and stationary periods.
- A mini portfolio separate from the active matching profile, once sample provenance and promotion are made reliable.
- Rendered Foundry reference capture as an offline experiment, not an unvalidated live rejection gate.

## What I Would Change About Our Approach

We added features faster than we verified the basic scan-to-detection-to-Foundry loop. I would now pause feature expansion and measure each part separately:

1. Registration: is the physical base mapped to the correct cell?
2. Identity: is this the intended ring rather than artwork, a hand, or another mini?
3. Temporal tracking: was a real placement confirmed without stationary false moves?
4. Delivery: did Foundry acknowledge the intended final position?
5. Gameplay: did vision, animation, and movement allowance behave correctly?

A better scan may help, but it cannot repair dropped commands, wrong calibration markers, or duplicate state-machine implementations. Increasing color thresholds or minimum blob sizes against Red alone is not an adequate acceptance criterion for Blue, Green, Yellow, or White.

The tap feature is currently a short-occlusion heuristic near a known mini, not actual hand/finger recognition. I would keep it experimental and disabled by default until labeled tap-versus-pickup footage validates it. Selection should not become a prerequisite for ordinary tracking.

The movement overlay currently represents a selection session, not a complete combat-turn movement ledger. Deselecting and reselecting starts a new allowance. Turn persistence, movement modes, and accurate reachable-cell shapes need an explicit scope before advertising rules enforcement.

Known-background comparison remains promising, but fogged pixels cannot be reconstructed from the map PNG alone. The current canvas snapshots also exclude DOM overlays such as Sarween's movement graphics, and camera/TV latency is not measured. Validate synchronization, display overlays, and color compensation with new paired footage before using residuals to suppress detections. All offline analysis entry points, including `rendered_reference.py`, should inherit the bounded-worker/resource controls.

## Recommended Travel Work

This is an order of work for the remaining 12 days, not a promise that elapsed days alone establish physical tracking quality.

### First: Repair The Confirmed Failures

Checkpoint the existing uncommitted regression work, then make small reviewable fixes with tests. Start with move acknowledgement/reconciliation and calibration consistency. Fix the multi-mini collision loop and pan accounting. Replace the source-overwriting build helper before using it again.

Acceptance: failed or unassigned moves remain visible and reconcile to the latest desired cell; stale scenes do not receive retries; scans cannot silently replace unrelated profiles; calibration markers agree; multi-mini results are independent of profile order; panning consumes zero movement allowance.

### Next: Make Offline Tests Representative

Extract the shared tracking state machine and record its missing inputs. Add fake-Foundry integration tests for disconnect/reconnect, assignment, rejected updates, deletion/rebinding, scene changes, and multiple clients.

Build a small offline labeling/review tool for the existing videos: frame stepping, visible scene-cell overlay, physical placement/settled-time labels, and uncertain/occluded intervals. Labels must come from visible footage or user confirmation, never from tracker predictions. Include stationary and fog-change negatives, and reserve a separate recording as a holdout. Where the footage does not show a cell unambiguously, preserve that uncertainty.

Acceptance: live and replay adapters produce identical state transitions for the same inputs; independently reviewed cases distinguish missed moves, identity errors, extra moves, and response delay; time/resource limits are enforced.

### Then: Reduce Setup Fragility

Make targeted mini scans non-destructive and reviewable. Add a concise preflight showing camera registration, authoritative display, active square grid, token bindings, profile readiness, and token vision readiness. Distinguish "camera locked", "connected", "tracking paused", and "last move acknowledged" instead of relying on a single connected indicator.

Move persistent user data to an application-support directory with versioned migration and atomic saves. Smoke-test packaging from a clean staging directory. Intel/Apple Silicon distribution can follow that foundation; it should not postpone the reliability repairs.

### Prepare The Return-Home Check

Prepare a short Red-only end-to-end smoke test first, then five-mini fog-off and fog-on captures using explicit highlighted placements. Include stationary periods, return to old squares, close neighbors, a separately reserved holdout, panning, scene changes, and deliberate reconnection. Capture taps separately. The detailed multi-mini capture checklist remains useful, but no new physical recording is required while away.

Defer new OBS integration, pre-gridded-map inference, auto-wall integration, attack/aura ranges, double-tap gestures, and automatic portfolio threshold expansion until these acceptance checks are in place.

## Changes Made In This Review

This report only. No application behavior, saved scans, Foundry settings, git branch, or existing user changes were modified. Temporary reproduction scripts were written under `/tmp` and run to completion.

## Follow-Up Implementation: 2026-09-21

After the review, the user approved implementation. The first repair pass addresses findings 1-3 and the single-display safety issue in finding 8:

- Added a scene-scoped, latest-position outbox, versioned acknowledgement protocol, bounded retries, reconnect/reassignment recovery, explicit failure status, and a Retry moves control. Stale command acknowledgements are rejected, and retries at an already-applied position do not replay animation or movement accounting.
- Only one Foundry client may own the relay. Assignment responses are scene-scoped; secondary connections are rejected instead of sharing queues and viewport state.
- Replaced automatic largest-blob scans with an explicit frozen-frame ring-pixel picker and magnified preview. Mixed patches are rejected and applying a sample requires user confirmation. Existing color curves and thresholds are retained unless replacement is explicitly chosen.
- Brightness calibration now explicitly uses its own legacy markers and geometry, samples the labeled ring patch, preserves other profiles, and avoids sampling a stale frame from an earlier brightness step. Profile writes are atomic and backed up. Imported curve points preserve their original brightness indices and are not newly marked user-verified without confirmation.
- Made the control panel scrollable after native UI testing exposed clipped lower controls.

Verification: Python discovery ran 99 tests (96 passed; the video check and two native UI checks were intentionally skipped). Both native UI checks were then run explicitly and passed. JavaScript module/capture and movement checks passed, including acknowledgements, retry idempotence, rejected updates, scene safety, and serialized delivery. Tests used fake Foundry sockets/hooks and synthetic camera images, not a live TV/camera session. No long replay was needed.

The installed local Foundry `module.js` and `module.json` were backed up under `/tmp/sarween-module-before-1.5.0` and updated to 1.5.0. Reload the world before the next live session. No saved scans or world/token data were modified by this implementation.

Remaining work includes the unsafe bundle helper, multi-mini collision filtering, shared live/replay state machine, map-pan budget accounting, independently reviewed footage, and physical validation of the new scan UI. This pass does not establish five-mini tracking accuracy.

## Second Repair Pass: Shared Tracking Decisions

The next approved pass fixes finding 5 and the duplicated decision logic in finding 6:

- Multi-mini collision filtering now sorts strongest-first with deterministic name-based ties and compares only against surviving detections. A rejected candidate cannot discard a valid third mini. Tests cover every ordering of a three-mini overlap chain and a five-mini set, including equal scores.
- `tracking_engine.py` owns anchors, last-seen times, 4-of-6 movement consensus, scene/pause resets, viewport remapping, and tap recognition. Both the actual live loop and headless replay call it. Clocks and controls are explicit inputs; mapping inputs are frozen per frame.
- Scene changes clear old state even while marker lock is lost. Pausing/resuming requires fresh confirmation, selected minis use the same shorter recovery timeout in both paths, and rescanning clears that mini's stale tracking state. Interrupted taps are cancelled when registration is unavailable.
- New timeline snapshots include mini selection and prediction-pause state. Replay consumes these without sending control messages to Foundry. Guided-capture evaluation explicitly ignores the intentional live prediction pause so it can evaluate the captured movements.
- Missing historical controls and nominal-FPS timing are disclosed in console and report output. Full recording fidelity remains open: per-frame wall-clock timing, recording-start tracking/background checkpoints, rescans, and recapture actions still need a capture/replay contract. Existing videos cannot retroactively supply these inputs.

The live/replay integration test runs the actual adapters with identical synthetic frames, scene/pan changes, selection, and pause transitions. It compares emitted movements and the state presented to the detector, not just two instances of the extracted class. Detection thresholds and saved scans were not changed.

Verification: Python discovery ran 119 tests (116 passed; the opt-in full-video check and two native UI checks skipped). Both JavaScript test suites passed. Separately, all five local video cases passed with 23/23 expected movements and no extra scored movements. Every emitted event, including its timestamp, matches the previous baseline. The report's code fingerprints matched the tracking files at the end of this pass. Videos ran sequentially with one OpenCV thread and a 90-second worker deadline; all workers exited and were reaped. Report: [shared-engine replay results](/Users/shafqat/Batcave/Sarween-Python/tracking_reports/shared-engine-20260921/report.md). These are single-mini, historically labeled recordings, not independently verified five-mini accuracy tests.

Remaining priorities: make the bundle helper non-destructive before using it; prevent map panning from spending movement budget; complete recording-state fidelity; independently label existing footage; validate real five-mini tracking and the scan UI after returning home. No hardware session is required for the next code repairs.

## Third Repair Pass: Panning And Build Safety

- Movement cause now reaches Foundry through the live callback, main entry point, and acknowledged outbox. The token-update hook recognizes `viewportTransform` and translates the saved movement path and Undo destinations to the new location, preserving distance already spent and remaining. The next physical move counts normally. Failed updates and duplicate delivery do not modify the budget.
- All four build shell entry points now delegate to a single staging builder. Current source plus the existing frozen-app overrides and spec data are copied to a temporary directory. PyInstaller runs there with an isolated cache/work directory. No source copy-back, Git checkout, or deletion of previous builds occurs. Successful artifacts get unique output directories and SHA-256 input manifests; failure/cancellation/timeout cleans up staging and stops the builder process group.
- The real project passed read-only preflight and a complete staging smoke test (71 inputs, about 218 MiB). Original input hashes remained unchanged, staged hashes matched the manifest, and the temporary output was removed. A full PyInstaller compilation and packaged-app launch were intentionally not performed. The legacy frozen setup and bundled user-data handling still need release-readiness work.

Verification: Python discovery ran 130 tests (127 passed, 3 opt-in skips), and both JavaScript suites passed. Tests cover the actual Foundry handler/hook, routing movement cause through Python, shifted Undo history, retries, real moves following pans, and build success/failure/timeout/interrupt from dirty workspaces. Build subprocess tests use a small fake PyInstaller module, not the real compiler. No long tracking replay was needed for these delivery/build changes.

The installed Foundry module is now 1.5.1. Its previous `module.js`, `movement_logic.mjs`, and `module.json` were saved under `/tmp/sarween-module-before-1.5.1.dEFEDP`; installed hashes match the repository files. Reload Foundry and restart Sarween before the next live check. No world/token data or mini profiles were changed.

Remaining priorities are independently verified footage labels, full capture/replay state fidelity, and real five-mini validation. Movement accounting still follows delivered token updates; reconstructing intermediate paths skipped by latest-position delivery is separate work. The build workflow is now non-destructive, but packaged-app runtime data migration, clean distribution inputs, signing, and platform validation remain open.

## Fourth Pass: Visual Evidence And Travel Preview

- Added a read-only footage extraction tool that uses only ArUco registration and recorded Foundry scene geometry. It saves clean/cell-labeled frames, contact sheets, input fingerprints, and optional manually chosen point-to-cell conversions. A missing marker produces an explicitly unregistered image rather than a guessed grid. Extraction is capped at 120 frames with OpenCV parallel workers disabled and a cooperative deadline.
- Visually inspected the full short recording `sarween_rec_20260911_135922.mp4` at approximately one-second intervals, then denser samples around placements. Circular base positions distinguish the physical Red mini from the blue digital token and its overhanging torso. Checked V16 initially, then J7, AK7, Y14, K21, and AL20. Frozen provenance includes base-center points, exact evidence-frame indices, hand-clear uncertainty intervals, and footage/sidecar hashes. This is sampled assistant review, not exhaustive frame-by-frame user confirmation. Historical labels had already been read, so it is explicitly not a blinded or held-out evaluation.
- Added a separate reviewed fixture without changing the legacy expectations. It includes initial acquisition and rejects extra movements over the entire clip, including stationary gaps. A bounded single-worker replay matched 6/6 positions (initial plus five moves), with zero extra outputs, in about 21 seconds. Relative to first registered hand-clear samples, the five response estimates range from 0.29 to 0.59 nominal-video seconds. Actual touchdown and live-clock timing remain uncertain. Report: `tracking_reports/visual-review-20260921/replay/report.md`.
- The blue digital token and its revealed area remain near the starting location in the inspected images. Therefore this footage is not evidence of successful historical Foundry delivery. It also does not validate moving fog, multiple colors, multi-mini identity, or the new scan workflow. No detector thresholds or saved profiles were changed to obtain the pass.
- Added `ui_preview.py`, which opens the real control panel and Mini Library with explicitly marked sample data. It bypasses setup, camera capture, Foundry networking, and tracking. Hardware/scan/record controls are disabled, UI state is in memory only, and Exit closes the process. The Preview menu provides ready, missing-marker, pending/failed-delivery, and disconnected examples. This gives UI work a hardware-free entry point; it is not the requested redesign yet.
- Recorded the requested personal DMG and UI work in the README backlog. Before producing a DMG, separate writable user data from bundled resources, remove developer-specific release inputs, and smoke-test an actual staged app. Packaging is not yet complete.

Native tests exercised the actual preview/library, all scenario changes, disabled hardware controls, and simulated retry without saving settings. Existing control-panel and ring-picker UI checks also passed. The preview was launched with a 45-second automatic close and exited normally. Screenshot inspection was blocked by macOS Computer Use permissions; no visual screenshot verification is claimed. All extraction, replay, test, and preview processes launched for this pass exited and were reaped.

Fast Python discovery: 138 tests, 134 passed and four intentional opt-in skips. The three native UI checks were run explicitly and passed separately. Both JavaScript suites passed (module tests require Node's `--experimental-vm-modules` flag). The reviewed video was evaluated separately from fast test discovery, with a 60-second worker deadline. The older unverified labels remain unchanged.

## Fifth Pass: Persistent User Data

The user approved the user-data migration as the next packaging prerequisite.

- Added a shared, import-side-effect-free path module. Development and frozen apps now use `~/Library/Application Support/Sarween/` for profiles, scan portfolios, hardware settings, assignments, camera calibration, and legacy capture data. New recordings and sidecars go into `Recordings/`; regenerated display assets use `Cache/`. Bundled assets remain read-only resources. Offline replay defaults and calibration fingerprints use the same active data paths; explicit frozen fixture profiles still work.
- Added a versioned, one-time migration with a file lock, no-overwrite atomic publication, SHA-256 verification, and a receipt. Existing destination files win; completed migrations never reimport older settings or resurrect deleted profiles. Incomplete migrations can resume. Corrupt JSON and symlinked legacy data are rejected rather than reset or followed silently. `SARWEEN_DATA_DIR` provides isolated development/test profiles and disables implicit legacy import when set.
- JSON state and camera-calibration saves now use atomic replacement with a previous-file `.bak`. Legacy capture CSVs remain byte-identical; their old capture paths are resolved to the copied files when read. Saved relative map paths retain their original-project reference when available. Custom maps and old recordings are deliberately not moved or copied.
- Updated source and frozen-app overrides together. Setup rereads preferences after migration instead of using stale import-time defaults. Brightness calibration now generates its marker tiles directly, fixing a fresh-data-folder failure that isolated tests exposed. This does not change ring-detection thresholds.
- Removed personal state from both bundle specs and added a staging guard against known personal-data filenames, captures, and recordings. Build preflight passed with 67 inputs; no full PyInstaller build or DMG was attempted.

Executed migration on this Mac: eight files copied, and all eight destination hashes matched their source hashes. A second read-only verification confirmed every original was unchanged. Receipt: `/Users/shafqat/Library/Application Support/Sarween/migration-v1.json`. No live camera, Foundry server, or hardware setup was started. The older files in the checkout remain backups, not the active data store; Git alone does not back up new scans in Application Support.

Verification: 159 Python tests ran, 155 passed and four opt-in checks skipped. The three native UI checks were run separately and passed; both JavaScript suites passed. Tests cover empty and populated destinations, interruptions, changed/corrupt sources, no clobbering, deleted-file handling, atomic save failure, read-only legacy sources, frozen resource paths, migration-before-service startup, copied capture references, preview isolation, and personal-data exclusion from bundles. Tests used isolated temporary data folders.

One bounded replay used the actual migrated Red profile and calibration. It passed all six reviewed positions, with no extra moves; every emitted event, timestamp, and score matched the pre-migration report exactly. Report: `tracking_reports/user-data-migration-20260921/report.md`. The worker used one OpenCV thread, a 60-second deadline, completed in about 21 seconds, and was reaped. All processes launched for this pass have stopped.

The next priorities after migration were hardware-optional startup and UI improvements, then an actual staged app launch and personal DMG. Signing/notarization, universal distribution, full capture-state fidelity, and real five-mini validation remain separate work. This migration does not constitute a packaged-app or physical tracking acceptance test.

## Sixth Pass: Hardware-Optional App Startup

- `python main.py` now opens the real Mini Library without enumerating devices, binding a server, or processing video. Portfolio browsing is read-only and separates verified samples from imported points. Saved token IDs are explicitly saved assignments, not connection confirmation.
- Connections and Diagnostics have separate home tabs. Starting the relay and starting a live session are explicit actions. The existing hardware setup is shared with a persistent Tk root; cancellation or a session error returns to the home window. Empty camera/display lists disable selection instead of indexing an empty list. Bundled setup no longer creates Tk at import time.
- The live control panel now has Session, Scans, and Diagnostics tabs. Its window is destroyed on session exit, including calibration transitions. Missing profiles no longer trigger a compulsory brightness scan. Carried recordings are released if calibration exits unexpectedly, and common camera-open/background-read failures release the capture handle.
- The Foundry relay has managed start/stop/error state and a joined shutdown. Binding failure is visible rather than disappearing in a background thread. No regression worker starts with the home window.

Verification: 169 Python tests ran, 162 passed and seven opt-in checks skipped. Six native/localhost checks passed separately, alongside the two preview data tests. These cover real home-window layout at 860x620 and 700x460, offline startup, cancelled and failed sessions, empty-device setup in both source and bundle overrides, existing scan dialogs, and stopping a relay with a connected WebSocket client. Both JavaScript suites passed; build preflight passed with 69 inputs. A real `main.py` launch with the migrated user library exited after an eight-second automatic close, without starting a relay or camera. All launched processes were reaped.

No TV/camera tracking, signed bundle launch, or screenshot-level visual verification was performed in this pass. Native geometry and interaction checks are not a replacement for that hardware QA. A DMG has not yet been built.

## Seventh Pass: Personal Apple Silicon DMG

- Built the real PyInstaller app using Python 3.13.9 and PyInstaller 6.16.0 on macOS 26.3/arm64. The final build is `dist/Sarween-j9k7r73g/Sarween.app`. A preliminary build is retained separately; no earlier app or installed copy was replaced. Builds completed in roughly 40 seconds each under 600-second process-group deadlines.
- Corrected the frozen spec to include only `maps/dnd1.jpg`, matching the source spec, instead of 217 MB of custom map assets. Removed an obsolete NumPy private hidden import and retained the build warning report. Verified all 36 staged source hashes against the unchanged checkout inputs and checked the app for excluded personal-state/recording filenames.
- Added an explicit auto-closing packaged startup check. It opens the real home window, loads lazy tracking/scan imports, detects a synthetic ArUco marker, exercises NumPy linear algebra, checks bundled assets and the inactive relay, writes a JSON report, and exits. Tests were launched outside the project with Conda/Python path overrides removed and a 30-second external deadline. Frozen logs now append rather than erase earlier diagnostic output.
- Clean-data and existing-user-data app launches passed. All eight existing state-file SHA-256 hashes were identical before and after. The app's deep/strict ad-hoc signature verification passed. This is local code signing, not Developer ID signing or notarization.
- Added a reusable bounded DMG packager with preserved bundle symlinks, an Applications shortcut, compressed-image checksum verification, temporary-directory cleanup, and no-clobber publication. Unit tests cover content, existing outputs, failed verification, and invalid input/deadlines.
- Created `dist/Sarween-0.1.0-Apple-Silicon.dmg`, 122,539,176 bytes. SHA-256: `68edd79154e9dd374dfaca4843d2408809206629b274957fadf4e9c4567b52e8`. Mounted it read-only, verified its app signature and Applications link, and passed another auto-closing launch from inside the image. The test volume was detached and every launched build/test process was reaped.

Verification: 173 Python tests ran, 166 passed and seven intentional opt-in checks skipped. The three final packaged-app smoke reports, warning report, and input manifest are retained in the final build directory. No camera, live Foundry connection, or long tracking replay was started. This personal build is Apple Silicon only; Intel/Universal distribution, Apple notarization, older-macOS support, screenshot-level visual QA, and physical camera/TV validation are not claimed.

## Eighth Pass: Replay, Transcripts And Map State

- Added Open recording and Try demo to the real home window. The native video reviewer has playback, frame stepping, seeking, speed, mini filtering, editable human labels, confirmation/rejection of predictions, transcript export, regression-case export, and scoring against a completed review. Human labels survive reanalysis; prediction decisions are scoped to an individual run. Raw analyses are retained separately.
- Added a generated, 24-second five-mini video with deterministic color profiles and scripted labels. The real pixel detector reports all 11 scripted acquisitions/moves, with zero extras. The bundle contains this synthetic footage, not personal videos or scans. Demo playback starts automatically while a bounded worker analyzes it.
- Added a faceted 2D map-state tab to replay and the live control panel. Replay seeks through analyzed position snapshots. Positions are dimmed when not actively tracked; registration/coordinates/grid dimensions remain visible. This is not a 3D reconstruction, a fog renderer, or a substitute for the raw camera view.
- New V3 recordings capture per-frame monotonic clocks, selected-mini/pause state, motion thresholds, initial engine/CV state, profile updates, and background recaptures. Repeated scene state is referenced instead of copied into every frame. Capture-time and tracking-time states remain distinct. Checkpoints use numeric NPZ arrays plus JSON, with no pickle, path containment, hashes, and size limits. Legacy footage retains explicit fidelity warnings; lossy encoding and incompletely recorded calibration phases remain limitations.
- New recordings automatically export timestamped CSV/text/JSON movement logs. Reviews also maintain automatic transcript copies under Application Support. Detected, human-reviewed, and acknowledged Foundry events remain separate. Acknowledgements are validated against the outbox, including coordinate mismatches, before entering the confirmed transcript. Missing detections are not presented as a complete physical history.
- GUI analysis runs in one isolated worker, with OpenCV parallel workers disabled, lower scheduling priority, progress, cancellation, a default 180-second deadline, and cleanup on close/live-session start. The child enforces its own deadline and exits if its parent disappears. Replay does not open a camera, connect to Foundry, or overwrite active profiles. Frozen apps dispatch the same worker through their bundled executable.
- Labeled only the first 8.5 seconds of a second real clip, 141952, using raw video and four-marker/scene registration before looking at detector output. The two observed positions were V16 and J7, and replay passes 2/2 with no extras. This is a sparse, single-mini prefix; most later sampled frames lack full marker registration, so they were deliberately left unscored. The existing visually reviewed 135922 clip still passes 6/6 with no extras. No detector thresholds or saved scans were tuned in this pass.

Verification: Python discovery ran 185 tests, 177 passed and eight opt-in checks skipped. Native home, replay, map-layout, scan and preview checks passed separately, including compact windows, seeking, persistent labels and cancellation of an active child. Both JavaScript suites passed using Node's experimental VM-module flag. A real synthetic recording started mid-session, included a background recapture and profile reset, then replayed with identical movement/frame outputs. Worker timeout and cleanup tests passed. Screenshot-level UI verification and physical TV/camera testing were not performed.

Built `dist/Sarween-hn007ne9/Sarween.app` and `dist/Sarween-0.2.0-Apple-Silicon.dmg`. The final bundle's 46 staged inputs match the checkout fingerprints and its deep/strict ad-hoc signature passes. Clean-data and existing-data-copy smoke tests each ran the actual bundled demo worker, found 11 events/five minis, and exited with the worker reaped. All eight original user-state files remained byte-identical. The app contains no personal state or recordings. Older builds/installers were preserved; nothing was installed over an existing app.

DMG: 123,353,869 bytes; SHA-256 `567d6ae4a161d3eb36818b5a2b92820cafec0b128c28f768aa7e774ed8f4cbc2`. Disk-image checksum verification passed. A third smoke test ran directly from the read-only mounted image and again passed all 11 events/five minis; the volume was detached afterward. This remains Apple Silicon only and is not Apple-notarized. Real five-mini fog-on accuracy, physical acceptance of the new recording contract, polished mini portfolios, true 3D, and public distribution/signing remain follow-up work.
