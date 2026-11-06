# Sarween work checkpoint — 6 November 2026

This checkpoint saves the source accumulated during the recent manual tracking,
Foundry workflow, printed-marker and direct iPhone camera work. Local raw videos,
user-specific profiles, signing assets, generated application bundles and bulky
pixel-analysis artifacts remain on the capture Mac and are not published.

## Implemented

- Inline hardware selection and cancellable Foundry/camera startup in the main
  window; compact control-panel layouts and developer-only source login bypass.
- VTT controls, physical TV-grid calibration, hardware profiles and camera
  calibration with a generated ChArUco board and hands-free capture.
- Foundry startup/display controls, actor-token creation and printed-marker
  registration controls plus the matching A4 print sheet under assets.
- Video-review application sources and reproducible observer recording/replay
  infrastructure; single-mover inference with explicit uncertainty and holds.
- Native iPhone Sarween Camera app and direct USB receiver, without OBS or a
  virtual-camera dependency. Actual portrait pixels, physical rear wide 1x,
  1080p at 15 FPS, one-frame backpressure, reconnection and explicit stale holds.
- Separate USB-camera profile binding: OBS/native calibration cannot apply to the
  direct feed, and USB calibration cannot apply to native/OBS inputs.

## Evidence and limits

The latest desktop suite ran 1,009 tests: 992 passed, 17 skipped. Native device and
simulator builds passed, as did 16 Swift protocol/flow tests. These code checks do
not establish physical miniature-tracking reliability.

A signed development app was installed using a free Personal Team. The production
USB path received 301 real 1080x1920 camera frames over about 20 seconds, observed
15 FPS with advancing sequence/phone timestamps and no receiver-slot drops. A
second sample received 895 frames over 60 seconds. The user then confirmed the
Sarween camera preview works. Sensor-to-display latency, sustained battery/heat,
useful table framing and miniature tracking were not established by those samples.
Torch-on hardware telemetry was not observed during the timed samples.

## Physical tests still blocked

Printed physical markers and a securely mounted rotatable CPL filter are pending.
The next physical acceptance test needs fresh calibration for the direct feed,
six minis moving one at a time, an untouched baseline, adjacency and hand
occlusion, then fog-only changes with stationary minis and moves after fog.
Score identity, origin/destination or justified border abstention, missed/extra
moves and hand-clear response delay. Keep observer output disconnected from
Foundry token movement until the acceptance criteria are met.

## Follow-up work implemented after the GitHub checkpoint

1. Accepted moves now append to a run-scoped movement-events.jsonl independently
   of bounded frame recording and preview refresh. Run/generation IDs distinguish
   rebindings and recording restarts; pending saves and corruption are visible.
   Readable TXT/CSV snapshots retain UTC/source/frame data and expose their own
   export failures separately from the durable JSONL status.
2. Supported presets are implemented on the phone. Standard 1080p/15 remains the
   default; Smooth 1080p/30 retains the exact Standard native format, and native
   Detail 4K/15 is offered only when supported and remains experimental. All modes
   passed synthetic framing tests; 28 Swift tests, device/simulator compilation
   and a signed device build passed. The installed working app was not replaced.
3. Location allowances now include accepted detector-family disagreement without
   moving centers or altering color/shape gates. Across 1,030 saved frames, all
   selected centers/radii/confidences stayed identical. The six-move clip still
   fails strict exact-cell acceptance: two of four required exact destinations,
   with an additional Salmon AM19/AM20 abstention instead of an optimistic AM20.
4. New live observer runs record a hold-known-display-changes-v1 policy. Known
   Foundry visual revisions latch a hold until explicit rebinding; quiet old TV
   pixels do not trigger automatic recovery. Recorded revision evidence replays
   identically. This is not automatic fog separation or recovery.
5. Preview status now identifies the exact JPEG by SHA-256 and capture/frame IDs,
   so readers can detect an image/status mismatch between their atomic writes.

The first GitHub checkpoint is 81f31c9 on codex/iphone-camera-and-tracking.
See PHYSICAL_ACCEPTANCE.md for the remaining hardware tests and
validation/2026-11-06.json for the final portable verification summary.

## Local evidence locations

The capture Mac retains artifacts/phone-camera-20261106/verification.json and
actual-usb-feed.json, the methodology review under
artifacts/methodology-review-20261105/, and prior replay/optical evidence under
artifacts/poc_single_mover_20261105/. These are not required to build the source.

## Final offline verification

From a clean committed-source checkout with isolated user data: 1,040 Python
tests ran, 1,023 passed and 17 skipped. All 69 JavaScript checks passed with the
VM-modules flag. The updated native app passed 28 Swift tests plus device,
simulator and signed-device builds. Nine localhost cross-language checks covered
synthetic Standard, Smooth, Detail and landscape JPEGs. These synthetic checks
are not phone/USB speed, heat, battery or tracking measurements.

The new signed iPhone build is prepared locally for installation when the user
returns. The working phone installation was preserved. Physical markers/CPL and
fresh independently reviewed recordings remain the blockers to physical tracking
acceptance. The saved six-move development clip still fails strict exact cells.

## User-return preset check

The new signed camera build was installed after the user returned. Short actual
USB samples confirmed all three native portrait dimensions. Standard delivered
15.0 FPS, Smooth 15.0 FPS against its 30 FPS target, and Detail 5.0 FPS against its
15 FPS target. Standard/Smooth optical identity matched; Detail differed.
Standard is restored and recommended. The test receiver stopped. Sustained
thermal/battery behavior and physical tracking are still unvalidated.
