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

## Unblocked work authorized next

1. Preserve every movement event in a durable run-scoped journal, independently
   of the bounded frame recording and one-second status export. The current
   status export can omit events between publication ticks.
2. Add a small supported camera-preset selector. Keep the validated 1080p/15 mode
   as default; treat 30 FPS and higher resolution as unvalidated until measured.
   Resolution/format geometry changes require their own verified calibration.
3. Include accepted detector-family disagreement in location uncertainty rather
   than selecting the most optimistic allowance. Preserve border abstentions.
4. Prepare explicit fog-transition holds and auditable test guidance. Existing
   observer detection remains a substantially static-background experiment;
   printed markers or polarization alone do not prove fog robustness.

## Local evidence locations

The capture Mac retains artifacts/phone-camera-20261106/verification.json and
actual-usb-feed.json, the methodology review under
artifacts/methodology-review-20261105/, and prior replay/optical evidence under
artifacts/poc_single_mover_20261105/. These are not required to build the source.
