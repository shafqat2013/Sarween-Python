# Sarween-Python
Camera Vision App for TTRPGs

## Foundry Viewport Tracking

Sarween's Foundry module displays ArUco IDs 10-13 as fixed overlays in the four
screen corners. They remain visible above fog while the GM pans or zooms even on
maps much larger than the TV. The module sends Foundry's exact canvas transform
to Python, so camera positions are converted to scene grid cells without moving
or nudging the map into alignment.

After the module connects, live Foundry sessions use the viewport markers.
Recorded regression videos continue to use the legacy scene-tile markers 0-3.
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

When recording is enabled, Sarween creates a video and timeline sidecar:

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
case uses prompted-to-confirmed time windows plus settling time; it references
the current `combo_profiles.json`, so improvements can be evaluated offline.

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

## Mini Library

The control panel's `Mini Library` button opens the player-mini roster. It shows
all five planned ring colors, whether each mini has a usable tracker profile,
how many verified appearance samples are saved, its token assignment in the
active Foundry scene, live recognition confidence, and its last known cell.

`Scan selected` waits up to 15 seconds for the chosen mini to move, then uses
the existing known-position calibration path. A successful scan updates the
live tracking profile and adds a verified, deduplicated portfolio sample.
`Full brightness scan` retains the existing multi-level TV calibration. Its
valid Lab curve points are imported into the portfolio automatically the next
time tracking starts. Existing `red10` data is migrated without changing its
tracking thresholds.

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

## Backlog

- Add a pre-gridded image setup mode that derives the Foundry grid dimensions,
  scale, and alignment from grid lines already baked into a map image.
- Evaluate how verified portfolio samples should update live color thresholds
  without reintroducing fog-of-war false positives. Add explicit room-lighting,
  map-brightness, and fog labels plus conservative outlier review.
- Add import/export and archive controls for minis that are not part of the
  default five-player roster.
