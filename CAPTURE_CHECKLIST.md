# Single-mover target comparison

This study uses grey, unpainted miniatures with 25 mm bases. Only one physical
mini moves at a time. Mini identity is defined by its starting position and the
movement sequence; ring color is not a required identity cue.

The guided capture records camera footage and independent placement labels.
It pauses the existing detector's token commands. Its digital tokens move when
you press **Placed**, so this capture does not demonstrate automatic tracking or
live gameplay response time. Single-mover inference is implemented separately
in the experimental mini observer, which keeps Foundry movement off. It has not
yet passed fresh live/fog acceptance. Automatic white/grey recovery remains
unimplemented. Use this checklist for controlled physical comparisons and the
observer for blind movement tests; their evidence serves different purposes.

| Test | Minis | Physical target | Other change |
| --- | --- | --- | --- |
| 1 | 6 | Rings: 30 mm inside, 38 mm outside; different colors | Baseline lighting; flashlight off |
| 2 | 4 | Rings: 26 mm inside, 32 mm outside; different colors | Same four participant labels and routes as Test 1 |
| 3 | 6 | No rings; reflective dot on top of each mini | iPhone flashlight on |
| 4 | 6 | Same minis and dot placement as Test 3 | Flashlight off |
| 5 | 6 | No rings or dots | Polarizing filter on the phone camera |
| 6 | TBD | Best target configuration from the earlier tests | White/grey display as a recovery experiment |

The nominal flat annular areas are 427 mm² for Test 1 and 273 mm² for Test 2.
Actual camera visibility, including any raised surfaces, is what we will compare.
Keep each ring centered and attached to the mini: a loose ring moving separately
would produce an incorrect position measurement. Dots on top of minis introduce
height and perspective offsets; their visible position must be evaluated against
the independently labeled **base center**.

## First recording: a short Test 1 pilot

1. Fit the six 30/38 mm rings and give the physical minis stable labels A-F.
   Labels are for following the script; the figure bodies may look identical.
2. Keep the camera at the current usable position, with the same capture format
   and calibration. Use the current OBS feed and keep the flashlight off.
3. Restart Sarween with `python main.py --dev-no-login`. In **VTT Controls**, click
   **Reload Foundry** once to load the installed six-participant module. Confirm
   Foundry connected and four-marker lock. Select a square-grid scene where
   moving a vision-enabled token changes the revealed area.
4. Keep the physical minis off the TV. Open **VTT Controls → Capture** and select
   all six rows. Pair each row with a different, one-square Foundry token that has
   enabled vision and a positive sight range. Rows A-F currently use the legacy
   labels Red, Blue, Yellow, Green, White, Orange; they are participant names for
   this experiment and do not require color scans or matching ring colors.
5. Choose **Fog on** and put this in **Notes**:

   ```text
   single-mover-study; test=1; base=25mm; rings=30/38mm; minis=6; grey-unpainted; torch=off; filter=none; pilot
   ```

6. Click **Start recording** once, in the Capture tab. Follow the prompts A0-F0,
   putting each mini's base center on its highlighted destination. After moving
   a mini, clear your hand and click **Placed**. Only move the prompted mini.
7. After all six initial placements, the guide records a stationary hold. Then
   follow A1 and B1. Keep the mini facing the same direction as its initial pose.
8. Click **Stop + save**. This intentionally saves a partial pilot. Leave the
   capture's tracking pause in place. Keep the video and its sidecar files.

Check the pilot before repeating or collecting the other configurations: confirm
all six pieces are visible, the video is complete, labels and geometry are
present, and capture timestamps are usable. Rendered Foundry references also
need checking: earlier recordings contained blank reference images, so the
presence of reference files alone is not evidence that they contain the map.

## Full comparison after the pilot is checked

- Use the full prompted sequence for each physical configuration. Keep the
  camera, map, grid scale, ambient lighting and participant-to-token pairing fixed.
  Change one test configuration at a time. Allow exposure to settle before starting.
- Use the same first four participants for Tests 1 and 2. Their generated target
  cells are the same whether four or six participants are selected. Compare those
  four directly and report the two additional Test 1 pieces separately.
- Update **Notes** with the test number, ring dimensions, flashlight and filter
  state. For Tests 3-4, preserve the exact dots and placement between recordings.
- For Test 5, confirm that the filter still leaves the map's ArUcos readable.
  Record its orientation; do not assume the darkest LCD image gives the best
  miniature measurement.
- Test 6 is a separate recovery test after a promising target is identified.
  Its live control is not ready. Do not substitute the camera-calibration routine
  or a hand-made display flash and call that an automatic recovery test.
- Reserve a repeat of the promising configuration for validation. Tune on the
  first recording and evaluate the repeat without changing thresholds for it.

We will report correct cells, missed and extra moves, identity swaps, and
recovery after occlusion. Video review supplies physical-stop/hand-clear times;
**Placed** clicks contain human reaction delay and are not precise latency labels.
Automatic position confirmation time, command time and Foundry acknowledgement
time are separate measurements. Guided token updates are not automatic-tracker
latency. A separate live test is required after an inference tracker exists.

## General capture reference

### Short smoke test for an individual participant

1. Exit Sarween if it is open. Reload the Foundry world to load Sarween 1.5.1.
2. Open a square-grid test scene, preferably your familiar 48x27 scene. Keep the
   current working camera path, including OBS if that is the selected source.
   Make the corresponding Foundry tokens available on the scene.
3. Keep all physical minis off the TV. Start Sarween:

   ```bash
   conda activate py_arm
   cd /Users/shafqat/Batcave/Sarween-Python
   python main.py
   ```

   If the login backend is restoring, use `python main.py --dev-no-login`
   instead for this local test. The home window will show
   `Developer mode — login bypassed`. Restart without the flag afterward to
   return to normal sign-in.

4. Select the TV, iPhone camera, and Foundry mode. Confirm Connected and Locked.
   All four markers must be visible to acquire the initial lock.
5. Open **VTT Controls → Capture** in Sarween. Choose only Red for this first
   check, choose the correct Red token, and set the fog label to match the scene.
6. Click **Start recording**. Do not click Record in the Session panel too.
7. After the empty-board baseline, place Red on the highlighted A0 cell. Move
   your hand away and click Placed promptly. The Red digital token should move
   there. Do not wait for tracking to move it: predictions are paused.
8. Wait for A1, move Red there, and click Placed. Then click **Stop + save** to save a
   partial test. Do not resume tracking yet.
9. Tell Codex this smoke test is done so the four saved files can be checked
   before spending time on the full dataset.

The guide selects the chosen digital tokens for token vision automatically.
Leave them selected during capture. It restores their original positions and
selection after saving. Use a test scene anyway; a browser crash/reload can
prevent restoration.

### Full dataset

Once the smoke test is verified:

- Record one all-five guide with fog off, Token Vision off, and global lighting
  on. Select Fog off as the recording label.
- Record the same guide with your usual fog on settings and Token Vision on.
  Each chosen token must have enabled vision and a positive range, such as 30ft.
  Select Fog on as the recording label.
- For each run, start with all physical minis off the TV. The guide places
  A0/B0/C0/D0/E0, holds all five stationary for 15 seconds, then alternates
  A1/B1/C1/D1/E1 through A5/B5/C5/D5/E5. Only move the prompted mini; leave the
  others exactly where they are. Center the ring/base in the highlighted cell.
- Confirm each placement after your hand clears the camera. The three-second
  settling interval is automatic; do not move another mini until the next label.
- After the final move, keep every mini stationary until the guide saves.
- Repeat the fog-on recording once in a separate run. Reserve this second
  video for verifying improvements, not choosing thresholds.
- Capture one solo guide for each color, keeping all other minis off the TV.
  These known-position views let us repair bad scans offline. Leave minis facing
  consistently during the main tests; a second solo run may use another facing.

Do not pan, resize the Foundry window, change scenes, change marker size, or
toggle fog during a guide. Geometry changes save a partial recording; use a
separate recording for each setup. If lock is lost, restore it before clicking
Placed. If you placed a mini in the wrong cell, correct it before confirming.

### Saved files and offline work

Every guide produces four files together in
`~/Library/Application Support/Sarween/Recordings/` on macOS. Older recordings
remain in the Sarween-Python folder and do not need to be moved:

```text
sarween_rec_TIMESTAMP.mp4
sarween_rec_TIMESTAMP.tracking.json
sarween_rec_TIMESTAMP.profiles.json
sarween_rec_TIMESTAMP.case.json
```

Keep all four files. The timeline stores geometry, lock changes, planned routes,
confirmed placement frames, token vision metadata, and a profile snapshot.
The case file contains expected movements from user confirmations, not from the
tracker's own guesses. Existing/bad scans do not invalidate the raw dataset.

The generated case uses the current profile library, so tests may fail until
all colors have usable profiles. To check one recording later:

```bash
conda activate py_arm
python tracking_regression.py check \
  "$HOME/Library/Application Support/Sarween/Recordings/sarween_rec_TIMESTAMP.case.json"
```

No TV, Foundry, or iPhone is needed for replay. File playback stops at EOF.
Run one case at a time, with a wall-clock timeout for any custom analysis.
