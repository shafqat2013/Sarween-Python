# Vacation Dataset Checklist

## First: a short smoke test

1. Exit Sarween if it is open. Reload the Foundry world to load Sarween 1.3.0.
2. Open a square-grid test scene, preferably your familiar 48x27 scene. Do not
   introduce OBS yet. Put Red, Blue, Yellow, Green, and White tokens on it.
3. Keep all physical minis off the TV. Start Sarween:

   ```bash
   conda activate py_arm
   cd /Users/shafqat/Batcave/Sarween-Python
   python main.py
   ```

4. Select the TV, iPhone camera, and Foundry mode. Confirm Connected and Locked.
   All four markers must be visible to acquire the initial lock.
5. Click the people icon in Foundry's Sarween bar. Choose only Red for this first
   check, choose the correct Red token, and set the fog label to match the scene.
6. Click Start camera recording. Do not click Record in the Python panel too.
7. After the empty-board baseline, place Red on the highlighted A0 cell. Move
   your hand away and click Placed promptly. The Red digital token should move
   there. Do not wait for tracking to move it: predictions are paused.
8. Wait for A1, move Red there, and click Placed. Then click Finish to save a
   partial test. Do not resume tracking yet.
9. Tell Codex this smoke test is done so the four saved files can be checked
   before spending time on the full dataset.

The guide selects the chosen digital tokens for token vision automatically.
Leave them selected during capture. It restores their original positions and
selection after saving. Use a test scene anyway; a browser crash/reload can
prevent restoration.

## Full dataset

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

## Saved files and offline work

Every guide produces four files together in the Sarween-Python folder:

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
python tracking_regression.py check sarween_rec_TIMESTAMP.case.json
```

No TV, Foundry, or iPhone is needed for replay. File playback stops at EOF.
Run one case at a time, with a wall-clock timeout for any custom analysis.
