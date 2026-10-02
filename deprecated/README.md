# Deprecated Archive

Archived on 2026-09-21. Nothing was permanently deleted. Files were moved with
their current contents, including uncommitted edits. This directory is not an
application package, a build input, or a runnable alternative installation.

## Source And Tools

| Current location | Original location | Why archived |
| --- | --- | --- |
| `tracking/band_tracking.py` | `band_tracking.py` | Older HSV band engine; current entry point uses V3/TrackingEngine. |
| `tracking/alt_band_tracking.py` | `alt_band_tracking.py` | Alternate older band engine; no current runtime caller. |
| `tracking/blob_tracking.py` | `blob_tracking.py` | Older blob engine; no current runtime caller. |
| `app_bundle_overrides/band_tracking.py` | `app_bundle_overrides/band_tracking.py` | Frozen-app copy of the retired engine. |
| `app_bundle_overrides/alt_band_tracking.py` | `app_bundle_overrides/alt_band_tracking.py` | Frozen-app copy of the retired alternate engine. |
| `tools/hsv_visualizer.py` | `hsv_visualizer.py` | Diagnostic for legacy HSV profiles; current ring profiles use Lab samples. |
| `tools/resource_path_snippet.py` | `resource_path_snippet.py` | Old compatibility example; active code imports `app_paths` directly. |
| `debug/` | `Debug Scripts/` | Old 4x4-marker test, fixed camera-index test and obsolete WebSocket protocol example. |
| `legacy/` | `Deprecated/` | All five files from the already-existing archive, preserved together. |

Archived debug programs may open hardware, bind a port, or reference old token
IDs. Do not run them as current tests. Use the supported app and `tests/` instead.
To experiment with an old engine, restore it in a separate Git worktree and
check its dependencies first; moving it back is not an automatic feature toggle.

## Builds And Generated Files

- `builds/legacy-build/` was the root `build/` directory from the older in-place
  PyInstaller workflow.
- `builds/dist/` contains the old root `dist/Sarween` and `dist/Sarween.app`, the
  0.1.0 DMG, and superseded build folders `Sarween-j9k7r73g`, `Sarween-kp6uryn8`,
  `Sarween-knihwx1m`, `Sarween-mnpq29dl`, and `Sarween-zqui_5fu`.
- `generated/blended_cache/` and `generated/blended_output.jpg` were old root
  artifacts. Current calibration writes generated assets to Application Support.

The verified current release remains at `dist/Sarween-hn007ne9/Sarween.app`
and `dist/Sarween-0.2.0-Apple-Silicon.dmg` relative to the project root. Its bytes
were not changed. Older reports retain their original absolute build paths;
replace an old `dist/` prefix with `deprecated/builds/dist/` to locate them.

Archiving moves files; it does not reclaim disk space. Builds, generated files,
and the previously ignored `legacy/` archive remain ignored by Git. Some old
generated artifacts were already tracked despite `.gitignore`; Git therefore
shows their former paths as deletions. Their local contents still exist here
and their earlier revisions remain in Git history. Archived source/tool files
are not ignored and can be included in the next cleanup commit. No Git index or
commit was changed by this cleanup.

Recordings, sidecars, saved profiles/calibration, map and marker assets, the
Auto-wall download, current test fixtures, and current builds were not archived.
