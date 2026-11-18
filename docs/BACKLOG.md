# Sarween backlog

Updated 18 November 2026, Asia/Bangkok.

The goal is reliable, responsive tracking of six physical minis with fog of war,
using one iPhone and one moving mini at a time. This is the authoritative list of
planned work. Optional experiments and later features live in [Parked work](PARKED.md).

## Current position

Working V1 is the user-confirmed static-map milestone saved on GitHub as
`working-v1-2026-11-09`; preserve it. The newer tracking candidate reproduced all
15 manually reviewed physical moves across three saved recordings, with no extra
events. Its Python suite ran 1,336 tests with 18 skips and no failures. A fresh
live fog test remains outstanding. See the
[validation record](validation/2026-11-09-ring-verification.json).

The newer tracking changes and Vision & Fog controls are still local,
uncommitted changes on `codex/after-working-v1` as of this update.

## Planned work

1. **Save the current candidate and refresh the handoff.** Pending; possible at
   the cafe. Commit and push the current work separately from Working V1. Keep
   the untested candidate distinct from a live-confirmed milestone, and update
   historical setup notes before treating them as current instructions.

2. **Add automatic avatar artwork rejection.** Experimental, not implemented;
   possible at the cafe. Cache Foundry avatar artwork and compare ambiguous
   candidate patches against its color and spatial pattern. Reject strong matches
   to exposed avatars without masking the entire token location. Test different
   artwork, scales and physical overlap. A manual proof on one avatar was
   promising. Keep this work separate from the frozen tracking candidate; it
   does not block testing that candidate live.

3. **Run fresh live six-mini fog acceptance and review its recording.**
   Implemented candidate; needs the
   table setup. Keep the current camera, rings and CPL configuration fixed, retain
   on-screen markers and visible avatars, and configure fog before binding.
   Include stationary minis, random single-mini moves, a repeated mover and hand
   occlusion. Check token following and the resulting fog changes. Record one or
   two short clips from this current CPL setup and review physical moves
   independently of Sarween's event log, including missed/extra moves and identity.
   Retain a fresh clip for checking future changes without tuning to its expected
   moves. Video review is part of this acceptance test, not a separate cafe
   project to audit the old archive. Existing CPL development clips remain useful
   regression checks. Preserve any failure before changing settings; save a new
   milestone only after a live pass.

4. **Measure placement accuracy and gameplay response.** Pending; recorded work
   and table testing. Measure neighboring-square uncertainty, actual hand-clear
   response time and camera-to-Foundry latency. Current CPU measurements do not
   establish end-to-end FPS or response time. Keep legitimate border uncertainty
   distinct from a correct exact-square result.

5. **Improve registration usability and redundancy.** Queued after the tracking
   baseline. Add a visual indication of the missing corner, support six markers
   for occlusion tolerance, and implement automatic sizing with a safe recovery
   margin. Logic and previews can be developed at the cafe; reliable detection
   through glare, hands and CPL needs the TV. Manual pattern width, white margin
   and inset controls already exist.

6. **Add independent Foundry controls.** Pending; mostly possible at the cafe.
   Expose Pause/Unpause and Enter/Exit Fullscreen independently of startup. The
   pause protocol exists, but the manual controls do not. Resolve the known macOS
   Accessibility problem with window placement; external-display acceptance
   requires the TV.

7. **Finish live acceptance of setup and existing VTT features.** Implemented;
   needs broader manual checks. Verify cold startup, reuse of an already-open
   world, inline connection progress, unpause and display placement. Check token
   creation from existing actors, hidden Foundry UI and Vision & Fog controls.
   These are acceptance tasks, not features that need rebuilding.

8. **Measure sustained iPhone camera behavior.** Pending; requires the phone,
   and the table for tracking latency. Check delivered FPS, USB reconnection,
   heat, battery use and behavior at the supported presets. Keep camera profiles
   matched to lens and capture format. The native USB feed and camera controls
   are already implemented.

9. **Finish the current Mac release path.** Pending; lower priority than live
   tracking. Verify normal sign-in and email delivery, the packaged Apple Silicon
   app and installer with the real camera/TV setup, and signing/distribution for
   the Mac and iPhone apps. Broader Mac compatibility is parked.

## Maintenance

Update this file and [Parked work](PARKED.md) when priorities or status change.
Move completed work out of the planned list while retaining its evidence in the
project history. Move parked work back only after an explicit decision or a
demonstrated need. Listing work does not authorize starting new code changes.

The native USB camera, saved hardware profiles, grid labels, marker-margin
controls, video-review app, inline setup, Foundry reuse and Vision & Fog are
implemented. Manual mini-to-token confirmation remains intentional. Active
development of OBS-watermark workarounds and legacy manual color/brightness
profile tuning is retired for the primary tracking path; retain existing
compatibility and recovery support until cleanup is separately approved.
