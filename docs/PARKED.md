# Sarween parked work

Updated 18 November 2026, Asia/Bangkok.

These ideas remain available, but are outside the current development plan.
Each has a reason to revisit it. The [backlog](BACKLOG.md) contains planned work.
Parking an item does not remove its existing implementation or authorize deleting
code.

1. **Printed markers and stronger CPL filtering.** Optional fallback; the print
   sheet and registration software are implemented, and the user has the printed
   cards and CPL. The current on-screen configuration has achieved 4/4 lock at
   roughly the 20% camera-brightness reading. Revisit paper markers if stronger
   LCD suppression, screen clutter or registration reliability warrants the extra
   setup. Their independence from LCD output can help when screen markers become
   too dark. Physical placement and dark-LCD acceptance remain untested.

2. **Further physical target comparisons.** Compare 38 mm and 32 mm rings;
   reflective stickers with flashlight on/off; and the contribution of CPL,
   stickers and rings independently. Revisit after a reliable baseline, or when
   simplifying required accessories. No-ring and sticker-only tracking are
   research into another detector or mode: the current verifier requires visible
   ring-band evidence. CPL effectiveness has not established that stickers or
   flashlight use are unnecessary.

3. **White or grey screen flashes for recovery.** Optional fallback. Revisit if
   passive tracking cannot resolve important uncertain cases, after assessing
   gameplay interruption and whether the flash supplies useful physical evidence.

4. **Alternative tracking hardware and large detection models.** Additional
   phones, LiDAR, infrared/retroreflective approaches and learned object detectors
   remain research options. Revisit if the single-camera ring approach cannot meet
   measured requirements across representative setups.

5. **Physical tap selection and richer game overlays.** Physical taps remain
   paused and off by default. Tap/double-tap interactions, attack ranges, aura
   ranges and richer action overlays can return after tracking is reliable and
   the user chooses the next gameplay features.

6. **Seamless map panning and zooming with minis present.** Revisit after the fog
   baseline is stable. The current shared tracker deliberately requires rebinding
   after unrelated viewport or registration changes; preserving a playable
   physical layout through map repositioning remains future work.

7. **Mini Library and profile workflow polish.** Naming, portfolio editing,
   import/export, archive controls and broader roster management may have product
   value later. Revisit when those workflows are needed. This does not revive
   legacy manual color-ID or brightness-curve tuning as a requirement for the
   current six-ring tracker.

8. **Automatic setup from map images with baked-in grids.** Detect existing grid
   lines and derive scale/alignment from a map image. Revisit after the core
   tracking and normal Foundry setup workflows are reliable.

9. **Broader Mac distribution and compatibility.** Intel/Universal builds and
   older macOS support are deferred until the current Apple Silicon release path
   and its hardware acceptance are complete, or user demand changes priorities.

10. **Broader historical video audits.** Review archived recordings when a
    specific failure, compatibility requirement or unresolved question warrants
    it. Pre-CPL or older OBS footage is not the primary acceptance material for
    the current USB/CPL/ring setup. Keep the already-reviewed CPL regression clips;
    review new current-setup clips as part of live acceptance. A large archive
    audit is not a prerequisite for cafe work or the next live test.

## Work excluded from both lists

Further OBS-watermark tuning for the primary iPhone feed and expansion of legacy
manual color/brightness learning are superseded by the native USB feed and
automatic ring sampling at binding. Existing compatibility can remain without
making it an active project. Automatic persistence of mini-to-token assignments
was intentionally dropped in favor of manual confirmation and should return only
if that preference changes. Completed feature implementations belong in project
history or acceptance checks, rather than either development list.
