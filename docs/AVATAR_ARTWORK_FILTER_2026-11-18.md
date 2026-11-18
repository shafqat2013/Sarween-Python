# Experimental avatar artwork filter

Updated 18 November 2026, Asia/Bangkok.

Implemented on `codex/avatar-artwork-filter`, separately from the saved
`ring-tracking-candidate-2026-11-18` and live-confirmed static-map Working V1.
Fresh live fog acceptance remains pending. This is an additional rejection cue;
it does not establish reliable tracking of arbitrary artwork.

## Behavior

The Foundry module automatically exports small isolated static token sprites,
including their tint, rotation, mirroring and alpha. It renders temporary owned
sprites and never extracts or mutates a live token mesh. Changed artwork clears
old authority, and scene, connection and viewport changes invalidate the cache.
The catalog refreshes independently of video/reference recording.

Every physical candidate still passes the existing size, ring-band and color
checks. Artwork comparison runs only when observations compete, such as an
apparently occupied old anchor plus a new arrival. Strong footprint, color and
spatial-pattern agreement can reject the exposed avatar. A token's position is
never masked wholesale. Identity continues to come from one unambiguous mover.
Physical rings overlapping artwork remain candidates when the full patch cannot
be explained by that artwork.

Unavailable, stale, unsupported, generic radial or uncertain artwork supplies no
veto. Dynamic token rings, custom visual effects, animated textures and faded
sprites are skipped. Only bounded static templates are supported; an elaborate
avatar is allowed to remain unknown. The matcher checks small local patches,
caches features and caps comparisons at 96 per ambiguous frame. Reaching the
budget keeps unchecked candidates; it does not guess.

Catalogs contain at most 32 thumbnails, each at most 96 pixels per side, within a
1 MiB validated payload. Journals save catalog snapshots as content-addressed
assets, include them in recording byte limits and verify them during replay.
Replay never fetches current Foundry artwork to reinterpret historical inputs.

## Evidence and limits

- Independent composites made from six actual Foundry SVG assets cover exposed
  artwork, dimming, same-color physical rings, overlap and partial occlusion.
  Across 48 cases, seven of eight eligible exposed cases were rejected, with no
  physical/occluded candidate rejected. A dimmed sword remained unknown. Running
  the cases with all six templates loaded produced the same result.
- An isolated probe of seven saved camera frames used automatically generated
  SVG/tint templates and detector-proposed centers. One of five exposed-avatar
  frames was rejected; four remained unknown. No physical candidates were
  rejected in those frames. The ring verifier was deliberately bypassed for this
  probe because it already rejects the recorded small avatar. Asset metadata
  comes from the current saved world; historical appearance is inferred.
- Integrated tests inject competing proposals while using real pixel matching.
  They check move confirmation after rejecting exposed artwork, physical overlap,
  missing identities, unchanged single-mover behavior, changed geometry and
  corrupt caches. They establish the software contract, not camera accuracy.
- Existing recorded movement regression and the full software checks are listed
  in the [validation record](validation/2026-11-18-avatar-artwork.json). Historical
  journals contain no automatic catalogs, so their success cannot prove the new
  artwork veto fired.

The isolated real-camera matcher probe used about 1.1 ms median CPU time. A
separate synthetic worst-case probe reached about 11 ms at the comparison cap.
These measurements exclude camera capture, catalog export, the ring detector,
recording and Foundry delivery. They do not establish end-to-end gameplay speed.

## Next acceptance

At the table, restart the source app and reload the matching Foundry module.
The old module remains compatible but supplies no artwork catalog. Keep the
current rings, camera and CPL fixed, configure fog before binding and follow the
[normal-session acceptance plan](PHYSICAL_ACCEPTANCE.md). Record a short session
with visible avatars, including a physical mini moved away from an exposed
avatar and a stationary mini overlapping artwork.

Check physical moves independently of the event log, and inspect each decision's
`avatar_artwork` evidence to distinguish a real veto from a ring-verifier result
or an abstention. Preserve failures before changing settings. No new live
milestone should be named until this feedback-loop test passes.
