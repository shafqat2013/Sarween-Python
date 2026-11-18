# Sarween candidate handoff

Updated 18 November 2026, Asia/Bangkok.

This checkpoint preserves the tracking and desktop-control changes developed
after Working V1. The candidate passes the saved-recording comparisons described
below; fresh live fog acceptance is still pending. Do not label it Working V2 or
replace the live-confirmed Working V1 recovery references.

## Checkpoint identification

- Candidate branch: `codex/ring-tracking-candidate-2026-11-18`.
- Candidate recovery tag: `ring-tracking-candidate-2026-11-18`.
- Working V1 branch: `codex/working-v1`.
- Working V1 recovery tag: `working-v1-2026-11-09`.
- Working V1 commit: `5745cd66bdf5c13b840a3605e8b4d77ac4694861`.

The source checkpoint was pushed to GitHub and verified on 18 November 2026 at
commit `1e760afc3de9ea80151b9b3ca1a5964967c8958b`. The recovery tag also includes
the documentation update recording publication. The candidate and Working V1
remain separate; live fog acceptance is still pending.

## What changed

- All ring proposals pass measured diameter, visible ring-band geometry and
  lighting-tolerant color checks before a representative measurement is chosen.
  Training also retains valid outer-ring measurements instead of allowing a
  precise inner-body fit to discard them.
- Starting-ring appearance compatibility constrains both old-position presence
  and new destinations. Identity still comes from one unambiguous departing mini;
  matching color alone does not assign an identity.
- Unexplained extra rings preserve uncertainty. They cannot silently become
  background, including when they appear during acquisition.
- Small searches around known positions use the same verifier as a full search.
  Missing or ambiguous evidence widens the search in the same frame. Periodic
  full-board audits check for extras, and local-only observations cannot update
  the whole background.
- **VTT Controls → Vision & Fog** exposes scene vision, fog exploration, global
  illumination, placed-token sight settings and selection of a token's TV view.
  Configure these before starting a session and binding minis.
- The maintained [backlog](BACKLOG.md) and [parked list](PARKED.md) distinguish
  implemented features awaiting acceptance, planned work and optional research.

Automatic avatar-artwork rejection is absent from this frozen checkpoint.
Subsequent experimental development is separate on `codex/avatar-artwork-filter`;
see its [implementation and validation](AVATAR_ARTWORK_FILTER_2026-11-18.md).
That work does not alter the candidate or Working V1 recovery references.

## Evidence and limits

The tracker source hashes match the
[9 November validation record](validation/2026-11-09-ring-verification.json):
868 exact saved frames, 15 independently reviewed physical moves matched and no
extra accepted events. Working V1 matched 12 of those 15 moves. The saved Working
V1 sequence remains 7/7, and the two failure sequences improve to 6/6 and 2/2.

The 9 November full Python suite ran 1,336 tests with 18 skips and no failures.
Fresh checkpoint checks are recorded in the
[18 November checkpoint record](validation/2026-11-18-candidate-checkpoint.json).

Controlled paired stationary engine measurements used about 45–47% less CPU than
Working V1. Full searches after departures cost more. These are engine/detector
measurements, not delivered camera FPS or end-to-end gameplay latency.

Saved replay does not generate new Foundry redraws in response to newly detected
moves. Exact-cell accuracy, arbitrary artwork, representative lighting and fresh
live fog behavior remain open. A screen image that genuinely looks like a
physical ring can still require additional evidence or an uncertain result.

## Resume at the table

Use the [current physical acceptance plan](PHYSICAL_ACCEPTANCE.md). Keep the
existing optics fixed for the first test: iPhone 13 Pro over USB, Wide 1x, Detail
4K portrait with its matching calibration, the 54-inch active display profile,
and six oversized rings. The saved Working V1 setup used torch off, locked
exposure/white balance, marker pattern 80, margin 12, inset 0 and a rough CPL LCD
brightness index around 20%. That index is a camera reference, not a calibrated
optical transmission measurement.

Configure fog before binding, leave avatars visible, use normal **Start session**,
confirm all six token assignments and acquisition, then make random single-mini
moves. No fresh live test of this candidate has been completed. Printed markers
are available but optional; the current on-screen setup has achieved 4/4 lock.

The read-only mini observer has a different display-change hold policy. Do not
use its holds as evidence that normal token-following sessions must stop after
every tracker-issued move. Unrelated manual scene/view changes still require
explicit rebinding in the normal session.

## Recover the candidate separately

From an existing clone, after quitting Sarween:

```sh
git fetch origin --tags
git worktree add ../Sarween-Candidate-2026-11-18 ring-tracking-candidate-2026-11-18
cd ../Sarween-Candidate-2026-11-18
conda activate py_arm
python main.py --dev-no-login
```

If the worktree already exists, open it rather than adding it again. Recovery
uses the source app; the installed Mac application bundle is not rebuilt by this
checkpoint. Foundry module changes require the corresponding module files and
a Foundry reload. Personal hardware profiles and calibration live separately in
the Mac's Sarween application data. The iPhone app, its signing and local media
also remain separate from Git source recovery.

For the live-confirmed static-map fallback, use [Working V1](../WORKING_V1.txt).
Do not overwrite its tag or branch when saving future candidates.
