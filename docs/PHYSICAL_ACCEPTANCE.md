# Physical tracking acceptance

Updated 18 November 2026, Asia/Bangkok.

The next test checks the current six-ring candidate in a real camera-to-Foundry
feedback loop with fog enabled. Saved replay passed all 15 reviewed moves, but
does not establish how newly detected moves and their resulting TV redraws behave
live. Use normal **Start session** for token following; the diagnostic mini
observer is read-only and has a different display-change hold policy.

## Keep the first setup fixed

Use the existing iPhone USB feed, Wide 1x and matching 4K portrait calibration,
54-inch active-screen profile, mounted CPL and six minis with oversized rings.
Retain the current on-screen marker settings and verify 4/4 lock. Printed markers
are an optional fallback, not a prerequisite for this test. The prior static-map
milestone used torch off with exposure and white balance locked.

Do not change capture mode, camera placement, CPL angle, pan or zoom during the
comparison. A matching lens/format calibration already exists; do not repeat
camera calibration merely because testing resumes. If a lens or capture format
changes, use the appropriate matching profile before continuing.

## Start the normal session

1. Fully quit any older Sarween process, then launch the source app with
   `python main.py --dev-no-login` in the normal project environment. The
   development login bypass is for this source test.
2. Connect Foundry and keep the scene on the TV. An already-open matching world
   can be reused. If automatic placement hits the known Accessibility error,
   place the window manually before binding.
3. In **VTT Controls → Vision & Fog**, enable **Token vision** and **Remember
   explored fog**, disable **Global illumination (daylight)** for an obvious
   first visibility test, and click **Apply scene settings**. Choose the test
   token, enable its vision, set a finite sight range in the displayed scene
   units, and click **Apply token settings**. Click **Show this token's view on
   TV**. Finish visibility changes before binding and leave avatars visible.
4. Start the session, choose the six physical base/ring centers and confirm their
   token assignments. Click order defines Mini 1–6. Keep hands clear until
   acquired 6/6 and visible 6/6. Clicks declare positions; acquisition verifies
   that the detector can see the rings. Physical taps remain off.

## Record one short test

1. Leave all six minis untouched for ten seconds. No physical move should be
   reported during that interval.
2. Make random moves one mini at a time, without supplying expected endpoints to
   the observer. Include all six, a repeated mover, adjacent minis and ordinary
   hand occlusion. Keep the camera and viewport fixed.
3. Include the previous failure: move a ring away so that its digital avatar is
   exposed at the old position. Sarween must not count that artwork as continuing
   physical presence or silently learn the real destination into the map.
4. Check that the selected token follows and changes vision/fog, then make another
   physical move. Tracker-issued token updates and acknowledgement timing must
   not prevent continued physical tracking.
5. Stop and save the short recording. Retain its manifest, exact inputs,
   decisions and movement-events journal. Preserve a failure before changing
   settings or software.

## Review the physical result

Use the existing video-review app to compare actual physical moves with the
reported moves. Record correct identity/order, missed and extra moves, origin and
destination or justified border uncertainty, acquisition failures and delay after
the hand clears. Exact-cell accuracy and uncertainty are separate outcomes; do
not force narrower uncertainty to obtain a pass.

Review one or two short clips from this current CPL setup rather than making a
large historical archive audit a prerequisite. Retain fresh footage for future
checks without tuning the tracker to its expected sequence. Accepted-event logs
alone are not a complete record of physical movements.

For save integrity, check that savedMoveCount equals acceptedMoveCount and that
pendingMoves is zero. Saved recordings and software tests support implementation
checks; only a fresh physical test can support a new live-confirmed checkpoint.

## Separate tests for unrelated changes

Manual scene, lighting, vision, camera-lock or viewport changes outside normal
tracker-issued token movement can intentionally hold tracking and require
rebinding. Test those safeguards separately with stationary minis; they must not
manufacture physical moves. This differs from the token-driven fog updates in
the main test above.

After a reliable baseline, optional printed-marker/CPL and other accessory
comparisons can be revisited from the [parked list](PARKED.md). Sustained camera
FPS, heat, battery behavior and end-to-end latency remain separate measurements;
higher resolution or a nominal FPS setting alone does not establish faster play.
