# Physical acceptance after the markers and CPL arrive

Use one fixed capture mode, camera mount, display profile and observer code
snapshot throughout each comparison. Keep Foundry token output and physical taps
off. Software unit/replay checks are preparation; the recordings below establish
physical performance only after independent human review.

## First useful table image

Install the matching printed markers at Actual Size, secure the CPL, and create
a calibration for the direct USB feed. Start with Standard 1080p/15. Verify focus,
marker lock and visible bases at the nearest and farthest TV squares. Measure
base diameter at both the camera and detector input; an upstream 4K image cannot
help detail discarded later by rectification/resizing.

Preserve the camera/profile/mode identity and the failure if six bases cannot be
acquired. Initial clicks assign IDs and approximate centers; they do not prove
that bases were detected. Wait for acquired 6/6 and visible 6/6 before moving.

## One short static-map recording

1. Leave all six minis untouched for ten seconds.
2. Move one mini at a time. Choose the sequence yourself without giving the
   observer expected endpoints. Include a repeated mover and leave one stationary.
3. Include neighboring bases and ordinary temporary hand occlusions. Clear the
   hand and wait between moves. Keep camera, pan and zoom fixed.
4. Stop the input recording; retain its manifest, exact inputs, decisions and the
   run's independent movement-events.jsonl. Confirm savedMoveCount agrees with
   acceptedMoveCount and pendingMoves is zero.

Human review scores identity and order, origin, destination or justified border
abstention, false/missed moves, acquisition failure and hand-clear response delay.
Record observations before tuning. Abstention is distinct from an exact-cell hit.
The archived 611-frame development clip still fails exact-cell acceptance; it is
not held-out evidence or a reason to force narrower uncertainty.

## Known fog change, then an independently bound physical move

Keep all physical minis still and change fog or lighting. The current observer
must hold on a received visual revision and report no movement. Do not wait for
an automatic recovery: explicit rebinding is required after seeing the new TV
image. After acquisition, make another physical move and review it separately.
This checks the hold, not automatic fog tracking or map/physical-object separation.

Automatic fog recovery remains future work requiring verified camera/display
pairing or equivalent physical foreground evidence. A relay notification, a quiet
timer, a dark LCD, or a digital circle at an old anchor cannot establish that by
itself. Preserve changes that arrive late and changes without notifications.

## Camera presets, only after the baseline is understood

Compare the same physical arrangement at Standard and Smooth using separate short
clips. Smooth must preserve the exact native format. Try Detail only if offered;
use a matching calibration, verify actual base pixels and measure response,
processor load, sustained heat and USB throughput. Higher FPS is more temporal
sampling, not proof of lower gameplay delay. Neither Smooth nor Detail phone performance has
been established by the synthetic server tests.
