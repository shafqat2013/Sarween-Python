# Sarween Camera for iPhone

This native iPhone app supplies a direct color-camera feed to Sarween on a Mac over a USB tunnel. The default Standard preset uses the physical rear Wide camera without digital zoom, full 1920×1080 capture at up to 15 FPS, and JPEG quality 0.94. Portrait output contains actual 1080×1920 pixels; landscape contains 1920×1080 pixels. Rotation is applied before transmission, with no letterboxing or mirroring. Video stabilization, geometric distortion correction, HDR and automatic preview downscaling are disabled to keep the capture geometry explicit.

## Physical camera selection

The **Rear camera** selector offers **Wide**, **Ultra Wide** and **Telephoto** only when that physical rear camera exists and supports Standard. Wide remains the default. The app never uses a virtual dual/triple camera or switches lenses automatically to meet a requested mode. Each lens has its own supported presets; switching to a lens that cannot supply the current preset explicitly selects Standard on that lens.

Tap **Pause**, then open **Controls → Rear camera** to change lenses. Resume and reconnect from Sarween. A lens change closes the old connection and creates a new session with a different optical source identity, including when the pixel dimensions remain 1080×1920. Wide keeps the original `back-wide` identity, so adding this selector does not rename existing Wide profiles. Ultra Wide uses `back-ultra-wide`; Telephoto uses `back-telephoto`. **Each lens needs its own matching calibration profile.** If a clip-on filter is fitted, center it over the selected physical lens and check the image for clipped corners before calibrating.

The selected physical lens always uses `videoZoomFactor = 1`; that means no added digital zoom on that lens. It is not the same as the main Camera app's displayed 1× across all lenses. The advertised `nativeFieldOfViewDegrees` is the selected native format's unrotated horizontal field of view, not a diagonal marketing figure. The same retained native format and FOV are used for configuration and metadata. Delayed callbacks from a removed camera output retain their old output-specific optics and cannot be relabelled as frames from the newly selected lens.

## Capture presets

| Preset | Native capture | Portrait output | Availability |
| --- | --- | --- | --- |
| Standard (default) | 1920×1080, up to 15 FPS | 1080×1920 | Largest native format FOV on the selected physical lens; preserves original Wide selection |
| Smooth | 1920×1080, up to 30 FPS | 1080×1920 | Only when that exact retained Standard format supports 30 FPS |
| Detail (experimental) | 3840×2160, up to 15 FPS | 2160×3840 | Only when a native format on the selected physical lens supports these dimensions and 15 FPS |

Unsupported choices are disabled. The app does not upscale or switch to another lens to enable a preset. Selecting another mode or orientation is allowed while stopped/paused, closes the old connection, and requires a fresh session/hello before new frames. Capture-mode capabilities are inspected after Camera permission is granted and before the listener becomes ready.

Smooth preserves Standard's exact native format and optical source identity; a frame-rate-only change therefore does not create a new geometric calibration identity. Detail and orientation changes use different dimension-based optical identities and require an appropriate verified calibration profile. Selected nominal FPS and preset are transmitted separately from optical identity. Actual delivered FPS can be lower because JPEG encoding, USB transfer, receiver processing, and thermal limits are bounded rather than queued. Detail has four times as many pixels per frame as 1080p and has not yet been validated in a sustained on-device test.

## Phone preview and controls

Opening the app with Camera permission starts a **local preview even without a Mac connection**. The full image fits the preview using `resizeAspect`; the app does not crop it to fill the screen. Connection and marker status appear in a small header, with **Pause/Resume** and a compact **Controls** menu below. No waiting panel covers the middle of the image.

JPEG encoding/transmission still starts only when the connected Sarween client requests frames. Local preview alone does not reserve a JPEG work ticket or queue video. Closing the Mac preview stops USB transmission while the phone's local preview continues. **Pause** on the phone overrides the Mac and stops the camera. Leaving the app stops capture and closes the connection. **Controls → Dim screen** shows a black view while keeping the camera active; tap to return. Locking the phone or switching apps is different and stops capture.

To change lens, capture quality or orientation: tap **Pause**, open **Controls**, choose the setting, then tap **Resume**. Light controls remain available while previewing. The selection persists across launches, and Wide remains the default for a new installation.

The **Torch** switch is available while the local preview runs on a selected camera that supports it and defaults to off. It turns off when the camera pauses or the app leaves the foreground. Closing only the Mac preview keeps the phone's local preview and light controls active. Frames report the actual torch state and level. Torch changes affect illumination and may require a lighting sample/background refresh in Sarween, but they do not change geometric camera identity.

**Controls → Lock exposure for CPL** holds the device's current exposure and white balance while the local preview runs, keeping brightness and color balance steady during a filter comparison. Pause/backgrounding restores continuous exposure and white balance. Frame telemetry includes actual `exposureMode`, `exposureLocked`, `whiteBalanceLocked` and white-balance gains. Focus behavior and optical source identity remain unchanged.

## Phone calibration and CPL tools

Open **Controls → Calibration guide** for placement and capture guidance. Start camera calibration on the Mac with this USB source. While the Mac reports an active calibration, the phone shows its readable instruction, saved-view count and optional countdown; internal reason codes are not displayed. **Capture view**, **Finish** and **Cancel** send requests to that active Mac workflow; the phone does not independently save a calibration. Finish remains unavailable until the host reports all required accepted views. The card hides when calibration ends, the connection closes or feedback becomes stale. Sweep/manual/timed/steady mode instructions describe varied useful views; a burst from one unchanged viewpoint is not sufficient coverage.

**Controls → CPL tool** shows a rough **LCD brightness** guide from the connected Mac. A bright reference is 100; lower readings mean a dimmer sampled LCD. It is a relative encoded-image index, not a physical filter angle or an optical-transmission percentage. Each fresh reading uses all four registration markers to align the same LCD area, so small camera/filter shifts do not invalidate the reference. Lens, scene/view, exposure, white balance and torch changes still require a comparable reference. Fit the filter, lock exposure, **Set bright reference** at its brightest view, rotate only the filter, then **Save target** at the chosen view. The confirmed guide is remembered on the phone separately for each lens, capture quality and orientation. Set a fresh bright reference each session; different maps and room lighting affect the guide. **Reset reference** clears the host measurement; **Forget saved guide** removes the remembered hint. Close the sheet with **Done** to use the large marker count, live reading and saved guide at the bottom, away from the camera clip. Missing readings show a dash and a reason in a reserved status area. Feedback overlays do not resize the camera preview. Actions require fresh compatible Mac feedback and are unavailable during camera calibration. Without a Mac preview/session, the local camera remains usable and the sheet explains that measurements need Sarween.

The phone's marker badge also comes from the Mac; there is no native ArUco detector in this app. It accepts the runtime expected four IDs (onscreen or printed), never a cached lock alone. Fresh feedback and advancing source-frame sequences are required. After three seconds without current evidence, the green count disappears; disconnect/pause/host stop clears it immediately. During ChArUco calibration the badge reads **Calibrating** rather than interpreting calibration-board IDs as TV registration.

## Short on-device measurements

On 6 November 2026, the iPhone 13 Pro delivered the declared portrait dimensions
through the production USB receiver in all three presets. After excluding ten
warmup frames, each sample covered approximately 15 seconds:

| Preset | Target FPS | Observed received FPS | Median JPEG encode time |
| --- | ---: | ---: | ---: |
| Standard | 15 | 15.0 | 24.3 ms |
| Smooth | 30 | 15.0 | 24.1 ms |
| Detail | 15 | 5.0 | 71.7 ms |

Keep Standard as the default. Mode capability and nominal FPS do not establish
actual delivery speed. These short checks do not measure sensor-to-display delay,
sustained heat/battery, table-image quality or tracking accuracy. Standard/Smooth
reported the same optical identity; Detail reported a different identity.

## Install on an iPhone for development

Requirements: a Mac with Xcode, an Apple account added to Xcode, an iPhone running iOS 17 or later, and a cable that supports data. An unsigned build cannot be installed on an ordinary iPhone.

1. Open `SarweenCamera.xcodeproj` in Xcode.
2. In **Xcode → Settings → Accounts** (the label may appear as **Apple Accounts**), add your Apple account if it is not already present. A Personal Team can be used for development; its provisioning expires and is not the permanent customer distribution method.
3. Select the blue **SarweenCamera** project, then the **SarweenCamera** target. Open **Signing & Capabilities**, leave **Automatically manage signing** enabled, and select your Team. If Xcode says `com.sarween.camera` is unavailable to your Team, choose a unique bundle identifier such as `com.yourname.sarweencamera`; the camera protocol does not depend on the bundle identifier.
4. Connect and unlock the iPhone. Approve **Allow accessory** on the Mac and **Trust This Computer** on the phone if prompted.
5. Select the actual iPhone as the run destination beside the **SarweenCamera** scheme in Xcode. Wait for Xcode to finish device preparation.
6. If Xcode requires **Developer Mode**, enable it under the iPhone's **Settings → Privacy & Security → Developer Mode**, follow the restart/confirmation steps, then return to Xcode. If Developer Mode is not yet listed, connecting the phone to Xcode and attempting the run first may expose it.
7. Choose **Product → Run**. Xcode builds, signs and installs the app. If iOS reports an untrusted developer, follow its device-management instructions for the development certificate, then run again.
8. Allow Camera access when Sarween Camera first opens. Keep it open on the phone. Start Sarween on the Mac and select its direct iPhone USB source. The phone should change from waiting to streaming when Sarween connects.

The desktop app needs its Sarween USB transport/receiver implementation. Opening this phone app alone does not publish a system-wide webcam. There is no OBS, FFmpeg or Homebrew dependency in this iPhone project. Successful unsigned builds establish compilation; actual installation, USB connectivity and camera output require the signed on-device test above.

For daily use after installation: open Sarween Camera and Sarween, connect the USB cable, and unlock the phone if needed. A new camera-calibration profile is required for this capture path; the OBS profile may have different cropping or lens correction.

## First feed check in Sarween

1. Keep Sarween Camera open on the iPhone after allowing Camera access.
2. Restart the Mac source app using the usual `python main.py --dev-no-login` command in the Sarween environment. The existing packaged desktop app has not been rebuilt for this prototype.
3. Choose **Start session**, then choose **Sarween Camera · iPhone / iPad USB** in the Camera list. Use **Refresh hardware** if the cable was connected after the list appeared.
4. Click **Preview Webcam**. This checks the raw feed before connecting Foundry or calibrating. Confirm the framing, orientation and lack of watermark. Press a key in the preview to close it and stop USB transmission. The phone's local preview continues until paused or backgrounded.
5. For later live sessions, choose **Create iPhone profile**, calibrate this feed, then confirm its profile. The new profile starts without any copied OBS calibration. Preview alone does not require a calibration profile.

The connection stage remains cancellable while the phone is locked, the app is closed or permission is missing. A disconnected live source supplies a tracking hold rather than repeating its last image as fresh evidence. Reconnection requires new marker registration and observer bindings. These safeguards do not establish miniature-tracking accuracy; that still requires the physical tests.

## Build without signing or launching

From the repository root:

```sh
./ios/SarweenCamera/build.sh device
./ios/SarweenCamera/build.sh simulator
```

These compile generic device/simulator app bundles under `/tmp/sarween-camera-device-build` and `/tmp/sarween-camera-simulator-build`. They do not install or launch either bundle. Set `SARWEEN_IOS_DERIVED_DATA` to choose another build directory. The simulator build does not establish that a physical camera is available.

Protocol and flow-control unit tests run on the Mac:

```sh
swift test --package-path ios/SarweenCamera
```

The package has no external dependencies. In a restricted execution environment, SwiftPM's additional nested sandbox may fail to start. The development checks used isolated `/tmp` cache/configuration paths and `--disable-sandbox` for the checked-in, dependency-free manifest; this does not change app entitlements.

`Tools/USBProtocolHarness.swift` can compile the production framing, flow gate and Network listener as a headless Mac executable with synthetic JPEG input. It is excluded from both the application and unit-test targets. This checks the socket protocol without opening a camera or simulator; it is not evidence of physical USB or on-device capture success.

The harness accepts `USBProtocolHarness synthetic.jpg [standard|smooth|detail] [portrait|landscape] [wide|ultra-wide|telephoto] [--exercise-feedback]`. The JPEG's stored pixel dimensions must match the selected mode/orientation. Its three lens capabilities and FOV values are invented protocol fixtures, not discovered camera hardware. The explicit feedback exercise mode prints parsed test feedback and sends one synthetic `calibration.capture` request per active test connection. It does not open a camera or calibrate anything.

## USB protocol v1

The phone listens on **127.0.0.1:52781 only**. The desktop transport forwards a host port through usbmux to this device port. The phone listener is not exposed on Wi-Fi.

Every packet contains:

1. Four bytes: unsigned big-endian UTF-8 JSON header length, 1–65,536 bytes.
2. The JSON object, including `type`, `protocolVersion: 1`, `sessionId` and integer `payloadBytes`.
3. Exactly `payloadBytes` bytes. Camera frames contain JPEG data; control packets contain zero payload bytes. The payload limit is 16 MiB.

On connection the phone sends `hello`, including `deviceName`, `deviceId` (app-installation identity), `sourceId` (stable optical configuration), and `captureFormat` with `width`, `height`, `fps`, `pixelFormat: "jpeg"`, `capturePreset`, orientation and camera settings. `lens` is `wide`, `ultra-wide` or `telephoto`. `nativeFieldOfViewDegrees` reports the selected format's unrotated horizontal FOV and is identical in hello and frames. Optional `availableCapturePresets` contains only the selected lens's supported presets, each with `id`, `label`, output `width`/`height`, `fps`, and `experimental`. `availableCameras` lists supported physical lenses, their label, torch support and per-lens preset capabilities/FOV. A new connection has a fresh session UUID; session UUIDs are not calibration identities.

The new phone advertises `hostFeedback: true` in `hello`. Optional `localPreviewRunning` in hello/pong reflects the local camera's actual running callback, independently of whether the Mac requested JPEG transmission. A compatible host may send zero-payload `feedback` packets with a nested `feedback` object: `kind` (`preview`, `tracking`, `calibration`), optional actual `sourceSeq`, and optional `markers`, `calibration`, `cpl` sections. The phone uses bounded known fields only, rejects older source sequences within a session, and expires repeated source-frame status after three seconds. Feedback without source pixels clears marker counts and the current CPL reading. Mode changes and disconnects reset all feedback.

Phone controls send zero-payload `request` packets with a UUID `requestId`, matching `sessionId`, and an approved `action`: `calibration.capture`, `calibration.finish`, `calibration.cancel`, `cpl.reference`, `cpl.target` or `cpl.reset`. Requests are gated on fresh valid host feedback and rate limited; an old Mac client receives no unexpected phone actions. They request an operation from the current host workflow rather than asserting that it succeeded.

The Mac sends `start` with the matching session ID and `ackFrames: true`. Each `frame` has a strictly increasing `seq`, actual width/height, `captureTimeNs`, `encodeDurationNs`, `rotationDegrees: 0`, `mirrored: false`, and capture settings. Optional `capturePreset` and `nominalFPS` identify the selected mode, while actual FPS must be measured from advancing frame timestamps. `captureTimeNs` is the camera presentation timestamp expressed in nanoseconds in the phone's monotonic clock domain; it is not directly comparable to the Mac's monotonic or wall clock. Optional exposure/focus/white-balance readouts describe the capture state.

The Mac acknowledges a decoded/published frame using `ack` with the matching session ID and `seq`. The phone permits one frame to be encoded/in transit at a time and drops new arrivals until both the send completion and matching ACK arrive. No queued video accumulates. A missing ACK closes the connection after five seconds so the desktop can reconnect. An explicit `ackFrames: false` mode waits only for send completion and is intended for minimal diagnostic clients.

`stop` stops streaming; `ping` receives `pong`. `status` messages report readiness. Unknown commands, invalid counters, nonempty command payloads, wrong protocol versions and wrong session IDs close the client connection. A single active Mac client is supported. Disconnecting, backgrounding, pausing or changing lens, preset or orientation invalidates the connection and outstanding frames.

This version transmits color images. LiDAR/TrueDepth capture, physical-mini detection and cross-device clock synchronization are separate future work.
