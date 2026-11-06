# Sarween Camera for iPhone

This native iPhone app supplies a direct color-camera feed to Sarween on a Mac over a USB tunnel. The default Standard preset uses the physical rear wide camera at 1×, full 1920×1080 capture at up to 15 FPS, and JPEG quality 0.94. Portrait output contains actual 1080×1920 pixels; landscape contains 1920×1080 pixels. Rotation is applied before transmission, with no letterboxing or mirroring. Video stabilization, geometric distortion correction, HDR and automatic preview downscaling are disabled to keep the capture geometry explicit.

## Capture presets

| Preset | Native capture | Portrait output | Availability |
| --- | --- | --- | --- |
| Standard (default) | 1920×1080, up to 15 FPS | 1080×1920 | Same physical-wide format selection as the original working feed |
| Smooth | 1920×1080, up to 30 FPS | 1080×1920 | Only when that exact retained Standard format supports 30 FPS |
| Detail (experimental) | 3840×2160, up to 15 FPS | 2160×3840 | Only when a native physical-wide format supports these dimensions and 15 FPS |

Unsupported choices are disabled. The app does not upscale or switch to another lens to enable a preset. Selecting another mode or orientation is allowed while stopped/paused, closes the old connection, and requires a fresh session/hello before new frames. Capture-mode capabilities are inspected after Camera permission is granted and before the listener becomes ready.

Smooth preserves Standard's exact native format and optical source identity; a frame-rate-only change therefore does not create a new geometric calibration identity. Detail and orientation changes use different dimension-based optical identities and require an appropriate verified calibration profile. Selected nominal FPS and preset are transmitted separately from optical identity. Actual delivered FPS can be lower because JPEG encoding, USB transfer, receiver processing, and thermal limits are bounded rather than queued. Detail has four times as many pixels per frame as 1080p and has not yet been validated in a sustained on-device test.

The app becomes ready when it is foregrounded and camera permission is granted. Capture starts only after the connected Sarween client requests it. **Pause camera** on the phone overrides the Mac. Leaving the app stops capture and closes the connection. **Dim screen** shows a black view while keeping the app active; locking the phone or switching apps is different and can interrupt capture.

The **Torch** switch is available while streaming and defaults to off. It turns off when streaming stops, the camera pauses, or the app leaves the foreground. Frames report the actual torch state and level. Torch changes affect illumination and may require a lighting sample/background refresh in Sarween, but they do not change geometric camera identity.

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
4. Click **Preview Webcam**. This checks the raw feed before connecting Foundry or calibrating. Confirm the framing, orientation and lack of watermark. Press a key in the preview to close it and stop capture.
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

## USB protocol v1

The phone listens on **127.0.0.1:52781 only**. The desktop transport forwards a host port through usbmux to this device port. The phone listener is not exposed on Wi-Fi.

Every packet contains:

1. Four bytes: unsigned big-endian UTF-8 JSON header length, 1–65,536 bytes.
2. The JSON object, including `type`, `protocolVersion: 1`, `sessionId` and integer `payloadBytes`.
3. Exactly `payloadBytes` bytes. Camera frames contain JPEG data; control packets contain zero payload bytes. The payload limit is 16 MiB.

On connection the phone sends `hello`, including `deviceName`, `deviceId` (app-installation identity), `sourceId` (stable optical configuration), and `captureFormat` with `width`, `height`, `fps`, `pixelFormat: "jpeg"`, `capturePreset`, orientation and camera settings. Optional `availableCapturePresets` contains only supported presets, each with `id`, `label`, output `width`/`height`, `fps`, and `experimental`. A new connection has a fresh session UUID; session UUIDs are not calibration identities.

The Mac sends `start` with the matching session ID and `ackFrames: true`. Each `frame` has a strictly increasing `seq`, actual width/height, `captureTimeNs`, `encodeDurationNs`, `rotationDegrees: 0`, `mirrored: false`, and capture settings. Optional `capturePreset` and `nominalFPS` identify the selected mode, while actual FPS must be measured from advancing frame timestamps. `captureTimeNs` is the camera presentation timestamp expressed in nanoseconds in the phone's monotonic clock domain; it is not directly comparable to the Mac's monotonic or wall clock. Optional exposure/focus/white-balance readouts describe the capture state.

The Mac acknowledges a decoded/published frame using `ack` with the matching session ID and `seq`. The phone permits one frame to be encoded/in transit at a time and drops new arrivals until both the send completion and matching ACK arrive. No queued video accumulates. A missing ACK closes the connection after five seconds so the desktop can reconnect. An explicit `ackFrames: false` mode waits only for send completion and is intended for minimal diagnostic clients.

`stop` stops streaming; `ping` receives `pong`. `status` messages report readiness. Unknown commands, invalid counters, nonempty command payloads, wrong protocol versions and wrong session IDs close the client connection. A single active Mac client is supported. Disconnecting, backgrounding, pausing or changing orientation invalidates the connection and outstanding frames.

This version transmits color images. LiDAR/TrueDepth capture, physical-mini detection and cross-device clock synchronization are separate future work.
