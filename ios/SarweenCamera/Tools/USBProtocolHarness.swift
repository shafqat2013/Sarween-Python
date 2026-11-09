// Headless protocol test only. Never included in the iPhone application target.
// Compile with WirePacket.swift, FrameFlowGate.swift, CapturePreset.swift and USBStreamServer.swift.
import Foundation
import ImageIO
import Network

struct EncodedCameraFrame {
    let jpeg: Data
    let width: Int
    let height: Int
    let captureTimeNs: Int64
    let encodeDurationNs: UInt64
    let metadata: [String: Any]
}

@main struct USBProtocolHarness {
    static func main() throws {
        guard (2...6).contains(CommandLine.arguments.count),
              CommandLine.arguments.count < 6 || CommandLine.arguments[5] == "--exercise-feedback" else {
            fputs("Usage: USBProtocolHarness synthetic.jpg [standard|smooth|detail] [portrait|landscape] [wide|ultra-wide|telephoto] [--exercise-feedback]\n", stderr)
            exit(2)
        }
        guard let preset = CameraCapturePreset(rawValue: CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "standard"),
              let orientation = CameraOrientation(rawValue: CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : "portrait"),
              let lens = CameraLens(rawValue: CommandLine.arguments.count > 4 ? CommandLine.arguments[4] : "wide") else {
            fputs("Invalid synthetic preset, orientation or lens.\n", stderr)
            exit(2)
        }
        let jpeg = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let width = preset.width(for: orientation)
        let height = preset.height(for: orientation)
        guard let image = CGImageSourceCreateWithData(jpeg as CFData, nil),
              CGImageSourceGetType(image) as String? == "public.jpeg",
              let properties = CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any],
              properties[kCGImagePropertyPixelWidth] as? Int == width,
              properties[kCGImagePropertyPixelHeight] as? Int == height else {
            fputs("Synthetic JPEG dimensions must match \(width)x\(height) for \(preset.rawValue)/\(orientation.rawValue).\n", stderr)
            exit(2)
        }
        // These are invented capabilities for protocol checks, not a hardware inventory.
        let cameras = CameraLens.allCases.map { lens -> CameraLensCapability in
            let fov: Double = lens == .ultraWide ? 115 : (lens == .telephoto ? 27 : 70)
            let formats = [CameraFormatCandidate(identifier: 1, width: 1920, height: 1080, fieldOfView: fov,
                              frameRates: [CameraFrameRateInterval(minimum: 1, maximum: 60)]),
                           CameraFormatCandidate(identifier: 2, width: 3840, height: 2160, fieldOfView: fov,
                              frameRates: [CameraFrameRateInterval(minimum: 1, maximum: 30)])]
            return CameraLensCapability(lens: lens, plans: CameraPresetPlanner.plans(for: formats), torchSupported: false)
        }
        let fov = cameras.first { $0.lens == lens }!.plans[preset]!.fieldOfView
        let server = USBStreamServer(
            installationID: "synthetic-protocol-test",
            deviceName: "Synthetic protocol test (no camera)", orientation: orientation,
            preset: preset, supportedPresets: CameraCapturePreset.allCases,
            lens: lens, availableCameras: cameras
        )
        server.onStatus = { print($0); fflush(stdout) }
        if CommandLine.arguments.count == 6 {
            var lastActionSession: String?
            server.onFeedback = { feedback, sessionID in
                var summary: [String: Any] = ["syntheticFeedback": true, "kind": feedback.kind.rawValue,
                    "markersComplete": feedback.markers?.complete ?? false,
                    "calibrationActive": feedback.calibration?.active ?? false]
                if let count = feedback.markers?.count { summary["markerCount"] = count }
                if let sequence = feedback.sourceSequence { summary["sourceSeq"] = sequence }
                if let data = try? JSONSerialization.data(withJSONObject: summary, options: .sortedKeys),
                   let text = String(data: data, encoding: .utf8) { print(text); fflush(stdout) }
                // Explicit synthetic integration mode only. No camera or actual
                // calibration exists in this harness. Exercise the production
                // reverse-control framing once per test connection.
                if feedback.calibration?.active == true, lastActionSession != sessionID {
                    lastActionSession = sessionID
                    server.sendAction(.calibrationCapture, expectedSessionID: sessionID)
                }
            }
        }
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "test.synthetic-frames"))
        timer.schedule(deadline: .now(), repeating: .nanoseconds(1_000_000_000 / preset.framesPerSecond))
        timer.setEventHandler {
            guard let ticket = server.gate.reserve() else { return }
            server.deliver(EncodedCameraFrame(
                jpeg: jpeg, width: width, height: height,
                captureTimeNs: Int64(DispatchTime.now().uptimeNanoseconds), encodeDurationNs: 1,
                metadata: ["syntheticTestFrame": true, "capturePreset": preset.rawValue,
                           "nominalFPS": preset.framesPerSecond, "lens": lens.rawValue,
                           "nativeFieldOfViewDegrees": fov]
            ), ticket: ticket)
        }
        timer.resume()
        server.setLocalEnabled(true)
        server.start()
        RunLoop.main.run()
        withExtendedLifetime(timer) {}
    }
}
