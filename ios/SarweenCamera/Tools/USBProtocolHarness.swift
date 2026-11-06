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
        guard (2...4).contains(CommandLine.arguments.count) else {
            fputs("Usage: USBProtocolHarness synthetic.jpg [standard|smooth|detail] [portrait|landscape]\n", stderr)
            exit(2)
        }
        guard let preset = CameraCapturePreset(rawValue: CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "standard"),
              let orientation = CameraOrientation(rawValue: CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : "portrait") else {
            fputs("Invalid synthetic preset or orientation.\n", stderr)
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
        let server = USBStreamServer(
            installationID: "synthetic-protocol-test",
            deviceName: "Synthetic protocol test (no camera)", orientation: orientation,
            preset: preset, supportedPresets: CameraCapturePreset.allCases
        )
        server.onStatus = { print($0); fflush(stdout) }
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "test.synthetic-frames"))
        timer.schedule(deadline: .now(), repeating: .nanoseconds(1_000_000_000 / preset.framesPerSecond))
        timer.setEventHandler {
            guard let ticket = server.gate.reserve() else { return }
            server.deliver(EncodedCameraFrame(
                jpeg: jpeg, width: width, height: height,
                captureTimeNs: Int64(DispatchTime.now().uptimeNanoseconds), encodeDurationNs: 1,
                metadata: ["syntheticTestFrame": true, "capturePreset": preset.rawValue,
                           "nominalFPS": preset.framesPerSecond]
            ), ticket: ticket)
        }
        timer.resume()
        server.setLocalEnabled(true)
        server.start()
        RunLoop.main.run()
        withExtendedLifetime(timer) {}
    }
}
