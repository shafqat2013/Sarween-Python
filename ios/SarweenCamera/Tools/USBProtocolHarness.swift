// Headless protocol test only. Never included in the iPhone application target.
// Compile with WirePacket.swift, FrameFlowGate.swift and USBStreamServer.swift.
import Foundation
import Network

enum CameraOrientation: String { case portrait, landscape }
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
        guard CommandLine.arguments.count == 2 else {
            fputs("Usage: USBProtocolHarness synthetic-1080x1920.jpg\n", stderr)
            exit(2)
        }
        let jpeg = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let server = USBStreamServer(
            installationID: "synthetic-protocol-test",
            deviceName: "Synthetic protocol test (no camera)", orientation: .portrait
        )
        server.onStatus = { print($0); fflush(stdout) }
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "test.synthetic-frames"))
        timer.schedule(deadline: .now(), repeating: .milliseconds(30))
        timer.setEventHandler {
            guard let ticket = server.gate.reserve() else { return }
            server.deliver(EncodedCameraFrame(
                jpeg: jpeg, width: 1080, height: 1920,
                captureTimeNs: Int64(DispatchTime.now().uptimeNanoseconds), encodeDurationNs: 1,
                metadata: ["syntheticTestFrame": true]
            ), ticket: ticket)
        }
        timer.resume()
        server.setLocalEnabled(true)
        server.start()
        RunLoop.main.run()
        withExtendedLifetime(timer) {}
    }
}
