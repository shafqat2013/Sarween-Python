import Foundation
import Network

struct CameraStreamFormat {
    let orientation: CameraOrientation
    var width: Int { orientation == .portrait ? 1080 : 1920 }
    var height: Int { orientation == .portrait ? 1920 : 1080 }
    var json: [String: Any] {
        ["width": width, "height": height, "fps": 15, "pixelFormat": "jpeg",
         "orientation": orientation.rawValue, "rotationDegrees": 0, "mirrored": false,
         "cameraPosition": "back", "lens": "wide", "zoom": 1.0,
         "stabilization": "off", "geometricDistortionCorrection": "off",
         "hdr": "off", "jpegQuality": 0.94, "colorSpace": "sRGB"]
    }
}

final class USBStreamServer {
    static let port: UInt16 = 52781
    let gate = CameraFrameFlowGate()
    var onStatus: ((String) -> Void)?
    var onRequested: ((Bool) -> Void)?
    var onFrameSent: ((UInt64) -> Void)?

    private let queue = DispatchQueue(label: "com.sarween.camera.usb")
    private let installationID: String
    private let deviceName: String
    private var format: CameraStreamFormat
    private var listener: NWListener?
    private var connection: NWConnection?
    private var sessionID: String?
    private var localEnabled = false
    private var controls: [Data] = []
    private var sendingControl = false
    private var decoder = CameraPacketDecoder(maximumIncomingPayload: 0)

    init(installationID: String, deviceName: String, orientation: CameraOrientation) {
        self.installationID = installationID
        self.deviceName = deviceName
        self.format = CameraStreamFormat(orientation: orientation)
    }

    func start() {
        queue.async { [weak self] in
            guard let self, self.listener == nil else { return }
            do {
                let parameters = NWParameters.tcp
                parameters.allowLocalEndpointReuse = true
                // usbmux connects to a port on the iPhone itself. Never expose the camera on Wi-Fi.
                parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: Self.port)!)
                let listener = try NWListener(using: parameters)
                self.listener = listener
                listener.stateUpdateHandler = { [weak self, weak listener] state in
                    guard let self, self.listener === listener else { return }
                    switch state {
                    case .ready: self.report("Connect your iPhone to the Mac and open Sarween.")
                    case .failed(let error):
                        self.report("USB connection unavailable: \(error.localizedDescription)")
                        self.stopOnQueue()
                    default: break
                    }
                }
                listener.newConnectionHandler = { [weak self] incoming in self?.accept(incoming) }
                listener.start(queue: self.queue)
            } catch { self.report("USB connection unavailable: \(error.localizedDescription)") }
        }
    }

    func stop() { queue.async { [weak self] in self?.stopOnQueue() } }

    func setLocalEnabled(_ enabled: Bool) {
        queue.async { [weak self] in
            guard let self else { return }
            self.localEnabled = enabled
            self.gate.setEnabled(enabled)
            if !enabled { self.disconnect() }
            else if self.connection != nil { self.sendStatus("ready") }
        }
    }

    func setOrientation(_ orientation: CameraOrientation) {
        queue.async { [weak self] in
            guard let self, self.format.orientation != orientation else { return }
            self.format = CameraStreamFormat(orientation: orientation)
            self.disconnect()
        }
    }

    func deliver(_ frame: EncodedCameraFrame, ticket: CameraFrameTicket) {
        queue.async { [weak self] in
            guard let self, let connection = self.connection,
                  self.sessionID == ticket.sessionID, self.gate.isPending(ticket) else { return }
            guard frame.width == self.format.width, frame.height == self.format.height else {
                self.sendStatus("formatError", message: "Unexpected camera dimensions. Pause and resume the camera.")
                self.gate.cancel(ticket)
                self.disconnect()
                return
            }
            var header = frame.metadata
            header.merge([
                "type": "frame", "protocolVersion": CameraWire.version,
                "sessionId": ticket.sessionID, "seq": ticket.sequence,
                "width": frame.width, "height": frame.height,
                "captureTimeNs": frame.captureTimeNs, "encodeDurationNs": frame.encodeDurationNs,
                "rotationDegrees": 0, "mirrored": false
            ]) { _, new in new }
            do {
                let data = try CameraWire.encode(header: header, payload: frame.jpeg)
                connection.send(content: data, completion: .contentProcessed { [weak self, weak connection] error in
                    guard let self, self.connection === connection, self.sessionID == ticket.sessionID else { return }
                    if let error {
                        self.report("USB disconnected: \(error.localizedDescription)")
                        self.disconnect()
                    } else {
                        self.gate.didSend(ticket)
                        self.onFrameSent?(ticket.sequence)
                    }
                })
                self.queue.asyncAfter(deadline: .now() + 5) { [weak self] in
                    guard let self, self.sessionID == ticket.sessionID, self.gate.isPending(ticket) else { return }
                    self.report("Waiting for Sarween. Reconnecting…")
                    self.disconnect()
                }
            } catch {
                self.gate.cancel(ticket)
                self.report("Could not encode the camera packet.")
                self.disconnect()
            }
        }
    }

    private func accept(_ incoming: NWConnection) {
        guard connection == nil else { incoming.cancel(); return }
        connection = incoming
        let id = UUID().uuidString
        sessionID = id
        decoder = CameraPacketDecoder(maximumIncomingPayload: 0)
        controls = []
        sendingControl = false
        gate.reset(sessionID: id)
        gate.setEnabled(localEnabled)
        incoming.stateUpdateHandler = { [weak self, weak incoming] state in
            guard let self, let incoming, self.connection === incoming else { return }
            switch state {
            case .ready:
                self.sendHello()
                self.receive(incoming)
                self.report(self.localEnabled ? "Connected. Waiting for Sarween to start the camera." : "Waiting for camera access on your iPhone.")
            case .failed, .cancelled: self.disconnect()
            default: break
            }
        }
        incoming.start(queue: queue)
    }

    private func receive(_ current: NWConnection) {
        current.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self, weak current] data, _, complete, error in
            guard let self, let current, self.connection === current else { return }
            if let data, !data.isEmpty {
                do {
                    for packet in try self.decoder.append(data) { try self.handle(packet) }
                } catch {
                    self.report("Sarween sent an incompatible camera message.")
                    self.disconnect()
                    return
                }
            }
            if complete || error != nil { self.disconnect(); return }
            self.receive(current)
        }
    }

    private func handle(_ packet: CameraWirePacket) throws {
        guard let type = packet.header["type"] as? String,
              CameraWire.unsignedInteger(packet.header["protocolVersion"]) == UInt64(CameraWire.version),
              packet.header["sessionId"] as? String == sessionID else {
            throw CameraWireError.invalidHeader
        }
        switch type {
        case "start":
            gate.setRequested(true, requiresAcknowledgement: packet.header["ackFrames"] as? Bool ?? true)
            onRequested?(true)
            sendStatus(localEnabled ? "starting" : "waitingForCamera")
        case "stop":
            gate.setRequested(false)
            onRequested?(false)
            sendStatus("stopped")
        case "ack":
            guard let sequence = CameraWire.unsignedInteger(packet.header["seq"]), sequence > 0 else {
                throw CameraWireError.invalidHeader
            }
            gate.acknowledge(sessionID: sessionID!, sequence: sequence)
        case "ping":
            var response: [String: Any] = ["type": "pong"]
            if let request = packet.header["requestId"] { response["requestId"] = request }
            sendControl(response)
        default: throw CameraWireError.invalidHeader
        }
    }

    private func sendHello() {
        sendControl([
            "type": "hello", "deviceName": deviceName, "deviceId": installationID,
            "sourceId": "sarween-iphone:\(installationID):back-wide:\(format.width)x\(format.height):rotation0:mirror0:zoom1:stabilization-off:gdc-off:hdr-off",
            "captureFormat": format.json, "cameraReady": localEnabled,
            "capabilities": ["ackFrames", "fixedOrientation", "physicalWideCamera"]
        ])
    }

    private func sendStatus(_ status: String, message: String? = nil) {
        var header: [String: Any] = ["type": "status", "status": status]
        if let message { header["message"] = message }
        sendControl(header)
    }

    private func sendControl(_ fields: [String: Any]) {
        guard sessionID != nil else { return }
        var header = fields
        header["protocolVersion"] = CameraWire.version
        header["sessionId"] = sessionID!
        guard let data = try? CameraWire.encode(header: header) else { return }
        guard controls.count < 8 else { disconnect(); return }
        controls.append(data)
        flushControl()
    }

    private func flushControl() {
        guard !sendingControl, !controls.isEmpty, let connection else { return }
        sendingControl = true
        let packet = controls.removeFirst()
        connection.send(content: packet, completion: .contentProcessed { [weak self, weak connection] error in
            guard let self, self.connection === connection else { return }
            self.sendingControl = false
            if error != nil { self.disconnect() }
            else { self.flushControl() }
        })
    }

    private func disconnect() {
        let previous = connection
        connection = nil
        sessionID = nil
        gate.reset(sessionID: nil)
        controls = []
        sendingControl = false
        previous?.cancel()
        onRequested?(false)
        if listener != nil { report(localEnabled ? "Connect your iPhone to the Mac and open Sarween." : "Camera paused.") }
    }

    private func stopOnQueue() {
        let previous = listener
        listener = nil
        previous?.cancel()
        disconnect()
    }

    private func report(_ message: String) { onStatus?(message) }
}
