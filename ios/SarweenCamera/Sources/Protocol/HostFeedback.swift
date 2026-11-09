import Foundation
import CoreFoundation

public enum CameraPhoneAction: String, CaseIterable {
    case calibrationCapture = "calibration.capture"
    case calibrationFinish = "calibration.finish"
    case calibrationCancel = "calibration.cancel"
    case cplReference = "cpl.reference"
    case cplTarget = "cpl.target"
    case cplReset = "cpl.reset"
}

public struct CameraMarkerFeedback {
    public let ids: [Int]
    public let requiredIDs: [Int]
    public let count: Int
    public let required: Int
    public let locked: Bool?
    public var complete: Bool {
        count >= required && locked != false && (requiredIDs.isEmpty || Set(requiredIDs).isSubset(of: Set(ids)))
    }
}

public struct CameraCalibrationFeedback {
    public let active: Bool
    public let captured: Int
    public let required: Int
    public let title: String
    public let message: String
    public let reason: String
    public let mode: String
    public let canFinish: Bool
    public let countdownSeconds: Double?
}

public struct CameraCPLFeedback {
    public let state: String
    public let index: Double?
    public let target: Double?
    public let message: String
}

public struct CameraHostFeedback {
    public enum Kind: String { case preview, tracking, calibration }
    public let kind: Kind
    public let sourceSequence: UInt64?
    public let markers: CameraMarkerFeedback?
    public let calibration: CameraCalibrationFeedback?
    public let cpl: CameraCPLFeedback?

    public init?(json: [String: Any]) {
        guard let rawKind = json["kind"] as? String, let kind = Kind(rawValue: rawKind) else { return nil }
        self.kind = kind
        if json["sourceSeq"] != nil {
            guard let sequence = CameraWire.unsignedInteger(json["sourceSeq"]), sequence <= UInt64(Int64.max) else { return nil }
            sourceSequence = sequence
        } else { sourceSequence = nil }
        if sourceSequence != nil, let data = json["markers"] as? [String: Any],
           let ids = Self.ids(data["ids"]), let expected = Self.ids(data["requiredIds"]), expected.count == 4,
           let count = Self.integer(data["count"], 0...4), count == ids.count,
           Self.integer(data["required"], 4...4) == 4, Set(ids).isSubset(of: Set(expected)),
           data["locked"] == nil || Self.boolean(data["locked"]) != nil {
            markers = CameraMarkerFeedback(ids: ids, requiredIDs: expected, count: count,
                                           required: 4, locked: Self.boolean(data["locked"]))
        } else { markers = nil }
        if let data = json["calibration"] as? [String: Any],
           let active = Self.boolean(data["active"]), let captured = Self.integer(data["captured"], 0...10_000),
           let required = Self.integer(data["required"], 1...10_000), captured <= required {
            let mode = Self.text(data["mode"], limit: 32)
            calibration = CameraCalibrationFeedback(active: active, captured: captured, required: required,
                title: Self.text(data["title"], limit: 100), message: Self.text(data["message"], limit: 500),
                reason: Self.text(data["reason"], limit: 180),
                mode: ["sweep", "manual", "timed", "steady"].contains(mode) ? mode : "",
                canFinish: Self.boolean(data["canFinish"]) == true && captured >= required,
                countdownSeconds: Self.number(data["countdownSeconds"], range: 0...3600))
        } else { calibration = nil }
        if let data = json["cpl"] as? [String: Any] {
            cpl = CameraCPLFeedback(state: Self.text(data["state"], limit: 64),
                index: sourceSequence == nil ? nil : Self.number(data["index"], range: 0...1_000_000),
                target: Self.number(data["target"], range: 0...1_000_000),
                message: Self.text(data["message"], limit: 500))
        } else { cpl = nil }
    }

    private static func text(_ value: Any?, limit: Int) -> String { String((value as? String ?? "").prefix(limit)) }
    private static func boolean(_ value: Any?) -> Bool? {
        guard let n = value as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else { return nil }
        return n.boolValue
    }
    private static func integer(_ value: Any?, _ range: ClosedRange<Int>) -> Int? {
        guard let n = CameraWire.unsignedInteger(value), n <= UInt64(range.upperBound), n >= UInt64(range.lowerBound) else { return nil }
        return Int(n)
    }
    private static func number(_ value: Any?, range: ClosedRange<Double>) -> Double? {
        guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
              n.doubleValue.isFinite, range.contains(n.doubleValue) else { return nil }
        return n.doubleValue
    }
    private static func ids(_ value: Any?) -> [Int]? {
        guard let values = value as? [Any], values.count <= 4 else { return nil }
        let result = values.compactMap { integer($0, 0...249) }
        guard result.count == values.count, Set(result).count == result.count else { return nil }
        return result
    }
}

/// Receipt freshness uses the iPhone's clock. Repeated feedback about the same
/// source frame cannot keep a green marker badge or active task alive forever.
public struct CameraHostFeedbackState {
    public private(set) var sessionID: String?
    private var feedback: CameraHostFeedback?
    private var receivedAt: TimeInterval = 0
    private var sourceAdvancedAt: TimeInterval = 0
    private var maximumSequence: UInt64?
    public init() {}
    public mutating func reset(sessionID: String?) {
        self.sessionID = sessionID
        feedback = nil; maximumSequence = nil; receivedAt = 0; sourceAdvancedAt = 0
    }
    @discardableResult public mutating func receive(_ value: CameraHostFeedback, sessionID: String, now: TimeInterval) -> Bool {
        guard self.sessionID == sessionID, now.isFinite else { return false }
        if let next = value.sourceSequence {
            if let previous = maximumSequence, next < previous { return false }
            if maximumSequence == nil || next > maximumSequence! { sourceAdvancedAt = now }
            maximumSequence = next
        }
        feedback = value; receivedAt = now
        return true
    }
    public func fresh(now: TimeInterval, maximumAge: TimeInterval = 3) -> CameraHostFeedback? {
        guard let feedback, sessionID != nil, now.isFinite, maximumAge > 0,
              now >= receivedAt, now-receivedAt < maximumAge else { return nil }
        if feedback.sourceSequence != nil {
            guard now >= sourceAdvancedAt, now-sourceAdvancedAt < maximumAge else { return nil }
        }
        return feedback
    }
}

/// A remembered guide is deliberately separate from a fresh LCD measurement.
/// Views keep the same slots for live, invalid and disconnected states so status
/// changes do not resize the camera image or make the diagnostic row disappear.
public struct CameraGuidePresentation {
    public let markerValue: String
    public let markerDetail: String
    public let markersComplete: Bool
    public let brightnessValue: String
    public let brightnessIsLive: Bool
    public let savedGuideText: String
    public let message: String

    public init(feedback: CameraHostFeedback?, hostConnected: Bool, cameraRunning: Bool,
                calibrationActive: Bool, savedGuide: Double? = nil) {
        let liveFeedback = cameraRunning && hostConnected ? feedback : nil
        let markers = calibrationActive ? nil : liveFeedback?.markers
        markerValue = markers.map { "\($0.count)/\($0.required)" } ?? "—"
        markersComplete = markers?.complete == true
        if calibrationActive { markerDetail = "Calibrating" }
        else if let markers {
            if markers.locked == false { markerDetail = "Lock pending" }
            else if markers.locked == true { markerDetail = markers.complete ? "Locked" : "Lock retained" }
            else { markerDetail = markers.complete ? "All visible" : "Visible now" }
        } else { markerDetail = cameraRunning && hostConnected ? "Waiting for fresh data" : "Needs Mac session" }

        let cpl = liveFeedback?.cpl
        let comparable = cpl?.state == "comparable" || cpl?.state == "target-matched"
        brightnessIsLive = cameraRunning && hostConnected && !calibrationActive && comparable && cpl?.index != nil
        brightnessValue = brightnessIsLive ? String(format: "%.1f", cpl!.index!) : "—"
        if let savedGuide, savedGuide.isFinite, savedGuide >= 0 {
            savedGuideText = String(format: "Saved guide: %.1f", savedGuide)
        } else { savedGuideText = "Saved guide: —" }

        if !cameraRunning { message = "Resume the camera for live marker and LCD measurements." }
        else if !hostConnected { message = "Open a camera preview or session in Sarween on the Mac for measurements." }
        else if calibrationActive { message = "Finish camera calibration before measuring the CPL setting." }
        else if feedback == nil { message = "Waiting for fresh measurements from Sarween on the Mac." }
        else if let text = cpl?.message, !text.isEmpty { message = text }
        else if brightnessIsLive { message = "Relative LCD brightness. Keep the map and room lighting fixed." }
        else { message = "Set a bright reference with the CPL fitted, then rotate only the filter." }
    }
}

public enum CameraPreviewPolicy {
    public static func shouldRun(active: Bool, authorized: Bool, capabilitiesReady: Bool, paused: Bool) -> Bool {
        active && authorized && capabilitiesReady && !paused
    }
    public static func canChangeMode(authorized: Bool, capabilitiesReady: Bool, paused: Bool, running: Bool) -> Bool {
        authorized && capabilitiesReady && paused && !running
    }
}
