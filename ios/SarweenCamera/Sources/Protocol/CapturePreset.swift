import Foundation

public enum CameraOrientation: String, CaseIterable, Identifiable {
    case portrait
    case landscape
    public var id: String { rawValue }
    public var label: String { self == .portrait ? "Portrait" : "Landscape" }
}

public enum CameraCapturePreset: String, CaseIterable, Identifiable {
    case standard
    case smooth
    case detail

    public var id: String { rawValue }
    public var sensorWidth: Int { self == .detail ? 3840 : 1920 }
    public var sensorHeight: Int { self == .detail ? 2160 : 1080 }
    public var framesPerSecond: Int { self == .smooth ? 30 : 15 }
    public var experimental: Bool { self == .detail }
    public var label: String {
        switch self {
        case .standard: return "Standard · 1080p · 15 FPS"
        case .smooth: return "Smooth · 1080p · 30 FPS"
        case .detail: return "Detail · 4K · 15 FPS (experimental)"
        }
    }
    public func width(for orientation: CameraOrientation) -> Int {
        orientation == .portrait ? sensorHeight : sensorWidth
    }
    public func height(for orientation: CameraOrientation) -> Int {
        orientation == .portrait ? sensorWidth : sensorHeight
    }
    public func capabilityJSON(orientation: CameraOrientation) -> [String: Any] {
        ["id": rawValue, "label": label, "width": width(for: orientation), "height": height(for: orientation),
         "fps": framesPerSecond, "experimental": experimental]
    }
    public func opticalSourceID(installationID: String, orientation: CameraOrientation) -> String {
        // Standard and Smooth are offered on the identical retained native format.
        // FPS is temporal metadata, while resolution/orientation change calibration.
        "sarween-iphone:\(installationID):back-wide:\(width(for: orientation))x\(height(for: orientation)):rotation0:mirror0:zoom1:stabilization-off:gdc-off:hdr-off"
    }
}

public struct CameraFrameRateInterval {
    public let minimum: Double
    public let maximum: Double
    public init(minimum: Double, maximum: Double) { self.minimum = minimum; self.maximum = maximum }
    public func supports(_ framesPerSecond: Int) -> Bool {
        minimum.isFinite && maximum.isFinite && minimum > 0 && minimum <= maximum
            && minimum <= Double(framesPerSecond) && Double(framesPerSecond) <= maximum
    }
}

/// Candidates come exclusively from the already-selected physical rear wide camera.
/// Identifiers refer to retained native format objects, never to another camera/lens.
public struct CameraFormatCandidate {
    public let identifier: Int
    public let width: Int
    public let height: Int
    public let fieldOfView: Double
    public let frameRates: [CameraFrameRateInterval]
    public init(identifier: Int, width: Int, height: Int, fieldOfView: Double, frameRates: [CameraFrameRateInterval]) {
        self.identifier = identifier; self.width = width; self.height = height
        self.fieldOfView = fieldOfView; self.frameRates = frameRates
    }
    public func supports(_ preset: CameraCapturePreset) -> Bool {
        width == preset.sensorWidth && height == preset.sensorHeight
            && fieldOfView.isFinite && fieldOfView > 0 && fieldOfView < 180
            && frameRates.contains { $0.supports(preset.framesPerSecond) }
    }
}

public struct CameraPresetPlan {
    public let preset: CameraCapturePreset
    public let formatIdentifier: Int
    public let fieldOfView: Double
    public var framesPerSecond: Int { preset.framesPerSecond }
    public func sharesOpticalFormat(with other: CameraPresetPlan) -> Bool {
        formatIdentifier == other.formatIdentifier && preset.sensorWidth == other.preset.sensorWidth
            && preset.sensorHeight == other.preset.sensorHeight && fieldOfView == other.fieldOfView
    }
}

public enum CameraPresetPlanner {
    public static func plans(for formats: [CameraFormatCandidate]) -> [CameraCapturePreset: CameraPresetPlan] {
        var result: [CameraCapturePreset: CameraPresetPlan] = [:]
        // Preserve the original Standard selection: exact 1080p, 15 FPS, largest FOV.
        if let standard = formats.filter({ $0.supports(.standard) }).max(by: { $0.fieldOfView < $1.fieldOfView }) {
            result[.standard] = CameraPresetPlan(preset: .standard, formatIdentifier: standard.identifier, fieldOfView: standard.fieldOfView)
            // FPS-only reuse is allowed only on this very same native format.
            // A different 1080p/30 format may have a different crop or optical mapping.
            if standard.supports(.smooth) {
                result[.smooth] = CameraPresetPlan(preset: .smooth, formatIdentifier: standard.identifier, fieldOfView: standard.fieldOfView)
            }
        }
        if let detail = formats.filter({ $0.supports(.detail) }).max(by: { $0.fieldOfView < $1.fieldOfView }) {
            result[.detail] = CameraPresetPlan(preset: .detail, formatIdentifier: detail.identifier, fieldOfView: detail.fieldOfView)
        }
        return result
    }
}
