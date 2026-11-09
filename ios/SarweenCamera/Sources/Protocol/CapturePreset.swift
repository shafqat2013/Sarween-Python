import Foundation

public enum CameraOrientation: String, CaseIterable, Identifiable {
    case portrait
    case landscape
    public var id: String { rawValue }
    public var label: String { self == .portrait ? "Portrait" : "Landscape" }
}

public enum CameraLens: String, CaseIterable, Identifiable {
    case wide
    case ultraWide = "ultra-wide"
    case telephoto
    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .wide: return "Wide"
        case .ultraWide: return "Ultra Wide"
        case .telephoto: return "Telephoto"
        }
    }
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
    public func opticalSourceID(installationID: String, orientation: CameraOrientation, lens: CameraLens = .wide) -> String {
        // Standard and Smooth are offered on the identical retained native format.
        // FPS is temporal metadata, while resolution/orientation change calibration.
        "sarween-iphone:\(installationID):back-\(lens.rawValue):\(width(for: orientation))x\(height(for: orientation)):rotation0:mirror0:zoom1:stabilization-off:gdc-off:hdr-off"
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

/// Candidates come exclusively from one already-selected physical rear camera.
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

public struct CameraPresetPlan: Equatable {
    public let preset: CameraCapturePreset
    public let formatIdentifier: Int
    public let fieldOfView: Double
    public var framesPerSecond: Int { preset.framesPerSecond }
    public func sharesOpticalFormat(with other: CameraPresetPlan) -> Bool {
        formatIdentifier == other.formatIdentifier && preset.sensorWidth == other.preset.sensorWidth
            && preset.sensorHeight == other.preset.sensorHeight && fieldOfView == other.fieldOfView
    }
}

public struct CameraLensCapability: Equatable {
    public let lens: CameraLens
    public let plans: [CameraCapturePreset: CameraPresetPlan]
    public let torchSupported: Bool
    public init(lens: CameraLens, plans: [CameraCapturePreset: CameraPresetPlan], torchSupported: Bool) {
        self.lens = lens; self.plans = plans; self.torchSupported = torchSupported
    }
    public var supportedPresets: [CameraCapturePreset] {
        CameraCapturePreset.allCases.filter { plans[$0] != nil }
    }
    public func capabilityJSON(orientation: CameraOrientation) -> [String: Any] {
        ["lens": lens.rawValue, "label": lens.label, "torchSupported": torchSupported,
         "availableCapturePresets": supportedPresets.map { preset -> [String: Any] in
             var result = preset.capabilityJSON(orientation: orientation)
             result["nativeFieldOfViewDegrees"] = plans[preset]!.fieldOfView
             return result
         }]
    }
}

public enum CameraLensPlanner {
    /// Only physical lenses with the baseline mode are selectable. Preserve the
    /// chosen lens/preset when supported, otherwise choose Wide/Standard explicitly.
    public static func selection(lens: CameraLens, preset: CameraCapturePreset,
                                 capabilities: [CameraLensCapability]) -> (lens: CameraLens, preset: CameraCapturePreset)? {
        let usable = capabilities.filter { $0.plans[.standard] != nil }
        guard let selected = usable.first(where: { $0.lens == lens })
                ?? usable.first(where: { $0.lens == .wide }) ?? usable.first else { return nil }
        return (selected.lens, selected.plans[preset] != nil ? preset : .standard)
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
