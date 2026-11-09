import Foundation

/// A remembered setup hint, never a live reading or an optical measurement.
/// A new bright reference is still needed in each Mac camera session.
public final class CameraCPLGuideStore {
    private struct Guide: Codable { let index: Double }
    private let defaults: UserDefaults
    public init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    private func key(lens: CameraLens, preset: CameraCapturePreset, orientation: CameraOrientation) -> String {
        "savedCPLGuide.v1.\(lens.rawValue).\(preset.rawValue).\(orientation.rawValue)"
    }

    public func value(lens: CameraLens, preset: CameraCapturePreset, orientation: CameraOrientation) -> Double? {
        guard let data = defaults.data(forKey: key(lens: lens, preset: preset, orientation: orientation)),
              let guide = try? JSONDecoder().decode(Guide.self, from: data),
              guide.index.isFinite, (0...1_000_000).contains(guide.index) else { return nil }
        return guide.index
    }

    /// The caller must pass freshly accepted feedback from the current session.
    /// Persist only a target confirmed by the host, never the button's intent.
    @discardableResult public func rememberConfirmedTarget(from feedback: CameraHostFeedback,
        lens: CameraLens, preset: CameraCapturePreset, orientation: CameraOrientation) -> Bool {
        guard feedback.sourceSequence != nil, let cpl = feedback.cpl,
              ["comparable", "target-matched"].contains(cpl.state), cpl.index != nil,
              let target = cpl.target, target.isFinite, (0...1_000_000).contains(target),
              target != value(lens: lens, preset: preset, orientation: orientation),
              let data = try? JSONEncoder().encode(Guide(index: target)) else { return false }
        defaults.set(data, forKey: key(lens: lens, preset: preset, orientation: orientation))
        return true
    }

    public func forget(lens: CameraLens, preset: CameraCapturePreset, orientation: CameraOrientation) {
        defaults.removeObject(forKey: key(lens: lens, preset: preset, orientation: orientation))
    }
}
