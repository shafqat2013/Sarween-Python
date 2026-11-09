import XCTest
@testable import SarweenCameraProtocol

final class CameraLensTests: XCTestCase {
    private func capability(_ lens: CameraLens, smooth: Bool = true, detail: Bool = false,
                            fov: Double = 70, torch: Bool = false) -> CameraLensCapability {
        var formats = [CameraFormatCandidate(identifier: 1, width: 1920, height: 1080, fieldOfView: fov,
            frameRates: [CameraFrameRateInterval(minimum: 1, maximum: smooth ? 60 : 15)])]
        if detail {
            formats.append(CameraFormatCandidate(identifier: 2, width: 3840, height: 2160, fieldOfView: fov,
                frameRates: [CameraFrameRateInterval(minimum: 1, maximum: 30)]))
        }
        return CameraLensCapability(lens: lens, plans: CameraPresetPlanner.plans(for: formats), torchSupported: torch)
    }

    func testLensWireNamesAreExplicitPhysicalLensChoices() {
        XCTAssertEqual(CameraLens.allCases.map(\.rawValue), ["wide", "ultra-wide", "telephoto"])
        XCTAssertNil(CameraLens(rawValue: "triple"))
        XCTAssertNil(CameraLens(rawValue: "dual"))
    }

    func testLensChoicePersistsWhenItsPresetIsSupported() throws {
        let selected = try XCTUnwrap(CameraLensPlanner.selection(lens: .ultraWide, preset: .detail,
            capabilities: [capability(.wide), capability(.ultraWide, detail: true, fov: 115)]))
        XCTAssertEqual(selected.lens, .ultraWide)
        XCTAssertEqual(selected.preset, .detail)
    }

    func testLensChangeFallsBackToStandardOnSameLensRatherThanChangingLens() throws {
        let selected = try XCTUnwrap(CameraLensPlanner.selection(lens: .telephoto, preset: .detail,
            capabilities: [capability(.wide, detail: true), capability(.telephoto, smooth: false, fov: 27)]))
        XCTAssertEqual(selected.lens, .telephoto)
        XCTAssertEqual(selected.preset, .standard)
    }

    func testMissingSavedLensFallsBackToWideWithSupportedPreset() throws {
        let selected = try XCTUnwrap(CameraLensPlanner.selection(lens: .ultraWide, preset: .smooth,
            capabilities: [capability(.telephoto), capability(.wide)]))
        XCTAssertEqual(selected.lens, .wide)
        XCTAssertEqual(selected.preset, .smooth)
    }

    func testLensWithoutBaselineCannotBeSelected() throws {
        let detailOnly = CameraLensCapability(lens: .ultraWide,
            plans: CameraPresetPlanner.plans(for: [CameraFormatCandidate(identifier: 8, width: 3840, height: 2160,
                fieldOfView: 115, frameRates: [CameraFrameRateInterval(minimum: 1, maximum: 30)])]), torchSupported: false)
        let selected = try XCTUnwrap(CameraLensPlanner.selection(lens: .ultraWide, preset: .detail,
            capabilities: [detailOnly, capability(.wide)]))
        XCTAssertEqual(selected.lens, .wide)
        XCTAssertEqual(selected.preset, .standard)
        XCTAssertNil(CameraLensPlanner.selection(lens: .ultraWide, preset: .detail, capabilities: [detailOnly]))
        XCTAssertNil(CameraLensPlanner.selection(lens: .wide, preset: .standard, capabilities: []))
    }

    func testSourceIdentitySeparatesSameResolutionPhysicalLenses() {
        let ids = CameraLens.allCases.map {
            CameraCapturePreset.standard.opticalSourceID(installationID: "phone-one", orientation: .portrait, lens: $0)
        }
        XCTAssertEqual(Set(ids).count, 3)
        XCTAssertTrue(ids[0].contains(":back-wide:1080x1920:"))
        XCTAssertTrue(ids[1].contains(":back-ultra-wide:1080x1920:"))
        XCTAssertTrue(ids[2].contains(":back-telephoto:1080x1920:"))
    }

    func testFPSOnlyReuseIsScopedToSelectedLens() {
        for lens in CameraLens.allCases {
            XCTAssertEqual(CameraCapturePreset.standard.opticalSourceID(installationID: "phone-one", orientation: .portrait, lens: lens),
                CameraCapturePreset.smooth.opticalSourceID(installationID: "phone-one", orientation: .portrait, lens: lens))
        }
    }

    func testCapabilityMetadataIncludesOnlyThisLensSupportedModesAndNativeFOV() throws {
        let camera = capability(.ultraWide, smooth: false, fov: 114.25, torch: false)
        let json = camera.capabilityJSON(orientation: .portrait)
        XCTAssertEqual(json["lens"] as? String, "ultra-wide")
        XCTAssertEqual(json["torchSupported"] as? Bool, false)
        let modes = try XCTUnwrap(json["availableCapturePresets"] as? [[String: Any]])
        XCTAssertEqual(modes.count, 1)
        XCTAssertEqual(modes[0]["id"] as? String, "standard")
        XCTAssertEqual(modes[0]["width"] as? Int, 1080)
        XCTAssertEqual(modes[0]["height"] as? Int, 1920)
        XCTAssertEqual(modes[0]["nativeFieldOfViewDegrees"] as? Double, 114.25)
        let packet = try CameraWire.encode(header: ["type": "hello", "availableCameras": [json]])
        var decoder = CameraPacketDecoder()
        let decoded = try decoder.append(packet)
        let cameras = try XCTUnwrap(decoded.first?.header["availableCameras"] as? [[String: Any]])
        XCTAssertEqual(cameras.first?["lens"] as? String, "ultra-wide")
    }
}
