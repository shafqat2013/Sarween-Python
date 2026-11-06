import XCTest
@testable import SarweenCameraProtocol

final class CapturePresetTests: XCTestCase {
    func format(_ id: Int, width: Int = 1920, height: Int = 1080, fov: Double = 70,
                minimum: Double = 1, maximum: Double = 30) -> CameraFormatCandidate {
        CameraFormatCandidate(identifier: id, width: width, height: height, fieldOfView: fov,
                              frameRates: [CameraFrameRateInterval(minimum: minimum, maximum: maximum)])
    }

    func testStandardPreservesExistingLargestFOVSelection() throws {
        let candidates = [format(1, fov: 60), format(2, fov: 76), format(3, fov: 68)]
        let plans = CameraPresetPlanner.plans(for: candidates)
        XCTAssertEqual(try XCTUnwrap(plans[.standard]).formatIdentifier, 2)
        XCTAssertEqual(plans[.standard]?.framesPerSecond, 15)
    }

    func testSmoothReusesExactStandardFormatRatherThanAnother30FPSFormat() throws {
        let plans = CameraPresetPlanner.plans(for: [format(1, fov: 75, maximum: 15), format(2, fov: 65, maximum: 60)])
        XCTAssertEqual(plans[.standard]?.formatIdentifier, 1)
        XCTAssertNil(plans[.smooth], "Do not silently select a different crop just to enable 30 FPS")
    }

    func testSmoothSharesOpticsWhenSameFormatSupports30FPS() throws {
        let plans = CameraPresetPlanner.plans(for: [format(8, maximum: 60)])
        let standard = try XCTUnwrap(plans[.standard])
        let smooth = try XCTUnwrap(plans[.smooth])
        XCTAssertEqual(smooth.framesPerSecond, 30)
        XCTAssertTrue(standard.sharesOpticalFormat(with: smooth))
    }

    func testFourKIsGatedByExactNativeDimensionsAnd15FPSSupport() {
        let plans = CameraPresetPlanner.plans(for: [
            format(0), format(1, width: 3840, height: 2160, minimum: 24, maximum: 60),
            format(2, width: 4096, height: 2160), format(3, width: 1280, height: 720)
        ])
        XCTAssertNil(plans[.detail])
        XCTAssertNotNil(plans[.standard])
    }

    func testSupportedFourKUsesNativeFormatAndDifferentOptics() throws {
        let plans = CameraPresetPlanner.plans(for: [format(1), format(2, width: 3840, height: 2160, maximum: 30)])
        let detail = try XCTUnwrap(plans[.detail])
        XCTAssertEqual(detail.formatIdentifier, 2)
        XCTAssertTrue(detail.preset.experimental)
        XCTAssertFalse(detail.sharesOpticalFormat(with: try XCTUnwrap(plans[.standard])))
    }

    func testNoUpscalingOrWrongResolutionFallback() {
        let plans = CameraPresetPlanner.plans(for: [format(0, width: 1280, height: 720), format(1, width: 3840, height: 2160)])
        XCTAssertNil(plans[.standard])
        XCTAssertNil(plans[.smooth])
        XCTAssertNotNil(plans[.detail])
    }

    func testAllDeclaredRangesAreConsidered() {
        let candidate = CameraFormatCandidate(identifier: 9, width: 1920, height: 1080, fieldOfView: 75,
            frameRates: [CameraFrameRateInterval(minimum: 1, maximum: 15), CameraFrameRateInterval(minimum: 30, maximum: 30)])
        XCTAssertNotNil(CameraPresetPlanner.plans(for: [candidate])[.smooth])
    }

    func testInvalidCapabilitiesDoNotEnablePresets() {
        let candidates = [format(0, fov: .nan), format(1, minimum: .nan), format(2, maximum: .infinity),
                          format(3, minimum: 30, maximum: 15), format(4, fov: -10)]
        XCTAssertTrue(CameraPresetPlanner.plans(for: candidates).isEmpty)
    }

    func testPortraitAndLandscapeDimensionsPreservePixelCount() {
        for preset in CameraCapturePreset.allCases {
            XCTAssertEqual(preset.width(for: .portrait), preset.height(for: .landscape))
            XCTAssertEqual(preset.height(for: .portrait), preset.width(for: .landscape))
            XCTAssertEqual(preset.width(for: .portrait) * preset.height(for: .portrait), preset.sensorWidth * preset.sensorHeight)
        }
        XCTAssertEqual(CameraCapturePreset.detail.width(for: .portrait), 2160)
        XCTAssertEqual(CameraCapturePreset.detail.height(for: .portrait), 3840)
    }

    func testCapabilityMetadataIsSupportedPresetAndOrientationSpecific() throws {
        let metadata = CameraCapturePreset.detail.capabilityJSON(orientation: .portrait)
        XCTAssertEqual(metadata["id"] as? String, "detail")
        XCTAssertEqual(metadata["width"] as? Int, 2160)
        XCTAssertEqual(metadata["height"] as? Int, 3840)
        XCTAssertEqual(metadata["fps"] as? Int, 15)
        XCTAssertEqual(metadata["experimental"] as? Bool, true)
        XCTAssertNoThrow(try CameraWire.encode(header: ["type": "hello", "availableCapturePresets": [metadata]]))
    }

    func testStandardPreservesPreviouslyShippedSourceIdentity() {
        XCTAssertEqual(CameraCapturePreset.standard.opticalSourceID(installationID: "phone-one", orientation: .portrait),
            "sarween-iphone:phone-one:back-wide:1080x1920:rotation0:mirror0:zoom1:stabilization-off:gdc-off:hdr-off")
    }

    func testFPSOnlyModeSharesSourceIdentityButFourKAndRotationDoNot() {
        let standard = CameraCapturePreset.standard.opticalSourceID(installationID: "phone-one", orientation: .portrait)
        XCTAssertEqual(standard, CameraCapturePreset.smooth.opticalSourceID(installationID: "phone-one", orientation: .portrait))
        XCTAssertNotEqual(standard, CameraCapturePreset.detail.opticalSourceID(installationID: "phone-one", orientation: .portrait))
        XCTAssertNotEqual(standard, CameraCapturePreset.standard.opticalSourceID(installationID: "phone-one", orientation: .landscape))
        XCTAssertNotEqual(standard, CameraCapturePreset.standard.opticalSourceID(installationID: "phone-two", orientation: .portrait))
    }
}
