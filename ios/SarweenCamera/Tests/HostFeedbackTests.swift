import XCTest
@testable import SarweenCameraProtocol

final class HostFeedbackTests: XCTestCase {
    private func fields(sequence: Any = 12) -> [String: Any] {
        ["kind": "preview", "sourceSeq": sequence,
         "markers": ["ids": [10, 11, 12, 13], "requiredIds": [10, 11, 12, 13], "count": 4, "required": 4, "locked": true],
         "cpl": ["state": "ready", "index": 37.5, "target": 40.0, "message": "Turn only the filter."]]
    }
    private func parsed(_ fields: [String: Any]? = nil) throws -> CameraHostFeedback {
        try XCTUnwrap(CameraHostFeedback(json: fields ?? self.fields()))
    }

    func testForegroundPreviewDoesNotRequireAHostAndDoesNotReserveJPEGWork() {
        XCTAssertTrue(CameraPreviewPolicy.shouldRun(active: true, authorized: true, capabilitiesReady: true, paused: false))
        let gate = CameraFrameFlowGate()
        gate.setEnabled(true)
        XCTAssertNil(gate.reserve(), "Local preview alone must never request JPEG encoding")
        gate.reset(sessionID: "host-session")
        XCTAssertNil(gate.reserve(), "Connecting a host without start still must not encode")
        gate.setRequested(true)
        XCTAssertNotNil(gate.reserve())
    }

    func testPreviewStopsWhenPausedBackgroundedOrUnauthorized() {
        XCTAssertFalse(CameraPreviewPolicy.shouldRun(active: false, authorized: true, capabilitiesReady: true, paused: false))
        XCTAssertFalse(CameraPreviewPolicy.shouldRun(active: true, authorized: false, capabilitiesReady: true, paused: false))
        XCTAssertFalse(CameraPreviewPolicy.shouldRun(active: true, authorized: true, capabilitiesReady: false, paused: false))
        XCTAssertFalse(CameraPreviewPolicy.shouldRun(active: true, authorized: true, capabilitiesReady: true, paused: true))
        XCTAssertFalse(CameraPreviewPolicy.canChangeMode(authorized: true, capabilitiesReady: true, paused: false, running: false))
        XCTAssertFalse(CameraPreviewPolicy.canChangeMode(authorized: true, capabilitiesReady: true, paused: true, running: true))
        XCTAssertTrue(CameraPreviewPolicy.canChangeMode(authorized: true, capabilitiesReady: true, paused: true, running: false))
    }

    func testKnownFeedbackFieldsAreParsedAndUnknownFieldsIgnored() throws {
        var input = fields(); input["futureOptionalField"] = ["anything": true]
        let feedback = try parsed(input)
        XCTAssertEqual(feedback.kind, .preview)
        XCTAssertEqual(feedback.sourceSequence, 12)
        XCTAssertEqual(feedback.markers?.count, 4)
        XCTAssertEqual(feedback.markers?.complete, true)
        XCTAssertEqual(feedback.cpl?.index, 37.5)
        XCTAssertEqual(feedback.cpl?.target, 40)
        XCTAssertEqual(feedback.cpl?.message, "Turn only the filter.")
    }

    func testReportedCountCannotHideMissingRequiredIDsOrUnlockedRegistration() throws {
        var input = fields()
        input["markers"] = ["ids": [10, 11, 12], "requiredIds": [10, 11, 12, 13], "count": 4, "required": 4]
        XCTAssertNil(try parsed(input).markers)
        input["markers"] = ["ids": [10, 11, 12, 13], "requiredIds": [10, 11, 12, 13], "count": 4, "required": 4, "locked": false]
        XCTAssertEqual(try parsed(input).markers?.complete, false)
        input["markers"] = ["ids": [20, 21, 22, 23], "requiredIds": [20, 21, 22, 23], "count": 4, "required": 4, "locked": true]
        XCTAssertEqual(try parsed(input).markers?.complete, true)
    }

    func testMalformedMarkerFieldsNeverProduceGreenBadge() throws {
        for markers: [String: Any] in [
            ["ids": [10, 10, 12, 13], "count": 4, "required": 4],
            ["ids": [10, 11, 12, 13], "count": true, "required": 4],
            ["ids": [10, 11, 12, 13], "count": 4, "required": 0],
            ["ids": [10, 11, 12, 13], "count": 4, "required": 4, "requiredIds": [-1]],
            ["ids": [10, 11, 12, 13], "count": 4, "required": 4, "locked": 1],
            ["ids": [10], "requiredIds": [10], "count": 1, "required": 1, "locked": true],
            ["ids": [10, 11, 12, 250], "requiredIds": [10, 11, 12, 250], "count": 4, "required": 4],
            ["ids": [10, 11, 12, 13], "requiredIds": [20, 21, 22, 23], "count": 4, "required": 4]
        ] {
            var input = fields(); input["markers"] = markers
            XCTAssertNil(try parsed(input).markers)
        }
    }

    func testMissingSourcePixelsClearOldMarkerCountsAndLCDReading() throws {
        var input = fields(); input.removeValue(forKey: "sourceSeq")
        let feedback = try parsed(input)
        XCTAssertNil(feedback.markers)
        XCTAssertNil(feedback.cpl?.index)
        var state = CameraHostFeedbackState(); state.reset(sessionID: "one")
        XCTAssertTrue(state.receive(try parsed(), sessionID: "one", now: 1))
        XCTAssertTrue(state.receive(feedback, sessionID: "one", now: 2))
        XCTAssertNil(state.fresh(now: 2)?.markers)
    }

    func testFreshnessExpiresAtThreeSecondsAndRepeatedFrameCannotRefreshIt() throws {
        var state = CameraHostFeedbackState(); state.reset(sessionID: "one")
        let feedback = try parsed()
        XCTAssertTrue(state.receive(feedback, sessionID: "one", now: 10))
        XCTAssertNotNil(state.fresh(now: 12.99))
        XCTAssertTrue(state.receive(feedback, sessionID: "one", now: 12.9))
        XCTAssertNil(state.fresh(now: 13), "Repeated sourceSeq must not keep old markers green")
        XCTAssertTrue(state.receive(try parsed(fields(sequence: 13)), sessionID: "one", now: 14))
        XCTAssertNotNil(state.fresh(now: 14))
    }

    func testOlderStatusAndOtherSessionsAreRejectedAndDisconnectClearsState() throws {
        var state = CameraHostFeedbackState(); state.reset(sessionID: "one")
        XCTAssertFalse(state.receive(try parsed(), sessionID: "two", now: 10))
        XCTAssertTrue(state.receive(try parsed(), sessionID: "one", now: 10))
        XCTAssertFalse(state.receive(try parsed(fields(sequence: 11)), sessionID: "one", now: 11))
        XCTAssertEqual(state.fresh(now: 11)?.sourceSequence, 12)
        state.reset(sessionID: nil)
        XCTAssertNil(state.fresh(now: 11))
        state.reset(sessionID: "new")
        XCTAssertTrue(state.receive(try parsed(fields(sequence: 1)), sessionID: "new", now: 12))
    }

    func testInvalidSourceSequenceIsNotAccepted() {
        for value: Any in [true, -1, 1.5, "12", Double.infinity] {
            XCTAssertNil(CameraHostFeedback(json: fields(sequence: value)))
        }
    }

    func testCalibrationFieldsAreBoundedAndActiveCanBeExplicitlyEnded() throws {
        var input = fields(); input["kind"] = "calibration"
        input["calibration"] = ["active": true, "captured": 7, "required": 10, "title": "ChArUco calibration",
            "message": String(repeating: "x", count: 900), "reason": "Hold still", "mode": "sweep", "canFinish": false,
            "countdownSeconds": 2.5]
        let feedback = try parsed(input)
        XCTAssertEqual(feedback.calibration?.captured, 7)
        XCTAssertEqual(feedback.calibration?.message.count, 500)
        XCTAssertEqual(feedback.calibration?.countdownSeconds, 2.5)
        XCTAssertEqual(feedback.calibration?.canFinish, false)
        input["calibration"] = ["active": true, "captured": 7, "required": 10, "canFinish": true]
        XCTAssertEqual(try parsed(input).calibration?.canFinish, false)
        input["calibration"] = ["active": true, "captured": 11, "required": 10, "canFinish": true]
        XCTAssertNil(try parsed(input).calibration)
        input["calibration"] = ["active": false, "captured": 10, "required": 10, "canFinish": false]
        XCTAssertEqual(try parsed(input).calibration?.active, false)
    }

    func testNonfiniteAndBooleanCPLReadingsAreIgnored() throws {
        for value: Any in [Double.nan, Double.infinity, -1, true] {
            var input = fields(); input["cpl"] = ["state": "invalid", "index": value]
            XCTAssertNil(try parsed(input).cpl?.index)
        }
    }

    func testOnlyApprovedPhoneActionsExist() {
        XCTAssertEqual(Set(CameraPhoneAction.allCases.map(\.rawValue)),
            ["calibration.capture", "calibration.finish", "calibration.cancel", "cpl.reference", "cpl.target", "cpl.reset"])
        XCTAssertNil(CameraPhoneAction(rawValue: "tracking.move"))
        XCTAssertNil(CameraPhoneAction(rawValue: "calibration.start"))
    }

    func testGuideShowsCurrentMeasurementAndSavedGuideAsDifferentValues() throws {
        var input = fields()
        input["cpl"] = ["state": "comparable", "index": 37.5, "message": "Rough LCD brightness guide."]
        let value = CameraGuidePresentation(feedback: try parsed(input), hostConnected: true,
            cameraRunning: true, calibrationActive: false, savedGuide: 28)
        XCTAssertEqual(value.markerValue, "4/4")
        XCTAssertEqual(value.markerDetail, "Locked")
        XCTAssertTrue(value.markersComplete)
        XCTAssertEqual(value.brightnessValue, "37.5")
        XCTAssertTrue(value.brightnessIsLive)
        XCTAssertEqual(value.savedGuideText, "Saved guide: 28.0")
        XCTAssertEqual(value.message, "Rough LCD brightness guide.")
    }

    func testInvalidMeasurementKeepsItsReasonAndDashEvenIfAnIndexIsPresent() throws {
        var input = fields()
        input["cpl"] = ["state": "invalid", "index": 37.5, "message": "Camera framing moved."]
        let value = CameraGuidePresentation(feedback: try parsed(input), hostConnected: true,
            cameraRunning: true, calibrationActive: false, savedGuide: 28)
        XCTAssertEqual(value.brightnessValue, "—")
        XCTAssertFalse(value.brightnessIsLive)
        XCTAssertEqual(value.message, "Camera framing moved.")
        XCTAssertEqual(value.savedGuideText, "Saved guide: 28.0")
        XCTAssertEqual(value.markerValue, "4/4", "CPL comparability and visible-marker count are separate")
    }

    func testExpiredFeedbackNeverShowsSavedGuideAsALiveReadingOrGreenMarkers() throws {
        var input = fields()
        input["cpl"] = ["state": "target-matched", "index": 28.0, "target": 28.0]
        var state = CameraHostFeedbackState(); state.reset(sessionID: "one")
        XCTAssertTrue(state.receive(try parsed(input), sessionID: "one", now: 10))
        let value = CameraGuidePresentation(feedback: state.fresh(now: 13), hostConnected: true,
            cameraRunning: true, calibrationActive: false, savedGuide: 28)
        XCTAssertEqual(value.brightnessValue, "—")
        XCTAssertEqual(value.markerValue, "—")
        XCTAssertFalse(value.markersComplete)
        XCTAssertFalse(value.brightnessIsLive)
        XCTAssertEqual(value.savedGuideText, "Saved guide: 28.0")
        XCTAssertFalse(value.message.isEmpty)
    }

    func testStoppedOrDisconnectedCameraClearsAllLiveGuideValues() throws {
        for (connected, running) in [(false, true), (true, false), (false, false)] {
            let value = CameraGuidePresentation(feedback: try parsed(), hostConnected: connected,
                cameraRunning: running, calibrationActive: false, savedGuide: 28)
            XCTAssertEqual(value.markerValue, "—")
            XCTAssertFalse(value.markersComplete)
            XCTAssertEqual(value.brightnessValue, "—")
            XCTAssertFalse(value.message.isEmpty)
            XCTAssertEqual(value.savedGuideText, "Saved guide: 28.0")
        }
    }

    func testVisibleMarkersDoNotInventARegistrationLock() throws {
        var input = fields()
        var markers = try XCTUnwrap(input["markers"] as? [String: Any])
        markers.removeValue(forKey: "locked"); input["markers"] = markers
        var value = CameraGuidePresentation(feedback: try parsed(input), hostConnected: true,
            cameraRunning: true, calibrationActive: false)
        XCTAssertEqual(value.markerValue, "4/4")
        XCTAssertEqual(value.markerDetail, "All visible")
        markers["locked"] = false; input["markers"] = markers
        value = CameraGuidePresentation(feedback: try parsed(input), hostConnected: true,
            cameraRunning: true, calibrationActive: false)
        XCTAssertEqual(value.markerDetail, "Lock pending")
        XCTAssertFalse(value.markersComplete)
    }

    func testMissingCPLReasonAndCalibrationStillHaveExplicitInstructions() throws {
        var input = fields(); input["cpl"] = ["state": "reference-needed"]
        var value = CameraGuidePresentation(feedback: try parsed(input), hostConnected: true,
            cameraRunning: true, calibrationActive: false)
        XCTAssertFalse(value.message.isEmpty)
        XCTAssertEqual(value.brightnessValue, "—")
        value = CameraGuidePresentation(feedback: try parsed(input), hostConnected: true,
            cameraRunning: true, calibrationActive: true)
        XCTAssertEqual(value.markerDetail, "Calibrating")
        XCTAssertEqual(value.markerValue, "—")
        XCTAssertFalse(value.brightnessIsLive)
        XCTAssertTrue(value.message.contains("Finish camera calibration"))
    }
}
