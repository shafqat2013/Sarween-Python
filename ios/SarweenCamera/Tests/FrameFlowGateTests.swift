import XCTest
@testable import SarweenCameraProtocol

final class FrameFlowGateTests: XCTestCase {
    func readyGate() -> CameraFrameFlowGate {
        let gate = CameraFrameFlowGate()
        gate.reset(sessionID: "one")
        gate.setEnabled(true)
        gate.setRequested(true)
        return gate
    }

    func testNeedsLocalConsentAndHostStart() {
        let gate = CameraFrameFlowGate()
        gate.reset(sessionID: "one")
        XCTAssertNil(gate.reserve())
        gate.setEnabled(true)
        XCTAssertNil(gate.reserve())
        gate.setRequested(true)
        XCTAssertEqual(gate.reserve()?.sequence, 1)
    }

    func testACKAndSendCompletionAreBothNecessary() throws {
        let gate = readyGate()
        let first = try XCTUnwrap(gate.reserve())
        XCTAssertNil(gate.reserve())
        gate.didSend(first)
        XCTAssertNil(gate.reserve())
        gate.acknowledge(sessionID: "one", sequence: 999)
        XCTAssertNil(gate.reserve())
        gate.acknowledge(sessionID: "one", sequence: first.sequence)
        XCTAssertEqual(gate.reserve()?.sequence, 2)
    }

    func testACKBeforeSendCompletionDoesNotQueueAnotherFrame() throws {
        let gate = readyGate()
        let first = try XCTUnwrap(gate.reserve())
        gate.acknowledge(sessionID: "one", sequence: first.sequence)
        XCTAssertNil(gate.reserve())
        gate.didSend(first)
        XCTAssertEqual(gate.reserve()?.sequence, 2)
    }

    func testOldSessionACKCannotReleaseNewFrame() throws {
        let gate = readyGate()
        let old = try XCTUnwrap(gate.reserve())
        gate.reset(sessionID: "two")
        gate.setRequested(true)
        let fresh = try XCTUnwrap(gate.reserve())
        gate.didSend(old)
        gate.acknowledge(sessionID: "one", sequence: old.sequence)
        XCTAssertTrue(gate.isPending(fresh))
        XCTAssertNil(gate.reserve())
        gate.didSend(fresh)
        gate.acknowledge(sessionID: "two", sequence: fresh.sequence)
        XCTAssertEqual(gate.reserve()?.sequence, 2)
    }

    func testNoACKModeStillWaitsForSendCompletion() throws {
        let gate = readyGate()
        gate.setRequested(true, requiresAcknowledgement: false)
        let first = try XCTUnwrap(gate.reserve())
        XCTAssertNil(gate.reserve())
        gate.didSend(first)
        XCTAssertEqual(gate.reserve()?.sequence, 2)
    }

    func testPauseDropsReservationAndRequiresExplicitResume() throws {
        let gate = readyGate()
        let first = try XCTUnwrap(gate.reserve())
        gate.setEnabled(false)
        gate.didSend(first)
        gate.acknowledge(sessionID: "one", sequence: first.sequence)
        XCTAssertNil(gate.reserve())
        gate.setEnabled(true)
        XCTAssertEqual(gate.reserve()?.sequence, 2)
        gate.setRequested(false)
        XCTAssertNil(gate.reserve())
    }
}
