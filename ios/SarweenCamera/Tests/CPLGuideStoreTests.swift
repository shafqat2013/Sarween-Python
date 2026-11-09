import Foundation
import XCTest
@testable import SarweenCameraProtocol

final class CPLGuideStoreTests: XCTestCase {
    func withStore(_ body: (CameraCPLGuideStore, UserDefaults) throws -> Void) rethrows {
        let name = "SarweenCPLGuideTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        try body(CameraCPLGuideStore(defaults: defaults), defaults)
    }
    func feedback(state: String = "target-matched", source: Bool = true, index: Double? = 32, target: Double? = 32) -> CameraHostFeedback {
        var cpl: [String: Any] = ["state": state]
        if let index { cpl["index"] = index }
        if let target { cpl["target"] = target }
        var input: [String: Any] = ["kind": "tracking", "cpl": cpl]
        if source { input["sourceSeq"] = 3 }
        return CameraHostFeedback(json: input)!
    }
    func test_confirmed_target_survives_reopening_but_is_scoped_to_capture_mode() {
        withStore { store, defaults in
            XCTAssertTrue(store.rememberConfirmedTarget(from: feedback(), lens: .wide, preset: .detail, orientation: .portrait))
            let reopened = CameraCPLGuideStore(defaults: defaults)
            XCTAssertEqual(reopened.value(lens: .wide, preset: .detail, orientation: .portrait), 32)
            XCTAssertNil(reopened.value(lens: .ultraWide, preset: .detail, orientation: .portrait))
            XCTAssertNil(reopened.value(lens: .wide, preset: .standard, orientation: .portrait))
            XCTAssertNil(reopened.value(lens: .wide, preset: .detail, orientation: .landscape))
            reopened.forget(lens: .wide, preset: .detail, orientation: .portrait)
            XCTAssertNil(store.value(lens: .wide, preset: .detail, orientation: .portrait))
        }
    }
    func test_invalid_or_unconfirmed_feedback_cannot_replace_a_saved_guide() {
        withStore { store, _ in
            store.rememberConfirmedTarget(from: feedback(), lens: .wide, preset: .detail, orientation: .portrait)
            for input in [feedback(state: "unavailable", target: 90), feedback(source: false, target: 90),
                          feedback(index: nil, target: 90), feedback(target: nil)] {
                XCTAssertFalse(store.rememberConfirmedTarget(from: input, lens: .wide, preset: .detail, orientation: .portrait))
                XCTAssertEqual(store.value(lens: .wide, preset: .detail, orientation: .portrait), 32)
            }
        }
    }
}
