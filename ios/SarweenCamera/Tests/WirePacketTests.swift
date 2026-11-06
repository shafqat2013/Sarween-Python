import Foundation
import XCTest
@testable import SarweenCameraProtocol

final class WirePacketTests: XCTestCase {
    func testFragmentedJPEGIsReassembledExactly() throws {
        let jpeg = Data([255, 216, 1, 2, 3, 255, 217])
        let encoded = try CameraWire.encode(header: ["type": "frame", "seq": 7], payload: jpeg)
        var decoder = CameraPacketDecoder()
        var output: [CameraWirePacket] = []
        for byte in encoded { output += try decoder.append(Data([byte])) }
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output.first?.payload, jpeg)
        XCTAssertEqual(output.first?.header["seq"] as? Int, 7)
        XCTAssertEqual(output.first?.header["payloadBytes"] as? Int, jpeg.count)
    }

    func testCoalescedCommandsAndPartialNextPacket() throws {
        let first = try CameraWire.encode(header: ["type": "start", "protocolVersion": 1])
        let second = try CameraWire.encode(header: ["type": "ack", "seq": 9])
        var decoder = CameraPacketDecoder(maximumIncomingPayload: 0)
        let initial = try decoder.append(first + second.prefix(3))
        XCTAssertEqual(initial.count, 1)
        XCTAssertEqual(initial[0].header["type"] as? String, "start")
        let final = try decoder.append(Data(second.dropFirst(3)))
        XCTAssertEqual(final.count, 1)
        XCTAssertEqual(final[0].header["type"] as? String, "ack")
    }

    func testBigEndianLength() throws {
        let encoded = try CameraWire.encode(header: ["type": "hello"])
        let prefix = Array(encoded.prefix(4))
        let length = prefix.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        XCTAssertEqual(Int(length), encoded.count - 4)
    }

    func testZeroAndOversizeHeaderAreRejected() {
        for prefix: [UInt8] in [[0, 0, 0, 0], [0, 1, 0, 1], [255, 255, 255, 255]] {
            var decoder = CameraPacketDecoder()
            XCTAssertThrowsError(try decoder.append(Data(prefix))) {
                XCTAssertEqual($0 as? CameraWireError, .invalidHeaderLength)
            }
        }
    }

    func testMalformedAndNonObjectJSONAreRejected() {
        for json in [Data("not-json".utf8), Data("[]".utf8)] {
            var decoder = CameraPacketDecoder()
            XCTAssertThrowsError(try decoder.append(Data([0, 0, 0, UInt8(json.count)]) + json)) {
                XCTAssertEqual($0 as? CameraWireError, .invalidHeader)
            }
        }
    }

    func testInvalidPayloadDeclarationsAreRejectedBeforeAllocation() throws {
        for value: Any in [-1, 1.5, true, "4", 16 * 1_024 * 1_024 + 1] {
            let json = try JSONSerialization.data(withJSONObject: ["payloadBytes": value])
            var decoder = CameraPacketDecoder()
            XCTAssertThrowsError(try decoder.append(Data([0, 0, 0, UInt8(json.count)]) + json)) {
                XCTAssertEqual($0 as? CameraWireError, .invalidPayloadLength)
            }
        }
    }

    func testControlDecoderRejectsUnexpectedPayload() throws {
        let encoded = try CameraWire.encode(header: ["type": "start"], payload: Data([1]))
        var decoder = CameraPacketDecoder(maximumIncomingPayload: 0)
        XCTAssertThrowsError(try decoder.append(encoded)) {
            XCTAssertEqual($0 as? CameraWireError, .invalidPayloadLength)
        }
    }

    func testEncodingOverridesCallerPayloadLength() throws {
        let encoded = try CameraWire.encode(header: ["payloadBytes": 999], payload: Data([1, 2]))
        var decoder = CameraPacketDecoder()
        XCTAssertEqual(try decoder.append(encoded).first?.header["payloadBytes"] as? Int, 2)
    }

    func testEmptyPayloadHello() throws {
        let encoded = try CameraWire.encode(header: ["type": "hello", "protocolVersion": 1])
        var decoder = CameraPacketDecoder()
        let output = try decoder.append(encoded)
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].payload.count, 0)
    }

    func testCommandCountersRejectBooleansFractionsAndNegativeNumbers() {
        XCTAssertEqual(CameraWire.unsignedInteger(7), 7)
        for value: Any in [true, false, -1, 1.5, "7"] {
            XCTAssertNil(CameraWire.unsignedInteger(value))
        }
    }
}
