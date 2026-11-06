import Foundation
import CoreFoundation

public enum CameraWireError: Error, Equatable {
    case invalidHeaderLength
    case invalidHeader
    case invalidPayloadLength
    case bufferOverflow
}

public struct CameraWirePacket {
    public let header: [String: Any]
    public let payload: Data
}

/// Shared framing for the USB-only Sarween camera connection.
/// Four-byte unsigned big-endian JSON length, JSON, then exactly payloadBytes bytes.
public enum CameraWire {
    public static let version = 1
    public static let maximumHeaderBytes = 65_536
    public static let maximumPayloadBytes = 16 * 1_024 * 1_024

    public static func unsignedInteger(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        return UInt64(number.stringValue)
    }

    public static func encode(header: [String: Any], payload: Data = Data()) throws -> Data {
        guard payload.count <= maximumPayloadBytes else { throw CameraWireError.invalidPayloadLength }
        var header = header
        header["payloadBytes"] = payload.count
        guard JSONSerialization.isValidJSONObject(header) else { throw CameraWireError.invalidHeader }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        guard !json.isEmpty, json.count <= maximumHeaderBytes else {
            throw CameraWireError.invalidHeaderLength
        }
        let count = UInt32(json.count)
        var packet = Data([
            UInt8((count >> 24) & 255), UInt8((count >> 16) & 255),
            UInt8((count >> 8) & 255), UInt8(count & 255)
        ])
        packet.append(json)
        packet.append(payload)
        return packet
    }

    static func payloadLength(_ header: [String: Any]) throws -> Int {
        guard let number = header["payloadBytes"] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite,
              number.doubleValue.rounded(.towardZero) == number.doubleValue,
              number.doubleValue >= 0,
              number.doubleValue <= Double(maximumPayloadBytes) else {
            throw CameraWireError.invalidPayloadLength
        }
        return number.intValue
    }
}

/// Incremental decoder. A single packet is bounded even if its sender fragments it.
public struct CameraPacketDecoder {
    private var buffer = Data()
    private var headerLength: Int?
    private var header: [String: Any]?
    private var payloadLength: Int?
    private let maximumIncomingPayload: Int

    public init(maximumIncomingPayload: Int = CameraWire.maximumPayloadBytes) {
        self.maximumIncomingPayload = max(0, min(maximumIncomingPayload, CameraWire.maximumPayloadBytes))
    }

    public mutating func append(_ bytes: Data) throws -> [CameraWirePacket] {
        // Network reads are capped at 64 KiB. Permit one full packet plus one read.
        guard buffer.count + bytes.count <= CameraWire.maximumHeaderBytes + maximumIncomingPayload + 65_540 else {
            throw CameraWireError.bufferOverflow
        }
        buffer.append(bytes)
        var packets: [CameraWirePacket] = []
        while true {
            if headerLength == nil {
                guard buffer.count >= 4 else { break }
                let prefix = Array(buffer.prefix(4))
                let length = (UInt32(prefix[0]) << 24) | (UInt32(prefix[1]) << 16)
                    | (UInt32(prefix[2]) << 8) | UInt32(prefix[3])
                guard length > 0, length <= CameraWire.maximumHeaderBytes else {
                    throw CameraWireError.invalidHeaderLength
                }
                buffer.removeFirst(4)
                headerLength = Int(length)
            }
            if header == nil {
                guard let length = headerLength, buffer.count >= length else { break }
                let data = Data(buffer.prefix(length))
                let object: Any
                do { object = try JSONSerialization.jsonObject(with: data) }
                catch { throw CameraWireError.invalidHeader }
                guard let decoded = object as? [String: Any] else { throw CameraWireError.invalidHeader }
                let size = try CameraWire.payloadLength(decoded)
                guard size <= maximumIncomingPayload else { throw CameraWireError.invalidPayloadLength }
                buffer.removeFirst(length)
                header = decoded
                payloadLength = size
            }
            guard let size = payloadLength, let decoded = header, buffer.count >= size else { break }
            packets.append(CameraWirePacket(header: decoded, payload: Data(buffer.prefix(size))))
            buffer.removeFirst(size)
            headerLength = nil
            header = nil
            payloadLength = nil
        }
        return packets
    }
}
