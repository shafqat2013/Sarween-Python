import Foundation

public struct CameraFrameTicket: Equatable {
    public let sessionID: String
    public let sequence: UInt64
}

/// One frame may be encoded or in transit. An ACK can precede send completion.
/// The gate opens only when both finish, so a slow reader never accumulates video.
public final class CameraFrameFlowGate {
    private let lock = NSLock()
    private var sessionID: String?
    private var enabled = false
    private var requested = false
    private var requiresAcknowledgement = true
    private var nextSequence: UInt64 = 0
    private var ticket: CameraFrameTicket?
    private var sent = false
    private var acknowledged = false

    public init() {}

    public func reset(sessionID: String?) {
        lock.lock(); defer { lock.unlock() }
        self.sessionID = sessionID
        nextSequence = 0
        requested = false
        ticket = nil
        sent = false
        acknowledged = false
    }

    public func setEnabled(_ enabled: Bool) {
        lock.lock(); defer { lock.unlock() }
        self.enabled = enabled
        if !enabled { ticket = nil }
    }

    public func setRequested(_ requested: Bool, requiresAcknowledgement: Bool = true) {
        lock.lock(); defer { lock.unlock() }
        self.requested = requested
        self.requiresAcknowledgement = requiresAcknowledgement
        if !requested { ticket = nil }
    }

    public func reserve() -> CameraFrameTicket? {
        lock.lock(); defer { lock.unlock() }
        guard enabled, requested, let sessionID, ticket == nil else { return nil }
        nextSequence += 1
        let next = CameraFrameTicket(sessionID: sessionID, sequence: nextSequence)
        ticket = next
        sent = false
        acknowledged = !requiresAcknowledgement
        return next
    }

    public func isPending(_ candidate: CameraFrameTicket) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return ticket == candidate
    }

    public func didSend(_ candidate: CameraFrameTicket) {
        lock.lock(); defer { lock.unlock() }
        guard ticket == candidate else { return }
        sent = true
        if acknowledged { ticket = nil }
    }

    public func acknowledge(sessionID: String, sequence: UInt64) {
        lock.lock(); defer { lock.unlock() }
        guard let current = ticket, current.sessionID == sessionID, current.sequence == sequence else { return }
        acknowledged = true
        if sent { ticket = nil }
    }

    public func cancel(_ candidate: CameraFrameTicket) {
        lock.lock(); defer { lock.unlock() }
        if ticket == candidate { ticket = nil }
    }
}
