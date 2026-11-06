import AVFoundation
import Combine
import SwiftUI
import UIKit

@MainActor
final class CameraViewModel: ObservableObject {
    @Published var status = "Connect your iPhone to the Mac and open Sarween."
    @Published var authorized = false
    @Published var paused = false
    @Published var running = false
    @Published var framesSent: UInt64 = 0
    @Published var orientation: CameraOrientation = .portrait
    @Published var dimmed = false
    @Published var torchEnabled = false
    let capture = CameraCapture()

    private var server: USBStreamServer!
    private var active = false
    private var requested = false
    private var requestingPermission = false
    private var lastFrameUpdate = DispatchTime.now().uptimeNanoseconds

    init() {
        if let saved = UserDefaults.standard.string(forKey: "cameraOrientation"), let value = CameraOrientation(rawValue: saved) {
            orientation = value
        }
        let installationID: String
        if let saved = UserDefaults.standard.string(forKey: "cameraInstallationID") { installationID = saved }
        else {
            installationID = UUID().uuidString
            UserDefaults.standard.set(installationID, forKey: "cameraInstallationID")
        }
        server = USBStreamServer(installationID: installationID, deviceName: UIDevice.current.name, orientation: orientation)
        server.onStatus = { [weak self] message in
            Task { @MainActor in
                guard let self, self.active, self.authorized, !self.paused else { return }
                self.status = message
            }
        }
        server.onRequested = { [weak self] value in
            Task { @MainActor in
                self?.requested = value
                self?.refreshCapture()
            }
        }
        server.onFrameSent = { [weak self] count in
            Task { @MainActor in
                guard let self else { return }
                let now = DispatchTime.now().uptimeNanoseconds
                if now - self.lastFrameUpdate > 500_000_000 {
                    self.framesSent = count
                    self.lastFrameUpdate = now
                }
            }
        }
        let gate = server.gate
        capture.reserveFrame = { gate.reserve() }
        capture.cancelFrame = { gate.cancel($0) }
        capture.onFrame = { [weak server] frame, ticket in server?.deliver(frame, ticket: ticket) }
        capture.onRunning = { [weak self] value in
            Task { @MainActor in
                guard let self else { return }
                self.running = value && self.active && self.authorized && !self.paused && self.requested
                UIApplication.shared.isIdleTimerDisabled = self.active && self.authorized && !self.paused
                if self.running { self.status = "Connected to Sarween. Camera streaming over USB." }
            }
        }
        capture.onTorchState = { [weak self] value in
            Task { @MainActor in self?.torchEnabled = value }
        }
        capture.onTorchError = { [weak self] message in
            Task { @MainActor in
                guard let self, self.active else { return }
                self.status = message
            }
        }
        capture.onError = { [weak self] message in
            Task { @MainActor in
                guard let self, self.active else { return }
                self.paused = true
                self.running = false
                self.torchEnabled = false
                self.status = message
                self.server.setLocalEnabled(false)
                self.capture.stop()
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    func setActive(_ isActive: Bool) {
        active = isActive
        if isActive {
            checkAuthorization()
            server.start()
        } else {
            requested = false
            running = false
            torchEnabled = false
            server.setLocalEnabled(false)
            server.stop()
            capture.stop()
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    private func checkAuthorization() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            authorized = true
            server.setLocalEnabled(active && !paused)
            refreshCapture()
        case .notDetermined:
            guard !requestingPermission else { return }
            requestingPermission = true
            AVCaptureDevice.requestAccess(for: .video) { [weak self] _ in
                Task { @MainActor in self?.requestingPermission = false; self?.checkAuthorization() }
            }
        default:
            authorized = false
            status = "Allow Camera access in Settings to use your iPhone with Sarween."
            server.setLocalEnabled(false)
            capture.stop()
        }
    }

    func togglePause() {
        paused.toggle()
        server.setLocalEnabled(active && authorized && !paused)
        refreshCapture()
        if paused { status = "Camera paused. Tap Resume camera when ready." }
    }

    func setTorch(_ enabled: Bool) {
        let value = enabled && active && authorized && !paused && running
        torchEnabled = value
        capture.setTorch(value)
    }

    func changeOrientation(_ value: CameraOrientation) {
        guard !running else { return }
        orientation = value
        UserDefaults.standard.set(value.rawValue, forKey: "cameraOrientation")
        server.setOrientation(value)
        refreshCapture()
    }

    private func refreshCapture() {
        if active && authorized && !paused && requested {
            capture.start(orientation: orientation)
        } else {
            running = false
            torchEnabled = false
            capture.stop()
        }
        UIApplication.shared.isIdleTimerDisabled = active && authorized && !paused
    }
}
