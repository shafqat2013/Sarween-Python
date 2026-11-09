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
    @Published var preset: CameraCapturePreset = .standard
    @Published var lens: CameraLens = .wide
    @Published var availableCameras: [CameraLensCapability] = []
    @Published var supportedPresets: [CameraCapturePreset] = []
    @Published var dimmed = false
    @Published var torchEnabled = false
    @Published var exposureLocked = false
    @Published var hostConnected = false
    @Published var actionNotice = ""
    @Published var controlNotice = ""
    @Published private var hostFeedback = CameraHostFeedbackState()
    @Published private var feedbackNow = ProcessInfo.processInfo.systemUptime
    @Published private var savedCPLGuideRevision = 0
    private let cplGuideStore = CameraCPLGuideStore()
    private var rememberCPLTargetEnabled = true
    let capture = CameraCapture()

    private var server: USBStreamServer!
    private var active = false
    private var requested = false
    private var requestingPermission = false
    private var loadingCapabilities = false
    private var capabilitiesReady = false
    private var lastFrameUpdate = DispatchTime.now().uptimeNanoseconds

    init() {
        if let saved = UserDefaults.standard.string(forKey: "cameraOrientation"), let value = CameraOrientation(rawValue: saved) {
            orientation = value
        }
        if let saved = UserDefaults.standard.string(forKey: "cameraCapturePreset"), let value = CameraCapturePreset(rawValue: saved) {
            preset = value
        }
        if let saved = UserDefaults.standard.string(forKey: "cameraLens"), let value = CameraLens(rawValue: saved) {
            lens = value
        }
        let installationID: String
        if let saved = UserDefaults.standard.string(forKey: "cameraInstallationID") { installationID = saved }
        else {
            installationID = UUID().uuidString
            UserDefaults.standard.set(installationID, forKey: "cameraInstallationID")
        }
        server = USBStreamServer(installationID: installationID, deviceName: UIDevice.current.name,
                                 orientation: orientation, preset: preset, lens: lens)
        server.onStatus = { [weak self] message in
            Task { @MainActor in
                guard let self, self.active, self.authorized, !self.paused else { return }
                self.status = message
            }
        }
        server.onRequested = { [weak self] value in
            Task { @MainActor in
                guard let self else { return }
                self.requested = value
                if !value { self.hostFeedback.reset(sessionID: self.hostFeedback.sessionID) }
                self.refreshCapture()
            }
        }
        server.onSessionChanged = { [weak self] sessionID in
            Task { @MainActor in
                guard let self else { return }
                self.hostConnected = sessionID != nil && self.active && !self.paused
                self.hostFeedback.reset(sessionID: self.hostConnected ? sessionID : nil)
                self.actionNotice = ""
            }
        }
        server.onFeedback = { [weak self] feedback, sessionID in
            Task { @MainActor in
                guard let self, self.active, self.hostConnected, !self.paused else { return }
                let now = ProcessInfo.processInfo.systemUptime
                if self.hostFeedback.receive(feedback, sessionID: sessionID, now: now) {
                    self.feedbackNow = now
                    self.actionNotice = ""
                    if self.rememberCPLTargetEnabled, let fresh = self.hostFeedback.fresh(now: now),
                       self.cplGuideStore.rememberConfirmedTarget(from: fresh, lens: self.lens,
                           preset: self.preset, orientation: self.orientation) {
                        self.savedCPLGuideRevision += 1
                    }
                }
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
                self.running = value && self.active && self.authorized && !self.paused
                self.server.setPreviewRunning(self.running)
                UIApplication.shared.isIdleTimerDisabled = self.active && self.authorized && !self.paused
                if self.running { self.status = self.requested ? "Streaming to Sarween over USB." : "Local preview ready." }
            }
        }
        capture.onTorchState = { [weak self] value in
            Task { @MainActor in self?.torchEnabled = value }
        }
        capture.onTorchError = { [weak self] message in
            Task { @MainActor in
                guard let self, self.active else { return }
                self.status = message
                self.controlNotice = message
            }
        }
        capture.onExposureState = { [weak self] value in
            Task { @MainActor in self?.exposureLocked = value }
        }
        capture.onExposureError = { [weak self] message in
            Task { @MainActor in self?.status = message; self?.controlNotice = message }
        }
        capture.onError = { [weak self] message in
            Task { @MainActor in
                guard let self, self.active else { return }
                self.paused = true
                self.running = false
                self.torchEnabled = false
                self.exposureLocked = false
                self.hostConnected = false
                self.hostFeedback.reset(sessionID: nil)
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
        } else {
            requested = false
            running = false
            torchEnabled = false
            exposureLocked = false
            hostConnected = false
            hostFeedback.reset(sessionID: nil)
            controlNotice = ""
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
            inspectCapabilities()
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
            server.stop()
            capture.stop()
        }
    }

    private func inspectCapabilities() {
        guard !loadingCapabilities else { return }
        loadingCapabilities = true
        if !paused { status = "Checking camera modes…" }
        capture.inspectCameras { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                self.loadingCapabilities = false
                guard self.authorized else { return }
                switch result {
                case .success(let cameras):
                    guard let selection = CameraLensPlanner.selection(lens: self.lens, preset: self.preset, capabilities: cameras) else { return }
                    self.availableCameras = cameras
                    self.lens = selection.lens
                    self.preset = selection.preset
                    self.supportedPresets = cameras.first { $0.lens == selection.lens }!.supportedPresets
                    self.capabilitiesReady = true
                    UserDefaults.standard.set(self.lens.rawValue, forKey: "cameraLens")
                    UserDefaults.standard.set(self.preset.rawValue, forKey: "cameraCapturePreset")
                    guard self.active else { return }
                    self.updateServerConfiguration()
                    self.server.setLocalEnabled(!self.paused)
                    self.server.start()
                    self.refreshCapture()
                case .failure(let error):
                    self.capabilitiesReady = false
                    self.server.setLocalEnabled(false)
                    self.server.stop()
                    self.capture.stop()
                    if self.active {
                        self.paused = true
                        self.status = error.localizedDescription
                    }
                }
            }
        }
    }

    var canChangeCaptureMode: Bool {
        CameraPreviewPolicy.canChangeMode(authorized: authorized, capabilitiesReady: capabilitiesReady, paused: paused, running: running)
    }

    var captureSummary: String {
        "Rear \(lens.label) · \(preset == .detail ? "4K" : "1080p") · up to \(preset.framesPerSecond) FPS"
    }

    var torchSupported: Bool { availableCameras.first { $0.lens == lens }?.torchSupported ?? false }

    var feedback: CameraHostFeedback? { hostFeedback.fresh(now: feedbackNow) }
    var calibration: CameraCalibrationFeedback? {
        guard let value = feedback?.calibration, value.active else { return nil }
        return value
    }
    var markers: CameraMarkerFeedback? { calibration == nil ? feedback?.markers : nil }
    var cpl: CameraCPLFeedback? { feedback?.cpl }
    var savedCPLGuide: Double? { cplGuideStore.value(lens: lens, preset: preset, orientation: orientation) }

    func forgetSavedCPLGuide() {
        cplGuideStore.forget(lens: lens, preset: preset, orientation: orientation)
        rememberCPLTargetEnabled = false
        savedCPLGuideRevision += 1
        actionNotice = "Saved setup guide cleared."
    }
    var connectionCaption: String {
        if paused { return status }
        if !running { return status }
        if hostConnected { return requested ? "Streaming to Mac over USB" : "Mac connected" }
        return "Local preview · Mac not connected"
    }
    var markerCaption: String {
        if calibration != nil { return "Calibrating" }
        guard let markers else { return hostConnected ? "Markers — · waiting" : "Markers need Mac" }
        return "Markers \(markers.count)/\(markers.required)"
    }
    var canUseHostTools: Bool { running && hostConnected && feedback != nil }
    var canUseCPL: Bool { canUseHostTools && calibration == nil }

    func refreshFeedbackFreshness() { feedbackNow = ProcessInfo.processInfo.systemUptime }

    private func updateServerConfiguration() {
        server.setConfiguration(orientation: orientation, preset: preset, supportedPresets: supportedPresets,
                                lens: lens, availableCameras: availableCameras)
    }

    func togglePause() {
        paused.toggle()
        if paused {
            hostConnected = false
            hostFeedback.reset(sessionID: nil)
        }
        if !paused && !capabilitiesReady { checkAuthorization() }
        server.setLocalEnabled(active && authorized && capabilitiesReady && !paused)
        refreshCapture()
        if paused { status = "Camera paused. Tap Resume camera when ready." }
    }

    func setTorch(_ enabled: Bool) {
        controlNotice = ""
        let value = enabled && active && authorized && !paused && running && torchSupported
        torchEnabled = value
        capture.setTorch(value)
    }

    func setExposureLocked(_ locked: Bool) {
        guard running, !paused else { return }
        controlNotice = ""
        capture.setExposureLocked(locked)
    }

    func request(_ action: CameraPhoneAction) {
        guard canUseHostTools, let sessionID = hostFeedback.sessionID else { return }
        switch action {
        case .calibrationCapture, .calibrationCancel:
            guard calibration != nil else { return }
        case .calibrationFinish:
            guard calibration?.canFinish == true else { return }
        case .cplReference:
            guard canUseCPL, exposureLocked, cpl != nil, feedback?.sourceSequence != nil else { return }
        case .cplTarget:
            guard canUseCPL, exposureLocked, cpl?.index != nil else { return }
            rememberCPLTargetEnabled = true
        case .cplReset: guard canUseCPL else { return }
        }
        actionNotice = "Request sent to Sarween…"
        server.sendAction(action, expectedSessionID: sessionID)
    }

    func changeOrientation(_ value: CameraOrientation) {
        guard canChangeCaptureMode, orientation != value else { return }
        orientation = value
        UserDefaults.standard.set(value.rawValue, forKey: "cameraOrientation")
        updateServerConfiguration()
        refreshCapture()
    }

    func changePreset(_ value: CameraCapturePreset) {
        guard canChangeCaptureMode, preset != value, supportedPresets.contains(value) else { return }
        preset = value
        UserDefaults.standard.set(value.rawValue, forKey: "cameraCapturePreset")
        updateServerConfiguration()
        refreshCapture()
    }

    func changeLens(_ value: CameraLens) {
        guard canChangeCaptureMode, lens != value,
              let selection = CameraLensPlanner.selection(lens: value, preset: preset, capabilities: availableCameras),
              selection.lens == value else { return }
        lens = selection.lens
        preset = selection.preset
        supportedPresets = availableCameras.first { $0.lens == lens }!.supportedPresets
        torchEnabled = false
        UserDefaults.standard.set(lens.rawValue, forKey: "cameraLens")
        UserDefaults.standard.set(preset.rawValue, forKey: "cameraCapturePreset")
        updateServerConfiguration()
        refreshCapture()
        status = paused ? "Camera paused. \(lens.label) selected; use a matching calibration profile in Sarween."
            : "\(lens.label) selected. Use a matching calibration profile in Sarween."
    }

    private func refreshCapture() {
        if CameraPreviewPolicy.shouldRun(active: active, authorized: authorized, capabilitiesReady: capabilitiesReady, paused: paused) {
            capture.start(orientation: orientation, preset: preset, lens: lens)
        } else {
            running = false
            torchEnabled = false
            exposureLocked = false
            capture.stop()
        }
        UIApplication.shared.isIdleTimerDisabled = active && authorized && !paused
    }
}
