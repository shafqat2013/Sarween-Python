import AVFoundation
import SwiftUI
import UIKit

@main
struct SarweenCameraApp: App {
    @StateObject private var model = CameraViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            CameraScreen(model: model)
                .onAppear { model.setActive(scenePhase == .active) }
                .onChange(of: scenePhase) { _, phase in model.setActive(phase == .active) }
        }
    }
}

struct CameraScreen: View {
    @ObservedObject var model: CameraViewModel
    @State private var showCPL = false
    @State private var showGuide = false
    private let freshnessTimer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            GeometryReader { geometry in
                CameraPreview(session: model.capture.session, orientation: model.orientation)
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .overlay(alignment: .top) { statusBar }
                    .overlay(alignment: .bottom) {
                        bottomTools(compact: geometry.size.width > geometry.size.height && geometry.size.height < 500)
                            .padding(.horizontal, 12).padding(.bottom, 8)
                    }
            }
            if model.dimmed {
                Color.black.ignoresSafeArea()
                    .overlay(alignment: .bottom) {
                        Text("Sarween Camera · tap to return")
                            .font(.footnote).foregroundStyle(.gray.opacity(0.5)).padding(24)
                    }
                    .contentShape(Rectangle()).onTapGesture { model.dimmed = false }
                    .accessibilityLabel("Dimmed camera screen. Double tap to return.")
                    .accessibilityAddTraits(.isButton)
            }
        }
        .foregroundStyle(.white).preferredColorScheme(.dark)
        .onReceive(freshnessTimer) { _ in model.refreshFeedbackFreshness() }
        .sheet(isPresented: $showCPL) {
            CPLToolSheet(model: model).presentationDetents([.medium, .large])
        }
        .sheet(isPresented: $showGuide) { CalibrationGuideSheet() }
    }

    private var statusBar: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle().fill(model.running ? Color.green : Color.secondary).frame(width: 8, height: 8)
                Text("Sarween Camera").font(.headline)
                Spacer()
            }
            HStack(spacing: 6) {
                Text(model.connectionCaption).accessibilityIdentifier("connectionStatus")
                Spacer(minLength: 0)
                Text("\(model.lens.label) · \(model.preset == .detail ? "4K" : "1080p")")
            }
            .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(.black.opacity(0.82))
    }

    private var guide: CameraGuidePresentation {
        CameraGuidePresentation(feedback: model.feedback, hostConnected: model.hostConnected,
            cameraRunning: model.running, calibrationActive: model.calibration != nil,
            savedGuide: model.savedCPLGuide)
    }

    @ViewBuilder private func bottomTools(compact: Bool) -> some View {
        if let calibration = model.calibration {
            VStack(spacing: 8) {
                calibrationCard(calibration)
                controlBar
            }
        } else if compact {
            HStack(alignment: .bottom, spacing: 8) {
                guideCard
                controlBar.frame(width: 230)
            }
        } else {
            VStack(spacing: 8) {
                guideCard
                controlBar
            }
        }
    }

    private var guideCard: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Markers").font(.caption)
                    Text(guide.markerValue).font(.system(size: 29, weight: .semibold)).monospacedDigit()
                        .foregroundStyle(guide.markersComplete ? Color.green : Color.secondary)
                        .accessibilityIdentifier("markerStatus")
                    Text(guide.markerDetail).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }.frame(maxWidth: .infinity, alignment: .leading)
                VStack(alignment: .leading, spacing: 1) {
                    Text("LCD brightness").font(.caption)
                    Text(guide.brightnessValue).font(.system(size: 29, weight: .semibold)).monospacedDigit()
                        .lineLimit(1).minimumScaleFactor(0.6).accessibilityIdentifier("lcdBrightness")
                    Text(guide.savedGuideText).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            Text(model.controlNotice.isEmpty ? guide.message : model.controlNotice)
                .font(.caption2).foregroundStyle(model.controlNotice.isEmpty ? Color.secondary : Color.yellow)
                .lineLimit(2).frame(maxWidth: .infinity, minHeight: 30, maxHeight: 30, alignment: .topLeading)
                .accessibilityIdentifier("cameraGuideReason")
        }
        .padding(12).background(.black.opacity(0.84), in: RoundedRectangle(cornerRadius: 16))
        .accessibilityElement(children: .contain).accessibilityIdentifier("cameraGuide")
    }

    private var controlBar: some View {
        HStack(spacing: 12) {
            if model.authorized {
                Button(action: model.togglePause) {
                    Label(model.paused ? "Resume" : "Pause", systemImage: model.paused ? "play.fill" : "pause.fill")
                        .frame(minHeight: 32)
                }.buttonStyle(.borderedProminent)
                Spacer()
                Menu {
                    Section("Camera — pause to change") {
                        Picker("Rear camera", selection: Binding(get: { model.lens }, set: model.changeLens)) {
                            ForEach(model.availableCameras, id: \.lens) { camera in Text(camera.lens.label).tag(camera.lens) }
                        }.disabled(!model.canChangeCaptureMode)
                        Picker("Capture quality", selection: Binding(get: { model.preset }, set: model.changePreset)) {
                            ForEach(CameraCapturePreset.allCases) { preset in
                                Text(preset.label + (model.supportedPresets.contains(preset) ? "" : " — unavailable"))
                                    .tag(preset).disabled(!model.supportedPresets.contains(preset))
                            }
                        }.disabled(!model.canChangeCaptureMode)
                        Picker("Orientation", selection: Binding(get: { model.orientation }, set: model.changeOrientation)) {
                            ForEach(CameraOrientation.allCases) { Text($0.label).tag($0) }
                        }.disabled(!model.canChangeCaptureMode)
                    }
                    Section("Light") {
                        Toggle("Torch", isOn: Binding(get: { model.torchEnabled }, set: model.setTorch))
                            .disabled(!model.running || !model.torchSupported)
                        Toggle("Lock exposure for CPL", isOn: Binding(get: { model.exposureLocked }, set: model.setExposureLocked))
                            .disabled(!model.running)
                    }
                    Section("Tools") {
                        Button("CPL tool", systemImage: "camera.filters") { showCPL = true }
                        Button("Calibration guide", systemImage: "viewfinder") { showGuide = true }
                        Button("Dim screen", systemImage: "moon.fill") { model.dimmed = true }
                    }
                } label: {
                    Label("Controls", systemImage: "slider.horizontal.3").frame(minHeight: 32)
                }.buttonStyle(.bordered)
            } else {
                Button("Allow camera in Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                }.buttonStyle(.borderedProminent)
            }
        }
        .padding(10).background(.black.opacity(0.82), in: RoundedRectangle(cornerRadius: 16))
    }

    private func calibrationCard(_ calibration: CameraCalibrationFeedback) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(calibration.title.isEmpty ? "Camera calibration" : calibration.title).font(.headline)
                Spacer()
                Text("\(calibration.captured)/\(calibration.required)").monospacedDigit().font(.headline)
            }
            ProgressView(value: Double(min(calibration.captured, calibration.required)), total: Double(calibration.required))
            if !calibration.message.isEmpty { Text(calibration.message).font(.callout).lineLimit(3) }
            if let countdown = calibration.countdownSeconds, countdown > 0 {
                Text("Next capture in \(Int(ceil(countdown)))s").font(.caption).monospacedDigit()
            }
            if !model.actionNotice.isEmpty { Text(model.actionNotice).font(.caption).foregroundStyle(.secondary) }
            HStack {
                Button("Capture view") { model.request(.calibrationCapture) }.buttonStyle(.borderedProminent)
                Button("Finish") { model.request(.calibrationFinish) }.disabled(!calibration.canFinish).buttonStyle(.bordered)
                Spacer(minLength: 0)
                Button("Cancel", role: .cancel) { model.request(.calibrationCancel) }.buttonStyle(.bordered)
            }.font(.callout).disabled(!model.canUseHostTools)
        }
        .padding(14).background(.black.opacity(0.9), in: RoundedRectangle(cornerRadius: 16))
        .accessibilityIdentifier("calibrationCard")
    }
}

struct CPLToolSheet: View {
    @ObservedObject var model: CameraViewModel
    @Environment(\.dismiss) private var dismiss
    private var guide: CameraGuidePresentation {
        CameraGuidePresentation(feedback: model.feedback, hostConnected: model.hostConnected,
            cameraRunning: model.running, calibrationActive: model.calibration != nil,
            savedGuide: model.savedCPLGuide)
    }
    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text("Markers \(guide.markerValue) · \(guide.markerDetail)").font(.subheadline.weight(.semibold))
                        .foregroundStyle(guide.markersComplete ? Color.green : Color.secondary)
                    Text("LCD brightness").font(.headline)
                    Text(guide.brightnessValue)
                        .font(.system(size: 48, weight: .semibold, design: .rounded)).monospacedDigit()
                    Text("Bright reference = 100. Lower means a dimmer LCD. This is a relative image reading, not a physical filter angle or optical transmission percentage.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Text(guide.savedGuideText).monospacedDigit()
                    Text(guide.message).font(.callout)
                }
                Section("Keep the scene and phone still") {
                    Toggle("Lock exposure", isOn: Binding(get: { model.exposureLocked }, set: model.setExposureLocked)).disabled(!model.running)
                    Text("Fit the CPL first. Lock exposure and white balance, rotate to the brightest LCD and set a fresh reference for this setup. Then turn only the filter. A saved guide is approximate and depends on the map and room lighting. Pause, a lens change or leaving this app releases both locks.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Button("Set bright reference") { model.request(.cplReference) }
                        .disabled(!model.canUseCPL || !model.exposureLocked || model.cpl == nil || model.feedback?.sourceSequence == nil)
                    Button("Save rough guide") { model.request(.cplTarget) }
                        .disabled(!model.canUseCPL || !model.exposureLocked || !guide.brightnessIsLive)
                    Button("Reset reference", role: .destructive) { model.request(.cplReset) }.disabled(!model.canUseCPL)
                    Button("Forget saved guide", role: .destructive) { model.forgetSavedCPLGuide() }
                        .disabled(model.savedCPLGuide == nil)
                    if model.calibration != nil { Text("Finish or cancel camera calibration before using the CPL tool.").font(.footnote) }
                }
                if !model.actionNotice.isEmpty { Text(model.actionNotice).font(.footnote) }
            }
            .navigationTitle("CPL tool").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }.preferredColorScheme(.dark)
    }
}

struct CalibrationGuideSheet: View {
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            List {
                Section("Start on the Mac") {
                    Text("In Sarween, choose this USB iPhone and start camera calibration. The board and capture mode are chosen there. Keep Sarween Camera open on the phone; live instructions and the saved-view count will appear here.")
                }
                Section("Capture useful views") {
                    Text("Move the phone slowly to show the board near the centre, edges and corners. Vary distance and tilt. Keep the complete board sharp and visible; stop briefly when instructed.")
                    Text("Sweep mode collects useful views as you move. Timed and steady modes capture when their on-screen conditions are met. In manual mode, hold still and tap Capture view. A burst from one unchanged viewpoint does not add useful coverage.")
                    Text("Watch the saved-view count. If it does not rise, follow the reason shown on the calibration card. Finish becomes available when the Mac has enough accepted views.")
                }
                Section("After calibration") {
                    Text("Select the matching lens, resolution and orientation profile in Sarween. Each physical lens needs its own calibration. Return the phone to its playing position and verify marker status on the Mac.")
                }
            }
            .navigationTitle("Calibration guide").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }.preferredColorScheme(.dark)
    }
}

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let orientation: CameraOrientation
    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }
    func updateUIView(_ view: PreviewUIView, context: Context) {
        if let connection = view.previewLayer.connection, connection.isVideoRotationAngleSupported(orientation.rotationAngle) {
            connection.videoRotationAngle = orientation.rotationAngle
        }
    }
}
