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

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 16) {
                HStack {
                    Text("Sarween Camera").font(.title2.bold())
                    Spacer()
                    Circle().fill(model.running ? Color.green : Color.secondary).frame(width: 12, height: 12)
                }
                CameraPreview(session: model.capture.session, orientation: model.orientation)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .overlay {
                        if !model.running {
                            VStack(spacing: 12) {
                                Image(systemName: model.paused ? "pause.circle" : "cable.connector").font(.system(size: 42))
                                Text(model.paused ? "Camera paused" : "Ready for Sarween").font(.headline)
                            }
                            .padding(24)
                            .background(.black.opacity(0.8), in: RoundedRectangle(cornerRadius: 16))
                        }
                    }
                Text(model.status)
                    .font(.body)
                    .multilineTextAlignment(.center)
                    .frame(minHeight: 55)
                    .accessibilityIdentifier("connectionStatus")
                if model.authorized {
                    Picker("Camera orientation", selection: Binding(get: { model.orientation }, set: { model.changeOrientation($0) })) {
                        ForEach(CameraOrientation.allCases) { Text($0.label).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(!model.canChangeCaptureMode)
                    Picker("Capture quality", selection: Binding(get: { model.preset }, set: { model.changePreset($0) })) {
                        ForEach(CameraCapturePreset.allCases) { preset in
                            Text(preset.label + (model.supportedPresets.contains(preset) ? "" : " — unavailable"))
                                .tag(preset)
                                .disabled(!model.supportedPresets.contains(preset))
                        }
                    }
                    .pickerStyle(.menu)
                    .disabled(!model.canChangeCaptureMode)
                    Toggle("Torch", isOn: Binding(get: { model.torchEnabled }, set: { model.setTorch($0) }))
                        .disabled(!model.running)
                    Text(model.running ? model.captureSummary : "Plug in by USB, open Sarween, and select this iPhone.")
                        .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    if model.preset.experimental {
                        Text("4K is experimental and needs a matching calibration profile. Actual speed depends on your phone and USB connection.")
                            .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    } else if model.running {
                        Text("Pause the camera to change capture quality or orientation.")
                            .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    HStack(spacing: 16) {
                        Button(model.paused ? "Resume camera" : "Pause camera", action: model.togglePause)
                            .buttonStyle(.borderedProminent)
                        Button("Dim screen") { model.dimmed = true }.buttonStyle(.bordered)
                    }
                } else {
                    Button("Open Settings") {
                        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                    }.buttonStyle(.borderedProminent)
                }
            }
            .padding(24)
            .foregroundStyle(.white)
            if model.dimmed {
                Color.black.ignoresSafeArea()
                    .overlay(alignment: .bottom) {
                        Text("Sarween Camera · tap to return")
                            .font(.footnote).foregroundStyle(.gray.opacity(0.5)).padding(24)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { model.dimmed = false }
                    .accessibilityLabel("Dimmed camera screen. Double tap to return.")
                    .accessibilityAddTraits(.isButton)
            }
        }
        .preferredColorScheme(.dark)
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
