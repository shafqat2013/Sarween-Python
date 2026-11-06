import AVFoundation
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum CameraOrientation: String, CaseIterable, Identifiable {
    case portrait
    case landscape
    var id: String { rawValue }
    var label: String { self == .portrait ? "Portrait" : "Landscape" }
    var rotationAngle: CGFloat { self == .portrait ? 90 : 0 }
}

struct EncodedCameraFrame {
    let jpeg: Data
    let width: Int
    let height: Int
    let captureTimeNs: Int64
    let encodeDurationNs: UInt64
    let metadata: [String: Any]
}

enum CameraCaptureError: LocalizedError {
    case unavailable
    case unsupportedFormat
    case outputUnavailable
    var errorDescription: String? {
        switch self {
        case .unavailable: return "The rear wide camera is unavailable."
        case .unsupportedFormat: return "This camera cannot provide the required 1080p format at 15 FPS."
        case .outputUnavailable: return "The camera output could not be configured."
        }
    }
}

final class CameraCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    var reserveFrame: (() -> CameraFrameTicket?)?
    var onFrame: ((EncodedCameraFrame, CameraFrameTicket) -> Void)?
    var cancelFrame: ((CameraFrameTicket) -> Void)?
    var onRunning: ((Bool) -> Void)?
    var onError: ((String) -> Void)?
    var onTorchState: ((Bool) -> Void)?
    var onTorchError: ((String) -> Void)?

    private let sessionQueue = DispatchQueue(label: "com.sarween.camera.capture")
    private let frameQueue = DispatchQueue(label: "com.sarween.camera.jpeg", qos: .userInitiated)
    private let imageContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private var configuredOrientation: CameraOrientation?
    private var device: AVCaptureDevice?
    private var output: AVCaptureVideoDataOutput?
    private var notifications: [NSObjectProtocol] = []

    override init() {
        super.init()
        let center = NotificationCenter.default
        notifications.append(center.addObserver(forName: .AVCaptureSessionWasInterrupted, object: session, queue: nil) { [weak self] _ in
            self?.onRunning?(false)
            self?.onError?("Camera interrupted. Keep Sarween Camera open and return to it to reconnect.")
        })
        notifications.append(center.addObserver(forName: .AVCaptureSessionRuntimeError, object: session, queue: nil) { [weak self] note in
            let error = note.userInfo?[AVCaptureSessionErrorKey] as? Error
            self?.onRunning?(false)
            self?.onError?(error?.localizedDescription ?? "The camera stopped unexpectedly.")
        })
    }

    deinit {
        notifications.forEach(NotificationCenter.default.removeObserver)
        let previousDevice = device
        let previousSession = session
        sessionQueue.async {
            if let previousDevice, previousDevice.hasTorch {
                do {
                    try previousDevice.lockForConfiguration()
                    previousDevice.torchMode = .off
                    previousDevice.unlockForConfiguration()
                } catch { }
            }
            if previousSession.isRunning { previousSession.stopRunning() }
        }
    }

    func start(orientation: CameraOrientation) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                if self.configuredOrientation != orientation { try self.configure(orientation: orientation) }
                if !self.session.isRunning { self.session.startRunning() }
                self.onRunning?(self.session.isRunning)
                if !self.session.isRunning { throw CameraCaptureError.unavailable }
            } catch {
                self.onRunning?(false)
                self.onError?(error.localizedDescription)
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.setTorchOnQueue(false)
            if self.session.isRunning { self.session.stopRunning() }
            self.onRunning?(false)
        }
    }

    func setTorch(_ enabled: Bool) {
        sessionQueue.async { [weak self] in self?.setTorchOnQueue(enabled) }
    }

    private func setTorchOnQueue(_ enabled: Bool) {
        guard let camera = device, camera.hasTorch else { onTorchState?(false); return }
        do {
            try camera.lockForConfiguration()
            defer { camera.unlockForConfiguration() }
            if enabled && session.isRunning {
                guard camera.isTorchAvailable, camera.isTorchModeSupported(.on) else {
                    onTorchState?(false)
                    onTorchError?("The torch is currently unavailable.")
                    return
                }
                try camera.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel)
            } else {
                camera.torchMode = .off
            }
            onTorchState?(camera.torchMode == .on)
        } catch {
            onTorchState?(false)
            onTorchError?("The torch could not be changed: \(error.localizedDescription)")
        }
    }

    private func configure(orientation: CameraOrientation) throws {
        setTorchOnQueue(false)
        if session.isRunning { session.stopRunning() }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        configuredOrientation = nil
        session.inputs.forEach(session.removeInput)
        session.outputs.forEach(session.removeOutput)
        session.sessionPreset = .inputPriority
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CameraCaptureError.unavailable
        }
        let formats = camera.formats.filter { format in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return dimensions.width == 1920 && dimensions.height == 1080
                && format.videoSupportedFrameRateRanges.contains { $0.minFrameRate <= 15 && $0.maxFrameRate >= 15 }
        }
        // Prefer the full field of view among matching physical-wide formats.
        guard let format = formats.max(by: { $0.videoFieldOfView < $1.videoFieldOfView }) else {
            throw CameraCaptureError.unsupportedFormat
        }
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else { throw CameraCaptureError.unavailable }
        session.addInput(input)
        try camera.lockForConfiguration()
        camera.activeFormat = format
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 15)
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 15)
        camera.videoZoomFactor = 1
        if camera.hasTorch { camera.torchMode = .off }
        if camera.isGeometricDistortionCorrectionSupported { camera.isGeometricDistortionCorrectionEnabled = false }
        camera.automaticallyAdjustsVideoHDREnabled = false
        if camera.activeFormat.isVideoHDRSupported { camera.isVideoHDREnabled = false }
        if camera.isFocusModeSupported(.continuousAutoFocus) { camera.focusMode = .continuousAutoFocus }
        if camera.isExposureModeSupported(.continuousAutoExposure) { camera.exposureMode = .continuousAutoExposure }
        if camera.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { camera.whiteBalanceMode = .continuousAutoWhiteBalance }
        camera.unlockForConfiguration()
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.automaticallyConfiguresOutputBufferDimensions = false
        output.deliversPreviewSizedOutputBuffers = false
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.setSampleBufferDelegate(self, queue: frameQueue)
        guard session.canAddOutput(output) else { throw CameraCaptureError.outputUnavailable }
        session.addOutput(output)
        guard let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(orientation.rotationAngle) else {
            throw CameraCaptureError.outputUnavailable
        }
        // AVFoundation rotates the delivered pixels. No EXIF-only rotation or letterboxing.
        connection.videoRotationAngle = orientation.rotationAngle
        if connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }
        if connection.isVideoStabilizationSupported { connection.preferredVideoStabilizationMode = .off }
        if connection.isCameraIntrinsicMatrixDeliverySupported { connection.isCameraIntrinsicMatrixDeliveryEnabled = true }
        self.device = camera
        self.output = output
        configuredOrientation = orientation
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let ticket = reserveFrame?() else { return }
        autoreleasepool {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { cancelFrame?(ticket); return }
            let started = DispatchTime.now().uptimeNanoseconds
            let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            guard timestamp.isValid, !timestamp.isIndefinite, timestamp.seconds.isFinite else { cancelFrame?(ticket); return }
            let image = CIImage(cvPixelBuffer: pixelBuffer)
            guard let rendered = imageContext.createCGImage(image, from: image.extent, format: .RGBA8, colorSpace: colorSpace) else {
                cancelFrame?(ticket); return
            }
            let bytes = NSMutableData()
            guard let destination = CGImageDestinationCreateWithData(bytes, UTType.jpeg.identifier as CFString, 1, nil) else {
                cancelFrame?(ticket); return
            }
            CGImageDestinationAddImage(destination, rendered, [kCGImageDestinationLossyCompressionQuality: 0.94] as CFDictionary)
            guard CGImageDestinationFinalize(destination) else { cancelFrame?(ticket); return }
            var metadata: [String: Any] = ["cameraPosition": "back", "lens": "wide", "zoom": 1.0,
                                           "stabilization": "off", "geometricDistortionCorrection": "off", "hdr": "off"]
            if let device {
                let gains = device.deviceWhiteBalanceGains
                metadata["lensPosition"] = device.lensPosition
                metadata["iso"] = device.iso
                metadata["exposureDurationNs"] = CMTimeConvertScale(device.exposureDuration, timescale: 1_000_000_000, method: .default).value
                metadata["whiteBalanceGains"] = [gains.redGain, gains.greenGain, gains.blueGain]
                metadata["torchOn"] = device.isTorchActive
                metadata["torchLevel"] = device.torchLevel
            }
            let frame = EncodedCameraFrame(
                jpeg: bytes as Data,
                width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer),
                captureTimeNs: CMTimeConvertScale(timestamp, timescale: 1_000_000_000, method: .default).value,
                encodeDurationNs: DispatchTime.now().uptimeNanoseconds - started,
                metadata: metadata
            )
            onFrame?(frame, ticket)
        }
    }
}
