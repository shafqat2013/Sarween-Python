import AVFoundation
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

extension CameraOrientation {
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
    case unsupportedPreset
    case outputUnavailable
    var errorDescription: String? {
        switch self {
        case .unavailable: return "The selected rear camera is unavailable."
        case .unsupportedFormat: return "This camera cannot provide the required 1080p format at 15 FPS."
        case .unsupportedPreset: return "This capture mode is unavailable on the selected camera. Choose Standard."
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
    var onExposureState: ((Bool) -> Void)?
    var onExposureError: ((String) -> Void)?

    private let sessionQueue = DispatchQueue(label: "com.sarween.camera.capture")
    private let frameQueue = DispatchQueue(label: "com.sarween.camera.jpeg", qos: .userInitiated)
    private let imageContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private var configuredOrientation: CameraOrientation?
    private var configuredPreset: CameraCapturePreset?
    private var configuredLens: CameraLens?
    private var device: AVCaptureDevice?
    private var output: AVCaptureVideoDataOutput?
    private var notifications: [NSObjectProtocol] = []
    // Retain the exact device and native format objects used by capability discovery.
    // In particular Smooth must never independently choose a different 1080p crop.
    private struct NativeCatalogue {
        let device: AVCaptureDevice
        let formats: [CameraCapturePreset: AVCaptureDevice.Format]
        let plans: [CameraCapturePreset: CameraPresetPlan]
    }
    private var catalogues: [CameraLens: NativeCatalogue]?
    // Delegate callbacks can outlive an output being removed. Bind every pixel
    // buffer to its immutable output/lens configuration, never to UI selection.
    private struct OutputConfiguration {
        let output: AVCaptureVideoDataOutput
        let device: AVCaptureDevice
        let lens: CameraLens
        let preset: CameraCapturePreset
        let fieldOfView: Double
    }
    private let outputLock = NSLock()
    private var outputConfiguration: OutputConfiguration?

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

    func inspectCameras(completion: @escaping (Result<[CameraLensCapability], Error>) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                let catalogues = try self.loadCatalogues()
                completion(.success(CameraLens.allCases.compactMap { lens in
                    guard let catalogue = catalogues[lens] else { return nil }
                    return CameraLensCapability(lens: lens, plans: catalogue.plans,
                        torchSupported: catalogue.device.hasTorch && catalogue.device.isTorchModeSupported(.on))
                }))
            } catch { completion(.failure(error)) }
        }
    }

    private func loadCatalogues() throws -> [CameraLens: NativeCatalogue] {
        if let catalogues { return catalogues }
        var result: [CameraLens: NativeCatalogue] = [:]
        for lens in CameraLens.allCases {
            let type: AVCaptureDevice.DeviceType
            switch lens {
            case .wide: type = .builtInWideAngleCamera
            case .ultraWide: type = .builtInUltraWideCamera
            case .telephoto: type = .builtInTelephotoCamera
            }
            // Physical devices only: a virtual dual/triple camera can switch lenses.
            guard let camera = AVCaptureDevice.default(type, for: .video, position: .back), !camera.isVirtualDevice else { continue }
            let nativeFormats = camera.formats
            let candidates = nativeFormats.enumerated().map { index, format in
                let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                return CameraFormatCandidate(identifier: index, width: Int(dimensions.width), height: Int(dimensions.height),
                    fieldOfView: Double(format.videoFieldOfView), frameRates: format.videoSupportedFrameRateRanges.map {
                        CameraFrameRateInterval(minimum: $0.minFrameRate, maximum: $0.maxFrameRate)
                    })
            }
            let plans = CameraPresetPlanner.plans(for: candidates)
            guard plans[.standard] != nil else { continue }
            var retained: [CameraCapturePreset: AVCaptureDevice.Format] = [:]
            for (preset, plan) in plans { retained[preset] = nativeFormats[plan.formatIdentifier] }
            result[lens] = NativeCatalogue(device: camera, formats: retained, plans: plans)
        }
        guard !result.isEmpty else { throw CameraCaptureError.unsupportedFormat }
        catalogues = result
        return result
    }

    func start(orientation: CameraOrientation, preset: CameraCapturePreset = .standard, lens: CameraLens = .wide) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                if self.configuredOrientation != orientation || self.configuredPreset != preset || self.configuredLens != lens {
                    try self.configure(orientation: orientation, preset: preset, lens: lens)
                }
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
            self.setExposureLockedOnQueue(false)
            if self.session.isRunning { self.session.stopRunning() }
            self.onRunning?(false)
        }
    }

    func setTorch(_ enabled: Bool) {
        sessionQueue.async { [weak self] in self?.setTorchOnQueue(enabled) }
    }

    func setExposureLocked(_ locked: Bool) {
        sessionQueue.async { [weak self] in self?.setExposureLockedOnQueue(locked) }
    }

    private func setExposureLockedOnQueue(_ locked: Bool) {
        guard let camera = device else { onExposureState?(false); return }
        do {
            try camera.lockForConfiguration()
            defer { camera.unlockForConfiguration() }
            let mode: AVCaptureDevice.ExposureMode = locked ? .locked : .continuousAutoExposure
            let whiteBalanceMode: AVCaptureDevice.WhiteBalanceMode = locked ? .locked : .continuousAutoWhiteBalance
            guard camera.isExposureModeSupported(mode), camera.isWhiteBalanceModeSupported(whiteBalanceMode),
                  !locked || session.isRunning else {
                onExposureState?(camera.exposureMode == .locked && camera.whiteBalanceMode == .locked)
                if locked { onExposureError?("Exposure and white balance lock is unavailable on this camera.") }
                return
            }
            camera.exposureMode = mode
            camera.whiteBalanceMode = whiteBalanceMode
            onExposureState?(camera.exposureMode == .locked && camera.whiteBalanceMode == .locked)
        } catch {
            onExposureState?(camera.exposureMode == .locked && camera.whiteBalanceMode == .locked)
            onExposureError?("Exposure could not be changed: \(error.localizedDescription)")
        }
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

    private func configure(orientation: CameraOrientation, preset: CameraCapturePreset, lens: CameraLens) throws {
        setTorchOnQueue(false)
        if session.isRunning { session.stopRunning() }
        outputLock.lock()
        outputConfiguration = nil
        outputLock.unlock()
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        configuredOrientation = nil
        configuredPreset = nil
        configuredLens = nil
        session.inputs.forEach(session.removeInput)
        session.outputs.forEach(session.removeOutput)
        session.sessionPreset = .inputPriority
        guard let catalogue = try loadCatalogues()[lens] else { throw CameraCaptureError.unavailable }
        let camera = catalogue.device
        guard let format = catalogue.formats[preset], let plan = catalogue.plans[preset] else { throw CameraCaptureError.unsupportedPreset }
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else { throw CameraCaptureError.unavailable }
        session.addInput(input)
        try camera.lockForConfiguration()
        camera.activeFormat = format
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: Int32(preset.framesPerSecond))
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: Int32(preset.framesPerSecond))
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
        configuredPreset = preset
        configuredLens = lens
        onExposureState?(false)
        outputLock.lock()
        outputConfiguration = OutputConfiguration(output: output, device: camera, lens: lens, preset: preset, fieldOfView: plan.fieldOfView)
        outputLock.unlock()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        outputLock.lock()
        let configuration = outputConfiguration
        outputLock.unlock()
        guard let configuration, configuration.output === output else { return }
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
            var metadata: [String: Any] = ["cameraPosition": "back", "lens": configuration.lens.rawValue, "zoom": 1.0,
                                           "stabilization": "off", "geometricDistortionCorrection": "off", "hdr": "off",
                                           "nativeFieldOfViewDegrees": configuration.fieldOfView,
                                           "capturePreset": configuration.preset.rawValue,
                                           "nominalFPS": configuration.preset.framesPerSecond]
            let device = configuration.device
            do {
                let gains = device.deviceWhiteBalanceGains
                metadata["lensPosition"] = device.lensPosition
                metadata["iso"] = device.iso
                metadata["exposureDurationNs"] = CMTimeConvertScale(device.exposureDuration, timescale: 1_000_000_000, method: .default).value
                metadata["exposureLocked"] = device.exposureMode == .locked
                switch device.exposureMode {
                case .locked: metadata["exposureMode"] = "locked"
                case .autoExpose: metadata["exposureMode"] = "auto"
                case .continuousAutoExposure: metadata["exposureMode"] = "continuous-auto"
                case .custom: metadata["exposureMode"] = "custom"
                @unknown default: metadata["exposureMode"] = "unknown"
                }
                metadata["whiteBalanceGains"] = [gains.redGain, gains.greenGain, gains.blueGain]
                metadata["whiteBalanceLocked"] = device.whiteBalanceMode == .locked
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
