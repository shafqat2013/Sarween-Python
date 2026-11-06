// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SarweenCameraProtocol",
    products: [.library(name: "SarweenCameraProtocol", targets: ["SarweenCameraProtocol"])],
    targets: [
        .target(name: "SarweenCameraProtocol", path: "Sources/Protocol"),
        .testTarget(name: "SarweenCameraProtocolTests", dependencies: ["SarweenCameraProtocol"], path: "Tests")
    ]
)
