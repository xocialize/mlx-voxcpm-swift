// swift-tools-version: 6.0
// VoxCPM2 — Flow Matching Text-to-Speech via LocDiT + AudioVAE (Swift/MLX port)

import PackageDescription

let package = Package(
    name: "VoxCPM",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "VoxCPM", targets: ["VoxCPM"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "VoxCPM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/VoxCPM"
        ),
        .testTarget(
            name: "VoxCPMTests",
            dependencies: ["VoxCPM"],
            path: "Tests/VoxCPMTests"
        ),
    ]
)
