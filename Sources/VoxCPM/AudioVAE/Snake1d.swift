// Snake1d.swift — Snake activation function
// Ports: audio_vae.py:83-93

import MLX
import MLXNN

/// Snake activation: x + (1 / (alpha + eps)) * sin(alpha * x)^2
/// Alpha is per-channel and learnable (loaded from weights).
public class Snake1d: Module, @unchecked Sendable {
    var alpha: MLXArray

    public init(channels: Int) {
        self.alpha = MLXArray.ones([1, 1, channels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (N, T, C), alpha: (1, 1, C)
        let sinVal = MLX.sin(alpha * x)
        return x + (1.0 / (alpha + 1e-9)) * (sinVal * sinVal)
    }
}
