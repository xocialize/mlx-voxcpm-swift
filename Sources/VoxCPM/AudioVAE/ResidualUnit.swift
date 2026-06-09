// ResidualUnit.swift — Snake → Conv(dilation) → Snake → Conv(1) → residual add
// Ports: audio_vae.py:96-131

import MLX
import MLXNN

/// Residual unit used in both encoder and decoder blocks.
/// Pattern: Snake1d → Conv1d(dilated) → Snake1d → Conv1d(k=1) → residual add
public class CausalResidualUnit: Module, @unchecked Sendable {
    let snake1: Snake1d
    let conv1: VoxCPMCausalConv1d
    let snake2: Snake1d
    let conv2: VoxCPMCausalConv1d

    public init(dim: Int = 16, dilation: Int = 1, kernel: Int = 7, groups: Int = 1) {
        let pad = ((kernel - 1) * dilation) / 2

        self.snake1 = Snake1d(channels: dim)
        self.conv1 = VoxCPMCausalConv1d(
            inChannels: dim, outChannels: dim,
            kernelSize: kernel, dilation: dilation, padding: pad, groups: groups
        )
        self.snake2 = Snake1d(channels: dim)
        self.conv2 = VoxCPMCausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1(x)
        h = conv1(h)
        h = snake2(h)
        h = conv2(h)
        return x + h
    }
}
