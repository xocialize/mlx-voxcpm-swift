// CausalConv1d.swift — Left-padded causal convolution
// Ports: audio_vae.py:11-44

import MLX
import MLXNN

/// Causal 1D convolution with left-only padding to preserve causality.
/// Input/output format: (N, T, C) — MLX channels-last convention.
///
/// Padding strategy: pads `padVal * 2` on the left side only, then runs a
/// zero-padding Conv1d. This ensures the output at time t depends only on
/// inputs at times <= t.
public class VoxCPMCausalConv1d: Module, @unchecked Sendable {
    let conv: Conv1d
    let padVal: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.padVal = padding
        self.conv = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if padVal > 0 {
            // Pad on the left (beginning of time dimension)
            // x is (N, T, C) — pad axis 1 with (padVal*2, 0)
            let padded = MLX.padded(x, widths: [.init((0, 0)), .init((padVal * 2, 0)), .init((0, 0))])
            return conv(padded)
        }
        return conv(x)
    }
}
