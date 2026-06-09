// CausalTransposeConv1d.swift — Causal transposed convolution with right trimming
// Ports: audio_vae.py:47-80

import MLX
import MLXNN

/// Causal transposed 1D convolution with right-side trimming for upsampling.
/// Input/output format: (N, T, C) — MLX channels-last convention.
///
/// After the transposed convolution, trims `padVal * 2 - outputPadding` samples
/// from the right to maintain causal alignment.
public class VoxCPMCausalTransposeConv1d: Module, @unchecked Sendable {
    let convT: ConvTransposed1d
    let padVal: Int
    let outputPadding: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        bias: Bool = true
    ) {
        self.padVal = padding
        self.outputPadding = outputPadding
        self.convT = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convT(x)

        // Trim from the right: padVal * 2 - outputPadding
        let trim = padVal * 2 - outputPadding
        if trim > 0 {
            let T = y.dim(1)
            y = y[0..., ..<(T - trim), 0...]
        }
        return y
    }
}
