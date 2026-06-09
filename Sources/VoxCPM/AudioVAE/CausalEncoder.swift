// CausalEncoder.swift — Strided conv stack for AudioVAE encoder
// Ports: audio_vae.py:134-206

import Foundation
import MLX
import MLXNN

/// Encoder block: 3x residual units (dilation 1, 3, 9) → Snake → strided conv
public class CausalEncoderBlock: Module, @unchecked Sendable {
    let res1: CausalResidualUnit
    let res2: CausalResidualUnit
    let res3: CausalResidualUnit
    let snake: Snake1d
    let conv: VoxCPMCausalConv1d

    public init(outputDim: Int = 16, inputDim: Int? = nil, stride: Int = 1, groups: Int = 1) {
        let inDim = inputDim ?? (outputDim / 2)
        self.res1 = CausalResidualUnit(dim: inDim, dilation: 1, groups: groups)
        self.res2 = CausalResidualUnit(dim: inDim, dilation: 3, groups: groups)
        self.res3 = CausalResidualUnit(dim: inDim, dilation: 9, groups: groups)
        self.snake = Snake1d(channels: inDim)
        self.conv = VoxCPMCausalConv1d(
            inChannels: inDim, outChannels: outputDim,
            kernelSize: 2 * stride, stride: stride,
            padding: Int(ceil(Double(stride) / 2.0))
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = res1(x)
        h = res2(h)
        h = res3(h)
        h = snake(h)
        h = conv(h)
        return h
    }
}

/// Causal encoder: Conv1d → N encoder blocks → fc_mu projection.
///
/// Default config (VoxCPM2): dModel=64, strides=[2,3,6,7,7], depthwise=true
/// Total downsampling: 2×3×6×7×7 = 1764×
/// Channel progression: 64 → 128 → 256 → 512 → 1024 → 2048
public class CausalEncoder: Module, @unchecked Sendable {
    let convIn: VoxCPMCausalConv1d
    let blocks: [CausalEncoderBlock]
    let fcMu: VoxCPMCausalConv1d

    public init(dModel: Int = 64, latentDim: Int = 32, strides: [Int] = [2, 4, 8, 8], depthwise: Bool = false) {
        self.convIn = VoxCPMCausalConv1d(inChannels: 1, outChannels: dModel, kernelSize: 7, padding: 3)

        var blocks: [CausalEncoderBlock] = []
        var currDim = dModel
        for stride in strides {
            let nextDim = currDim * 2
            let groups = depthwise ? (nextDim / 2) : 1
            blocks.append(CausalEncoderBlock(outputDim: nextDim, inputDim: currDim, stride: stride, groups: groups))
            currDim = nextDim
        }
        self.blocks = blocks

        self.fcMu = VoxCPMCausalConv1d(inChannels: currDim, outChannels: latentDim, kernelSize: 3, padding: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in blocks {
            h = block(h)
        }
        return fcMu(h)
    }
}
