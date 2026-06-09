// CausalDecoder.swift — Transposed conv stack for AudioVAE decoder
// Ports: audio_vae.py:239-330

import Foundation
import MLX
import MLXNN

/// Decoder block: Snake → TransposeConv(strided) → 3x residual units (dilation 1, 3, 9)
public class CausalDecoderBlock: Module, @unchecked Sendable {
    let snake: Snake1d
    let convT: VoxCPMCausalTransposeConv1d
    let res1: CausalResidualUnit
    let res2: CausalResidualUnit
    let res3: CausalResidualUnit

    public init(inputDim: Int = 16, outputDim: Int = 8, stride: Int = 1, groups: Int = 1) {
        self.snake = Snake1d(channels: inputDim)
        self.convT = VoxCPMCausalTransposeConv1d(
            inChannels: inputDim, outChannels: outputDim,
            kernelSize: 2 * stride, stride: stride,
            padding: Int(ceil(Double(stride) / 2.0)),
            outputPadding: stride % 2
        )
        self.res1 = CausalResidualUnit(dim: outputDim, dilation: 1, groups: groups)
        self.res2 = CausalResidualUnit(dim: outputDim, dilation: 3, groups: groups)
        self.res3 = CausalResidualUnit(dim: outputDim, dilation: 9, groups: groups)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake(x)
        h = convT(h)
        h = res1(h)
        h = res2(h)
        h = res3(h)
        return h
    }
}

/// Causal decoder: Conv1d → N decoder blocks (with sr_cond) → Snake → Conv1d → Tanh.
///
/// Default config (VoxCPM2): channels=2048, rates=[8,6,5,2,2,2], depthwise=true
/// Total upsampling: 8×6×5×2×2×2 = 1920×
/// Channel progression: 2048 → 1024 → 512 → 256 → 128 → 64 → 32
public class CausalDecoder: Module, @unchecked Sendable {
    let convIn: [VoxCPMCausalConv1d]  // 1 or 2 layers depending on depthwise
    let blocks: [CausalDecoderBlock]
    let srCondLayers: [SampleRateConditionLayer]  // FiLM conditioning per block
    let snakeOut: Snake1d
    let convOut: VoxCPMCausalConv1d

    /// Sample rate bucket for conditioning (set before decoding)
    public var srBucket: Int = 3  // default: highest quality (48kHz, bucket 3 for boundaries [20k,30k,40k])

    public init(
        inputChannel: Int,
        channels: Int,
        rates: [Int],
        depthwise: Bool = false,
        dOut: Int = 1,
        numSrBuckets: Int = 4
    ) {
        // Input conv — depthwise separable or standard
        if depthwise {
            self.convIn = [
                VoxCPMCausalConv1d(
                    inChannels: inputChannel, outChannels: inputChannel,
                    kernelSize: 7, padding: 3, groups: inputChannel
                ),
                VoxCPMCausalConv1d(
                    inChannels: inputChannel, outChannels: channels,
                    kernelSize: 1
                ),
            ]
        } else {
            self.convIn = [
                VoxCPMCausalConv1d(
                    inChannels: inputChannel, outChannels: channels,
                    kernelSize: 7, padding: 3
                ),
            ]
        }

        var blocks: [CausalDecoderBlock] = []
        var srCondLayers: [SampleRateConditionLayer] = []
        for (i, stride) in rates.enumerated() {
            let inputDim = channels / (1 << i)
            let outputDim = channels / (1 << (i + 1))
            let groups = depthwise ? outputDim : 1
            blocks.append(CausalDecoderBlock(
                inputDim: inputDim, outputDim: outputDim,
                stride: stride, groups: groups
            ))
            // sr_cond modulates the input to each block (matches inputDim)
            srCondLayers.append(SampleRateConditionLayer(channels: inputDim, numBuckets: numSrBuckets))
        }
        self.blocks = blocks
        self.srCondLayers = srCondLayers

        let finalDim = channels / (1 << rates.count)
        self.snakeOut = Snake1d(channels: finalDim)
        self.convOut = VoxCPMCausalConv1d(inChannels: finalDim, outChannels: dOut, kernelSize: 7, padding: 3)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in convIn {
            h = layer(h)
        }
        for (block, srCond) in zip(blocks, srCondLayers) {
            h = srCond(h, bucketIndex: srBucket)  // FiLM conditioning before each block
            h = block(h)
        }
        h = snakeOut(h)
        h = convOut(h)
        return MLX.tanh(h)
    }
}
