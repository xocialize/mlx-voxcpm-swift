// LongRoPE.swift — MiniCPM LongRoPE with short/long factors and scaling
// Ports: minicpm.py:21-75

import Foundation
import MLX
import MLXNN

/// MiniCPM-style LongRoPE with configurable short/long frequency factors.
/// Returns (cos, sin) position embeddings for given position IDs.
public class MiniCPMLongRoPE: Module, @unchecked Sendable {
    let config: LMConfig
    let dim: Int
    let base: Float
    let maxPositionEmbeddings: Int
    let originalMaxPositionEmbeddings: Int
    let shortFactor: MLXArray
    let longFactor: MLXArray
    let scalingFactor: Float
    let invFreq: MLXArray

    public init(config: LMConfig) {
        self.config = config
        self.dim = config.headDim
        self.base = config.ropeTheta
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = config.originalMaxPositionEmbeddings

        self.shortFactor = MLXArray(config.ropeShortFactor)
        self.longFactor = MLXArray(config.ropeLongFactor)

        let scale = Float(config.maxPositionEmbeddings) / Float(config.originalMaxPositionEmbeddings)
        self.scalingFactor = sqrt(
            1.0 + log(max(scale, 1.0)) / log(Float(config.originalMaxPositionEmbeddings))
        )

        let halfDim = dim / 2
        let exponents = MLXArray(0 ..< halfDim).asType(.float32) / Float(halfDim)
        self.invFreq = 1.0 / MLX.pow(MLXArray(base), exponents)
    }

    public func callAsFunction(_ positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let seqLen = positionIds.max().item(Int.self) + 1
        let factors = seqLen > originalMaxPositionEmbeddings ? longFactor : shortFactor

        let t = MLXArray(0 ..< seqLen).asType(.float32)
        let factorsF32 = factors.asType(.float32)

        // freqs = t[:, None] * (1/factors[None, :]) * invFreq[None, :]
        // shapes: (L, 1) * (1, D/2) * (1, D/2) → (L, D/2)
        let oneOverFactors = MLX.divide(MLXArray(Float32(1.0)), factorsF32.expandedDimensions(axis: 0))
        let freqs = t.expandedDimensions(axis: 1)
            * oneOverFactors
            * invFreq.expandedDimensions(axis: 0)

        // emb = cat(freqs, freqs) → (L, D)
        let emb = MLX.concatenated([freqs, freqs], axis: -1)

        let cosVals = MLX.cos(emb) * scalingFactor
        let sinVals = MLX.sin(emb) * scalingFactor

        // Index by positionIds to support arbitrary position sequences
        return (cos: cosVals[positionIds], sin: sinVals[positionIds])
    }
}
