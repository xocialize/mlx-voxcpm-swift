// TimestepEmbedding.swift — Sinusoidal position embedding + MLP projection
// Ports: dit.py:11-44

import Foundation
import MLX
import MLXNN

/// Sinusoidal positional embedding for diffusion timesteps.
/// Output shape: (N, dim) from input (N,) scalar timesteps.
public class SinusoidalPosEmb: Module, @unchecked Sendable {
    let dim: Int

    public init(dim: Int) {
        precondition(dim % 2 == 0, "SinusoidalPosEmb dim must be even")
        self.dim = dim
    }

    public func callAsFunction(_ x: MLXArray, scale: Float = 1000.0) -> MLXArray {
        var inp = x.asType(.float32)
        if inp.ndim < 1 {
            inp = inp.reshaped([1])
        }

        let halfDim = dim / 2
        let logBase = Float(log(10000.0)) / Float(halfDim - 1)
        let freqs = MLX.exp(MLXArray(0 ..< halfDim).asType(.float32) * (-logBase))

        // (N, 1) * (1, halfDim) → (N, halfDim)
        let scaleArr = MLXArray(scale)
        let args = MLX.multiply(scaleArr, inp.expandedDimensions(axis: -1)) * freqs.expandedDimensions(axis: 0)

        return MLX.concatenated([MLX.sin(args), MLX.cos(args)], axis: -1)
    }
}

/// Two-layer MLP that projects sinusoidal embedding to model hidden space.
/// Linear → SiLU → Linear
public class TimestepMLP: Module, @unchecked Sendable {
    let linear1: Linear
    let linear2: Linear

    public init(inChannels: Int, timeEmbedDim: Int, outDim: Int? = nil) {
        self.linear1 = Linear(inChannels, timeEmbedDim)
        self.linear2 = Linear(timeEmbedDim, outDim ?? timeEmbedDim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)
        h = silu(h)
        h = linear2(h)
        return h
    }
}
