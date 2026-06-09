// RMSNorm.swift — RMS normalization using MLX fast kernel
// Ports: minicpm.py:10-18

import MLX
import MLXFast
import MLXNN

/// RMSNorm using `MLXFast.rmsNorm` for fused GPU execution.
public class VoxCPMRMSNorm: Module, @unchecked Sendable {
    var weight: MLXArray
    let eps: Float

    public init(dims: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dims])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}
