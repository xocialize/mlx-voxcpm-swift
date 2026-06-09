// MLP.swift — SwiGLU MLP
// Ports: minicpm.py:152-166

import MLX
import MLXNN

/// SwiGLU MLP: down(silu(gate(x)) * up(x))
public class VoxCPMMLP: Module, @unchecked Sendable {
    let gateProj: Linear
    let upProj: Linear
    let downProj: Linear

    public init(config: LMConfig) {
        self.gateProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.upProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.downProj = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
