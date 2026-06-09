// ScalarQuantization.swift — FSQ layer (Linear → tanh → round → Linear)
// Ports: voxcpm.py:16-28

import MLX
import MLXNN

/// Scalar quantization layer used as information bottleneck between TSLM and RALM.
///
/// Forward: inProj → tanh → round(x * scale) / scale → outProj
public class ScalarQuantizationLayer: Module, @unchecked Sendable {
    let scale: Int
    let inProj: Linear
    let outProj: Linear

    public init(inDim: Int, outDim: Int, latentDim: Int = 64, scale: Int = 9) {
        self.scale = scale
        self.inProj = Linear(inDim, latentDim)
        self.outProj = Linear(latentDim, outDim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inProj(x)
        h = MLX.tanh(h)
        let s = Float(scale)
        h = MLX.round(h * s) / s
        return outProj(h)
    }
}
