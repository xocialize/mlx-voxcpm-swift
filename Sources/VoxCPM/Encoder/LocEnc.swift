// LocEnc.swift — Bidirectional encoder with CLS token extraction
// Ports: encoder.py:8-45

import MLX
import MLXRandom
import MLXNN

/// VoxCPM2 LocEnc: encodes audio latent patches into conditioning tokens.
///
/// For each patch: prepends a learnable special (CLS) token, runs bidirectional
/// attention over (P+1) positions, extracts CLS output (index 0) as the per-patch
/// representation.
///
/// Input: (B, T, P, D) — batch of T patches, each P frames of D-dim latents
/// Output: (B, T, H) — per-patch hidden representations
///
/// Ports: encoder.py VoxCPMLocEnc
public class VoxCPMLocEnc: Module, @unchecked Sendable {
    let config: LMConfig
    var specialToken: MLXArray
    let inProj: Linear
    let encoder: MiniCPMModel

    public init(config: LMConfig, inputDim: Int = 64) {
        self.config = config
        self.specialToken = MLXRandom.normal([1, 1, 1, config.hiddenSize])
        self.inProj = Linear(inputDim, config.hiddenSize)
        self.encoder = MiniCPMModel(config: config)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let P = x.dim(2)
        _ = P

        // Project input features to hidden size: (B, T, P, D) → (B, T, P, H)
        let projected = inProj(x)

        // Expand CLS token: (1, 1, 1, H) → (B, T, 1, H)
        let clsTokens = MLX.broadcast(
            specialToken,
            to: [B, T, 1, config.hiddenSize]
        )

        // Prepend CLS: (B, T, P+1, H)
        let withCLS = MLX.concatenated([clsTokens, projected], axis: 2)

        // Flatten batch and time: (B*T, P+1, H)
        let P1 = withCLS.dim(2)
        let H = withCLS.dim(3)
        let flat = withCLS.reshaped([B * T, P1, H])

        // Bidirectional attention (isCausal: false)
        let (output, _) = encoder(inputsEmbeds: flat, isCausal: false)

        // Extract CLS token output (index 0): (B*T, H)
        let clsOutput = output[0..., 0, 0...]

        // Reshape back: (B, T, H)
        return clsOutput.reshaped([B, T, -1])
    }
}
