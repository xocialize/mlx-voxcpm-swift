// LocDiT.swift — Bidirectional DiT with multi-token prefix conditioning
// Ports: voxcpm/modules/locdit/local_dit_v2.py (official VoxCPM2)

import MLX
import MLXNN

/// VoxCPM2 LocDiT: bidirectional transformer that predicts velocity fields for flow matching.
///
/// Input sequence construction (VoxCPM2):
///   mu is (B, 2*H_dit) from concat(lm_proj, res_proj), reshaped to (B, 2, H_dit)
///   [mu1, mu2 | t | cond_proj(prev_patch) | in_proj(z_noisy)]
///
/// The decoder runs with `isCausal: false` (full bidirectional attention).
/// Output: velocity prediction extracted at z_noisy positions only.
///   Slice: hidden[:, cond_len + mu_tokens + 1 :, :]
///
/// Ports: local_dit_v2.py VoxCPMLocDiT
public class VoxCPMLocDiT: Module, @unchecked Sendable {
    let config: LMConfig
    let inChannels: Int

    let inProj: Linear
    let condProj: Linear
    let outProj: Linear

    let timeEmbeddings: SinusoidalPosEmb
    let timeMLP: TimestepMLP
    let deltaTimeMLP: TimestepMLP

    let decoder: MiniCPMModel

    public init(config: LMConfig, inChannels: Int = 64) {
        self.config = config
        self.inChannels = inChannels

        self.inProj = Linear(inChannels, config.hiddenSize)
        self.condProj = Linear(inChannels, config.hiddenSize)
        self.outProj = Linear(config.hiddenSize, inChannels)

        self.timeEmbeddings = SinusoidalPosEmb(dim: config.hiddenSize)
        self.timeMLP = TimestepMLP(inChannels: config.hiddenSize, timeEmbedDim: config.hiddenSize)
        self.deltaTimeMLP = TimestepMLP(inChannels: config.hiddenSize, timeEmbedDim: config.hiddenSize)

        self.decoder = MiniCPMModel(config: config)
    }

    /// Forward pass.
    /// - Parameters:
    ///   - x: Noisy latent `z_noisy`, shape (N, C, T) — channels-first
    ///   - mu: Conditioning from TSLM+RALM, shape (N, 2*H_dit) — concatenated projections
    ///   - t: Timestep values, shape (N,)
    ///   - cond: Previous patch conditioning, shape (N, C, T') — channels-first
    ///   - dt: Delta timestep values, shape (N,)
    /// - Returns: Predicted velocity, shape (N, C, T) — channels-first
    public func callAsFunction(
        _ x: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        cond: MLXArray,
        dt: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let hiddenSize = config.hiddenSize

        // Cast all inputs to float32 — bf16 on Apple Silicon Metal causes audio glitches
        // (confirmed by upstream PR #263: OpenBMB/VoxCPM)
        let xCL = x.transposed(0, 2, 1).asType(.float32)
        let xProj = inProj(xCL)

        let condCL = cond.transposed(0, 2, 1).asType(.float32)
        let condProjOut = condProj(condCL)
        let condLen = condCL.dim(1)

        // Timestep embeddings (ensure float32)
        let tF32 = t.asType(.float32)
        let dtF32 = dt.asType(.float32)
        let tEmb = timeMLP(timeEmbeddings(tF32))
        let dtEmb = deltaTimeMLP(timeEmbeddings(dtF32))
        let tComb = (tEmb + dtEmb).expandedDimensions(axis: 1)

        // Reshape mu from (B, 2*H) to (B, 2, H) — two conditioning tokens
        let muReshaped = mu.asType(.float32).reshaped([B, -1, hiddenSize])
        let muTokens = muReshaped.dim(1)  // should be 2

        // Concatenate prefix: [mu1, mu2, t, cond, x]
        let hidden = MLX.concatenated([muReshaped, tComb, condProjOut, xProj], axis: 1)

        // Bidirectional attention (isCausal: false)
        let (output, _) = decoder(inputsEmbeds: hidden, isCausal: false)

        // Extract velocity at x positions: skip mu tokens + t token + cond tokens
        let prefixLen = muTokens + 1 + condLen
        let velocityHidden = output[0..., prefixLen..., 0...]

        // Project back to latent dim and transpose to channels-first
        let velocity = outProj(velocityHidden)
        return velocity.transposed(0, 2, 1)
    }
}
