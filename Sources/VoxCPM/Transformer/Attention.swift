// Attention.swift — GQA attention with causal/bidirectional masking
// Ports: minicpm.py:78-149

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - RoPE helpers

/// Rotate the second half of features to the front, negated.
/// x[..., :d/2], x[..., d/2:] → [-x[..., d/2:], x[..., :d/2]]
func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<halfDim]
    let x2 = x[.ellipsis, halfDim...]
    return MLX.concatenated([MLX.negative(x2), x1], axis: -1)
}

/// Apply rotary position embeddings to query and key tensors.
/// q, k: (B, L, H, D);  cos, sin: (B, L, 1, D) or broadcastable
func applyRotaryPosEmb(
    q: MLXArray, k: MLXArray,
    cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    // Expand cos/sin for the head dimension: (B, L, D) → (B, L, 1, D)
    let cosExp = cos[.ellipsis, .newAxis, 0...]
    let sinExp = sin[.ellipsis, .newAxis, 0...]

    let qEmbed = (q * cosExp) + (rotateHalf(q) * sinExp)
    let kEmbed = (k * cosExp) + (rotateHalf(k) * sinExp)
    return (qEmbed, kEmbed)
}

// MARK: - Attention

/// Group Query Attention with RoPE and optional causal masking.
/// Supports both causal (TSLM/RALM) and bidirectional (LocEnc/LocDiT) modes.
public class VoxCPMAttention: Module, @unchecked Sendable {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int

    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear

    public init(config: LMConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim  // Uses kvChannels if set, else hiddenSize/numHeads

        self.qProj = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self.kProj = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self.vProj = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self.oProj = Linear(numHeads * headDim, config.hiddenSize, bias: false)
    }

    public func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim)

        // Apply RoPE
        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)

        // KV cache: concatenate past keys/values (cache format: (B, L_past, H_kv, D))
        if let (kCache, vCache) = cache {
            k = MLX.concatenated([kCache, k], axis: 1)
            v = MLX.concatenated([vCache, v], axis: 1)
        }
        let newCache = (k, v)

        // Transpose to (B, H, L, D) for attention
        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        let scale = 1.0 / sqrt(Float(headDim))
        let out = MLXFast.scaledDotProductAttention(
            queries: qT, keys: kT, values: vT,
            scale: scale, mask: mask
        )

        // (B, H, L, D) → (B, L, H, D) → (B, L, H*D)
        let reshaped = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oProj(reshaped), newCache)
    }
}
