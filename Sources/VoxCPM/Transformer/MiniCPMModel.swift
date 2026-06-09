// MiniCPMModel.swift — MiniCPM-4 backbone shared by TSLM, RALM, LocEnc, LocDiT
// Ports: minicpm.py:169-260

import Foundation
import MLX
import MLXNN

/// MiniCPM decoder layer: pre-norm → attention → residual → pre-norm → MLP → residual
/// With optional muP depth scaling.
public class MiniCPMDecoderLayer: Module, @unchecked Sendable {
    let selfAttn: VoxCPMAttention
    let mlp: VoxCPMMLP
    let inputLayernorm: VoxCPMRMSNorm
    let postAttentionLayernorm: VoxCPMRMSNorm
    let scaleDepth: Float
    let numHiddenLayers: Int
    let useMup: Bool

    public init(config: LMConfig) {
        self.selfAttn = VoxCPMAttention(config: config)
        self.mlp = VoxCPMMLP(config: config)
        self.inputLayernorm = VoxCPMRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self.postAttentionLayernorm = VoxCPMRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self.scaleDepth = config.scaleDepth
        self.numHiddenLayers = config.numHiddenLayers
        self.useMup = config.useMup
    }

    public func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        // Pre-norm → attention → residual
        var r = x
        var h = inputLayernorm(x)
        let (attnOut, newCache) = selfAttn(h, cos: cos, sin: sin, mask: mask, cache: cache)

        if useMup {
            h = r + attnOut * (scaleDepth / sqrt(Float(numHiddenLayers)))
        } else {
            h = r + attnOut
        }

        // Pre-norm → MLP → residual
        r = h
        let mlpOut = mlp(postAttentionLayernorm(h))

        if useMup {
            h = r + mlpOut * (scaleDepth / sqrt(Float(numHiddenLayers)))
        } else {
            h = r + mlpOut
        }

        return (h, newCache)
    }
}

/// MiniCPM model: embedding (if vocab_size > 0) + N decoder layers + final RMSNorm + LongRoPE.
/// Supports both causal and bidirectional attention via `isCausal` parameter.
public class MiniCPMModel: Module, @unchecked Sendable {
    let config: LMConfig
    var embedTokens: Embedding?
    let layers: [MiniCPMDecoderLayer]
    let norm: VoxCPMRMSNorm
    let rope: MiniCPMLongRoPE?

    public init(config: LMConfig) {
        self.config = config
        self.embedTokens = config.vocabSize > 0
            ? Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
            : nil
        self.layers = (0 ..< config.numHiddenLayers).map { _ in MiniCPMDecoderLayer(config: config) }
        self.norm = VoxCPMRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        // noRope=true disables RoPE (used by RALM)
        self.rope = config.noRope ? nil : MiniCPMLongRoPE(config: config)
    }

    public func callAsFunction(
        inputsEmbeds: MLXArray? = nil,
        inputIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil,
        isCausal: Bool = true
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        // Resolve input embeddings
        var h: MLXArray
        if let embeds = inputsEmbeds {
            h = embeds
        } else if let ids = inputIds, let embed = embedTokens {
            h = embed(ids)
        } else {
            fatalError("Either inputsEmbeds or inputIds must be provided")
        }

        let B = h.dim(0)
        let L = h.dim(1)
        _ = B // suppress unused warning

        // Position offset from KV cache
        let offset: Int
        if let firstCache = cache?.first {
            offset = firstCache.0.dim(1) // k shape is (B, L_past, H_kv, D)
        } else {
            offset = 0
        }

        // Compute RoPE position embeddings (nil when noRope=true, e.g. RALM)
        let cosBatch: MLXArray
        let sinBatch: MLXArray
        if let rope = rope {
            let positionIds = MLXArray(Int32(offset) ..< Int32(offset + L))
            let (cos, sin) = rope(positionIds)
            cosBatch = cos.expandedDimensions(axis: 0)
            sinBatch = sin.expandedDimensions(axis: 0)
        } else {
            // No RoPE — provide zeros (attention will still work, just without position info)
            let headDim = config.headDim
            cosBatch = MLXArray.ones([1, L, headDim]).asType(.float32)
            sinBatch = MLXArray.zeros([1, L, headDim]).asType(.float32)
        }

        // Generate causal mask if needed
        var attnMask = mask
        if attnMask == nil && isCausal && L > 1 {
            // Upper-triangular -inf mask: positions where col > row are masked
            let totalLen = offset + L
            let rowPos = MLXArray(0 ..< Int32(L)).reshaped([L, 1])
            let colPos = MLXArray(0 ..< Int32(totalLen)).reshaped([1, totalLen])
            // Allow j <= i + offset (causal): mask where colPos > rowPos + offset
            let causal = colPos .> (rowPos + Int32(offset))
            attnMask = MLX.where(causal, MLXArray(Float(-1e9)), MLXArray(Float(0)))
            attnMask = attnMask!.reshaped([1, 1, L, totalLen])
        }
        // If isCausal == false, mask stays nil → full bidirectional attention

        // Run through layers
        var newCaches: [(MLXArray, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (layerOut, newCache) = layer(h, cos: cosBatch, sin: sinBatch, mask: attnMask, cache: layerCache)
            h = layerOut
            newCaches.append(newCache)
        }

        h = norm(h)
        return (h, newCaches)
    }
}
