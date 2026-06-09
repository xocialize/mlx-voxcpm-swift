// VoxCPMConfig.swift — Configuration structs for VoxCPM2
// Ported from: config.py (mlx_audio/tts/models/voxcpm/config.py)

import Foundation

// MARK: - LM Configuration (MiniCPM-4 backbone)

/// Configuration for the MiniCPM-4 language model backbone.
/// Shared by TSLM, RALM, LocEnc, and LocDiT (with overridden fields).
public struct LMConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var intermediateSize: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScalingType: String
    public var ropeLongFactor: [Float]
    public var ropeShortFactor: [Float]
    public var scaleEmb: Int
    public var dimModelBase: Int
    public var scaleDepth: Float
    public var originalMaxPositionEmbeddings: Int
    public var maxPositionEmbeddings: Int
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var useMup: Bool
    /// Per-head dimension for K/V (and Q) projections. When set, overrides `hiddenSize / numAttentionHeads`.
    public var kvChannels: Int?
    /// When true, disables RoPE for this model (used by RALM).
    public var noRope: Bool

    /// Effective head dimension: kvChannels if set, otherwise hiddenSize / numAttentionHeads.
    public var headDim: Int {
        kvChannels ?? (hiddenSize / numAttentionHeads)
    }

    public init(
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 2,
        intermediateSize: Int = 4096,
        vocabSize: Int = 73448,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10000.0,
        ropeScalingType: String = "longrope",
        ropeLongFactor: [Float] = [],
        ropeShortFactor: [Float] = [],
        scaleEmb: Int = 12,
        dimModelBase: Int = 256,
        scaleDepth: Float = 1.4,
        originalMaxPositionEmbeddings: Int = 32768,
        maxPositionEmbeddings: Int = 32768,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        useMup: Bool = false,
        kvChannels: Int? = nil,
        noRope: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScalingType = ropeScalingType
        self.ropeLongFactor = ropeLongFactor
        self.ropeShortFactor = ropeShortFactor
        self.scaleEmb = scaleEmb
        self.dimModelBase = dimModelBase
        self.scaleDepth = scaleDepth
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.useMup = useMup
        self.kvChannels = kvChannels
        self.noRope = noRope
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case scaleEmb = "scale_emb"
        case dimModelBase = "dim_model_base"
        case scaleDepth = "scale_depth"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case useMup = "use_mup"
        case kvChannels = "kv_channels"
        case noRope = "no_rope"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 73448
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        scaleEmb = try container.decodeIfPresent(Int.self, forKey: .scaleEmb) ?? 12
        dimModelBase = try container.decodeIfPresent(Int.self, forKey: .dimModelBase) ?? 256
        scaleDepth = try container.decodeIfPresent(Float.self, forKey: .scaleDepth) ?? 1.4
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        useMup = try container.decodeIfPresent(Bool.self, forKey: .useMup) ?? false
        kvChannels = try container.decodeIfPresent(Int.self, forKey: .kvChannels)
        noRope = try container.decodeIfPresent(Bool.self, forKey: .noRope) ?? false

        // rope_scaling is a nested object in the JSON config
        if let ropeScaling = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling) {
            ropeScalingType = ropeScaling.type ?? "longrope"
            ropeLongFactor = ropeScaling.longFactor ?? []
            ropeShortFactor = ropeScaling.shortFactor ?? []
            originalMaxPositionEmbeddings = ropeScaling.originalMaxPositionEmbeddings ?? 32768
        } else {
            ropeScalingType = "longrope"
            ropeLongFactor = []
            ropeShortFactor = []
            originalMaxPositionEmbeddings = try container.decodeIfPresent(
                Int.self, forKey: .originalMaxPositionEmbeddings
            ) ?? 32768
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(scaleEmb, forKey: .scaleEmb)
        try container.encode(dimModelBase, forKey: .dimModelBase)
        try container.encode(scaleDepth, forKey: .scaleDepth)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(bosTokenId, forKey: .bosTokenId)
        try container.encode(eosTokenId, forKey: .eosTokenId)
        try container.encode(useMup, forKey: .useMup)
        try container.encodeIfPresent(kvChannels, forKey: .kvChannels)
        try container.encode(noRope, forKey: .noRope)
        let ropeScaling = RopeScaling(
            type: ropeScalingType,
            longFactor: ropeLongFactor,
            shortFactor: ropeShortFactor,
            originalMaxPositionEmbeddings: originalMaxPositionEmbeddings
        )
        try container.encode(ropeScaling, forKey: .ropeScaling)
    }
}

/// Nested rope_scaling object in HuggingFace config.json
struct RopeScaling: Codable, Sendable {
    var type: String?
    var longFactor: [Float]?
    var shortFactor: [Float]?
    var originalMaxPositionEmbeddings: Int?

    enum CodingKeys: String, CodingKey {
        case type
        case longFactor = "long_factor"
        case shortFactor = "short_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }
}

// MARK: - Encoder Configuration

/// Configuration for LocEnc (bidirectional encoder).
public struct EncoderConfig: Codable, Sendable {
    public var hiddenDim: Int
    public var ffnDim: Int
    public var numHeads: Int
    public var numLayers: Int
    public var kvChannels: Int?

    public init(
        hiddenDim: Int = 1024,
        ffnDim: Int = 4096,
        numHeads: Int = 16,
        numLayers: Int = 8,
        kvChannels: Int? = nil
    ) {
        self.hiddenDim = hiddenDim
        self.ffnDim = ffnDim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.kvChannels = kvChannels
    }

    enum CodingKeys: String, CodingKey {
        case hiddenDim = "hidden_dim"
        case ffnDim = "ffn_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case kvChannels = "kv_channels"
    }
}

// MARK: - CFM Configuration

/// Configuration for Conditional Flow Matching.
public struct CFMConfig: Codable, Sendable {
    public var sigmaMin: Float
    public var solver: String
    public var tScheduler: String
    public var inferenceCfgRate: Float

    public init(
        sigmaMin: Float = 1e-6,
        solver: String = "euler",
        tScheduler: String = "log-norm",
        inferenceCfgRate: Float = 2.0
    ) {
        self.sigmaMin = sigmaMin
        self.solver = solver
        self.tScheduler = tScheduler
        self.inferenceCfgRate = inferenceCfgRate
    }

    enum CodingKeys: String, CodingKey {
        case sigmaMin = "sigma_min"
        case solver
        case tScheduler = "t_scheduler"
        case inferenceCfgRate = "inference_cfg_rate"
    }
}

// MARK: - DiT Configuration

/// Configuration for LocDiT (bidirectional DiT with flow matching).
public struct DiTConfig: Codable, Sendable {
    public var hiddenDim: Int
    public var ffnDim: Int
    public var numHeads: Int
    public var numLayers: Int
    public var cfmConfig: CFMConfig
    public var kvChannels: Int?

    public init(
        hiddenDim: Int = 1024,
        ffnDim: Int = 4096,
        numHeads: Int = 16,
        numLayers: Int = 8,
        cfmConfig: CFMConfig = CFMConfig(),
        kvChannels: Int? = nil
    ) {
        self.hiddenDim = hiddenDim
        self.ffnDim = ffnDim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.cfmConfig = cfmConfig
        self.kvChannels = kvChannels
    }

    enum CodingKeys: String, CodingKey {
        case hiddenDim = "hidden_dim"
        case ffnDim = "ffn_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case cfmConfig = "cfm_config"
        case kvChannels = "kv_channels"
    }
}

// MARK: - AudioVAE Configuration

/// Configuration for AudioVAE V2 (DAC-based encoder/decoder).
public struct AudioVAEConfig: Codable, Sendable {
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var sampleRate: Int
    /// Output sample rate — decoder upsamples asymmetrically (e.g. 16kHz input → 48kHz output).
    public var outSampleRate: Int?

    public init(
        encoderDim: Int = 64,
        encoderRates: [Int] = [2, 3, 6, 7, 7],
        latentDim: Int = 64,
        decoderDim: Int = 2048,
        decoderRates: [Int] = [7, 7, 6, 3, 2],
        sampleRate: Int = 44100,
        outSampleRate: Int? = nil
    ) {
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.latentDim = latentDim
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.sampleRate = sampleRate
        self.outSampleRate = outSampleRate
    }

    /// Product of encoder rates — total temporal downsampling factor.
    public var hopLength: Int {
        encoderRates.reduce(1, *)
    }

    /// Effective output sample rate: outSampleRate if set, else sampleRate.
    public var effectiveOutputSampleRate: Int {
        outSampleRate ?? sampleRate
    }

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case latentDim = "latent_dim"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case sampleRate = "sample_rate"
        case outSampleRate = "out_sample_rate"
    }
}

// MARK: - Top-Level Model Configuration

/// Top-level configuration for the VoxCPM2 model.
public struct ModelArgs: Codable, Sendable {
    public var lmConfig: LMConfig
    public var encoderConfig: EncoderConfig
    public var ditConfig: DiTConfig
    public var audioVaeConfig: AudioVAEConfig
    public var patchSize: Int
    public var featDim: Int
    public var scalarQuantizationLatentDim: Int
    public var scalarQuantizationScale: Int
    public var residualLmNumLayers: Int
    public var residualLmNoRope: Bool
    public var maxLength: Int

    public init(
        lmConfig: LMConfig = LMConfig(),
        encoderConfig: EncoderConfig = EncoderConfig(),
        ditConfig: DiTConfig = DiTConfig(),
        audioVaeConfig: AudioVAEConfig = AudioVAEConfig(),
        patchSize: Int = 4,
        featDim: Int = 64,
        scalarQuantizationLatentDim: Int = 256,
        scalarQuantizationScale: Int = 9,
        residualLmNumLayers: Int = 8,
        residualLmNoRope: Bool = false,
        maxLength: Int = 8192
    ) {
        self.lmConfig = lmConfig
        self.encoderConfig = encoderConfig
        self.ditConfig = ditConfig
        self.audioVaeConfig = audioVaeConfig
        self.patchSize = patchSize
        self.featDim = featDim
        self.scalarQuantizationLatentDim = scalarQuantizationLatentDim
        self.scalarQuantizationScale = scalarQuantizationScale
        self.residualLmNumLayers = residualLmNumLayers
        self.residualLmNoRope = residualLmNoRope
        self.maxLength = maxLength
    }

    enum CodingKeys: String, CodingKey {
        case lmConfig = "lm_config"
        case encoderConfig = "encoder_config"
        case ditConfig = "dit_config"
        case audioVaeConfig = "audio_vae_config"
        case patchSize = "patch_size"
        case featDim = "feat_dim"
        case scalarQuantizationLatentDim = "scalar_quantization_latent_dim"
        case scalarQuantizationScale = "scalar_quantization_scale"
        case residualLmNumLayers = "residual_lm_num_layers"
        case residualLmNoRope = "residual_lm_no_rope"
        case maxLength = "max_length"
    }

    /// Load ModelArgs from a config.json file.
    public static func load(from url: URL) throws -> ModelArgs {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(ModelArgs.self, from: data)
    }
}
