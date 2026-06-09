// VoxCPMModel.swift — Top-level VoxCPM2 model with autoregressive generation
// Ports: voxcpm.py:31-463

import Foundation
import MLX
import MLXNN

/// Result of a VoxCPM2 generation.
public struct VoxCPMGenerationResult {
    /// Generated audio waveform, shape (T,) — mono at model sample rate.
    public let audio: MLXArray
    /// Number of audio samples generated.
    public let samples: Int
    /// Sample rate in Hz (typically 44100).
    public let sampleRate: Int
    /// Number of autoregressive patches generated.
    public let patchCount: Int
    /// Wall-clock generation time in seconds.
    public let elapsedSeconds: Double
    /// Real-time factor (audio_duration / elapsed_time).
    public var realTimeFactor: Double {
        let audioDuration = Double(samples) / Double(sampleRate)
        return elapsedSeconds > 0 ? audioDuration / elapsedSeconds : 0
    }
}

/// VoxCPM2 top-level model.
///
/// Architecture: TSLM (MiniCPM-4) → FSQ → RALM → LocDiT (flow matching) → AudioVAE decode
/// With optional voice cloning via: ref audio → AudioVAE encode → LocEnc → conditioning
///
/// Generation is autoregressive over latent patches (patch_size=4).
/// Each patch runs n_timesteps Euler steps through LocDiT with CFG-zero-star.
///
/// Ports: voxcpm.py Model
public class VoxCPMModel: Module, @unchecked Sendable {
    public let args: ModelArgs

    // LM Backbone (TSLM — 24-layer causal)
    let baseLm: MiniCPMModel

    // Residual LM (RALM — 8-layer causal, no embedding)
    let residualLm: MiniCPMModel

    // Encoder (LocEnc — 8-layer bidirectional)
    let featEncoder: VoxCPMLocEnc

    // DiT / CFM (LocDiT + Euler solver)
    let featDecoder: UnifiedCFM

    // Projections
    let fsqLayer: ScalarQuantizationLayer
    let encToLmProj: Linear
    let lmToDitProj: Linear
    let resToDitProj: Linear

    // Concat-projection fusion for RALM input (VoxCPM2 upgrade over additive residuals)
    let fusionConcatProj: Linear

    // Stop predictor
    let stopProj: Linear
    let stopHead: Linear

    // Audio VAE
    public let audioVae: AudioVAE

    public init(args: ModelArgs) {
        self.args = args

        // TSLM
        self.baseLm = MiniCPMModel(config: args.lmConfig)

        // RALM — same architecture, fewer layers, no vocab embedding, optionally no RoPE
        var resConfig = args.lmConfig
        resConfig.numHiddenLayers = args.residualLmNumLayers
        resConfig.vocabSize = 0
        resConfig.noRope = args.residualLmNoRope
        self.residualLm = MiniCPMModel(config: resConfig)

        // LocEnc
        var encLmConfig = args.lmConfig
        encLmConfig.hiddenSize = args.encoderConfig.hiddenDim
        encLmConfig.intermediateSize = args.encoderConfig.ffnDim
        encLmConfig.numAttentionHeads = args.encoderConfig.numHeads
        encLmConfig.numHiddenLayers = args.encoderConfig.numLayers
        encLmConfig.kvChannels = args.encoderConfig.kvChannels
        encLmConfig.vocabSize = 0
        self.featEncoder = VoxCPMLocEnc(config: encLmConfig, inputDim: args.featDim)

        // LocDiT
        var ditLmConfig = args.lmConfig
        ditLmConfig.hiddenSize = args.ditConfig.hiddenDim
        ditLmConfig.intermediateSize = args.ditConfig.ffnDim
        ditLmConfig.numAttentionHeads = args.ditConfig.numHeads
        ditLmConfig.numHiddenLayers = args.ditConfig.numLayers
        ditLmConfig.kvChannels = args.ditConfig.kvChannels
        ditLmConfig.vocabSize = 0
        let estimator = VoxCPMLocDiT(config: ditLmConfig, inChannels: args.featDim)
        self.featDecoder = UnifiedCFM(
            inChannels: args.featDim,
            cfmParams: args.ditConfig.cfmConfig,
            estimator: estimator
        )

        // Projections
        self.fsqLayer = ScalarQuantizationLayer(
            inDim: args.lmConfig.hiddenSize,
            outDim: args.lmConfig.hiddenSize,
            latentDim: args.scalarQuantizationLatentDim,
            scale: args.scalarQuantizationScale
        )
        self.encToLmProj = Linear(args.encoderConfig.hiddenDim, args.lmConfig.hiddenSize)
        self.lmToDitProj = Linear(args.lmConfig.hiddenSize, args.ditConfig.hiddenDim)
        self.resToDitProj = Linear(args.lmConfig.hiddenSize, args.ditConfig.hiddenDim)

        // Concat-projection fusion: Linear(2*H → H) for RALM input
        self.fusionConcatProj = Linear(2 * args.lmConfig.hiddenSize, args.lmConfig.hiddenSize)

        // Stop predictor
        self.stopProj = Linear(args.lmConfig.hiddenSize, args.lmConfig.hiddenSize)
        self.stopHead = Linear(args.lmConfig.hiddenSize, 2, bias: false)

        // AudioVAE
        self.audioVae = AudioVAE(config: args.audioVaeConfig)
    }

    // MARK: - Prompt Audio Encoding

    /// Encode reference audio into patch-aligned latent features for voice cloning.
    ///
    /// - Parameter audio: Raw waveform, shape (T,) — mono, at model sample rate
    /// - Returns: Patch-aligned features, shape (audioLength, patchSize, featDim)
    public func encodePromptAudio(_ audio: MLXArray) -> MLXArray {
        let patchLen = args.patchSize * audioVae.hopLength
        var wav = audio.asType(.float32)

        // Left-pad to align to patch boundary
        let remainder = wav.dim(0) % patchLen
        if remainder != 0 {
            let padSize = patchLen - remainder
            wav = MLX.padded(wav, widths: [.init((padSize, 0))])
        }

        // Encode: (1, 1, T) → (1, T', D)
        let audioInput = wav.reshaped([1, 1, -1])
        var audioFeat = audioVae.encode(audioInput, sampleRate: audioVae.sampleRate)
        audioFeat = audioFeat.squeezed(axis: 0)  // (T', D)

        // Reshape into patches: (T', D) → (audioLength, patchSize, D)
        let tPrime = audioFeat.dim(0)
        let audioLength = tPrime / args.patchSize
        let trimmed = audioFeat[..<(audioLength * args.patchSize)]
        return trimmed.reshaped([audioLength, args.patchSize, -1])
    }

    // MARK: - Generation

    /// Generate speech audio from token IDs.
    ///
    /// ## Voice cloning modes
    ///
    /// The optional `refAudio` and `promptAudio` parameters select one of four
    /// generation modes, following the official VoxCPM2 pattern:
    ///
    /// | Mode | `refAudio` | `promptAudio` | Token / feat layout |
    /// |------|-----------|---------------|---------------------|
    /// | **Zero-shot** | nil | nil | `[text..., 101]` |
    /// | **Reference-only** | ref audio | nil | `[103, ref_audio..., 104, text..., 101]` |
    /// | **Continuation (Ultimate Cloning)** | nil | prompt audio | `[text..., 101, prompt_audio...]` |
    /// | **Combined** | ref audio | prompt audio | `[103, ref_audio..., 104, text..., 101, prompt_audio...]` |
    ///
    /// - **Reference-only** transfers voice characteristics (timbre, style) from
    ///   a brief clean sample, best for vocal identity cloning with text content
    ///   unrelated to the ref audio.
    /// - **Continuation / Ultimate Cloning** is the highest-fidelity cloning mode.
    ///   It requires the prompt audio's transcript to be **prepended** to the
    ///   target text in `inputIds` at tokenization time. The model is warm-started
    ///   from real audio features (not zeros) and generates a smooth continuation.
    ///   The returned `audio` contains only the newly-generated continuation;
    ///   the prompt audio itself is not re-synthesized or included (it's used
    ///   as conditioning only).
    /// - **Combined** uses both references — reference audio for isolation of
    ///   "what voice" from "what to say" while prompt audio anchors the starting
    ///   audio state.
    ///
    /// ## Token ID conventions
    ///
    /// - `101` = `<audio_start>` (inserted automatically)
    /// - `103` = `<ref_start>`, `104` = `<ref_end>` (inserted automatically in ref modes)
    ///
    /// ## Parameters
    ///
    /// - Parameters:
    ///   - inputIds: Text token IDs (from tokenizer), shape (L,). In continuation /
    ///     combined modes, this should be `tokenizer(prompt_text + target_text)`.
    ///   - maxTokens: Maximum autoregressive patches to generate (default 4096)
    ///   - minLen: Minimum patches before stop predictor can fire (default 2)
    ///   - refAudio: Optional reference audio for timbre cloning. Waveform (T,)
    ///     at the encoder sample rate.
    ///   - refInputIds: Unused — kept for API compatibility.
    ///   - promptAudio: Optional prompt audio for continuation-mode warm-start.
    ///     Waveform (T,) at the encoder sample rate. Caller must prepend
    ///     `prompt_text` to `inputIds` when this is set.
    ///   - inferenceTimesteps: Euler ODE steps per patch (default 10, try 7 for speed)
    ///   - cfgValue: Classifier-free guidance scale (default 2.0, range 1.5-2.5)
    ///   - temperature: Noise temperature for flow matching (default 1.0)
    ///   - retryBadcase: Retry if generation length exceeds ratio threshold (default true)
    ///   - retryMaxTimes: Maximum retry attempts (default 3)
    ///   - retryRatioThreshold: Patch-to-token ratio that triggers retry (default 6.0)
    public func generate(
        inputIds: MLXArray,
        maxTokens: Int = 4096,
        minLen: Int = 2,
        refAudio: MLXArray? = nil,
        refInputIds: MLXArray? = nil,
        promptAudio: MLXArray? = nil,
        inferenceTimesteps: Int = 10,
        cfgValue: Float = 2.0,
        temperature: Float = 1.0,
        retryBadcase: Bool = true,
        retryMaxTimes: Int = 3,
        retryRatioThreshold: Float = 6.0
    ) -> VoxCPMGenerationResult {
        let tokenCount = inputIds.dim(0)
        for attempt in 0 ..< (retryBadcase ? retryMaxTimes : 1) {
            let result = _generateOnce(
                inputIds: inputIds, maxTokens: maxTokens, minLen: minLen,
                refAudio: refAudio, refInputIds: refInputIds,
                promptAudio: promptAudio,
                inferenceTimesteps: inferenceTimesteps, cfgValue: cfgValue,
                temperature: temperature
            )
            // Check for badcase: too many patches relative to token count
            if retryBadcase && Float(result.patchCount) >= Float(tokenCount) * retryRatioThreshold {
                if attempt < retryMaxTimes - 1 {
                    continue // retry
                }
            }
            return result
        }
        // Shouldn't reach here, but generate once as fallback
        return _generateOnce(
            inputIds: inputIds, maxTokens: maxTokens, minLen: minLen,
            refAudio: refAudio, refInputIds: refInputIds,
            promptAudio: promptAudio,
            inferenceTimesteps: inferenceTimesteps, cfgValue: cfgValue,
            temperature: temperature
        )
    }

    /// Generate speech with an optional burn-in lead-in phrase prepended.
    ///
    /// **Why this exists:** The autoregressive LocDiT generation loop conditions
    /// each patch on the previous one. For zero-shot (no `refAudio`) generation,
    /// the first patch has zero conditioning, which creates audible artifacts at
    /// the very start — most pronounced on certain "cold-start" words like
    /// `"The"` in English. Callers typically mask this with a 50 ms output
    /// fade-in, but a fade only hides the artifact, it doesn't eliminate it.
    ///
    /// **How this fixes it:** This method prepends a short neutral phrase
    /// (the "burn-in") to the user's tokens, runs the generation loop once,
    /// then trims the audio proportionally so the caller gets back only the
    /// audio that corresponds to their original text. By the time the
    /// generation reaches the user's tokens, the autoregressive state has
    /// "warmed up" — the current patch conditions on real audio features from
    /// the burn-in, not zeros.
    ///
    /// **Trim heuristic:** Token-proportional. We assume burn-in and user
    /// text generate audio at roughly the same tokens-per-patch rate, so
    /// `trimPatches = round(burnInTokenCount / totalTokenCount × totalPatchCount)`.
    /// A small safety pad (1 patch) is then un-trimmed back onto the front of
    /// the audio and masked with a short fade-in, handling boundary cases
    /// where the actual burn-in ran slightly longer than the proportional guess.
    ///
    /// **Cost:** One burn-in phrase worth of extra generation (~5–15 patches
    /// for a short English phrase). No extra model runs vs `generate()`.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs of the user's text (tokenize without the burn-in).
    ///   - burnInIds: Token IDs of the burn-in phrase (tokenize separately).
    ///       Suggested English: `tokenizer.encode(text: "Let me tell you. ")`.
    ///       Pass empty (`MLXArray([])`) to fall back to `generate()` behavior.
    ///   - fadeInMs: Length of the boundary fade-in applied to the safety-pad
    ///       samples, in milliseconds. 15–30 ms is typical. Default 20.
    ///   - Other params: forwarded to `generate()`.
    /// - Returns: A `VoxCPMGenerationResult` whose `audio` corresponds only to
    ///   the user's `inputIds` (burn-in audio trimmed off). `patchCount` is
    ///   also reduced to reflect only the user-text patches. `elapsedSeconds`
    ///   includes the full generation time (burn-in isn't free).
    public func generateWithBurnIn(
        inputIds: MLXArray,
        burnInIds: MLXArray,
        fadeInMs: Double = 20.0,
        maxTokens: Int = 4096,
        minLen: Int = 2,
        refAudio: MLXArray? = nil,
        refInputIds: MLXArray? = nil,
        promptAudio: MLXArray? = nil,
        inferenceTimesteps: Int = 10,
        cfgValue: Float = 2.0,
        temperature: Float = 1.0,
        retryBadcase: Bool = true,
        retryMaxTimes: Int = 3,
        retryRatioThreshold: Float = 6.0
    ) -> VoxCPMGenerationResult {
        let burnInTokens = burnInIds.dim(0)
        // Degenerate case: no burn-in → behave exactly like generate()
        if burnInTokens == 0 {
            return generate(
                inputIds: inputIds, maxTokens: maxTokens, minLen: minLen,
                refAudio: refAudio, refInputIds: refInputIds,
                promptAudio: promptAudio,
                inferenceTimesteps: inferenceTimesteps, cfgValue: cfgValue,
                temperature: temperature,
                retryBadcase: retryBadcase, retryMaxTimes: retryMaxTimes,
                retryRatioThreshold: retryRatioThreshold
            )
        }

        // Concatenate burn-in with user tokens
        let combinedIds = MLX.concatenated([burnInIds, inputIds], axis: 0)
        let totalTokens = combinedIds.dim(0)  // = burnInTokens + userTokens

        // Run normal generation on the combined sequence
        let fullResult = generate(
            inputIds: combinedIds,
            maxTokens: maxTokens,
            minLen: minLen,
            refAudio: refAudio,
            refInputIds: refInputIds,
            promptAudio: promptAudio,
            inferenceTimesteps: inferenceTimesteps,
            cfgValue: cfgValue,
            temperature: temperature,
            retryBadcase: retryBadcase,
            retryMaxTimes: retryMaxTimes,
            retryRatioThreshold: retryRatioThreshold
        )

        // Proportional trim — how many of the generated patches belong to burn-in.
        // Subtract 1 patch of safety pad so we don't clip the first user word;
        // the pad gets a fade-in below.
        let patchCount = fullResult.patchCount
        let proportionalBurnInPatches =
            Int((Double(burnInTokens) / Double(totalTokens) * Double(patchCount)).rounded())
        let safetyPad = 1
        let patchesToTrim = max(0, proportionalBurnInPatches - safetyPad)

        // Patches → samples. AudioVAE decode rate: encoder_rate_product × patchSize.
        // In VoxCPM2, 4 patches (1 token of audio) = ~160 ms at 48 kHz, but the
        // precise patch→samples ratio is totalSamples / totalPatches — exact
        // by construction.
        let samplesPerPatch = patchCount > 0 ? fullResult.samples / patchCount : 0
        var trimSamples = patchesToTrim * samplesPerPatch
        trimSamples = max(0, min(trimSamples, fullResult.samples - 1))

        let remainingSamples = fullResult.samples - trimSamples
        if remainingSamples <= 0 {
            // Catastrophic trim — fall back to full audio. Shouldn't happen
            // unless user text is empty relative to burn-in.
            return fullResult
        }

        // Slice the audio
        var trimmed = fullResult.audio[trimSamples...]

        // Apply fade-in across the first `fadeInMs` of the trimmed audio to mask
        // any residual boundary discontinuity.
        let sampleRate = fullResult.sampleRate
        let fadeSamples = min(Int(fadeInMs * 0.001 * Double(sampleRate)), remainingSamples)
        if fadeSamples > 1 {
            let ramp = MLXArray(0 ..< fadeSamples).asType(.float32)
                / Float(max(fadeSamples - 1, 1))
            // Keep tail untouched: multiply first fadeSamples by ramp, rest by 1.
            let ones = MLXArray.ones([remainingSamples - fadeSamples]).asType(.float32)
            let mask = MLX.concatenated([ramp, ones], axis: 0)
            trimmed = trimmed * mask
            eval(trimmed)
        }

        return VoxCPMGenerationResult(
            audio: trimmed,
            samples: remainingSamples,
            sampleRate: sampleRate,
            patchCount: max(0, patchCount - patchesToTrim),
            elapsedSeconds: fullResult.elapsedSeconds
        )
    }

    private func _generateOnce(
        inputIds: MLXArray,
        maxTokens: Int,
        minLen: Int,
        refAudio: MLXArray?,
        refInputIds: MLXArray?,
        promptAudio: MLXArray?,
        inferenceTimesteps: Int,
        cfgValue: Float,
        temperature: Float
    ) -> VoxCPMGenerationResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // scale_emb is only applied when use_mup=True (official VoxCPM2 convention)
        let scaleEmb = args.lmConfig.useMup ? Float(args.lmConfig.scaleEmb) : Float(1)
        let audioStartToken = Int32(101)
        let patchSize = args.patchSize
        let featDim = args.featDim

        var textMask: MLXArray?
        var audioMask: MLXArray?
        var featEmbed: MLXArray?
        var combinedEmbed: MLXArray
        var prefixFeatCond: MLXArray

        if let refAudio = refAudio {
            // ---- Reference-only OR Combined mode ----
            // Ref-only token sequence:
            //   [103, ref_audio..., 104, text..., 101]
            // Combined (ref + prompt continuation):
            //   [103, ref_audio..., 104, text..., 101, prompt_audio...]
            //
            // _make_ref_prefix pattern from official voxcpm2.py:
            //   tokens: [103, zeros(refLen), 104]
            //   feats:  [zero_patch, ref_feat..., zero_patch]
            //   t_mask: [1, 0...0, 1]   (103/104 are text-masked)
            //   a_mask: [0, 1...1, 0]   (ref audio is audio-masked)
            let refStartToken = Int32(103)
            let refEndToken = Int32(104)

            // Encode reference audio into patches
            let refFeat = encodePromptAudio(refAudio)  // (refLen, P, D)
            let refLen = refFeat.dim(0)
            let zeroPatch = MLXArray.zeros([1, patchSize, featDim]).asType(.float32)

            // Build ref prefix
            let refTokens = MLX.concatenated([
                MLXArray([refStartToken]),
                MLXArray.zeros([refLen]).asType(.int32),
                MLXArray([refEndToken]),
            ], axis: 0)
            let refFeats = MLX.concatenated([zeroPatch, refFeat, zeroPatch], axis: 0)
            let refTMask = MLX.concatenated([
                MLXArray([Int32(1)]),
                MLXArray.zeros([refLen]).asType(.int32),
                MLXArray([Int32(1)]),
            ], axis: 0)
            let refAMask = MLX.concatenated([
                MLXArray([Int32(0)]),
                MLXArray.ones([refLen]).asType(.int32),
                MLXArray([Int32(0)]),
            ], axis: 0)

            // Build text segment: text_tokens + audio_start_token
            let textIds = MLX.concatenated([inputIds, MLXArray([audioStartToken])], axis: 0)
            let textLength = textIds.dim(0)
            let textPadFeat = MLXArray.zeros([textLength, patchSize, featDim]).asType(.float32)

            // Optional prompt audio suffix (Combined mode)
            if let promptAudio = promptAudio {
                let promptFeat = encodePromptAudio(promptAudio)  // (promptLen, P, D), left-padded
                let promptLen = promptFeat.dim(0)

                // [ref | text+101 | prompt]
                let promptPadTokens = MLXArray.zeros([promptLen]).asType(.int32)
                let allTokens = MLX.concatenated(
                    [refTokens, textIds, promptPadTokens], axis: 0
                ).expandedDimensions(axis: 0)
                let allFeats = MLX.concatenated(
                    [refFeats, textPadFeat, promptFeat], axis: 0
                ).expandedDimensions(axis: 0)

                let textMaskVec = MLX.concatenated([
                    refTMask,
                    MLXArray.ones([textLength]).asType(.int32),
                    MLXArray.zeros([promptLen]).asType(.int32),
                ], axis: 0).asType(.float32)
                let audioMaskVec = MLX.concatenated([
                    refAMask,
                    MLXArray.zeros([textLength]).asType(.int32),
                    MLXArray.ones([promptLen]).asType(.int32),
                ], axis: 0).asType(.float32)

                textMask = textMaskVec.expandedDimensions(axis: 0)
                audioMask = audioMaskVec.expandedDimensions(axis: 0)

                featEmbed = encToLmProj(featEncoder(allFeats))
                let textEmbed = MLX.multiply(
                    MLXArray(scaleEmb), baseLm.embedTokens!(allTokens)
                )
                combinedEmbed = MLX.multiply(textMask!.expandedDimensions(axis: -1), textEmbed)
                    + MLX.multiply(audioMask!.expandedDimensions(axis: -1), featEmbed!)

                // Warm start: last position is prompt audio (real features)
                let totalT = allFeats.dim(1)
                prefixFeatCond = allFeats[0..., (totalT - 1), 0..., 0...]
            } else {
                // Reference-only (no prompt continuation)
                let allTokens = MLX.concatenated(
                    [refTokens, textIds], axis: 0
                ).expandedDimensions(axis: 0)
                let allFeats = MLX.concatenated(
                    [refFeats, textPadFeat], axis: 0
                ).expandedDimensions(axis: 0)

                let textMaskVec = MLX.concatenated([
                    refTMask,
                    MLXArray.ones([textLength]).asType(.int32),
                ], axis: 0).asType(.float32)
                let audioMaskVec = MLX.concatenated([
                    refAMask,
                    MLXArray.zeros([textLength]).asType(.int32),
                ], axis: 0).asType(.float32)

                textMask = textMaskVec.expandedDimensions(axis: 0)
                audioMask = audioMaskVec.expandedDimensions(axis: 0)

                featEmbed = encToLmProj(featEncoder(allFeats))
                let textEmbed = MLX.multiply(
                    MLXArray(scaleEmb), baseLm.embedTokens!(allTokens)
                )
                combinedEmbed = MLX.multiply(textMask!.expandedDimensions(axis: -1), textEmbed)
                    + MLX.multiply(audioMask!.expandedDimensions(axis: -1), featEmbed!)

                // Last position is text (zero features) — cold start
                let totalT = allFeats.dim(1)
                prefixFeatCond = allFeats[0..., (totalT - 1), 0..., 0...]
            }

        } else if let promptAudio = promptAudio {
            // ---- Continuation-only mode (Ultimate Cloning without separate ref) ----
            // Token sequence: [text_tokens, 101, prompt_audio_pad_tokens]
            //   t_mask: [1..., 1, 0...]   (text + audio_start are text)
            //   a_mask: [0..., 0, 1...]   (prompt audio is audio)
            let textIds = MLX.concatenated([inputIds, MLXArray([audioStartToken])], axis: 0)
            let textLength = textIds.dim(0)

            let promptFeat = encodePromptAudio(promptAudio)  // (promptLen, P, D)
            let promptLen = promptFeat.dim(0)

            let textPadFeat = MLXArray.zeros([textLength, patchSize, featDim]).asType(.float32)
            let promptPadTokens = MLXArray.zeros([promptLen]).asType(.int32)

            let allTokens = MLX.concatenated(
                [textIds, promptPadTokens], axis: 0
            ).expandedDimensions(axis: 0)
            let allFeats = MLX.concatenated(
                [textPadFeat, promptFeat], axis: 0
            ).expandedDimensions(axis: 0)

            let textMaskVec = MLX.concatenated([
                MLXArray.ones([textLength]).asType(.int32),
                MLXArray.zeros([promptLen]).asType(.int32),
            ], axis: 0).asType(.float32)
            let audioMaskVec = MLX.concatenated([
                MLXArray.zeros([textLength]).asType(.int32),
                MLXArray.ones([promptLen]).asType(.int32),
            ], axis: 0).asType(.float32)

            textMask = textMaskVec.expandedDimensions(axis: 0)
            audioMask = audioMaskVec.expandedDimensions(axis: 0)

            featEmbed = encToLmProj(featEncoder(allFeats))
            let textEmbed = MLX.multiply(
                MLXArray(scaleEmb), baseLm.embedTokens!(allTokens)
            )
            combinedEmbed = MLX.multiply(textMask!.expandedDimensions(axis: -1), textEmbed)
                + MLX.multiply(audioMask!.expandedDimensions(axis: -1), featEmbed!)

            // Warm start from last prompt audio patch
            let totalT = allFeats.dim(1)
            prefixFeatCond = allFeats[0..., (totalT - 1), 0..., 0...]

        } else {
            // ---- Zero-shot: no cloning ----
            let allIds = MLX.concatenated([inputIds, MLXArray([audioStartToken])], axis: 0)
            combinedEmbed = MLX.multiply(
                MLXArray(scaleEmb),
                baseLm.embedTokens!(allIds.expandedDimensions(axis: 0))
            )
            prefixFeatCond = MLXArray.zeros([1, patchSize, featDim]).asType(.float32)
        }

        // ---- Prefill: run TSLM over combined embeddings ----
        var (encOutputs, lmCache) = baseLm(inputsEmbeds: combinedEmbed, isCausal: true)

        // Apply FSQ to audio positions (or all if no cloning)
        if let tm = textMask, let am = audioMask {
            encOutputs = MLX.multiply(am.expandedDimensions(axis: -1), fsqLayer(encOutputs))
                + MLX.multiply(tm.expandedDimensions(axis: -1), encOutputs)
        }

        var lmHidden = encOutputs[0..., (-1)..., 0...].squeezed(axis: 1)  // (1, H)

        if textMask == nil {
            lmHidden = fsqLayer(lmHidden)
        }

        // RALM prefill input — concat-projection fusion (VoxCPM2)
        var residualInput: MLXArray
        if let am = audioMask, let fe = featEmbed {
            let audioContrib = MLX.multiply(am.expandedDimensions(axis: -1), fe)
            let concatInput = MLX.concatenated([encOutputs, audioContrib], axis: -1)
            residualInput = fusionConcatProj(concatInput)
        } else {
            // No cloning — concat with zeros and project
            let zeros = MLXArray.zeros(like: encOutputs)
            let concatInput = MLX.concatenated([encOutputs, zeros], axis: -1)
            residualInput = fusionConcatProj(concatInput)
        }

        var (residualOutputs, resCache) = residualLm(inputsEmbeds: residualInput, isCausal: true)
        var residualHidden = residualOutputs[0..., (-1)..., 0...].squeezed(axis: 1)

        // ---- Autoregressive generation loop ----
        //
        // Note on continuation modes: the official Python seeds
        // `pred_feat_seq` with the prompt audio patches so the final VAE
        // decode sees them for smooth boundary continuity (then trims via
        // context_len). We skip this seeding for simplicity — the warm
        // `prefixFeatCond` already carries the continuation signal into
        // the autoregressive loop, so the generated patches are voice-
        // continuous. The minor VAE-boundary benefit of seeding is worth
        // revisiting if streaming decode (async iterator yielding
        // per-chunk audio) is added — see official voxcpm2.py line 1069+
        // for the Python reference pattern.
        var predFeatSeq: [MLXArray] = []

        for i in 0 ..< maxTokens {
            // Project TSLM + RALM hidden to DiT space — CONCATENATION (not addition)
            let ditH = MLX.concatenated([lmToDitProj(lmHidden), resToDitProj(residualHidden)], axis: -1)

            // Previous patch as conditioning (channels-first for LocDiT)
            let condIn = prefixFeatCond.transposed(0, 2, 1)  // (B, D, P)

            // Flow matching: generate one latent patch
            var predFeat = featDecoder.sample(
                mu: ditH,
                nTimesteps: inferenceTimesteps,
                patchSize: patchSize,
                cond: condIn,
                temperature: temperature,
                cfgValue: cfgValue
            )
            // predFeat is (B, D, P) channels-first → transpose to (B, P, D)
            predFeat = predFeat.transposed(0, 2, 1)
            predFeatSeq.append(predFeat)

            // Encode generated patch for next step
            let currEmbed = encToLmProj(
                featEncoder(predFeat.expandedDimensions(axis: 1))  // (B, 1, P, D) → (B, 1, H)
            )

            // Stop prediction
            let stopLogits = stopHead(silu(stopProj(lmHidden)))
            eval(stopLogits)
            let stopFlag = MLX.argMax(stopLogits, axis: -1).item(Int.self)
            if i > minLen && stopFlag == 1 {
                break
            }

            // TSLM step: advance one position with KV cache
            let (newLmOut, newLmCache) = baseLm(
                inputsEmbeds: currEmbed, cache: lmCache, isCausal: true
            )
            lmCache = newLmCache

            var lmHiddenNext = newLmOut[0..., (-1)..., 0...].squeezed(axis: 1)
            lmHiddenNext = fsqLayer(lmHiddenNext)

            // RALM step — concat-projection fusion (VoxCPM2)
            let resConcat = MLX.concatenated([
                lmHiddenNext.expandedDimensions(axis: 1), currEmbed
            ], axis: -1)
            let resIn = fusionConcatProj(resConcat)
            let (newResOut, newResCache) = residualLm(
                inputsEmbeds: resIn, cache: resCache, isCausal: true
            )
            resCache = newResCache
            residualHidden = newResOut[0..., (-1)..., 0...].squeezed(axis: 1)

            lmHidden = lmHiddenNext
            prefixFeatCond = predFeat
        }

        // ---- Decode all patches to audio ----
        let allFeats = MLX.concatenated(predFeatSeq, axis: 1)  // (B, totalPatches*P, D)
        let audio = audioVae.decode(allFeats)
        let audioFlat = audio.reshaped([-1])
        eval(audioFlat)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        return VoxCPMGenerationResult(
            audio: audioFlat,
            samples: audioFlat.dim(0),
            sampleRate: args.audioVaeConfig.effectiveOutputSampleRate,
            patchCount: predFeatSeq.count,
            elapsedSeconds: elapsed
        )
    }
}
