// WeightSanitizer.swift — Weight loading, weight_norm fusion, and key remapping
// Ports: voxcpm.py:105-203 (model sanitize) + audio_vae.py:386-551 (VAE sanitize)

import Foundation
import MLX
import MLXNN

/// Handles weight loading and conversion for VoxCPM2 models.
///
/// Key responsibilities:
/// 1. Fuse weight_norm (weight_v + weight_g → weight) for AudioVAE
/// 2. Remap PyTorch key names to Swift module paths
/// 3. Transpose Conv1d/ConvTranspose1d weights for MLX layout
/// 4. Handle shape mismatches via automatic transposition
public enum WeightSanitizer {

    // MARK: - Weight Norm Fusion

    /// Fuse weight normalization tensors: weight = weight_g * (weight_v / ||weight_v||)
    ///
    /// Scans for `*.weight_g` / `*.weight_v` pairs, computes fused weight,
    /// and returns a dict with `*.weight` entries replacing the pairs.
    public static func fuseWeightNorm(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var fused: [String: MLXArray] = [:]
        var processed: Set<String> = []

        let keys = Array(weights.keys)
        for key in keys {
            if processed.contains(key) { continue }

            if key.hasSuffix(".weight_g") {
                let base = String(key.dropLast(9))  // remove ".weight_g"
                let vKey = base + ".weight_v"
                if let g = weights[key], let v = weights[vKey] {
                    // Fuse: w = g * (v / ||v||)
                    let vFlat = v.reshaped([v.dim(0), -1])
                    let norm = MLX.sqrt(MLX.sum(vFlat * vFlat, axis: 1)).reshaped(g.shape)
                    let w = MLX.multiply(g, v / (norm + 1e-9))
                    fused[base + ".weight"] = w
                    processed.insert(key)
                    processed.insert(vKey)
                    continue
                }
            }
            if key.hasSuffix(".weight_v") { continue }  // skip, handled with _g

            fused[key] = weights[key]
        }

        return fused
    }

    // MARK: - AudioVAE Key Remapping

    /// Remap AudioVAE PyTorch key names to Swift module paths.
    ///
    /// PyTorch layout:
    ///   encoder.block.0 → conv_in
    ///   encoder.block.N (N>=1) → blocks[N-1]
    ///   decoder.model.0/1 → conv_in (depthwise layers)
    ///   decoder.model.2..N+1 → blocks[0..N-1]
    ///   decoder.model.N+2 → snake_out
    ///   decoder.model.N+3 → conv_out
    ///
    /// Within encoder blocks: block.{0,1,2,3,4} → {res1,res2,res3,snake,conv}
    /// Within decoder blocks: block.{0,1,2,3,4} → {snake,conv_t,res1,res2,res3}
    /// Within residual units: block.{0,1,2,3} → {snake1,conv1,snake2,conv2}
    public static func remapAudioVAEKeys(_ weights: [String: MLXArray], decoderRates: [Int]) -> [String: MLXArray] {
        var remapped: [String: MLXArray] = [:]
        let numDecBlocks = decoderRates.count

        for (key, value) in weights {
            let parts = key.split(separator: ".").map(String.init)
            let newKey = remapParts(parts, numDecBlocks: numDecBlocks)
            remapped[newKey] = value
        }

        return remapped
    }

    private static func remapParts(_ parts: [String], numDecBlocks: Int) -> String {
        var result: [String] = []
        var i = 0

        while i < parts.count {
            let p = parts[i]

            if p == "encoder" && i + 1 < parts.count && parts[i + 1] == "block" && i + 2 < parts.count {
                result.append("encoder")
                let idx = Int(parts[i + 2])!
                if idx == 0 {
                    result.append("convIn")
                    i += 3
                    continue
                } else {
                    result.append("blocks")
                    result.append(String(idx - 1))
                    i += 3
                    continue
                }
            }

            if p == "decoder" && i + 1 < parts.count && parts[i + 1] == "model" && i + 2 < parts.count {
                result.append("decoder")
                let idx = Int(parts[i + 2])!
                if idx == 0 {
                    result.append("convIn")
                    result.append("0")
                    i += 3
                    continue
                } else if idx == 1 {
                    result.append("convIn")
                    result.append("1")
                    i += 3
                    continue
                } else if idx >= 2 && idx < 2 + numDecBlocks {
                    result.append("blocks")
                    result.append(String(idx - 2))
                    i += 3
                    continue
                } else if idx == 2 + numDecBlocks {
                    result.append("snakeOut")
                    i += 3
                    continue
                } else if idx == 2 + numDecBlocks + 1 {
                    result.append("convOut")
                    i += 3
                    continue
                }
            }

            // Block-level remapping within encoder/decoder blocks
            if p == "block" && i + 1 < parts.count, let idx = Int(parts[i + 1]) {
                let isEncoder = result.contains("encoder") && result.contains("blocks")
                let isDecoder = result.contains("decoder") && result.contains("blocks")
                let isResidual = result.last == "res1" || result.last == "res2" || result.last == "res3"

                if isResidual {
                    // Inside residual unit: block.{0,1,2,3} → {snake1,conv1,snake2,conv2}
                    let mapping = [0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"]
                    result.append(mapping[idx] ?? "unknown_\(idx)")
                    i += 2
                    continue
                } else if isEncoder {
                    // Encoder block: block.{0,1,2,3,4} → {res1,res2,res3,snake,conv}
                    let mapping = [0: "res1", 1: "res2", 2: "res3", 3: "snake", 4: "conv"]
                    result.append(mapping[idx] ?? "unknown_\(idx)")
                    i += 2
                    continue
                } else if isDecoder {
                    // Decoder block: block.{0,1,2,3,4} → {snake,convT,res1,res2,res3}
                    let mapping = [0: "snake", 1: "convT", 2: "res1", 3: "res2", 4: "res3"]
                    result.append(mapping[idx] ?? "unknown_\(idx)")
                    i += 2
                    continue
                }
            }

            result.append(p)
            i += 1
        }

        return result.joined(separator: ".")
    }

    // MARK: - Shape Matching

    /// Sanitize model weights: transpose mismatched shapes, add missing RoPE buffers.
    ///
    /// For Conv1d weights: tries `transpose(0, 2, 1)` if shapes don't match.
    /// For Linear weights: tries `transpose()` if shapes don't match.
    public static func sanitizeModelWeights(
        _ weights: [String: MLXArray],
        modelParams: [String: MLXArray]
    ) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            guard let expectedShape = modelParams[key]?.shape else {
                // Key not in model — keep anyway (might be needed later)
                result[key] = value
                continue
            }

            if value.shape == expectedShape {
                result[key] = value
            } else if value.ndim == 3 && expectedShape.count == 3 {
                // Conv1d weight — try transpose(0, 2, 1)
                let transposed = value.transposed(0, 2, 1)
                if transposed.shape == expectedShape {
                    result[key] = transposed
                } else {
                    result[key] = value
                }
            } else if value.ndim == 2 && expectedShape.count == 2 {
                // Linear weight — try transpose
                let transposed = value.transposed()
                if transposed.shape == expectedShape {
                    result[key] = transposed
                } else {
                    result[key] = value
                }
            } else {
                result[key] = value
            }
        }

        // Add missing RoPE parameters from model (computed at init, not in weights file)
        for (key, value) in modelParams {
            if result[key] == nil && key.contains("rope") {
                result[key] = value
            }
        }

        return result
    }

    // MARK: - BF16 Pre-Sanitized Key Remapping

    /// Remap keys from mlx-community/VoxCPM2-bf16 weight files to Swift module paths.
    ///
    /// The bf16 weights are already sanitized (weight_norm fused, PyTorch keys remapped to
    /// Python MLX module paths). This function converts from Python snake_case convention
    /// to Swift camelCase, and inserts wrapper property names (`.conv`, `.convT`).
    ///
    /// Key transformations:
    /// - `audio_vae` → `audioVae`
    /// - `base_lm` → `baseLm`
    /// - `residual_lm` → `residualLm`
    /// - `self_attn` → `selfAttn`
    /// - `q_proj` → `qProj`, `k_proj` → `kProj`, etc.
    /// - `input_layernorm` → `inputLayernorm`
    /// - `post_attention_layernorm` → `postAttentionLayernorm`
    /// - `gate_proj` → `gateProj`, `up_proj` → `upProj`, `down_proj` → `downProj`
    /// - `conv_in` → `convIn`, `conv_out` → `convOut`, `conv_t` → `convT`
    /// - `snake_out` → `snakeOut`, `fc_mu` → `fcMu`
    /// - `feat_encoder` → `featEncoder`, `feat_decoder` → `featDecoder`
    /// - `enc_to_lm_proj` → `encToLmProj`, etc.
    /// - `fsq_layer` → `fsqLayer`
    /// - `stop_proj` → `stopProj`, `stop_head` → `stopHead`
    /// - `fusion_concat_proj` → `fusionConcatProj`
    /// - `in_proj` → `inProj`, `out_proj` → `outProj`, `cond_proj` → `condProj`
    /// - `time_mlp` → `timeMLP`, `delta_time_mlp` → `deltaTimeMLP`
    /// - `time_embeddings` → `timeEmbeddings`
    /// - `special_token` → `specialToken`
    /// - `embed_tokens` → `embedTokens`
    /// - `inv_freq` → `invFreq`
    /// - `long_factor` → `longFactor`, `short_factor` → `shortFactor`
    /// - `blocks.layers.N` → `blocks.N`
    /// - `conv_in.layers.N` → `convIn.N`
    /// - AudioVAE conv wrappers: insert `.conv` for CausalConv1d, `.convT` for CausalTransposeConv1d
    public static func remapBF16Keys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            let newKey = remapSingleBF16Key(key)
            result[newKey] = value
        }

        return result
    }

    /// Convert a single weight key from Python snake_case to Swift camelCase module path.
    static func remapSingleBF16Key(_ key: String) -> String {
        var k = key

        // Static segment replacements (order matters — longer matches first)
        let replacements: [(String, String)] = [
            ("audio_vae", "audioVae"),
            ("base_lm", "baseLm"),
            ("residual_lm", "residualLm"),
            ("feat_encoder", "featEncoder"),
            ("feat_decoder", "featDecoder"),
            ("enc_to_lm_proj", "encToLmProj"),
            ("lm_to_dit_proj", "lmToDitProj"),
            ("res_to_dit_proj", "resToDitProj"),
            ("fusion_concat_proj", "fusionConcatProj"),
            ("fsq_layer", "fsqLayer"),
            ("stop_proj", "stopProj"),
            ("stop_head", "stopHead"),
            ("post_attention_layernorm", "postAttentionLayernorm"),
            ("input_layernorm", "inputLayernorm"),
            ("self_attn", "selfAttn"),
            ("embed_tokens", "embedTokens"),
            ("special_token", "specialToken"),
            ("delta_time_mlp", "deltaTimeMLP"),
            ("time_embeddings", "timeEmbeddings"),
            ("time_mlp", "timeMLP"),
            ("gate_proj", "gateProj"),
            ("up_proj", "upProj"),
            ("down_proj", "downProj"),
            ("q_proj", "qProj"),
            ("k_proj", "kProj"),
            ("v_proj", "vProj"),
            ("o_proj", "oProj"),
            ("in_proj", "inProj"),
            ("out_proj", "outProj"),
            ("cond_proj", "condProj"),
            ("conv_in", "convIn"),
            ("conv_out", "convOut"),
            ("conv_t", "convT"),
            ("snake_out", "snakeOut"),
            ("fc_mu", "fcMu"),
            ("inv_freq", "invFreq"),
            ("long_factor", "longFactor"),
            ("short_factor", "shortFactor"),
            ("_sr_boundaries", "srBoundaries"),
            ("sr_cond_layers", "srCondLayers"),
            ("scale_embed", "scaleEmbed"),
            ("bias_embed", "biasEmbed"),
            ("linear_1", "linear1"),
            ("linear_2", "linear2"),
            ("rms_norm_eps", "rmsNormEps"),
        ]

        for (old, new) in replacements {
            k = k.replacingOccurrences(of: old, with: new)
        }

        // Remove `.layers` only from AudioVAE array paths:
        //   blocks.layers.0 → blocks.0
        //   convIn.layers.0 → convIn.0
        // Do NOT remove from transformer paths (baseLm.layers.0 must stay)
        k = k.replacingOccurrences(of: "blocks.layers.", with: "blocks.")
        k = k.replacingOccurrences(of: "convIn.layers.", with: "convIn.")

        // Insert .conv wrapper for AudioVAE CausalConv1d weights
        // Pattern: audioVae.{encoder|decoder}...{conv|convIn|convOut|fcMu|conv1|conv2}.{weight|bias}
        // These need an extra .conv because VoxCPMCausalConv1d wraps Conv1d
        k = insertConvWrapper(k)

        return k
    }

    /// Insert `.conv` or `.convT` wrapper for AudioVAE conv layers.
    ///
    /// The weight file has flat keys like `audioVae.decoder.blocks.0.convT.weight`
    /// but our Swift module has `audioVae.decoder.blocks.0.convT.convT.weight`
    /// because VoxCPMCausalTransposeConv1d wraps ConvTransposed1d in a `.convT` property,
    /// and VoxCPMCausalConv1d wraps Conv1d in a `.conv` property.
    private static func insertConvWrapper(_ key: String) -> String {
        guard key.hasPrefix("audioVae.") else { return key }

        let parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 3 else { return key }

        let terminal = parts.last!
        guard terminal == "weight" || terminal == "bias" else { return key }

        // Walk backwards to find the conv-like parent
        // Patterns that need .conv insertion:
        //   audioVae.*.conv.{weight|bias} → audioVae.*.conv.conv.{weight|bias}
        //   audioVae.*.convIn.{weight|bias} → audioVae.*.convIn.conv.{weight|bias}
        //   audioVae.*.convOut.{weight|bias} → audioVae.*.convOut.conv.{weight|bias}
        //   audioVae.*.fcMu.{weight|bias} → audioVae.*.fcMu.conv.{weight|bias}
        //   audioVae.*.conv1.{weight|bias} → audioVae.*.conv1.conv.{weight|bias}
        //   audioVae.*.conv2.{weight|bias} → audioVae.*.conv2.conv.{weight|bias}
        //   audioVae.*.convT.{weight|bias} → audioVae.*.convT.convT.{weight|bias}
        //   audioVae.decoder.convIn.N.{weight|bias} → audioVae.decoder.convIn.N.conv.{weight|bias}

        let parent = parts[parts.count - 2]

        // Named conv layers that wrap VoxCPMCausalConv1d (which has .conv property)
        let causalConvNames: Set<String> = [
            "conv", "convIn", "convOut", "fcMu", "conv1", "conv2",
        ]

        // ConvTranspose1d: convT wraps VoxCPMCausalTransposeConv1d (which has .convT property)
        if parent == "convT" {
            var newParts = parts
            newParts.insert("convT", at: parts.count - 1)
            return newParts.joined(separator: ".")
        }

        if causalConvNames.contains(parent) {
            var newParts = parts
            newParts.insert("conv", at: parts.count - 1)
            return newParts.joined(separator: ".")
        }

        // Array index parent (e.g. convIn.0.weight → convIn.0.conv.weight)
        // This is for decoder.convIn which is an array of VoxCPMCausalConv1d
        if let _ = Int(parent), parts.count >= 4 {
            let grandparent = parts[parts.count - 3]
            if grandparent == "convIn" {
                var newParts = parts
                newParts.insert("conv", at: parts.count - 1)
                return newParts.joined(separator: ".")
            }
        }

        return key
    }

    // MARK: - Full AudioVAE Sanitization Pipeline

    /// Complete sanitization for AudioVAE weights:
    /// 1. Filter out fc_logvar (not needed for inference)
    /// 2. Fuse weight_norm pairs
    /// 3. Remap PyTorch keys to Swift module paths
    /// 4. Transpose Conv1d weights for MLX layout
    public static func sanitizeAudioVAEWeights(
        _ weights: [String: MLXArray],
        modelParams: [String: MLXArray],
        decoderRates: [Int]
    ) -> [String: MLXArray] {
        // 1. Filter out fc_logvar
        var filtered = weights.filter { !$0.key.contains("fc_logvar") }

        // 2. Fuse weight_norm
        filtered = fuseWeightNorm(filtered)

        // 3. Remap keys
        filtered = remapAudioVAEKeys(filtered, decoderRates: decoderRates)

        // 4. Shape matching (transpose conv weights etc.)
        filtered = sanitizeModelWeights(filtered, modelParams: modelParams)

        return filtered
    }
}
