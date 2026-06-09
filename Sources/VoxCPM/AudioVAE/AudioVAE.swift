// AudioVAE.swift — AudioVAE V2 encoder/decoder wrapper
// Ports: audio_vae.py:333-384

import Foundation
import MLX
import MLXNN

/// AudioVAE V2: DAC-based variational autoencoder for audio.
///
/// Encoder: 44.1 kHz waveform → latent (25 Hz, 64-dim)
/// Decoder: latent → 44.1 kHz waveform
///
/// Encoder rates: [2, 3, 6, 7, 7] = 1764x downsampling
/// Decoder rates: [7, 7, 6, 3, 2] = 1764x upsampling
public class AudioVAE: Module, @unchecked Sendable {
    let config: AudioVAEConfig
    let encoder: CausalEncoder
    let decoder: CausalDecoder
    public let hopLength: Int
    public let sampleRate: Int

    /// Output sample rate (48kHz for VoxCPM2)
    public let outSampleRate: Int

    public init(config: AudioVAEConfig) {
        self.config = config
        self.hopLength = config.hopLength
        self.sampleRate = config.sampleRate
        self.outSampleRate = config.effectiveOutputSampleRate

        self.encoder = CausalEncoder(
            dModel: config.encoderDim,
            latentDim: config.latentDim,
            strides: config.encoderRates,
            depthwise: true
        )
        self.decoder = CausalDecoder(
            inputChannel: config.latentDim,
            channels: config.decoderDim,
            rates: config.decoderRates,
            depthwise: true,
            dOut: 1
        )

        // Set sample rate bucket for decoder conditioning
        // boundaries = [20000, 30000, 40000]: 48kHz → bucket 3
        decoder.srBucket = sampleRateBucket(
            sampleRate: outSampleRate,
            boundaries: [20000, 30000, 40000]
        )
    }

    /// Encode waveform to latent representation.
    ///
    /// Input: (N, 1, T) or (N, T, 1) or (N, T) waveform at sample_rate
    /// Output: (N, T', D) latent features where T' = ceil(T / hopLength)
    public func encode(_ x: MLXArray, sampleRate: Int? = nil) -> MLXArray {
        var h = x
        // Ensure (N, T, C) format
        if h.ndim == 2 {
            h = h.expandedDimensions(axis: -1)  // (N, T) → (N, T, 1)
        }
        // If channels < time, it's in PyTorch (N, C, T) format — transpose
        if h.dim(1) < h.dim(2) {
            h = h.transposed(0, 2, 1)
        }

        h = preprocess(h, sampleRate: sampleRate)
        return encoder(h)
    }

    /// Decode latent to waveform.
    ///
    /// Input: (N, T, C) latent
    /// Output: (N, T) waveform
    public func decode(_ z: MLXArray) -> MLXArray {
        let out = decoder(z)
        return out.squeezed(axis: -1)  // (N, T, 1) → (N, T)
    }

    /// Pad input to be a multiple of hopLength.
    private func preprocess(_ audioData: MLXArray, sampleRate: Int? = nil) -> MLXArray {
        let length = audioData.dim(1)
        let rightPad = (Int(ceil(Double(length) / Double(hopLength))) * hopLength) - length
        if rightPad > 0 {
            return MLX.padded(audioData, widths: [.init((0, 0)), .init((0, rightPad)), .init((0, 0))])
        }
        return audioData
    }
}
