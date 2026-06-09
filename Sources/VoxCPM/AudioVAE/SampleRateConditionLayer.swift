// SampleRateConditionLayer.swift — FiLM-style scale-bias conditioning on target sample rate
// Weights: scale_embed.weight (numBuckets, channels), bias_embed.weight (numBuckets, channels)

import MLX
import MLXNN

/// FiLM-style conditioning layer that modulates decoder activations based on target sample rate.
///
/// Each layer has `scaleEmbed` and `biasEmbed` — Embedding lookup tables indexed by
/// sample rate bucket. Output: `x * scale + bias` where scale/bias are per-channel.
///
/// Weight shapes: (numBuckets, channels) — e.g. (4, 2048) for the first decoder block.
public class SampleRateConditionLayer: Module, @unchecked Sendable {
    let scaleEmbed: Embedding
    let biasEmbed: Embedding

    public init(channels: Int, numBuckets: Int = 4) {
        self.scaleEmbed = Embedding(embeddingCount: numBuckets, dimensions: channels)
        self.biasEmbed = Embedding(embeddingCount: numBuckets, dimensions: channels)
    }

    /// Apply scale-bias modulation.
    /// - Parameters:
    ///   - x: Input tensor (N, T, C)
    ///   - bucketIndex: Which sample rate bucket (0-3)
    public func callAsFunction(_ x: MLXArray, bucketIndex: Int) -> MLXArray {
        let idx = MLXArray(Int32(bucketIndex))
        let scale = scaleEmbed(idx)  // (C,)
        let bias = biasEmbed(idx)    // (C,)
        // FiLM: x * scale + bias, broadcast over (N, T, C)
        return x * scale + bias
    }
}

/// Determines the sample rate bucket index from boundaries.
///
/// boundaries = [20000, 30000, 40000]:
/// - sr < 20000 → bucket 0
/// - 20000 <= sr < 30000 → bucket 1
/// - 30000 <= sr < 40000 → bucket 2
/// - sr >= 40000 → bucket 3
public func sampleRateBucket(sampleRate: Int, boundaries: [Int]) -> Int {
    for (i, boundary) in boundaries.enumerated() {
        if sampleRate < boundary {
            return i
        }
    }
    return boundaries.count
}
