// ModelLoader.swift — Load VoxCPM2 model from a local directory
// Handles config.json, safetensors shards, tokenizer.json

import Foundation
import MLX
import MLXNN
import Hub
import Tokenizers

/// Loads a VoxCPM2 model from a local HuggingFace download directory.
///
/// Expected directory layout (from `huggingface_hub.snapshot_download`):
/// ```
/// VoxCPM2-bf16/
/// ├── config.json
/// ├── model-00001.safetensors
/// ├── model-00002.safetensors
/// ├── model.safetensors.index.json
/// ├── tokenizer.json
/// └── tokenizer_config.json
/// ```
public enum ModelLoader {

    /// Load result containing model, tokenizer, and config.
    public struct LoadResult {
        public let model: VoxCPMModel
        public let tokenizer: any Tokenizer
        public let config: ModelArgs
        /// Model parameter keys that **no** weight in the loaded file filled — these remain at
        /// random init. A non-empty list means the weights don't match this architecture (e.g. an
        /// incompatible HuggingFace revision or a differently-converted checkpoint), and inference
        /// will produce garbage audio with no other symptom. Callers should treat non-empty as a
        /// hard error. Empty == every parameter was loaded.
        public let missingKeys: [String]
    }

    /// Load the VoxCPM2 model and tokenizer from a local directory.
    ///
    /// - Parameter directory: Path to the downloaded model directory
    /// - Returns: LoadResult with model, tokenizer, and config
    public static func load(from directory: URL) async throws -> LoadResult {
        // 1. Load config
        let configURL = directory.appendingPathComponent("config.json")
        let config = try ModelArgs.load(from: configURL)

        // 2. Create model (allocates parameters with random init)
        let model = VoxCPMModel(args: config)

        // 3. Load weight shards
        let weights = try loadWeightShards(from: directory)

        // 4. Remap keys from Python snake_case to Swift camelCase
        let remapped = WeightSanitizer.remapBF16Keys(weights)

        // 5. Filter to model keys and cast all weights to float32
        // (bf16 on Apple Silicon Metal causes audio glitches — upstream PR #263)
        let modelKeySet = Set(model.parameters().flattened().map { $0.0 })
        let filtered = remapped
            .filter { modelKeySet.contains($0.key) }
            .mapValues { $0.asType(.float32) }

        // Parity check: any model parameter the weights didn't fill stays at random init.
        // `update(verify: .noUnusedKeys)` only catches the opposite (extra weights), so surface
        // the unfilled keys for the caller to reject. (Empty == fully loaded.)
        let missingKeys = modelKeySet.subtracting(filtered.keys).sorted()

        // 6. Unflatten and apply weights
        let nested = ModuleParameters.unflattened(filtered)
        try model.update(parameters: nested, verify: [.noUnusedKeys])

        // 7. Load tokenizer
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: directory,
            strict: false
        )

        return LoadResult(model: model, tokenizer: tokenizer, config: config, missingKeys: missingKeys)
    }

    /// Load config only (no weights, no GPU).
    public static func loadConfig(from directory: URL) throws -> ModelArgs {
        let configURL = directory.appendingPathComponent("config.json")
        return try ModelArgs.load(from: configURL)
    }

    /// Load tokenizer only (no model, no GPU).
    public static func loadTokenizer(from directory: URL) async throws -> any Tokenizer {
        try await AutoTokenizer.from(modelFolder: directory, strict: false)
    }

    // MARK: - Internal

    /// Load all safetensors shards from a directory.
    public static func loadWeightShards(from directory: URL) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]

        // Check for shard index
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            // Sharded model — load each shard file
            let indexData = try Data(contentsOf: indexURL)
            let index = try JSONSerialization.jsonObject(with: indexData) as! [String: Any]
            let weightMap = index["weight_map"] as! [String: String]
            let shardFiles = Set(weightMap.values)

            for shardFile in shardFiles.sorted() {
                let shardURL = directory.appendingPathComponent(shardFile)
                let shard = try MLX.loadArrays(url: shardURL)
                for (key, value) in shard {
                    weights[key] = value
                }
            }
        } else {
            // Single safetensors file
            let modelURL = directory.appendingPathComponent("model.safetensors")
            weights = try MLX.loadArrays(url: modelURL)
        }

        return weights
    }
}
