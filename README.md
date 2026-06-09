# mlx-voxcpm-swift

A Swift/MLX port of **VoxCPM2** — a flow-matching text-to-speech model (LocDiT + AudioVAE) that
produces 48 kHz speech with zero-shot synthesis, text-driven voice design, and reference-audio voice
cloning on Apple silicon.

This is the **standalone inference engine**: a full, hand-written MLX implementation (TSLM MiniCPM-4
backbone → FSQ → RALM → LocDiT flow-matching head → AudioVAE decode). It has no dependency on
MLXEngine; the MLXEngine `tts` model package
[`mlx-voxcpm2-tts-swift`](https://github.com/xocialize/mlx-voxcpm2-tts-swift) wraps it to expose the
canonical contract.

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (MLX, MLXNN, MLXFast, MLXRandom)
- [swift-transformers](https://github.com/huggingface/swift-transformers) (Hub + Tokenizers)

## Usage

```swift
import VoxCPM

// Load weights + tokenizer from a local model directory (HF layout: config.json,
// model*.safetensors [+ index], tokenizer.json).
let result = try await ModelLoader.load(from: modelDirectory)

// Tokenize text with result.tokenizer, then synthesize (zero-shot shown):
let output = result.model.generate(inputIds: inputIds)   // -> VoxCPMGenerationResult
// output.audio : 48 kHz mono MLXArray   output.sampleRate : 48000
```

## Weights

Designed against [`mlx-community/VoxCPM2-bf16`](https://huggingface.co/mlx-community/VoxCPM2-bf16)
(Apache-2.0, ~4.96 GB). The loader reads any local directory in that layout and casts weights to
float32 (bf16 is glitchy on Metal — see `PORTING.md`), so expect ~10 GB resident during inference.

## Testing

`swift test` covers config parsing and pure logic. **GPU/Metal tests require Xcode's build system**
(the SwiftPM CLI cannot compile Metal shaders).

## Notes

See [`PORTING.md`](PORTING.md) for the architecture map, exact config parameters, the MLX-Swift
gotchas (float64 GPU crashes, conv1d channels-last ordering, weight transposition), and the
documented bug-fixes that got this to intelligible speech. Some paths in those notes reference the
original development environment.

## License

MIT — the Swift port. VoxCPM2 weights are licensed by their publisher (Apache-2.0); review the model
card before redistribution.
