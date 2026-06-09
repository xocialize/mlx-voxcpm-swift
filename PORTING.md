# com.xocialize.voxcpm

Swift/MLX port of VoxCPM2 — flow matching text-to-speech with LocDiT + AudioVAE. Produces 48 kHz speech from text with voice design and voice cloning. **Status: intelligible speech output achieved.**

## Architecture

VoxCPM2 is a three-stage autoregressive-diffusion hybrid:

1. **TSLM** (MiniCPM-4, 28L causal) — Text → coarse semantic-prosodic tokens via FSQ bottleneck
2. **RALM** (8L causal, no RoPE) — Refines TSLM output with acoustic detail via fusion_concat_proj
3. **LocDiT** (12L bidirectional DiT) — Flow matching generates continuous latent patches from TSLM+RALM conditioning (2 mu prefix tokens)
4. **AudioVAE V2** — Decodes latent patches to 48 kHz waveform with SampleRateConditionLayer

```
Text → [TSLM] → FSQ → [RALM] → proj → [LocDiT (flow matching)] → [AudioVAE decode] → audio
                                          ↑
                              [LocEnc] ← ref audio → [AudioVAE encode]
```

### Generation loop (autoregressive)

Each iteration generates one patch (`patch_size=4` latent frames). The LocDiT runs `n_timesteps` Euler steps (default 10) with CFG-zero-star per patch. The `dit_hidden` is the **concatenation** (not sum) of `lm_to_dit_proj(lm_hidden)` and `res_to_dit_proj(residual_hidden)`, giving shape `(B, 2*H_dit)`. This is reshaped to `(B, 2, H_dit)` as two mu prefix tokens in the LocDiT. Stop predictor MLP on TSLM hidden state terminates generation.

### Key conventions (from official voxcpm pip v2.0.2)

- **Euler step**: subtraction `x = x - dt * v`, time goes 1→0 (t=1 noise, t=0 clean)
- **scale_emb**: only applied when `useMup=true` (default false → scale_emb=1.0)
- **scale_depth**: only applied when `useMup=true` (default false → no depth scaling)
- **RALM noRope**: `residual_lm_no_rope=true` — RALM has no positional encoding
- **Voice design**: parentheses syntax `(description)text` — tokenizer handles it
- **Output**: 48 kHz waveform (encoder input 16 kHz, decoder upsamples 1920×)

## Component Map

| Component | Swift File | Python Reference | Status |
|-----------|-----------|-----------------|--------|
| Config structs | `Config/VoxCPMConfig.swift` | `config.py` | **Done** |
| MiniCPM backbone | `Transformer/MiniCPMModel.swift` | `minicpm.py:205-260` | **Done** (Phase 1) |
| GQA Attention | `Transformer/Attention.swift` | `minicpm.py:96-149` | **Done** (Phase 1) |
| SwiGLU MLP | `Transformer/MLP.swift` | `minicpm.py:152-166` | **Done** (Phase 1) |
| RMSNorm | `Transformer/RMSNorm.swift` | `minicpm.py:10-18` | **Done** (Phase 1) |
| LongRoPE | `Transformer/LongRoPE.swift` | `minicpm.py:21-75` | **Done** (Phase 1) |
| LocDiT | `FlowMatching/LocDiT.swift` | `dit.py:47-94` | **Done** (Phase 2) |
| UnifiedCFM | `FlowMatching/UnifiedCFM.swift` | `dit.py:97-197` | **Done** (Phase 2) |
| Timestep embedding | `FlowMatching/TimestepEmbedding.swift` | `dit.py:11-44` | **Done** (Phase 1) |
| LocEnc | `Encoder/LocEnc.swift` | `encoder.py:8-45` | **Done** (Phase 2) |
| AudioVAE | `AudioVAE/AudioVAE.swift` | `audio_vae.py:333-384` | **Done** (Phase 4) |
| CausalEncoder | `AudioVAE/CausalEncoder.swift` | `audio_vae.py:166-206` | **Done** (Phase 4) |
| CausalDecoder | `AudioVAE/CausalDecoder.swift` | `audio_vae.py:277-330` | **Done** (Phase 4) |
| Snake1d | `AudioVAE/Snake1d.swift` | `audio_vae.py:83-93` | **Done** (Phase 1) |
| CausalConv1d | `AudioVAE/CausalConv1d.swift` | `audio_vae.py:11-44` | **Done** (Phase 4) |
| CausalTransposeConv1d | `AudioVAE/CausalTransposeConv1d.swift` | `audio_vae.py:47-80` | **Done** (Phase 4) |
| ResidualUnit | `AudioVAE/ResidualUnit.swift` | `audio_vae.py:96-131` | **Done** (Phase 4) |
| VoxCPM Model | `Model/VoxCPMModel.swift` | `voxcpm.py:31-463` | **Done** (Phase 5) |
| ScalarQuantization | `Model/ScalarQuantization.swift` | `voxcpm.py:16-28` | **Done** (Phase 1) |
| Weight sanitizer | `Weights/WeightSanitizer.swift` | `voxcpm.py:105-203`, `audio_vae.py:386-551` | **Done** (Phase 4) |

## Python MLX Reference

Complete working Python MLX port at:
```
com.xocialize.rosetta-dub/rosettacast-tts-server/.venv/lib/python3.12/site-packages/mlx_audio/tts/models/voxcpm/
```

Files: `voxcpm.py`, `dit.py`, `encoder.py`, `audio_vae.py`, `minicpm.py`, `config.py`

## Key Parameters (from real config.json)

| Parameter | Value |
|-----------|-------|
| LM hidden size | 2048 |
| LM layers (TSLM) | 28 |
| LM attention heads | 16 |
| LM KV heads | 2 (GQA) |
| LM kv_channels (head dim) | 128 |
| LM intermediate size | 6144 |
| RALM layers | 8 |
| Encoder hidden/layers/heads | 1024 / 12 / 16 |
| DiT hidden/layers/heads | 1024 / 12 / 16 |
| Patch size | 4 |
| Feature dim (latent) | 64 |
| FSQ latent dim / scale | 512 / 9 |
| AudioVAE encoder rates | [2, 5, 8, 8] (640× downsample) |
| AudioVAE decoder rates | [8, 6, 5, 2, 2, 2] (960× upsample) |
| AudioVAE decoder channels | 2048 |
| Encoder sample rate | 16000 Hz (input) |
| Decoder sample rate | 48000 Hz (output) |
| CFG scale | 2.0 |
| Inference timesteps | 10 |
| Total weight size | ~4.96 GB (bf16, 2 shards) |
| Model parameter keys | 820 (807 matched to Swift model) |

## Weight Files

From HuggingFace `openbmb/VoxCPM2`:
- `model.safetensors` (4.58 GB) — TSLM + RALM + LocEnc + LocDiT + FSQ + projections + stop predictor
- `audiovae.pth` (377 MB) — AudioVAE V2 (requires weight_norm fusion + key remapping)

Quantized variants from `mlx-community/`: `VoxCPM2-bf16`, `VoxCPM2-8bit`, `VoxCPM2-4bit`

## Existing Code Reuse

These components from sibling packages map directly to VoxCPM2's architecture:

| Existing | Package | VoxCPM2 Equivalent |
|----------|---------|-------------------|
| `TalkerAttention` (GQA + RoPE) | qwen3-tts | TSLM/RALM attention |
| `QuantizedMLP` (SwiGLU) | qwen3-tts | TSLM/RALM MLP |
| `FlexibleLinear` | qwen3-tts | Runtime quantization detection |
| `CausalConv1d` | qwen3-tts, mimi-codec-encoder | AudioVAE convolutions |
| `CausalTransposeConv1d` | qwen3-tts | AudioVAE decoder upsampling |
| `SnakeBeta` | qwen3-tts | AudioVAE Snake activation (variant) |
| `CommonWeightLoader` | qwen3-tts (AudioCommon) | Safetensors loading |
| `HuggingFaceDownloader` | qwen3-tts (AudioCommon) | Model fetching |

**Note:** VoxCPM2 uses a custom MiniCPM-4 backbone (not Qwen3), so the transformer is ported fresh from `minicpm.py` rather than adapting the Qwen3 Talker directly. The conv/activation/weight infrastructure transfers.

## Build Status

All 20 components implemented. All phases complete.

| Phase | Components | Status |
|-------|-----------|--------|
| 1. Foundations | Snake1d, FSQ, timestep embeddings, MiniCPM backbone (RMSNorm, LongRoPE, GQA Attention, SwiGLU MLP, DecoderLayer, Model) | **Done** |
| 2. Flow matching | LocDiT, UnifiedCFM (Euler + sway + CFG-zero-star), LocEnc | **Done** |
| 4. AudioVAE V2 | CausalConv1d, CausalTransposeConv1d, ResidualUnit, CausalEncoder, CausalDecoder, AudioVAE, WeightSanitizer | **Done** |
| 5. Integration | VoxCPMModel (autoregressive generate loop + voice cloning pathway) | **Done** |

**Tests:** 31 GPU tests (Xcode) + 15 pure logic tests (SPM) = 46 total, all passing.

## Weight Loading

Weights downloaded from `mlx-community/VoxCPM2-bf16` (pre-sanitized for MLX).

```swift
// Full model + tokenizer loading:
let result = try await ModelLoader.load(
    from: URL(fileURLWithPath: "VoxCPM/models/downloads/VoxCPM2-bf16")
)
let model = result.model       // VoxCPMModel with loaded weights
let tokenizer = result.tokenizer  // HuggingFace tokenizer

// Config-only (no GPU):
let config = try ModelLoader.loadConfig(from: url)

// Tokenizer-only (no GPU):
let tokenizer = try await ModelLoader.loadTokenizer(from: url)
```

**Key mapping:** Weight file uses Python snake_case (`self_attn.q_proj`), Swift uses camelCase (`selfAttn.qProj`). `WeightSanitizer.remapBF16Keys()` handles the conversion. 822 model keys loaded, including SampleRateConditionLayer (Embedding-based FiLM conditioning per decoder block, bucket index 3 for 48kHz).

**Weight location:** `VoxCPM/models/downloads/VoxCPM2-bf16/`

### Current status: Near production quality ✅

Intelligible English speech with voice design and voice cloning. 7 voice design tests produce varied, recognizable voices. Voice cloning via VoxCPM2 reference-only mode working. Quality approaching official web demo.

**12 bugs found and fixed:**
1. Euler step: subtraction, time 1→0 (was addition 0→1)
2. Output sample rate: 48kHz (was 16kHz)
3. SampleRateConditionLayer: Embedding FiLM per decoder block (was missing)
4. `scale_emb`: only when `useMup=true` (was inverted — 12× too large)
5. `dit_hidden`: concatenation not addition (2 mu prefix tokens in LocDiT)
6. `scale_depth`: gated by `useMup` (was unconditional)
7. RALM `noRope`: disabled (was applying position encoding incorrectly)
8. bf16→float32: cast LocDiT/CFM inputs (upstream PR #263 Metal precision fix)
9. Stop predictor: parameterized `minLen=2` (was hardcoded 5)
10. First-patch: 15ms fade-in (masks zero-conditioning artifact)
11. `zero_init_steps`: `nSteps+1` for official parity
12. Voice cloning: VoxCPM2 `_make_ref_prefix` pattern (was VoxCPM 1.x mode)

**Full float32 inference implemented** — all weights cast to fp32 in ModelLoader. RTF 0.6-0.7 on M4 Pro 64GB. Noticeable quality improvement.

**Remaining quality observations:**
- Initial word sensitivity: certain starting words ("The") produce artifacts from zero conditioning. Lead-in phrases help. Clone mode unaffected (non-zero conditioning from ref audio).
- Voice description specificity: more detailed descriptions = better voices ("A 25 year old female..." > "A professional female...")
- 50ms fade-in masks most initial artifacts

**Easy wins implemented:**
- Retry badcase: auto-retry up to 3× when patch count > 6× token count
- Temperature parameter exposed (default 1.0)
- All tuning knobs at top of `VoxCPM Debugger/main.swift` with documented ranges
- 50ms fade-in on output

**Remaining refinement opportunities:**
- Smart burn-in strategy (auto-prepend neutral lead-in, trim extra patches) — certain starting words ("The") produce artifacts from cold start
- SampleRateConditionLayer placement A/B (before vs after block)
- Proper sinc resampling for voice cloning ref audio
- Streaming decode for long-form output
- Ultimate Cloning mode (continuation with transcript — `prompt_latents` + `prompt_text`)

**HuggingFace demo analysis** (`VoxCPM/VoxCPM-Demo/app.py`):
- Confirms our runtime defaults match (cfg=2.0, temp=1.0, no output normalization)
- Uses nano-vLLM backend (not standard voxcpm) — quality gap may be from vLLM optimizations
- Three modes: Voice Design `(control)text`, Controllable Cloning (ref+control), Ultimate Cloning (ref+transcript)
- Control instruction disabled during Ultimate Cloning mode
- Max 50s reference audio, denoise on input only (ZipEnhancer), max 2000 patches

## Testing

GPU tests require Xcode's build system (SPM CLI cannot compile Metal shaders).

| Runner | Location | Metal? | Purpose |
|--------|----------|--------|---------|
| `swift test` | `Tests/VoxCPMTests/` | No | Config parsing, pure logic |
| Xcode (Cmd+U) | `VoxCPM Debugger/VoxCPMTestRunner/` | Yes | Shape tests, numerical validation, reference tensor comparisons |

The **VoxCPM Debugger** Xcode project (`../VoxCPM Debugger/`) provides both a CLI tool target for interactive debugging and an XCTest bundle for GPU tests. See [`VoxCPM Debugger/CLAUDE.md`](../VoxCPM%20Debugger/CLAUDE.md) for usage.

```bash
# GPU tests
cd "VoxCPM Debugger"
xcodebuild -project "VoxCPM Debugger.xcodeproj" -scheme VoxCPMTestRunner -destination "platform=macOS" test

# Pure logic tests
cd com.xocialize.voxcpm
swift test
```

## MLX-Swift Gotchas

### float64 crashes the GPU

MLX-Swift does not support `float64` on the GPU — any float64 tensor reaching a GPU op causes a fatal crash. Swift `Double` literals silently create float64 MLXArrays.

**Rules:**
- Always use `Float32` when constructing MLXArray from literals: `MLXArray([Float32(0.5)])` not `MLXArray([0.5])`
- Cast `MLXArray.linspace` output: `.asType(.float32)` — linspace can return float64
- Wrap Swift `Float` in `MLXArray()` before multiplying with MLXArray to avoid `Duration` type ambiguity: `MLXArray(cfgValue) * tensor` not `cfgValue * tensor`
- Use `MLX.multiply()` / `MLX.divide()` to disambiguate when the compiler confuses `Float * MLXArray` with `Duration`
- Cast any externally-supplied tensor: `input.asType(.float32)` in public API entry points

### Conv1d dimension ordering

MLX-Swift Conv1d uses channels-last `(N, T, C)` throughout. PyTorch uses channels-first `(N, C, T)`. The LocDiT and UnifiedCFM use channels-first at their public API boundary (matching the Python reference) and transpose internally.

### Weight transposition

- Conv1d: PyTorch `[O, I, K]` → MLX `[O, K, I]` — `transpose(0, 2, 1)` at conversion time
- ConvTranspose1d: PyTorch `[I, O, K]` → MLX `[I, K, O]` — `transpose(0, 2, 1)`
- Linear: no transposition needed
- GroupNorm: must init with `pytorchCompatible: true`

### Upstream `mlx-audio` PR #641 module path renamed mid-PR

When cross-referencing the upstream Python port at [Blaizzy/mlx-audio PR #641](https://github.com/Blaizzy/mlx-audio/pull/641), be aware: the module was originally named `mlx_audio/tts/models/voxcpm/` in the initial commit (`bf03666`), then **renamed to `mlx_audio/tts/models/voxcpm2/`** in commits `c7c79b8` through `33f3f45` to disambiguate from a separate, already-merged VoxCPM 1.x module (`mlx_audio/tts/models/voxcpm/` from PR #293). The 7-item bug list we filed against the PR on 2026-04-17 referenced the old `voxcpm/voxcpm.py` paths; line numbers in our comments don't map to the current file. All 7 items were fixed during the rename. See PR #641 conversation for the full audit trail.

The practical implication: **always sanity-check that `voxcpm/` vs `voxcpm2/` matches the commit you're diffing against** before claiming a bug exists upstream. The blame trail will look stale otherwise.

## Reference Code

| Resource | Location | Purpose |
|----------|----------|---------|
| **Official VoxCPM2 source** (AUTHORITATIVE) | `pip install voxcpm` → `/Library/Frameworks/Python.framework/.../site-packages/voxcpm/` | v2.0.2 — the ground truth for all implementation details |
| Official model code | `voxcpm/model/voxcpm2.py` | `_inference()` method (line 964), `_generate()` (line 452) |
| Official LocDiT V2 | `voxcpm/modules/locdit/local_dit_v2.py` | Multi-token prefix `[mu1, mu2, t, cond, x]` |
| Official UnifiedCFM | `voxcpm/modules/locdit/unified_cfm.py` | Euler solver, CFG-zero-star, sway sampling |
| Official MiniCPM4 | `voxcpm/modules/minicpm4/model.py` | `forward_step`, KV cache, RoPE, scale_depth gating |
| Official AudioVAE V2 | `voxcpm/modules/audiovae/audio_vae_v2.py` | SampleRateConditionLayer, CausalDecoder |
| Python MLX port | `com.xocialize.rosetta-dub/rosettacast-tts-server/.venv/.../mlx_audio/tts/models/voxcpm2/` | [PR #641](https://github.com/Blaizzy/mlx-audio/pull/641), unmerged. **Note path change:** module was renamed `voxcpm/` → `voxcpm2/` mid-PR (see Gotchas above). All 7 items from our 2026-04-17 bug list (incl. `kv_channels` and `sr_cond_layers`) were folded in during that rename — verified end-to-end on 2026-04-25. |
| f5-tts-swift | `VoxCPM/references/f5-tts-swift/` | Swift DiT + flow matching patterns |
| Porting prerequisites | `VoxCPM/VoxCPM.md` | Component gap analysis + phased build order |
| Research round 1 | `VoxCPM/voxCPM_research.md` | Euler sign, sample rate, sr_cond_layers, fusion_concat_proj |
| Research round 2 | `VoxCPM/voxCPM_research2.md` | Weight_norm, depthwise conv, scale_emb, official generate flow |
| Research round 3 | `VoxCPM/voxCPM_research3.md` | First-patch warmup, voice cloning tokens, CFG tuning, AudioVAE chunking |
| Research round 3r1 | `VoxCPM/voxCPM_research3r1.md` | Updated with official source verification — 12 ranked findings |
| Research round 3r2 | `VoxCPM/voxCPM_research3r2.md` | Final verified review + **bf16 precision finding (PR #263)** |

**IMPORTANT:** The Python MLX port (`mlx_audio`) has several bugs and does NOT match the official VoxCPM2 source. Always cross-reference against the official `voxcpm` pip package (v2.0.2) for implementation details.

## Dependencies

- `mlx-swift` 0.30.0+ (MLX, MLXNN, MLXFast, MLXRandom)
- `swift-transformers` 1.1.6+ (Hub for HuggingFace model loading)
