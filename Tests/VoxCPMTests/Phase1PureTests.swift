// Phase1PureTests.swift — Non-MLX tests that don't require Metal runtime

import XCTest
@testable import VoxCPM

/// Tests that can run under `swift test` without the Metal library.
/// MLX-dependent tests must be run from Xcode (known SPM CLI limitation).
final class Phase1PureTests: XCTestCase {

    // MARK: - Config validation for Phase 1 components

    func testLMConfigHeadDim() {
        let config = LMConfig()
        let headDim = config.hiddenSize / config.numAttentionHeads
        XCTAssertEqual(headDim, 64, "Default head dim should be 1024/16 = 64")
    }

    func testResidualLMConfig() {
        var config = LMConfig()
        config.numHiddenLayers = 8  // RALM override
        config.vocabSize = 0        // RALM has no embedding
        XCTAssertEqual(config.numHiddenLayers, 8)
        XCTAssertEqual(config.vocabSize, 0)
    }

    func testDiTConfigForLocDiT() {
        let ditConfig = DiTConfig()
        XCTAssertEqual(ditConfig.hiddenDim, 1024)
        XCTAssertEqual(ditConfig.numLayers, 8)
        XCTAssertEqual(ditConfig.cfmConfig.solver, "euler")
        XCTAssertEqual(ditConfig.cfmConfig.inferenceCfgRate, 2.0)
    }

    func testEncoderConfigForLocEnc() {
        let encConfig = EncoderConfig()
        XCTAssertEqual(encConfig.hiddenDim, 1024)
        XCTAssertEqual(encConfig.numLayers, 8)
        XCTAssertEqual(encConfig.numHeads, 16)
    }

    func testFSQParameters() {
        let args = ModelArgs()
        XCTAssertEqual(args.scalarQuantizationLatentDim, 256)
        XCTAssertEqual(args.scalarQuantizationScale, 9)
        // 9 levels: {-4..4} * delta → 9^256 theoretical codebook size
    }

    func testLongRoPEScalingFactor() {
        // Verify the scaling factor formula: sqrt(1 + log(max(scale,1)) / log(origMax))
        let config = LMConfig()
        let scale = Float(config.maxPositionEmbeddings) / Float(config.originalMaxPositionEmbeddings)
        let expected = sqrt(1.0 + log(max(scale, 1.0)) / log(Float(config.originalMaxPositionEmbeddings)))
        // When max == original, scale = 1.0, log(1) = 0 → factor = 1.0
        XCTAssertEqual(expected, 1.0, accuracy: 1e-6)
    }

    func testAudioVAEHopLength() {
        let config = AudioVAEConfig()
        // 2 * 3 * 6 * 7 * 7 = 1764
        XCTAssertEqual(config.hopLength, 1764)
        // At 44100 Hz: 44100 / 1764 = 25 Hz latent rate
        XCTAssertEqual(44100 / config.hopLength, 25)
    }
}
