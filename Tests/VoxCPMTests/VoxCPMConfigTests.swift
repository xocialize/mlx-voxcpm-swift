// VoxCPMConfigTests.swift — Config parsing smoke tests

import XCTest
@testable import VoxCPM

final class VoxCPMConfigTests: XCTestCase {

    func testDefaultModelArgs() {
        let args = ModelArgs()

        XCTAssertEqual(args.lmConfig.hiddenSize, 1024)
        XCTAssertEqual(args.lmConfig.numHiddenLayers, 24)
        XCTAssertEqual(args.lmConfig.numAttentionHeads, 16)
        XCTAssertEqual(args.lmConfig.numKeyValueHeads, 2)
        XCTAssertEqual(args.lmConfig.intermediateSize, 4096)
        XCTAssertEqual(args.lmConfig.vocabSize, 73448)
        XCTAssertEqual(args.lmConfig.ropeScalingType, "longrope")
        XCTAssertEqual(args.lmConfig.useMup, false)

        XCTAssertEqual(args.residualLmNumLayers, 8)
        XCTAssertEqual(args.patchSize, 4)
        XCTAssertEqual(args.featDim, 64)
        XCTAssertEqual(args.scalarQuantizationLatentDim, 256)
        XCTAssertEqual(args.scalarQuantizationScale, 9)

        XCTAssertEqual(args.encoderConfig.hiddenDim, 1024)
        XCTAssertEqual(args.encoderConfig.numLayers, 8)

        XCTAssertEqual(args.ditConfig.hiddenDim, 1024)
        XCTAssertEqual(args.ditConfig.numLayers, 8)
        XCTAssertEqual(args.ditConfig.cfmConfig.solver, "euler")
        XCTAssertEqual(args.ditConfig.cfmConfig.inferenceCfgRate, 2.0)

        XCTAssertEqual(args.audioVaeConfig.encoderDim, 64)
        XCTAssertEqual(args.audioVaeConfig.encoderRates, [2, 3, 6, 7, 7])
        XCTAssertEqual(args.audioVaeConfig.decoderDim, 2048)
        XCTAssertEqual(args.audioVaeConfig.decoderRates, [7, 7, 6, 3, 2])
        XCTAssertEqual(args.audioVaeConfig.sampleRate, 44100)
        XCTAssertEqual(args.audioVaeConfig.hopLength, 1764)
    }

    func testModelArgsJSONRoundTrip() throws {
        let original = ModelArgs()
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(ModelArgs.self, from: data)

        XCTAssertEqual(decoded.lmConfig.hiddenSize, original.lmConfig.hiddenSize)
        XCTAssertEqual(decoded.patchSize, original.patchSize)
        XCTAssertEqual(decoded.audioVaeConfig.sampleRate, original.audioVaeConfig.sampleRate)
        XCTAssertEqual(decoded.ditConfig.cfmConfig.solver, original.ditConfig.cfmConfig.solver)
    }

    func testLMConfigWithRopeScaling() throws {
        let json = """
        {
            "hidden_size": 2048,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 8192,
            "vocab_size": 73448,
            "rope_scaling": {
                "type": "longrope",
                "long_factor": [1.0, 1.0, 1.0],
                "short_factor": [1.0, 1.0, 1.0],
                "original_max_position_embeddings": 32768
            },
            "scale_emb": 12,
            "scale_depth": 1.4
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(LMConfig.self, from: data)

        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.numHiddenLayers, 32)
        XCTAssertEqual(config.numKeyValueHeads, 4)
        XCTAssertEqual(config.ropeScalingType, "longrope")
        XCTAssertEqual(config.ropeLongFactor, [1.0, 1.0, 1.0])
        XCTAssertEqual(config.ropeShortFactor, [1.0, 1.0, 1.0])
        XCTAssertEqual(config.originalMaxPositionEmbeddings, 32768)
    }
}
