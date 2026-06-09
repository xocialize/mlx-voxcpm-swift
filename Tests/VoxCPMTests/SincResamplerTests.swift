// SincResamplerTests.swift — Pure Swift tests for the SincResampler.

import XCTest
@testable import VoxCPM

final class SincResamplerTests: XCTestCase {

    // MARK: - Identity cases

    func testIdentityWhenRatesMatch() {
        let input: [Float] = [0.1, 0.2, -0.3, 0.4, -0.5]
        let output = SincResampler.resample(audio: input, from: 16000, to: 16000)
        XCTAssertEqual(output, input, "Same rate should return input unchanged")
    }

    func testEmptyInputReturnsEmpty() {
        let output = SincResampler.resample(audio: [], from: 44100, to: 16000)
        XCTAssertTrue(output.isEmpty)
    }

    // MARK: - Length scaling

    func testUpsamplingLengthScales() {
        // 10 samples @ 8kHz → ~20 samples @ 16kHz
        let input = [Float](repeating: 0.0, count: 10)
        let output = SincResampler.resample(audio: input, from: 8000, to: 16000)
        XCTAssertEqual(output.count, 20)
    }

    func testDownsamplingLengthScales() {
        // 10 samples @ 48kHz → ~3 samples @ 16kHz (ratio 1/3)
        let input = [Float](repeating: 0.0, count: 10)
        let output = SincResampler.resample(audio: input, from: 48000, to: 16000)
        // Expect round(10 × 16000 / 48000) = round(3.33) = 3
        XCTAssertEqual(output.count, 3)
    }

    // MARK: - Signal fidelity

    func testDCPreservation() {
        // A constant (DC-only) signal must survive resampling at full amplitude.
        let input = [Float](repeating: 0.7, count: 1000)
        let output = SincResampler.resample(audio: input, from: 44100, to: 16000)
        XCTAssertGreaterThan(output.count, 300)

        // Interior samples (away from edges where the kernel truncates) should
        // be very close to 0.7.
        let interior = Array(output[50..<(output.count - 50)])
        let mean = interior.reduce(0.0) { $0 + Double($1) } / Double(interior.count)
        XCTAssertEqual(mean, 0.7, accuracy: 1e-3, "DC should be preserved to 0.1% at interior")
    }

    func testSineWavePreservesFrequency() {
        // Generate a 1 kHz sine at 48 kHz source rate, resample to 16 kHz,
        // then verify the dominant frequency is still near 1 kHz.
        let srcRate = 48000
        let dstRate = 16000
        let freq: Double = 1000
        let duration: Double = 0.1  // 100 ms
        let srcCount = Int(Double(srcRate) * duration)

        var input = [Float](repeating: 0, count: srcCount)
        for i in 0 ..< srcCount {
            let t = Double(i) / Double(srcRate)
            input[i] = Float(sin(2 * .pi * freq * t))
        }

        let output = SincResampler.resample(audio: input, from: srcRate, to: dstRate)

        // Count zero crossings in the output (ignoring first/last 10% for edge effects).
        // A 1 kHz sine has 2 crossings per cycle. Over 80 ms (middle 80% of 100 ms):
        // expected crossings ≈ 2 × 1000 × 0.08 = 160.
        let guardN = output.count / 10
        var crossings = 0
        for i in (guardN + 1) ..< (output.count - guardN) {
            if (output[i - 1] >= 0) != (output[i] >= 0) { crossings += 1 }
        }
        // Allow ±3 crossings for edge effects / rounding.
        XCTAssertEqual(crossings, 160, accuracy: 3,
                       "1 kHz sine should yield ~160 zero crossings in middle 80 ms")
    }

    // MARK: - Anti-aliasing (downsampling)

    func testHighFrequencyAttenuatedOnDownsample() {
        // Put energy above the destination Nyquist (e.g., 10 kHz content when
        // downsampling 44.1 kHz → 16 kHz, destination Nyquist is 8 kHz).
        // A correct anti-aliasing resampler should reject this, producing
        // near-silence. A naive resampler would alias it down into the audible
        // range at full amplitude.
        let srcRate = 44100
        let dstRate = 16000
        let freq: Double = 10000  // Above dst Nyquist of 8 kHz
        let duration: Double = 0.1
        let srcCount = Int(Double(srcRate) * duration)

        var input = [Float](repeating: 0, count: srcCount)
        for i in 0 ..< srcCount {
            let t = Double(i) / Double(srcRate)
            input[i] = Float(sin(2 * .pi * freq * t))
        }

        let output = SincResampler.resample(audio: input, from: srcRate, to: dstRate)

        // Interior RMS should be much lower than input RMS (which is ~0.707 for a sine).
        let guardN = output.count / 10
        let interior = Array(output[guardN..<(output.count - guardN)])
        let rms = sqrt(interior.reduce(0.0) { $0 + Double($1 * $1) } / Double(interior.count))
        XCTAssertLessThan(rms, 0.1,
                          "Frequencies above destination Nyquist should be attenuated (got RMS \(rms))")
    }
}
