// UnifiedCFM.swift — Euler ODE solver with sway sampling + CFG-zero-star
// Ports: dit.py:97-197
// IMPORTANT: This matches the Python MLX port convention (1→0, subtraction)
// which is what the bf16 weights were converted for.

import Foundation
import MLX
import MLXRandom
import MLXNN

/// Conditional Flow Matching sampler with Euler integration.
///
/// Convention (matching Python MLX port / VoxCPM bf16 weights):
/// - Time goes from t=1 (noise) to t=0 (clean data)
/// - Euler step: z_{t-dt} = z_t - dt * v(z_t, t)
/// - dt is positive (t decreases each step)
public class UnifiedCFM: Module, @unchecked Sendable {
    let inChannels: Int
    let estimator: VoxCPMLocDiT
    let cfmParams: CFMConfig

    public init(inChannels: Int, cfmParams: CFMConfig, estimator: VoxCPMLocDiT) {
        self.inChannels = inChannels
        self.estimator = estimator
        self.cfmParams = cfmParams
    }

    /// Euler ODE integration matching Python MLX port exactly.
    public func solveEuler(
        x: MLXArray,
        tSpan: MLXArray,
        mu: MLXArray,
        cond: MLXArray,
        cfgValue: Float = 1.0,
        useCfgZeroStar: Bool = true
    ) -> MLXArray {
        let nSteps = tSpan.dim(0) - 1
        var currentX = x

        // Extract timesteps as floats
        eval(tSpan)
        var tValues: [Float] = []
        for i in 0 ... nSteps {
            tValues.append(tSpan[i].item(Float.self))
        }

        // t_span goes 1→0, dt = t_span[0] - t_span[1] (positive)
        var tCurrent = tValues[0]
        var dtCurrent = tValues[0] - tValues[1]

        // Official uses len(t_span) = nSteps+1; match for exact parity
        let zeroInitSteps = max(1, Int(Float(nSteps + 1) * 0.04))

        for step in 1 ... nSteps {
            let dphiDt: MLXArray

            if useCfgZeroStar && step <= zeroInitSteps {
                dphiDt = MLXArray.zeros(like: currentX)
            } else {
                let b = currentX.dim(0)
                let batchSize = 2 * b

                let xIn = MLX.concatenated([currentX, currentX], axis: 0)
                let muIn = MLX.concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)

                let tVal = MLXArray(Array(repeating: tCurrent, count: batchSize))
                // mean_mode=False → dt_in = zeros
                let dtVal = MLXArray.zeros([batchSize])

                let condIn = MLX.concatenated([cond, cond], axis: 0)

                let out = estimator(xIn, mu: muIn, t: tVal, cond: condIn, dt: dtVal)

                let condResult = out[..<b]
                let uncondResult = out[b...]

                let cfg = MLXArray(cfgValue)
                if useCfgZeroStar {
                    let posFlat = condResult.reshaped([b, -1])
                    let negFlat = uncondResult.reshaped([b, -1])

                    let posMulNeg = MLX.multiply(posFlat, negFlat)
                    let negMulNeg = MLX.multiply(negFlat, negFlat)
                    let dotProd = MLX.sum(posMulNeg, axis: 1, keepDims: true)
                    let sqNorm = MLX.sum(negMulNeg, axis: 1, keepDims: true) + 1e-8
                    let stStar = MLX.divide(dotProd, sqNorm).reshaped([b, 1, 1])

                    let uncondScaled = MLX.multiply(uncondResult, stStar)
                    dphiDt = uncondScaled + MLX.multiply(cfg, condResult - uncondScaled)
                } else {
                    dphiDt = uncondResult + MLX.multiply(cfg, condResult - uncondResult)
                }
            }

            // Euler step: x = x - dt * v (SUBTRACTION — official VoxCPM2 convention)
            // Time goes 1→0, dt is positive, subtraction walks from noise to data
            currentX = currentX - dtCurrent * dphiDt
            tCurrent = tCurrent - dtCurrent
            eval(currentX)

            // Update dt for next step
            if step < nSteps {
                dtCurrent = tCurrent - tValues[step + 1]
            }
        }

        return currentX
    }

    /// Sample a latent patch.
    public func sample(
        mu: MLXArray,
        nTimesteps: Int,
        patchSize: Int,
        cond: MLXArray,
        temperature: Float = 1.0,
        cfgValue: Float = 1.0
    ) -> MLXArray {
        let B = mu.dim(0)

        // Initialize noise in float32 — bf16 precision causes audio glitches on Metal (PR #263)
        let z = MLXRandom.normal([B, inChannels, patchSize]).asType(.float32) * temperature

        // Timestep schedule: linspace 1→0 (official VoxCPM2: t=1 is noise, t=0 is clean)
        var tSpan = MLXArray.linspace(Float32(1), Float32(0), count: nTimesteps + 1)
            .asType(.float32)

        // Sway sampling
        let piHalf = MLXArray(Float32.pi / 2.0)
        tSpan = tSpan + (MLX.cos(MLX.multiply(piHalf, tSpan)) - 1.0 + tSpan)
        eval(tSpan)

        return solveEuler(
            x: z, tSpan: tSpan, mu: mu, cond: cond,
            cfgValue: cfgValue, useCfgZeroStar: true
        )
    }
}
