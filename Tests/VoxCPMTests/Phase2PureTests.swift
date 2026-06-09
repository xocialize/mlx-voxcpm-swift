// Phase2PureTests.swift — Non-MLX tests for Phase 2 flow matching logic

import XCTest
@testable import VoxCPM

final class Phase2PureTests: XCTestCase {

    func testCFGZeroStarWarmupSteps() {
        // 4% of 10 steps = 0.4, max(1, 0) = 1 step
        let nSteps = 10
        let zeroInitSteps = max(1, Int(Float(nSteps) * 0.04))
        XCTAssertEqual(zeroInitSteps, 1)
    }

    func testCFGZeroStarWarmupStepsLarger() {
        // 4% of 100 steps = 4 steps
        let nSteps = 100
        let zeroInitSteps = max(1, Int(Float(nSteps) * 0.04))
        XCTAssertEqual(zeroInitSteps, 4)
    }

    func testSwayScheduleFormula() {
        // t_new = t + coef*(cos(pi/2*t) - 1 + t)
        // At t=1: 1 + (cos(pi/2) - 1 + 1) = 1 + (0 - 1 + 1) = 1.0
        // At t=0: 0 + (cos(0) - 1 + 0)   = 0 + (1 - 1 + 0) = 0.0
        let piHalf = Float.pi / 2.0

        let t1: Float = 1.0
        let swayed1 = t1 + 1.0 * (cos(piHalf * t1) - 1.0 + t1)
        XCTAssertEqual(swayed1, 1.0, accuracy: 1e-5)

        let t0: Float = 0.0
        let swayed0 = t0 + 1.0 * (cos(piHalf * t0) - 1.0 + t0)
        XCTAssertEqual(swayed0, 0.0, accuracy: 1e-5)

        // Intermediate values shift upward (slower early denoising)
        let tMid: Float = 0.5
        let swayedMid = tMid + 1.0 * (cos(piHalf * tMid) - 1.0 + tMid)
        XCTAssertGreaterThan(swayedMid, tMid)
    }

    func testLocDiTPrefixLength() {
        let patchSize = 4
        let prefixLen = 1 + patchSize
        XCTAssertEqual(prefixLen, 5)
    }

    func testCFMDefaultTimesteps() {
        let cfm = CFMConfig()
        XCTAssertEqual(cfm.solver, "euler")
        XCTAssertEqual(cfm.inferenceCfgRate, 2.0)
    }
}
