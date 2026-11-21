# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (C) 2026 mplx <jennifer@mplx.dev>
# pragma-jennifer-version: >=0.27.0

/**
 * The project's own tests - gradient checks against hand-derived derivatives.
 *
 * The overlay splices autograd.j in front, so the engine (leaf / add / mul / ...)
 * is reachable by bare name. Each test calls `resetTo(0)` first, because the tape
 * is a single global arena shared by every test in the file.
 *
 * Run with `jennifer test autograd_test.j`.
 *
 * @module autograd_test
 * @author mplx <jennifer@mplx.dev>
 * @license MIT
 * @see autograd.j
 */

use testing;

/**
 * Checks the product rule: d(a*b)/da = b and d(a*b)/db = a.
 *
 * @return {null}
 */
func testMulGrad() {
    resetTo(0);
    def ai as int init leaf(3.0);
    def bi as int init leaf(4.0);
    def pi as int init mul($ai, $bi);
    backward($pi);
    testing.assertEqual(valueOf($pi), 12.0);
    testing.assertEqual(gradOf($ai), 4.0);
    testing.assertEqual(gradOf($bi), 3.0);
}

/**
 * Checks that a value used twice accumulates gradient from both paths.
 *
 * With `f = a*a`, df/da = 2a - the case that needs `accum` to add rather than
 * assign.
 *
 * @return {null}
 */
func testReusedNodeAccumulates() {
    resetTo(0);
    def ai as int init leaf(5.0);
    def f as int init mul($ai, $ai);
    backward($f);
    testing.assertEqual(valueOf($f), 25.0);
    testing.assertEqual(gradOf($ai), 10.0); # 2 * 5
}

/**
 * Checks that add / sub pass gradient through as +1 / -1 per operand.
 *
 * `(a+b) - b` cancels, so b's two contributions must sum to zero.
 *
 * @return {null}
 */
func testAddSubGrad() {
    resetTo(0);
    def ai as int init leaf(2.0);
    def bi as int init leaf(7.0);
    def s as int init sub(add($ai, $bi), $bi); # (a+b) - b = a
    backward($s);
    testing.assertEqual(valueOf($s), 2.0);
    testing.assertEqual(gradOf($ai), 1.0);
    testing.assertEqual(gradOf($bi), 0.0); # +1 from add, -1 from sub -> 0
}

/**
 * Checks tanh backprop: d/dx tanh(x) = 1 - tanh(x)^2.
 *
 * @return {null}
 */
func testTanhGrad() {
    resetTo(0);
    def xi as int init leaf(0.5);
    def yi as int init tanhOp($xi);
    backward($yi);
    def t as float init valueOf($yi);
    testing.assertEqual(gradOf($xi), 1.0 - $t * $t);
}

/**
 * Checks the power rule: d/dx x^3 = 3x^2.
 *
 * @return {null}
 */
func testPowGrad() {
    resetTo(0);
    def xi as int init leaf(2.0);
    def yi as int init powc($xi, 3.0);
    backward($yi);
    testing.assertEqual(valueOf($yi), 8.0);
    testing.assertEqual(gradOf($xi), 12.0); # 3 * 2^2
}

/**
 * Checks that a negative exponent differentiates correctly: d/dx x^-1 = -1/x^2.
 *
 * @return {null}
 */
func testPowNegativeExponent() {
    resetTo(0);
    def xi as int init leaf(4.0);
    def yi as int init powc($xi, -1.0);
    backward($yi);
    testing.assertEqual(valueOf($yi), 0.25);
    testing.assertEqual(gradOf($xi), -0.0625); # -1 / 4^2
}

/**
 * Checks logarithm backprop: d/dx ln(x) = 1/x.
 *
 * @return {null}
 */
func testLogGrad() {
    resetTo(0);
    def xi as int init leaf(4.0);
    def yi as int init logOp($xi);
    backward($yi);
    testing.assertEqual(gradOf($xi), 0.25);
}

/**
 * Checks exponential backprop: d/dx exp(x) = exp(x).
 *
 * @return {null}
 */
func testExpGrad() {
    resetTo(0);
    def xi as int init leaf(1.5);
    def yi as int init expOp($xi);
    backward($yi);
    testing.assertEqual(gradOf($xi), valueOf($yi));
}

/**
 * Checks that `ln` and `exp` invert each other, forward and backward.
 *
 * `ln(exp(x)) == x`, so the gradient through the pair must come back as 1. Both
 * checks carry a tolerance: a transcendental round trip is not required to land
 * on the exact bit pattern, and `testing` has no approximate assertion.
 *
 * @return {null}
 */
func testExpLogRoundTrip() {
    resetTo(0);
    def xi as int init leaf(0.75);
    def yi as int init logOp(expOp($xi));
    backward($yi);
    testing.assertTrue(math.abs(valueOf($yi) - 0.75) < 1.0e-12);
    testing.assertTrue(math.abs(gradOf($xi) - 1.0) < 1.0e-12);
}

/**
 * Checks the rectifier on both sides of its kink.
 *
 * Active (x > 0) passes the gradient through unchanged; inactive (x < 0) blocks
 * it, and the node's own value is 0.
 *
 * @return {null}
 */
func testReluGrad() {
    resetTo(0);
    def pos as int init leaf(3.0);
    def rp as int init reluOp($pos);
    backward($rp);
    testing.assertEqual(valueOf($rp), 3.0);
    testing.assertEqual(gradOf($pos), 1.0);

    resetTo(0);
    def neg as int init leaf(-3.0);
    def rn as int init reluOp($neg);
    backward($rn);
    testing.assertEqual(valueOf($rn), 0.0);
    testing.assertEqual(gradOf($neg), 0.0);
}

/**
 * Checks that relu blocks the gradient at exactly 0, its undefined point.
 *
 * The convention here matches micrograd's `(out.data > 0) * out.grad`.
 *
 * @return {null}
 */
func testReluAtZero() {
    resetTo(0);
    def zero as int init leaf(0.0);
    def r as int init reluOp($zero);
    backward($r);
    testing.assertEqual(valueOf($r), 0.0);
    testing.assertEqual(gradOf($zero), 0.0);
}

/**
 * Checks the new ops composed with the old ones, through a shared node.
 *
 * `f = relu(x) * ln(x)` at x = e reaches x by two paths, so the gradients must
 * add: `d/dx = ln(x) + relu(x)/x`, which at x = e is `1 + 1 = 2`.
 *
 * @return {null}
 */
func testMixedOpsAccumulate() {
    resetTo(0);
    def xi as int init leaf(math.E);
    def f as int init mul(reluOp($xi), logOp($xi));
    backward($f);
    testing.assertTrue(math.abs(valueOf($f) - math.E) < 1.0e-12);
    testing.assertTrue(math.abs(gradOf($xi) - 2.0) < 1.0e-12);
}

/**
 * Checks that `addc` matches `add` against a constant leaf, value and gradient.
 *
 * @return {null}
 */
func testAddcMatchesAdd() {
    resetTo(0);
    def a as int init leaf(3.0);
    def folded as int init addc($a, 2.5);
    backward($folded);
    testing.assertEqual(valueOf($folded), 5.5);
    testing.assertEqual(gradOf($a), 1.0);

    resetTo(0);
    def b as int init leaf(3.0);
    def viaLeaf as int init add($b, leaf(2.5));
    backward($viaLeaf);
    testing.assertEqual(valueOf($viaLeaf), 5.5);
    testing.assertEqual(gradOf($b), 1.0);
}

/**
 * Checks that `mulc` matches `mul` against a constant leaf, value and gradient.
 *
 * @return {null}
 */
func testMulcMatchesMul() {
    resetTo(0);
    def a as int init leaf(3.0);
    def folded as int init mulc($a, -2.0);
    backward($folded);
    testing.assertEqual(valueOf($folded), -6.0);
    testing.assertEqual(gradOf($a), -2.0);

    resetTo(0);
    def b as int init leaf(3.0);
    def viaLeaf as int init mul($b, leaf(-2.0));
    backward($viaLeaf);
    testing.assertEqual(valueOf($viaLeaf), -6.0);
    testing.assertEqual(gradOf($b), -2.0);
}

/**
 * Checks that a folded node really is one node rather than two.
 *
 * The whole point of `mulc` is the node it does not allocate, so the count is
 * worth asserting - a refactor that quietly reintroduced the leaf would still
 * compute the right answer.
 *
 * @return {null}
 */
func testFoldingAllocatesOneNode() {
    resetTo(0);
    def a as int init leaf(3.0);
    def folded as int init mulc($a, 2.0);
    testing.assertEqual(valueOf($folded), 6.0);
    testing.assertEqual(leaf(0.0), 2); # leaf + mulc == 2 nodes so far
    
    resetTo(0);
    def b as int init leaf(3.0);
    def viaLeaf as int init mul($b, leaf(2.0));
    testing.assertEqual(valueOf($viaLeaf), 6.0);
    testing.assertEqual(leaf(0.0), 3); # leaf + leaf + mul == 3
}

/**
 * Checks the invariant that lets `backward` skip its zeroing pass.
 *
 * A fresh node is born with a zero gradient, and `resetTo` zeroes the prefix it
 * keeps. If either stopped holding, `backward` would accumulate onto stale values
 * and every gradient after the first step would be wrong.
 *
 * @return {null}
 */
func testGradientsStartZero() {
    resetTo(0);
    def a as int init leaf(3.0);
    def b as int init leaf(4.0);
    def p as int init mul($a, $b);
    testing.assertEqual(gradOf($a), 0.0);
    testing.assertEqual(gradOf($p), 0.0);

    backward($p);
    testing.assertEqual(gradOf($a), 4.0);

    resetTo(2); # keep both leaves, drop the product
    testing.assertEqual(gradOf($a), 0.0);
    testing.assertEqual(gradOf($b), 0.0);
}

/**
 * Checks that a second `backward` over one graph accumulates unless `zeroGrads`
 * is called first.
 *
 * This is the cost of dropping the zeroing pass, and it is worth pinning down so
 * the behaviour is documented by a test rather than discovered.
 *
 * @return {null}
 */
func testSecondBackwardNeedsZeroGrads() {
    resetTo(0);
    def a as int init leaf(3.0);
    def b as int init leaf(4.0);
    def p as int init mul($a, $b);

    backward($p);
    testing.assertEqual(gradOf($a), 4.0);

    backward($p); # without zeroing, contributions add up
    testing.assertEqual(gradOf($a), 8.0);

    zeroGrads();
    backward($p);
    testing.assertEqual(gradOf($a), 4.0);
}
