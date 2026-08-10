# autograd_test.j - gradient checks for the engine microGPT runs on. Run with:
#
#     jennifer test autograd_test.j
#
# The overlay splices autograd.j in front, so the engine (leaf / add / mul / powc /
# logOp / ...) is reachable by bare name. Each test resets the global tape first.

use testing;
use math;

# d(a*b)/da = b, d(a*b)/db = a
func testMulGrad() {
    resetTo(0);
    def ai as int init leaf(3.0);
    def bi as int init leaf(4.0);
    def p as int init mul($ai, $bi);
    backward($p);
    testing.assertEqual(valueOf($p), 12.0);
    testing.assertEqual(gradOf($ai), 4.0);
    testing.assertEqual(gradOf($bi), 3.0);
}

# The property the arena design exists to preserve: a node used on several paths
# accumulates gradient from all of them. f = a*a -> df/da = 2a.
func testReusedNodeAccumulates() {
    resetTo(0);
    def ai as int init leaf(5.0);
    def f as int init mul($ai, $ai);
    backward($f);
    testing.assertEqual(valueOf($f), 25.0);
    testing.assertEqual(gradOf($ai), 10.0);
}

# Scalar-constant ops: addc passes gradient through (1), mulc scales it by c.
func testConstOps() {
    resetTo(0);
    def ai as int init leaf(2.0);
    def s as int init addc($ai, 10.0);   # a + 10
    def m as int init mulc($ai, 3.0);    # a * 3
    backward($s);
    testing.assertEqual(valueOf($s), 12.0);
    testing.assertEqual(gradOf($ai), 1.0);
    backward($m);
    testing.assertEqual(valueOf($m), 6.0);
    testing.assertEqual(gradOf($ai), 3.0);
}

# powc: f = x^3, df/dx = 3x^2.
func testPowGrad() {
    resetTo(0);
    def xi as int init leaf(2.0);
    def f as int init powc($xi, 3.0);
    backward($f);
    testing.assertEqual(valueOf($f), 8.0);
    testing.assertEqual(gradOf($xi), 3.0 * math.pow(2.0, 2.0));
}

# logOp: f = ln(x), df/dx = 1/x.
func testLogGrad() {
    resetTo(0);
    def xi as int init leaf(4.0);
    def f as int init logOp($xi);
    backward($f);
    testing.assertEqual(valueOf($f), math.ln(4.0));
    testing.assertEqual(gradOf($xi), 1.0 / 4.0);
}

# expOp: f = exp(x), df/dx = exp(x).
func testExpGrad() {
    resetTo(0);
    def xi as int init leaf(1.0);
    def f as int init expOp($xi);
    backward($f);
    testing.assertEqual(gradOf($xi), math.exp(1.0));
}

# reluOp: gradient is 1 where the input is positive, 0 where it is not.
func testReluGrad() {
    resetTo(0);
    def pos as int init leaf(3.0);
    def fp as int init reluOp($pos);
    backward($fp);
    testing.assertEqual(valueOf($fp), 3.0);
    testing.assertEqual(gradOf($pos), 1.0);

    resetTo(0);
    def neg as int init leaf(-2.0);
    def fn as int init reluOp($neg);
    backward($fn);
    testing.assertEqual(valueOf($fn), 0.0);
    testing.assertEqual(gradOf($neg), 0.0);
}

# A small composite exercising the chain rule end to end:
# f = ln(exp(a) * b) with a=0, b=5  ->  f = ln(5); df/da = 1, df/db = 1/5.
func testChainRule() {
    resetTo(0);
    def ai as int init leaf(0.0);
    def bi as int init leaf(5.0);
    def f as int init logOp(mul(expOp($ai), $bi));
    backward($f);
    testing.assertEqual(valueOf($f), math.ln(5.0));
    testing.assertEqual(gradOf($ai), 1.0);
    testing.assertEqual(gradOf($bi), 1.0 / 5.0);
}
