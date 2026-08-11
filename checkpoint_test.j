# checkpoint_test.j - round-trip tests for both checkpoint formats. Run with:
#
#     jennifer test checkpoint_test.j
#
# The overlay splices checkpoint.j (and via it model.j + autograd.j) in front, so
# the model and checkpoint functions are reachable by bare name. Each test builds a
# tiny random model, saves it, reloads it, and asserts the parameters come back
# bit-for-bit - the property that makes a checkpoint reusable.

use testing;
use math;
use fs;
use strings;

# assertParamsEqual asserts two parameter vectors are element-wise identical.
func assertParamsEqual(a as list of float, b as list of float) {
    testing.assertEqual(len($a), len($b));
    def i as int init 0;
    while ($i < len($a)) {
        testing.assertEqual($a[$i], $b[$i]);
        $i = $i + 1;
    }
}

# JSON: save then load reproduces every parameter exactly.
func testJsonRoundTrip() {
    math.randSeed(7);
    setVocab("abcde");
    buildAllMatrices();
    def before as list of float init collectParams();
    saveJson("roundtrip.tmp.json");
    loadJson("roundtrip.tmp.json");
    assertParamsEqual($before, collectParams());
    fs.remove("roundtrip.tmp.json");
}

# FS (flat text): same exact round-trip through the plain-text format.
func testFsRoundTrip() {
    math.randSeed(9);
    setVocab("xyz");
    buildAllMatrices();
    def before as list of float init collectParams();
    saveFs("roundtrip.tmp.ckpt");
    loadFs("roundtrip.tmp.ckpt");
    assertParamsEqual($before, collectParams());
    fs.remove("roundtrip.tmp.ckpt");
}

# The tokenizer is part of the checkpoint: loading restores the exact vocabulary,
# even after it has been clobbered in between.
func testVocabPreserved() {
    math.randSeed(3);
    setVocab("abc");
    buildAllMatrices();
    saveJson("vocab.tmp.json");
    setVocab("zzzz"); # clobber the live vocabulary
    loadJson("vocab.tmp.json");
    testing.assertEqual(strings.join($uchars, ""), "abc");
    testing.assertEqual($vocabSize, 4);
    fs.remove("vocab.tmp.json");
}

# The two formats agree: JSON and FS load the same model from the same weights.
func testFormatsAgree() {
    math.randSeed(5);
    setVocab("abcdef");
    buildAllMatrices();
    def before as list of float init collectParams();
    saveJson("agree.tmp.json");
    saveFs("agree.tmp.ckpt");
    loadJson("agree.tmp.json");
    def viaJson as list of float init collectParams();
    loadFs("agree.tmp.ckpt");
    def viaFs as list of float init collectParams();
    assertParamsEqual($viaJson, $viaFs);
    assertParamsEqual($before, $viaFs);
    fs.remove("agree.tmp.json");
    fs.remove("agree.tmp.ckpt");
}
