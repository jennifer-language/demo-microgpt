# demo-microgpt - the model: a char-level GPT-2-shape transformer on the scalar
# autograd engine (autograd.j). Shared by the trainer (microgpt.j) and the
# sampler (sample.j), so a checkpoint saved by one loads into the other.
#
# The learnable parameters are the FIRST `numParams` leaf nodes on the tape, in a
# fixed creation order (wte, wpe, lm_head, then per-layer). buildAllMatrices()
# rebuilds them from a param SOURCE - random Gaussians when training, or the saved
# floats when loading a checkpoint - which is what makes save / load exact.
#
# jennifer: >=0.24.0

include "autograd.j";
use math;
use lists;
use strings;
use convert;

# --- model shape (fixed; a checkpoint records these and load verifies them) ---
def const NLAYER as int init 1;               # transformer depth
def const NEMBD as int init 8;                # embedding width
def const NHEAD as int init 2;                # attention heads
def const HEADDIM as int init NEMBD // NHEAD; # per-head dimension
def const BLOCKSIZE as int init 16;           # max context / attention window

# --- tokenizer + parameter state (set at runtime) -----------------------------
def uchars as list of string init [];         # token id -> character
def bos as int init 0;                        # Beginning-Of-Sequence token id
def vocabSize as int init 0;
def stateDict as map of string to list of list of int init {};
def numParams as int init 0;                  # params are tape[0 .. numParams-1]
def keysCache as list of list of list of int init [];
def valuesCache as list of list of list of int init [];

# --- parameter source: random when training, replayed floats when loading -----
def loadMode as bool init false;
def loadFloats as list of float init [];
def loadCursor as int init 0;

# gauss draws from N(mean, std^2) via Box-Muller (math has sin/cos).
func gauss(mean as float, std as float) {
    def u1 as float init math.rand();
    def u2 as float init math.rand();
    if ($u1 < 1e-12) {
        $u1 = 1e-12;
    }
    def z as float init math.sqrt(-2.0 * math.ln($u1)) * math.cos(2.0 * math.PI * $u2);
    return $mean + $std * $z;
}

# nextParam yields the next parameter value: a saved float in load mode, else a
# fresh Gaussian. This is the single seam that makes buildAllMatrices reusable for
# both random init and checkpoint load.
func nextParam() {
    if ($loadMode) {
        def x as float init $loadFloats[$loadCursor];
        $loadCursor = $loadCursor + 1;
        return $x;
    }
    return gauss(0.0, 0.08);
}

# buildMatrix returns an nout x nin grid of fresh leaf-parameter node indices,
# each seeded from nextParam().
func buildMatrix(nout as int, nin as int) {
    def m as list of list of int init [];
    def r as int init 0;
    while ($r < $nout) {
        def row as list of int init [];
        def c as int init 0;
        while ($c < $nin) {
            $row[] = leaf(nextParam());
            $c = $c + 1;
        }
        $m[] = $row;
        $r = $r + 1;
    }
    return $m;
}

# layerKey builds the per-layer parameter name, e.g. "layer0.attn_wq".
func layerKey(li as int, name as string) {
    return "layer" + convert.toString($li) + "." + $name;
}

# buildAllMatrices (re)creates every parameter leaf node in a fixed order and
# records numParams. Called once after the vocab is known (training) or after a
# checkpoint's floats are staged in loadFloats (loading).
func buildAllMatrices() {
    resetTo(0);
    $stateDict = {};
    $stateDict["wte"] = buildMatrix($vocabSize, NEMBD);
    $stateDict["wpe"] = buildMatrix(BLOCKSIZE, NEMBD);
    $stateDict["lm_head"] = buildMatrix($vocabSize, NEMBD);
    def li as int init 0;
    while ($li < NLAYER) {
        $stateDict[layerKey($li, "attn_wq")] = buildMatrix(NEMBD, NEMBD);
        $stateDict[layerKey($li, "attn_wk")] = buildMatrix(NEMBD, NEMBD);
        $stateDict[layerKey($li, "attn_wv")] = buildMatrix(NEMBD, NEMBD);
        $stateDict[layerKey($li, "attn_wo")] = buildMatrix(NEMBD, NEMBD);
        $stateDict[layerKey($li, "mlp_fc1")] = buildMatrix(4 * NEMBD, NEMBD);
        $stateDict[layerKey($li, "mlp_fc2")] = buildMatrix(NEMBD, 4 * NEMBD);
        $li = $li + 1;
    }
    $numParams = len($tape);
}

# --- tokenizer ----------------------------------------------------------------

# setVocab installs a vocabulary from a string of its characters (used by both
# buildVocab and checkpoint load, so the two agree on the char -> id mapping).
func setVocab(vocab as string) {
    $uchars = strings.chars($vocab);
    $bos = len($uchars);
    $vocabSize = len($uchars) + 1;
}

# buildVocab derives the sorted unique-character vocabulary from the documents.
func buildVocab(docs as list of string) {
    def chars as list of string init [];
    for (def doc in $docs) {
        for (def ch in strings.chars($doc)) {
            if (not lists.contains($chars, $ch)) {
                $chars[] = $ch;
            }
        }
    }
    setVocab(strings.join(lists.sort($chars), ""));
}

# charId maps a character to its token id (a small linear scan; no lists.indexOf).
func charId(ch as string) {
    def i as int init 0;
    while ($i < len($uchars)) {
        if ($uchars[$i] == $ch) {
            return $i;
        }
        $i = $i + 1;
    }
    return -1;
}

# tokenize surrounds a document's char ids with BOS on both sides.
func tokenize(doc as string) {
    def toks as list of int init [$bos];
    for (def ch in strings.chars($doc)) {
        $toks[] = charId($ch);
    }
    $toks[] = $bos;
    return $toks;
}

# --- KV cache -----------------------------------------------------------------

func resetKV() {
    $keysCache = [];
    $valuesCache = [];
    def li as int init 0;
    while ($li < NLAYER) {
        def ek as list of list of int init [];
        $keysCache[] = $ek;
        def ev as list of list of int init [];
        $valuesCache[] = $ev;
        $li = $li + 1;
    }
}

# appendKV pushes the current step's key / value vectors into layer li's cache
# (read-modify-write, since a chained `$keysCache[$li][]` append is not supported).
func appendKV(li as int, k as list of int, v as list of int) {
    def kl as list of list of int init $keysCache[$li];
    $kl[] = $k;
    $keysCache[$li] = $kl;
    def vl as list of list of int init $valuesCache[$li];
    $vl[] = $v;
    $valuesCache[$li] = $vl;
}

# --- model primitives (all return node indices) -------------------------------

# linear: matrix-vector product, out[o] = sum_k w[o][k] * x[k].
func linear(x as list of int, w as list of list of int) {
    def out as list of int init [];
    for (def wo in $w) {
        def acc as int init mul($wo[0], $x[0]);
        def k as int init 1;
        while ($k < len($x)) {
            $acc = add($acc, mul($wo[$k], $x[$k]));
            $k = $k + 1;
        }
        $out[] = $acc;
    }
    return $out;
}

# softmaxNodes: numerically-stable softmax over a list of logit nodes.
func softmaxNodes(logits as list of int) {
    def maxv as float init valueOf($logits[0]);
    for (def l in $logits) {
        if (valueOf($l) > $maxv) {
            $maxv = valueOf($l);
        }
    }
    def exps as list of int init [];
    for (def l in $logits) {
        $exps[] = expOp(addc($l, -$maxv));
    }
    def total as int init $exps[0];
    def k as int init 1;
    while ($k < len($exps)) {
        $total = add($total, $exps[$k]);
        $k = $k + 1;
    }
    def tinv as int init powc($total, -1.0);
    def out as list of int init [];
    for (def e in $exps) {
        $out[] = mul($e, $tinv);
    }
    return $out;
}

# rmsnorm: root-mean-square normalization (GPT-2's layernorm, biasless variant).
func rmsnorm(x as list of int) {
    def ss as int init mul($x[0], $x[0]);
    def k as int init 1;
    while ($k < len($x)) {
        $ss = add($ss, mul($x[$k], $x[$k]));
        $k = $k + 1;
    }
    def ms as int init mulc($ss, 1.0 / convert.toFloat(len($x)));
    def scale as int init powc(addc($ms, 1e-5), -0.5);
    def out as list of int init [];
    for (def xi in $x) {
        $out[] = mul($xi, $scale);
    }
    return $out;
}

# gpt forwards one token at position posId, extending the KV cache, and returns
# the logits over the next token.
func gpt(tokenId as int, posId as int) {
    def tokEmb as list of int init $stateDict["wte"][$tokenId];
    def posEmb as list of int init $stateDict["wpe"][$posId];
    def x as list of int init [];
    def m as int init 0;
    while ($m < NEMBD) {
        $x[] = add($tokEmb[$m], $posEmb[$m]);
        $m = $m + 1;
    }
    $x = rmsnorm($x);

    def li as int init 0;
    while ($li < NLAYER) {
        # 1) multi-head causal self-attention
        def xRes as list of int init $x;
        def xn as list of int init rmsnorm($x);
        def q as list of int init linear($xn, $stateDict[layerKey($li, "attn_wq")]);
        def k as list of int init linear($xn, $stateDict[layerKey($li, "attn_wk")]);
        def v as list of int init linear($xn, $stateDict[layerKey($li, "attn_wv")]);
        appendKV($li, $k, $v);
        def steps as int init len($keysCache[$li]);
        def invSqrt as float init 1.0 / math.sqrt(convert.toFloat(HEADDIM));
        def xAttn as list of int init [];
        def h as int init 0;
        while ($h < NHEAD) {
            def hs as int init $h * HEADDIM;
            def attnLogits as list of int init [];
            def t as int init 0;
            while ($t < $steps) {
                def dotacc as int init mul($q[$hs], $keysCache[$li][$t][$hs]);
                def j as int init 1;
                while ($j < HEADDIM) {
                    $dotacc = add($dotacc, mul($q[$hs + $j], $keysCache[$li][$t][$hs + $j]));
                    $j = $j + 1;
                }
                $attnLogits[] = mulc($dotacc, $invSqrt);
                $t = $t + 1;
            }
            def weights as list of int init softmaxNodes($attnLogits);
            def jj as int init 0;
            while ($jj < HEADDIM) {
                def acc as int init mul($weights[0], $valuesCache[$li][0][$hs + $jj]);
                def tt as int init 1;
                while ($tt < $steps) {
                    $acc = add($acc, mul($weights[$tt], $valuesCache[$li][$tt][$hs + $jj]));
                    $tt = $tt + 1;
                }
                $xAttn[] = $acc;
                $jj = $jj + 1;
            }
            $h = $h + 1;
        }
        def attnOut as list of int init linear($xAttn, $stateDict[layerKey($li, "attn_wo")]);
        def xa as list of int init [];
        def m2 as int init 0;
        while ($m2 < NEMBD) {
            $xa[] = add($attnOut[$m2], $xRes[$m2]);
            $m2 = $m2 + 1;
        }
        $x = $xa;

        # 2) MLP block
        def xRes2 as list of int init $x;
        def xn2 as list of int init rmsnorm($x);
        def hidden as list of int init linear($xn2, $stateDict[layerKey($li, "mlp_fc1")]);
        def activated as list of int init [];
        for (def hi in $hidden) {
            $activated[] = reluOp($hi);
        }
        def down as list of int init linear($activated, $stateDict[layerKey($li, "mlp_fc2")]);
        def xm as list of int init [];
        def m3 as int init 0;
        while ($m3 < NEMBD) {
            $xm[] = add($down[$m3], $xRes2[$m3]);
            $m3 = $m3 + 1;
        }
        $x = $xm;
        $li = $li + 1;
    }
    return linear($x, $stateDict["lm_head"]);
}

# sumNodes folds a list of nodes into a single sum node.
func sumNodes(xs as list of int) {
    def acc as int init $xs[0];
    def i as int init 1;
    while ($i < len($xs)) {
        $acc = add($acc, $xs[$i]);
        $i = $i + 1;
    }
    return $acc;
}

# sampleWeighted returns an index chosen in proportion to weights (random.choices).
func sampleWeighted(weights as list of float) {
    def total as float init 0.0;
    for (def w in $weights) {
        $total = $total + $w;
    }
    def r as float init math.rand() * $total;
    def cum as float init 0.0;
    def i as int init 0;
    while ($i < len($weights)) {
        $cum = $cum + $weights[$i];
        if ($r < $cum) {
            return $i;
        }
        $i = $i + 1;
    }
    return len($weights) - 1;
}

# generateName autoregressively samples one name at the given temperature.
func generateName(temperature as float) {
    resetTo($numParams);
    resetKV();
    def tokenId as int init $bos;
    def chars as list of string init [];
    def posId as int init 0;
    while ($posId < BLOCKSIZE) {
        def logits as list of int init gpt($tokenId, $posId);
        def scaled as list of int init [];
        for (def l in $logits) {
            $scaled[] = mulc($l, 1.0 / $temperature);
        }
        def probs as list of int init softmaxNodes($scaled);
        def weights as list of float init [];
        for (def p in $probs) {
            $weights[] = valueOf($p);
        }
        $tokenId = sampleWeighted($weights);
        if ($tokenId == $bos) {
            break;
        }
        $chars[] = $uchars[$tokenId];
        $posId = $posId + 1;
    }
    return strings.join($chars, "");
}
