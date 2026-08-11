# demo-microgpt - checkpoint save / load, in two formats.
#
# A trained model is fully captured by three things: the model SHAPE (a few ints),
# the TOKENIZER (the char vocabulary), and the WEIGHTS (the numParams parameter
# floats, read off the front of the tape). Everything else - the computation graph
# - is transient and rebuilt on every forward pass, so it is never saved.
#
# Two formats, same contents:
#   * JSON  - structured and portable (via the `json` library).
#   * FS    - a flat text file (shape line, vocab line, one float per line) written
#             with `fs`; trivially diff-able, no json dependency.
#
# Loading replays the saved floats through model.j's buildAllMatrices, so the
# parameter leaf nodes are recreated in the exact same order they were trained in.
#
# jennifer: >=0.24.0

include "./model.j";
use json;
use fs;
use strings;
use convert;

# The JSON shape. Structs encode straight to JSON objects; `params` becomes an
# array of numbers, `vocab` a string of the vocabulary characters.
def struct Checkpoint {
    nLayer as int,
    nEmbd as int,
    nHead as int,
    blockSize as int,
    vocab as string,
    params as list of float
};

# collectParams reads the numParams parameter values off the front of the tape.
func collectParams() {
    def ps as list of float init [];
    def i as int init 0;
    while ($i < $numParams) {
        $ps[] = valueOf($i);
        $i = $i + 1;
    }
    return $ps;
}

# checkShape rejects a checkpoint whose fixed model dimensions do not match this
# build (the vocabulary size may differ - it is driven by the saved vocab).
func checkShape(nl as int, ne as int, nh as int, bs as int) {
    if ($nl != NLAYER or $ne != NEMBD or $nh != NHEAD or $bs != BLOCKSIZE) {
        throw Error{
            kind: "checkpoint",
            message: "checkpoint shape does not match this model build",
            file: "", line: 0, col: 0
        };
    }
}

# stageAndBuild installs a vocabulary and replays the saved parameter floats,
# rebuilding every parameter leaf node in creation order. It verifies that exactly
# the right number of floats was consumed.
func stageAndBuild(vocab as string, ps as list of float) {
    setVocab($vocab);
    $loadMode = true;
    $loadFloats = $ps;
    $loadCursor = 0;
    buildAllMatrices();
    $loadMode = false;
    if ($loadCursor != len($ps)) {
        throw Error{
            kind: "checkpoint",
            message: "checkpoint parameter count does not match the model",
            file: "", line: 0, col: 0
        };
    }
}

# --- JSON format --------------------------------------------------------------

func saveJson(path as string) {
    def ck as Checkpoint init Checkpoint{
        nLayer: NLAYER, nEmbd: NEMBD, nHead: NHEAD, blockSize: BLOCKSIZE,
        vocab: strings.join($uchars, ""),
        params: collectParams()
    };
    fs.writeString($path, json.encodePretty($ck));
}

func loadJson(path as string) {
    def doc as json.Value init json.decode(fs.readString($path));
    checkShape(
        json.asInt($doc, "/nLayer"), json.asInt($doc, "/nEmbd"),
        json.asInt($doc, "/nHead"), json.asInt($doc, "/blockSize"));
    def ps as list of float init [];
    def np as int init json.length($doc, "/params");
    def i as int init 0;
    while ($i < $np) {
        $ps[] = json.asFloat($doc, "/params/" + convert.toString($i));
        $i = $i + 1;
    }
    stageAndBuild(json.asString($doc, "/vocab"), $ps);
}

# --- FS (flat text) format ----------------------------------------------------
# Layout:
#   line 1: nLayer nEmbd nHead blockSize numParams   (space-separated ints)
#   line 2: the vocabulary characters, concatenated
#   line 3+: one parameter float per line, in creation order

func saveFs(path as string) {
    def lines as list of string init [];
    $lines[] = convert.toString(NLAYER) + " " + convert.toString(NEMBD) + " "
        + convert.toString(NHEAD) + " " + convert.toString(BLOCKSIZE) + " "
        + convert.toString($numParams);
    $lines[] = strings.join($uchars, "");
    def i as int init 0;
    while ($i < $numParams) {
        $lines[] = convert.toString(valueOf($i));
        $i = $i + 1;
    }
    fs.writeString($path, strings.join($lines, "\n"));
}

func loadFs(path as string) {
    def lines as list of string init strings.split(fs.readString($path), "\n");
    def hdr as list of string init strings.split(strings.trim($lines[0]), " ");
    checkShape(convert.toInt($hdr[0]), convert.toInt($hdr[1]),
        convert.toInt($hdr[2]), convert.toInt($hdr[3]));
    def vocab as string init $lines[1];
    def ps as list of float init [];
    def i as int init 2;
    while ($i < len($lines)) {
        def t as string init strings.trim($lines[$i]);
        if (len($t) > 0) {
            $ps[] = convert.toFloat($t);
        }
        $i = $i + 1;
    }
    stageAndBuild($vocab, $ps);
}
