#!/usr/bin/env -S jennifer run
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (C) 2026 mplx <jennifer@mplx.dev>
# pragma-jennifer-version: >=0.27.0

# demo-microgpt - Karpathy's microGPT, ported to Jennifer, with data-parallel
# training and checkpointing.
#
# The model (model.j) runs on a scalar autograd engine (autograd.j). Jennifer has
# value semantics and no closures, so - unlike Python's shared-mutable `Value`
# graph - every scalar is an INDEX into one global `tape`, and backward is a
# reverse walk over it.
#
# CONCURRENCY: each training step processes a BATCH of documents *in parallel*.
# Because `spawn` deep-copies the enclosing scope, each worker gets its OWN private
# copy of the tape, computes the gradients for its document independently (no shared
# mutable state, no races - guaranteed by value semantics), and returns them. The
# main thread averages the batch's gradients and does one Adam update. That is
# textbook data-parallel SGD, and it falls out naturally from Jennifer's isolation
# model. BATCH defaults to the core count, so training uses the whole CPU.
#
# Usage:
#   jennifer run microgpt.j --steps 200 --batch 8 --save model.json

include "./checkpoint.j"; # brings model.j + autograd.j

use io;
use os;
use math;
use lists;
use strings;
use convert;
use fs;
use task;

import "args.j" as args;

# --- command line -------------------------------------------------------------
def parser as args.Parser init args.parser(
    "microgpt",
    "Train a char-level GPT on names and save a checkpoint");
$parser = args.intFlag($parser, "steps", "s", 100, "training steps (parameter updates)");
$parser = args.intFlag(
    $parser,
    "batch",
    "b",
    os.NCPU,
    "documents per step, run in parallel (default: core count)");
$parser = args.flag(
    $parser,
    "input",
    "i",
    "input.txt",
    "training data (one name per line); falls back to a built-in list");
$parser = args.flag($parser, "save", "o", "model.json", "checkpoint output path");
$parser = args.flag($parser, "format", "f", "json", "checkpoint format");
$parser = args.choices($parser, ["json", "fs"]);
$parser = args.intFlag($parser, "samples", "n", 20, "names to sample after training");
def cli as args.Result init args.parse($parser, os.ARGS);
if ($cli.done) {
    io.printf("%s\n", $cli.helpText);
    exit 0;
}
def numSteps as int init args.asInt($cli, "steps");
def batch as int init args.asInt($cli, "batch");
if ($batch < 1) {
    $batch = 1;
}
def inputPath as string init args.asString($cli, "input");
def savePath as string init args.asString($cli, "save");
def saveFmt as string init args.asString($cli, "format");
def numSamples as int init args.asInt($cli, "samples");

math.randSeed(42); # let there be order among chaos

# --- a small default corpus (used when the input file is absent) --------------
def const DEFAULTNAMES as list of string init [
    "emma",
    "olivia",
    "ava",
    "isabella",
    "sophia",
    "mia",
    "amelia",
    "harper",
    "evelyn",
    "abigail",
    "emily",
    "ella",
    "aria",
    "scarlett",
    "grace",
    "chloe",
    "liam",
    "noah",
    "oliver",
    "elijah",
    "james",
    "william",
    "benjamin",
    "lucas",
    "henry",
    "theodore",
    "jack",
    "levi",
    "mateo",
    "asher",
    "ethan",
    "daniel",
    "luna",
    "aurora",
    "nova",
    "hazel",
    "zoe",
    "lily",
    "nora",
    "leah",
    "mason",
    "logan",
    "jacob",
    "michael",
    "samuel",
    "david",
    "joseph",
    "carter"
];

# --- dataset ------------------------------------------------------------------
def docs as list of string init [];
if (fs.exists($inputPath)) {
    for (def line in strings.split(fs.readString($inputPath), "\n")) {
        def trimmed as string init strings.trim($line);
        if (len($trimmed) > 0) {
            $docs[] = $trimmed;
        }
    }
} else {
    $docs = DEFAULTNAMES;
}
$docs = lists.shuffle($docs);
io.printf("num docs: %d\n", len($docs));

buildVocab($docs);
io.printf("vocab size: %d\n", $vocabSize);
buildAllMatrices(); # random init (loadMode is false)
io.printf("num params: %d\n", $numParams);

# --- per-document gradient computation (runs inside a spawned worker) ----------
# A GradResult is what one parallel worker returns: the document's loss plus the
# gradient of that loss w.r.t. every parameter.
def struct GradResult {
    loss as float,
    grads as list of float
};

# computeGrads forwards one document, backprops, and reads the parameter gradients
# off the worker's OWN tape copy. It never touches shared state, so any number of
# these run in parallel safely.
func computeGrads(docIdx as int) {
    def tokens as list of int init tokenize($docs[$docIdx]);
    def n as int init BLOCKSIZE;
    if (len($tokens) - 1 < $n) {
        $n = len($tokens) - 1;
    }
    resetKV();
    def losses as list of int init [];
    def posId as int init 0;
    while ($posId < $n) {
        def logits as list of int init gpt($tokens[$posId], $posId);
        def probs as list of int init softmaxNodes($logits);
        $losses[] = mulc(logOp($probs[$tokens[$posId + 1]]), -1.0);
        $posId = $posId + 1;
    }
    def lossNode as int init mulc(sumNodes($losses), 1.0 / convert.toFloat($n));
    backward($lossNode);
    def grads as list of float init [];
    def i as int init 0;
    while ($i < $numParams) {
        $grads[] = gradOf($i);
        $i = $i + 1;
    }
    return GradResult{loss: valueOf($lossNode), grads: $grads};
}

# --- training: data-parallel minibatch SGD with Adam --------------------------
def mBuf as list of float init [];
def vBuf as list of float init [];
def pInit as int init 0;
while ($pInit < $numParams) {
    $mBuf[] = 0.0;
    $vBuf[] = 0.0;
    $pInit = $pInit + 1;
}
def learningRate as float init 0.01;
def beta1 as float init 0.85;
def beta2 as float init 0.99;
def epsAdam as float init 1e-8;

io.printf(
    "training %d steps, batch %d (parallel across %d cores)...\n",
    $numSteps,
    $batch,
    os.NCPU);
def step as int init 0;
while ($step < $numSteps) {
    resetTo($numParams); # the tape is just the params; each worker copies it
    
    # fan out: BATCH documents, each gradient computed on its own core
    def jobs as list of task of GradResult init [];
    def w as int init 0;
    while ($w < $batch) {
        def docIdx as int init ($step * $batch + $w) % len($docs);
        $jobs[] = spawn {
            return computeGrads($docIdx);
        };
        $w = $w + 1;
    }
    def results as list of GradResult init task.waitAll($jobs);

    # average the batch loss (for reporting)
    def avgLoss as float init 0.0;
    def b as int init 0;
    while ($b < $batch) {
        $avgLoss = $avgLoss + $results[$b].loss;
        $b = $b + 1;
    }
    $avgLoss = $avgLoss / convert.toFloat($batch);

    # Adam update on the averaged gradient
    def lrT as float init $learningRate *
        (1.0 - convert.toFloat($step) / convert.toFloat($numSteps));
    def bc1 as float init 1.0 - math.pow($beta1, convert.toFloat($step + 1));
    def bc2 as float init 1.0 - math.pow($beta2, convert.toFloat($step + 1));
    def i as int init 0;
    while ($i < $numParams) {
        def g as float init 0.0;
        def bb as int init 0;
        while ($bb < $batch) {
            $g = $g + $results[$bb].grads[$i];
            $bb = $bb + 1;
        }
        $g = $g / convert.toFloat($batch);
        $mBuf[$i] = $beta1 * $mBuf[$i] + (1.0 - $beta1) * $g;
        $vBuf[$i] = $beta2 * $vBuf[$i] + (1.0 - $beta2) * $g * $g;
        def mHat as float init $mBuf[$i] / $bc1;
        def vHat as float init $vBuf[$i] / $bc2;
        $tape[$i].data = $tape[$i].data - $lrT * $mHat / (math.sqrt($vHat) + $epsAdam);
        $i = $i + 1;
    }

    if (($step + 1) % 10 == 0) {
        io.printf("step %d / %d  loss %f|prec=4\n", $step + 1, $numSteps, $avgLoss);
    }
    $step = $step + 1;
}

# --- save the trained model ---------------------------------------------------
if ($saveFmt == "json") {
    saveJson($savePath);
} else {
    saveFs($savePath);
}
io.printf("saved %s checkpoint -> %s\n", $saveFmt, $savePath);

# --- inference ----------------------------------------------------------------
io.printf("\n--- inference (new, hallucinated names) ---\n");
def si as int init 0;
while ($si < $numSamples) {
    io.printf("sample %d: %s\n", $si + 1, generateName(0.5));
    $si = $si + 1;
}
