#!/usr/bin/env -S jennifer run
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (C) 2026 mplx <jennifer@mplx.dev>
# pragma-jennifer-version: >=0.27.0

# demo-microgpt - load a trained checkpoint and generate names (no training).
#
# This is the "train once, sample many times" half: it reconstructs the model from
# a checkpoint (weights + tokenizer + shape) and only runs inference, so it starts
# in a fraction of a second. Loads either checkpoint format.
#
# Usage:
#   jennifer run sample.j --load model.json --samples 20
#   jennifer run sample.j --load model.ckpt --format fs

include "./checkpoint.j"; # brings model.j + autograd.j
use io;
use os;
use math;
use convert;

import "args.j" as args;

def parser as args.Parser init args.parser(
    "sample",
    "Load a microGPT checkpoint and generate names");
$parser = args.flag($parser, "load", "l", "model.json", "checkpoint to load");
$parser = args.flag($parser, "format", "f", "json", "checkpoint format");
$parser = args.choices($parser, ["json", "fs"]);
$parser = args.intFlag($parser, "samples", "n", 20, "names to generate");
$parser = args.floatFlag($parser, "temperature", "t", 0.5, "sampling temperature, in (0, 1]");
def cli as args.Result init args.parse($parser, os.ARGS);
if ($cli.done) {
    io.printf("%s\n", $cli.helpText);
    exit 0;
}

math.randSeed(42);
def loadPath as string init args.asString($cli, "load");
def fmt as string init args.asString($cli, "format");
def numSamples as int init args.asInt($cli, "samples");
def temperature as float init args.asFloat($cli, "temperature");

if ($fmt == "json") {
    loadJson($loadPath);
} else {
    loadFs($loadPath);
}
io.printf(
    "loaded %s checkpoint <- %s  (vocab %d, %d params)\n",
    $fmt,
    $loadPath,
    $vocabSize,
    $numParams);

io.printf("\n--- sampled names (temperature %f) ---\n", $temperature);
def si as int init 0;
while ($si < $numSamples) {
    io.printf("sample %d: %s\n", $si + 1, generateName($temperature));
    $si = $si + 1;
}
