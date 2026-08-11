# demo-microgpt

Andrej Karpathy's [microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- a from-scratch, char-level GPT trained on names - ported to
[Jennifer](https://github.com/jennifer-language/jennifer), as a standalone example
of a real Jennifer project.

Same architecture as the original (GPT-2 shape:
multi-head causal self-attention + MLP, RMSNorm, no biases, ReLU), running on a
hand-rolled scalar autograd engine. No tensors, no NumPy - every number is one
scalar in a computation graph.

On top it adds three things worth a look in Jennifer:

- **Data-parallel training** that uses the whole CPU (`spawn` + value semantics).
- **Checkpointing** in two formats (JSON and a flat text file).
- A real **command-line interface** via the `args` module.

```
$ jennifer run microgpt.j --steps 80 --batch 8
num docs: 32033
vocab size: 27
num params: 1328
training 80 steps, batch 8 (parallel across 8 cores)...
step 10 / 80  loss 3.0593
...
step 80 / 80  loss 2.3042
saved json checkpoint -> model.json

--- inference (new, hallucinated names) ---
sample  1: jina
sample  2: saran
sample  3: marian
sample  4: arani
...
```

## Why the autograd is not a straight port

Karpathy's `Value` class is elegant because of two things Jennifer deliberately
does **not** have:

- **Closures** - each `Value` stores a `_backward` closure capturing its operands.
  Jennifer's `func` is a handle to a top-level method, with no capture.
- **Shared mutable objects** - the graph is a DAG of shared `Value` nodes, and
  `child.grad += ...` mutates a node reached from many parents. Jennifer has **value
  semantics**: structs and lists are *copied* on assignment, so a node cannot be
  shared and mutated in place from two places.

So the autograd (`autograd.j`) is an **arena / tape** design: every scalar is a
`Node` in one global list `tape`, a node references its operands by integer
**index**, forward appends nodes (which keeps the tape in topological order), and
`backward` is a single reverse walk dispatching on an `Op` enum. A parameter reused
in many places still accumulates gradient correctly because every parent sits at a
higher index and is visited first.

A graphical introduction to microgpt can be found at
https://lxb12123.github.io/microgpt-3d-tutorial/.

## Data-parallel training (uses the whole CPU)

Here the same "limitation" pays off. When you `spawn` a block, Jennifer
**deep-copies the enclosing scope** into the new goroutine - so each worker gets its
**own private copy of the tape**. That is exactly what data-parallel SGD wants:

- each step fans out a **batch** of documents, one `spawn`ed worker per document;
- every worker builds its graph, backprops, and reads its gradients off its *own*
  tape copy - no shared mutable state, so **no locks and no data races** (guaranteed
  by value semantics, not by careful discipline);
- the main thread averages the batch's gradients and does one Adam update.

`--batch` defaults to the core count, so training spreads across all cores. Measured
on an 8-core machine, the same document budget trains **~3x faster** wall-clock at
~450-490% CPU. It is not a perfect 8x because each step has a barrier (it waits for
the slowest worker, and names vary in length) plus the cost of the per-`spawn` scope
copy - but it is genuine multicore parallelism that falls straight out of the
isolation model. `--batch 1` gives the original's serial, one-document-per-step SGD.

## Checkpointing (save / reuse a trained model)

A trained model is just three things: the **shape** (a few ints), the **tokenizer**
(the character vocabulary), and the **weights** (the `numParams` parameter floats,
read off the front of the tape). The computation graph is transient and never
saved. `checkpoint.j` writes those in two formats:

- **JSON** (`model.json`) - structured and portable, via the `json` library.
- **FS** (`model.ckpt`) - a flat text file (shape line, vocab line, one float per
  line) via `fs`; trivially diff-able, no json dependency.

Both preserve full float precision and round-trip bit-for-bit (see
`checkpoint_test.j`). Loading replays the saved floats through the *same*
parameter-construction order used in training, so `sample.j` reconstructs the exact
model and only runs inference - the fast "train once, sample many times" path.

## Running it

```sh
# train (parallel), save a JSON checkpoint, sample some names
jennifer run microgpt.j --steps 200 --batch 8 --save model.json

# reload the checkpoint and just generate (starts in a fraction of a second)
jennifer run sample.j --load model.json --samples 20

# the flat-text format instead
jennifer run microgpt.j --format fs --save model.ckpt
jennifer run sample.j   --format fs --load model.ckpt

# tests
jennifer test autograd_test.j     # engine gradient checks
jennifer test checkpoint_test.j   # save/load round-trips
```

Pass `--help` to either program for the full flag list. By default training uses a
small built-in list of ~50 names; drop an **`input.txt`** (one name per line) next
to the scripts to train on your own - e.g. the classic
[names.txt](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt)
(32K names). On the real corpus the model quickly starts producing plausible names
(`jina`, `saran`, `marian`, `dahan`, `minili`, ...).

## Scaling toward the original

The model shape is fixed at the top of `model.j` (`NEMBD=8`, `NHEAD=2`, `NLAYER=1`,
`BLOCKSIZE=16`); the original uses `16 / 4 / 1 / 16`. Raising them makes samples
better and each step slower, roughly linearly in the node count. A tree-walking
interpreter doing *scalar* autograd over a transformer is slow by design - the point
is clarity, not throughput - which is exactly why the data parallelism is nice to
have. Runtime knobs (`--steps`, `--batch`, `--input`, `--save`, `--format`,
`--samples`, `--temperature`) are CLI flags and do not change the model shape, so a
checkpoint stays loadable.

## Files

- `autograd.j` - the scalar reverse-mode autodiff engine (arena / tape).
- `model.j` - the GPT model, tokenizer, and sampler (shared by trainer and sampler).
- `checkpoint.j` - save / load in JSON and flat-text formats.
- `microgpt.j` - the trainer: parallel data-parallel SGD + Adam, saves a checkpoint.
- `sample.j` - loads a checkpoint and generates names (no training).
- `autograd_test.j`, `checkpoint_test.j` - `jennifer test` overlays.

## License

MIT - see [LICENSE](LICENSE). A port of Karpathy's microGPT, which is MIT-licensed.
The `args.j` module is part of the (LGPL) Jennifer distribution and is imported at
runtime, not vendored here.
