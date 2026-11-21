# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (C) 2026 mplx <jennifer@mplx.dev>
# pragma-jennifer-version: >=0.27.0

/**
 * demo-micrograd: a tiny scalar reverse-mode autodiff engine, in Jennifer.
 *
 * Jennifer has no closures and uses value semantics, so this is deliberately NOT a
 * closure-per-node port of Karpathy's micrograd. It is an ARENA / tape design:
 *
 *   * every value is a Node in one global list (`tape`);
 *   * a node references its operands by integer INDEX, never by a shared struct
 *     (value semantics would copy a struct, so shared mutable nodes are impossible);
 *   * backward is a single reverse pass that dispatches on an `Op` enum instead of a
 *     per-node closure.
 *
 * The key trick: a node is always appended AFTER its operands, so the tape is
 * already in topological order - walking it in reverse index order is exactly
 * reverse-topological order, and no explicit sort is needed.
 *
 * v2 adds four operations to v1's five - `powc`, `logOp`, `expOp`, `reluOp` - and
 * the `Node.c` field that `powc` needs to carry a constant exponent.
 *
 * Every function here addresses a node by its tape index, so an index outside
 * `0..len(tape)` raises a catchable out-of-range Error.
 *
 * This file is `include`d (a textual splice), not `import`ed: a module top level is
 * declarations-only and could not hold the mutable global `tape` the arena needs.
 *
 * @module autograd
 * @author mplx <jennifer@mplx.dev>
 * @license MIT
 * @see https://github.com/karpathy/micrograd
 */

use math;

/**
 * The operation that produced a node.
 *
 * `Leaf` is an input or a learnable parameter (no operands); `Add`, `Sub` and `Mul`
 * are binary; `Tanh`, `Log`, `Exp` and `Relu` are unary; `AddC`, `MulC` and `Pow`
 * are unary in a node with a constant alongside it in `Node.c`.
 *
 * Adding a variant here leaves `backward`'s
 * `match` non-exhaustive until its backward rule is written, so an op cannot ship
 * without one.
 */
def enum Op {
    Leaf,
    Add,
    Sub,
    Mul,
    AddC,
    MulC,
    Tanh,
    Pow,
    Log,
    Exp,
    Relu
};

/**
 * One value in the computation graph.
 *
 * @field data {float} the forward value
 * @field grad {float} accumulates d(output)/d(this) during backward
 * @field op {Op} the operation that produced this node
 * @field a {int} first operand's index into `tape`, or -1 for none
 * @field b {int} second operand's index into `tape`, or -1 for none
 * @field c {float} the constant operand of `AddC` / `MulC` / `Pow`; unused otherwise
 */
def struct Node {
    data as float,
    grad as float,
    op as Op,
    a as int,
    b as int,
    c as float
};

# The arena. Every node created below is appended here; its index is its identity.
# (A doc comment attaches to a func / struct / enum / const, so this mutable global
# keeps a plain comment.)
def tape as list of Node init [];

/**
 * Appends a node to the tape.
 *
 * @param d {float} the node's forward value
 * @param o {Op} the operation that produced it
 * @param a {int} first operand's tape index, or -1 for none
 * @param b {int} second operand's tape index, or -1 for none
 * @param c {float} the constant operand, for the ops that take one; else 0.0
 * @return {int} the new node's index, which is its identity from here on
 */
func node(d as float, o as Op, a as int, b as int, c as float) {
    $tape[] = Node{data: $d, grad: 0.0, op: $o, a: $a, b: $b, c: $c};
    return len($tape) - 1;
}

# --- forward: each op computes its value and records how it was made ----------

/**
 * Creates a leaf: an input or a learnable parameter, with no operands.
 *
 * @param x {float} the value
 * @return {int} the new node's tape index
 */
func leaf(x as float) {
    return node($x, Op.Leaf, -1, -1, 0.0);
}

/**
 * Adds two nodes.
 *
 * @param i {int} left operand's tape index
 * @param j {int} right operand's tape index
 * @return {int} tape index of a node holding `i + j`
 */
func add(i as int, j as int) {
    return node($tape[$i].data + $tape[$j].data, Op.Add, $i, $j, 0.0);
}

/**
 * Subtracts one node from another.
 *
 * @param i {int} minuend's tape index
 * @param j {int} subtrahend's tape index
 * @return {int} tape index of a node holding `i - j`
 */
func sub(i as int, j as int) {
    return node($tape[$i].data - $tape[$j].data, Op.Sub, $i, $j, 0.0);
}

/**
 * Multiplies two nodes.
 *
 * @param i {int} left operand's tape index
 * @param j {int} right operand's tape index
 * @return {int} tape index of a node holding `i * j`
 */
func mul(i as int, j as int) {
    return node($tape[$i].data * $tape[$j].data, Op.Mul, $i, $j, 0.0);
}

/**
 * Adds a constant to a node, without putting the constant on the tape.
 *
 * `addc(i, k)` computes exactly what `add(i, leaf(k))` does, in one node instead
 * of two. See `mulc` for what this is and is not worth.
 *
 * @param i {int} operand's tape index
 * @param k {float} the constant to add
 * @return {int} tape index of a node holding `i + k`
 */
func addc(i as int, k as float) {
    return node($tape[$i].data + $k, Op.AddC, $i, -1, $k);
}

/**
 * Multiplies a node by a constant, without putting the constant on the tape.
 *
 * `mulc(i, k)` computes exactly what `mul(i, leaf(k))` does, in one node instead
 * of two, and its backward rule reads `c` instead of another tape entry.
 *
 * This is a **small** win, and v5 measured it rather than assuming: on the
 * two-moons run it removes 13 nodes out of 2313 per batch (0.6% of the tape) and
 * 2.1% of wall clock - more than the node count suggests, because `MulC` reads its
 * constant from `c` instead of making `backward` fetch a second tape entry. Useful,
 * but not the headline: the change that mattered was not building the L2 graph at
 * all.
 *
 * @param i {int} operand's tape index
 * @param k {float} the constant to multiply by
 * @return {int} tape index of a node holding `i * k`
 */
func mulc(i as int, k as float) {
    return node($tape[$i].data * $k, Op.MulC, $i, -1, $k);
}

/**
 * Applies tanh to a node - the engine's one nonlinearity.
 *
 * Named `tanhOp` rather than `tanh` so it does not collide with `math.tanh`.
 *
 * @param i {int} operand's tape index
 * @return {int} tape index of a node holding `tanh(i)`
 */
func tanhOp(i as int) {
    return node(math.tanh($tape[$i].data), Op.Tanh, $i, -1, 0.0);
}

/**
 * Raises a node to a constant power.
 *
 * The exponent is a plain `float` kept in `Node.c`, not a node, so no gradient
 * flows to it - the same restriction micrograd puts on `__pow__`. A node exponent
 * would need `d/dy x^y = x^y * ln(x)`, which is undefined for `x <= 0`.
 *
 * @param i {int} base's tape index
 * @param k {float} the constant exponent
 * @return {int} tape index of a node holding `i ** k`
 * @throws {Error} if `i ** k` is undefined (e.g. a negative base to a fractional
 *   exponent) or overflows to a non-finite value
 */
func powc(i as int, k as float) {
    return node(math.pow($tape[$i].data, $k), Op.Pow, $i, -1, $k);
}

/**
 * Takes the natural logarithm of a node.
 *
 * Named `logOp` rather than `ln` so it does not collide with `math.ln`.
 *
 * @param i {int} operand's tape index
 * @return {int} tape index of a node holding `ln(i)`
 * @throws {Error} if the operand is not positive
 */
func logOp(i as int) {
    return node(math.ln($tape[$i].data), Op.Log, $i, -1, 0.0);
}

/**
 * Exponentiates a node.
 *
 * Named `expOp` rather than `exp` so it does not collide with `math.exp`.
 *
 * @param i {int} operand's tape index
 * @return {int} tape index of a node holding `exp(i)`
 * @throws {Error} if the result overflows to a non-finite value
 */
func expOp(i as int) {
    return node(math.exp($tape[$i].data), Op.Exp, $i, -1, 0.0);
}

/**
 * Applies the rectifier `max(0, x)` to a node.
 *
 * Kinked at 0, where the derivative does not exist; by the usual convention the
 * backward rule passes nothing through at exactly 0.
 *
 * @param i {int} operand's tape index
 * @return {int} tape index of a node holding `max(0, i)`
 */
func reluOp(i as int) {
    return node(math.max(0.0, $tape[$i].data), Op.Relu, $i, -1, 0.0);
}

# --- backward: one reverse pass, dispatching on the op ------------------------

/**
 * Adds `g` into node i's gradient.
 *
 * Accumulating (rather than assigning) is what makes a value reused by several
 * consumers collect the gradient of every path through it.
 *
 * @param i {int} the node's tape index
 * @param g {float} the gradient contribution to add
 * @return {null}
 */
func accum(i as int, g as float) {
    $tape[$i].grad = $tape[$i].grad + $g;
}

/**
 * Sets every gradient on the tape back to zero.
 *
 * Only needed to run `backward` twice over the same graph - the normal training
 * loop never calls it, because `resetTo` already zeroes the nodes that survive and
 * `node` gives every fresh node a zero gradient. See `backward`.
 *
 * @return {null}
 */
func zeroGrads() {
    def i as int init 0;
    while ($i < len($tape)) {
        $tape[$i].grad = 0.0;
        $i = $i + 1;
    }
}

/**
 * Runs the backward pass, filling in every node's `grad`.
 *
 * Seeds d(out)/d(out) = 1, then walks the tape in reverse (= reverse-topological)
 * order, pushing each node's gradient to its operands by the chain rule. Because a
 * node's parents all sit at higher indices, its gradient is fully accumulated by
 * the time the walk reaches it. The `match` on `Op` is exhaustive, so every
 * operation is guaranteed to have a backward rule.
 *
 * **Precondition: every gradient on the tape is 0.** Up to v4 this function
 * started by zeroing the whole tape, which measured at 10.7% of a training run -
 * work that was almost entirely redundant, since `node` is born with `grad: 0.0`
 * and `resetTo` zeroes the prefix it keeps. v5 drops the pass and maintains the
 * invariant at the two places that can break it instead. Call `zeroGrads` first if
 * you need to run `backward` twice over one graph, or gradients will accumulate
 * across both passes.
 *
 * @param out {int} tape index of the output node to differentiate
 * @return {null}
 */
func backward(out as int) {
    $tape[$out].grad = 1.0;
    def i as int init len($tape) - 1;
    while ($i >= 0) {
        def n as Node init $tape[$i];
        match ($n.op) {
            when Leaf {
                # an input / parameter has no operands
            }
            when Add {
                accum($n.a, $n.grad);
                accum($n.b, $n.grad);
            }
            when Sub {
                accum($n.a, $n.grad);
                accum($n.b, -$n.grad);
            }
            when Mul {
                accum($n.a, $tape[$n.b].data * $n.grad);
                accum($n.b, $tape[$n.a].data * $n.grad);
            }
            when AddC {
                # d/dx (x + c) = 1; the constant gets no gradient
                accum($n.a, $n.grad);
            }
            when MulC {
                # d/dx (x * c) = c
                accum($n.a, $n.c * $n.grad);
            }
            when Tanh {
                # d/dx tanh(x) = 1 - tanh(x)^2, and n.data == tanh(operand)
                accum($n.a, (1.0 - $n.data * $n.data) * $n.grad);
            }
            when Pow {
                # d/dx x^c = c * x^(c-1); the exponent is a constant, so it gets
                # no gradient of its own
                accum($n.a, $n.c * math.pow($tape[$n.a].data, $n.c - 1.0) * $n.grad);
            }
            when Log {
                # d/dx ln(x) = 1/x
                accum($n.a, (1.0 / $tape[$n.a].data) * $n.grad);
            }
            when Exp {
                # d/dx exp(x) = exp(x), and n.data == exp(operand)
                accum($n.a, $n.data * $n.grad);
            }
            when Relu {
                # the rectifier passes gradient through only where it is active
                if ($tape[$n.a].data > 0.0) {
                    accum($n.a, $n.grad);
                }
            }
        }
        $i = $i - 1;
    }
}

# --- reading / updating nodes (the tape is the single source of truth) --------

/**
 * Reads a node's forward value.
 *
 * @param i {int} the node's tape index
 * @return {float} the node's `data`
 */
func valueOf(i as int) {
    return $tape[$i].data;
}

/**
 * Reads a node's gradient, as left by the last `backward` call.
 *
 * @param i {int} the node's tape index
 * @return {float} the node's `grad`
 */
func gradOf(i as int) {
    return $tape[$i].grad;
}

/**
 * Truncates the tape to its first `keep` nodes and zeroes their gradients.
 *
 * A training loop rebuilds the graph each step on top of a fixed prefix of
 * parameter leaves: `resetTo(numParams)` drops last step's graph while those
 * leading parameter nodes (and their indices) persist. `resetTo(0)` empties it.
 *
 * Zeroing the survivors is what lets `backward` skip its own zeroing pass: this
 * touches the ~100 nodes that outlive the step instead of the ~2300 that did not.
 *
 * @param keep {int} how many leading nodes to keep
 * @return {null}
 */
func resetTo(keep as int) {
    $tape = $tape[..$keep];
    def i as int init 0;
    while ($i < len($tape)) {
        $tape[$i].grad = 0.0;
        $i = $i + 1;
    }
}

/**
 * Nudges a parameter node one gradient-descent step against its gradient.
 *
 * @param i {int} the parameter node's tape index
 * @param lr {float} the learning rate
 * @return {null}
 */
func sgd(i as int, lr as float) {
    $tape[$i].data = $tape[$i].data - $lr * $tape[$i].grad;
}
