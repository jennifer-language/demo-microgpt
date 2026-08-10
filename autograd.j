# demo-microgpt: the scalar reverse-mode autodiff engine, in Jennifer.
#
# Jennifer has no closures and uses value semantics, so this is deliberately NOT a
# closure-per-node port of Karpathy's `Value`. It is an ARENA / tape design:
#
#   * every value is a Node in one global list (`tape`);
#   * a node references its operands by integer INDEX, never by a shared struct
#     (value semantics would copy a struct, so shared mutable nodes are impossible);
#   * backward is a single reverse pass that dispatches on an `Op` enum instead of a
#     per-node closure.
#
# The key trick: a node is always appended AFTER its operands, so the tape is
# already in topological order - walking it in reverse index order is exactly
# reverse-topological order, and no explicit sort is needed. A node with a
# constant operand (a fixed exponent, or a scalar add / mul) stores it in `c`.

use math;
use lists;

# The operation that produced a node (Leaf = an input or a learnable parameter).
# The scalar-constant variants (AddC / MulC / Pow) fold a Python `Value op float`
# into one node instead of a throwaway constant Leaf.
def enum Op { Leaf, Add, Sub, Mul, AddC, MulC, Pow, Log, Exp, Relu };

# One value in the computation graph. `a` / `b` are operand indices into `tape`
# (-1 = none); `c` is the constant operand for AddC / MulC / Pow (else unused);
# `grad` accumulates d(output)/d(this) during backward.
def struct Node {
    data as float,
    grad as float,
    op as Op,
    a as int,
    b as int,
    c as float
};

# The arena. Every node created below is appended here; its index is its identity.
def tape as list of Node init [];

# node appends a node and returns its index.
func node(d as float, o as Op, a as int, b as int, c as float) {
    $tape[] = Node{data: $d, grad: 0.0, op: $o, a: $a, b: $b, c: $c};
    return len($tape) - 1;
}

# --- forward: each op computes its value and records how it was made ----------

func leaf(x as float) {
    return node($x, Op.Leaf, -1, -1, 0.0);
}

func add(i as int, j as int) {
    return node($tape[$i].data + $tape[$j].data, Op.Add, $i, $j, 0.0);
}

func sub(i as int, j as int) {
    return node($tape[$i].data - $tape[$j].data, Op.Sub, $i, $j, 0.0);
}

func mul(i as int, j as int) {
    return node($tape[$i].data * $tape[$j].data, Op.Mul, $i, $j, 0.0);
}

# value + constant, value * constant, value ** constant (the Python `Value op float`
# and `__pow__` cases). Folded into a single node with the constant in `c`.
func addc(i as int, k as float) {
    return node($tape[$i].data + $k, Op.AddC, $i, -1, $k);
}

func mulc(i as int, k as float) {
    return node($tape[$i].data * $k, Op.MulC, $i, -1, $k);
}

func powc(i as int, k as float) {
    return node(math.pow($tape[$i].data, $k), Op.Pow, $i, -1, $k);
}

func logOp(i as int) {
    return node(math.ln($tape[$i].data), Op.Log, $i, -1, 0.0);
}

func expOp(i as int) {
    return node(math.exp($tape[$i].data), Op.Exp, $i, -1, 0.0);
}

func reluOp(i as int) {
    return node(math.max(0.0, $tape[$i].data), Op.Relu, $i, -1, 0.0);
}

# --- backward: one reverse pass, dispatching on the op ------------------------

# accum adds `g` into node i's gradient.
func accum(i as int, g as float) {
    $tape[$i].grad = $tape[$i].grad + $g;
}

# backward zeroes every gradient, seeds d(out)/d(out) = 1, then walks the tape in
# reverse (= reverse-topological) order, pushing each node's gradient to its
# operands by the chain rule. Because a node's parents all sit at higher indices,
# its gradient is fully accumulated by the time the walk reaches it.
func backward(out as int) {
    def i as int init 0;
    while ($i < len($tape)) {
        $tape[$i].grad = 0.0;
        $i = $i + 1;
    }
    $tape[$out].grad = 1.0;
    $i = len($tape) - 1;
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
                accum($n.a, $n.grad);
            }
            when MulC {
                accum($n.a, $n.c * $n.grad);
            }
            when Pow {
                # d/dx x^c = c * x^(c-1)
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
                if ($tape[$n.a].data > 0.0) {
                    accum($n.a, $n.grad);
                }
            }
        }
        $i = $i - 1;
    }
}

# --- reading / updating nodes (the tape is the single source of truth) --------

func valueOf(i as int) {
    return $tape[$i].data;
}

func gradOf(i as int) {
    return $tape[$i].grad;
}

# resetTo drops every node after the first `keep`, so the training loop can rebuild
# the graph each step while the leading `keep` parameter nodes persist.
func resetTo(keep as int) {
    $tape = lists.slice($tape, 0, $keep);
    def i as int init 0;
    while ($i < len($tape)) {
        $tape[$i].grad = 0.0;
        $i = $i + 1;
    }
}
