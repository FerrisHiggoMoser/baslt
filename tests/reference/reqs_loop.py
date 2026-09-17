"""A slow, sample-by-sample reading of requirement expressions, for checking the vectorized evaluator.

Trees are tuples: ("num", x), ("sig", name), ("t",), (op, child, ...). Values are Python floats (NaN for
missing); conditions are True, False or None (unknown). Written from docs/requirements.md, not from the evaluator.
"""

from __future__ import annotations

import math

NUM_OPS = ("add", "sub", "mul", "div", "neg", "abs", "max", "min", "clip", "where")
COND_OPS = ("cmp", "and", "or", "not")


def render(tree) -> str:
    op = tree[0]
    if op == "num":
        return repr(tree[1])
    if op == "sig":
        return tree[1]
    if op == "t":
        return "t"
    if op in ("add", "sub", "mul", "div"):
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        return f"({render(tree[1])} {symbol} {render(tree[2])})"
    if op == "neg":
        return f"(-{render(tree[1])})"
    if op in ("abs", "max", "min", "clip"):
        return f"{op}({', '.join(render(child) for child in tree[1:])})"
    if op == "where":
        return f"where({render(tree[1])}, {render(tree[2])}, {render(tree[3])})"
    if op == "cmp":
        return f"({render(tree[2])} {tree[1]} {render(tree[3])})"
    if op in ("and", "or"):
        return f"({render(tree[1])} {op} {render(tree[2])})"
    if op == "not":
        return f"(not {render(tree[1])})"
    if op == "label":
        return f"({tree[1]} == '{tree[2]}')"
    raise ValueError(op)


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def value(tree, env: dict, i: int):
    op = tree[0]
    if op == "num":
        return float(tree[1])
    if op == "sig":
        return float(env[tree[1]][i])
    if op == "t":
        return float(env["t"][i])
    if op in ("add", "sub", "mul"):
        a, b = value(tree[1], env, i), value(tree[2], env, i)
        try:
            return a + b if op == "add" else (a - b if op == "sub" else a * b)
        except OverflowError:
            return math.nan
    if op == "div":
        a, b = value(tree[1], env, i), value(tree[2], env, i)
        if math.isnan(a) or math.isnan(b):
            return math.nan
        if b == 0:
            if a == 0:
                return math.nan
            return math.copysign(math.inf, a) * math.copysign(1.0, b)
        return a / b
    if op == "neg":
        return -value(tree[1], env, i)
    if op == "abs":
        return abs(value(tree[1], env, i))
    if op in ("max", "min"):
        values = [value(child, env, i) for child in tree[1:]]
        present = [v for v in values if not math.isnan(v)]
        if not present:
            return math.nan
        return max(present) if op == "max" else min(present)
    if op == "clip":
        x, lo, hi = (value(child, env, i) for child in tree[1:])
        if math.isnan(x) or math.isnan(lo) or math.isnan(hi):
            return math.nan
        return min(max(x, lo), hi)
    if op == "where":
        test = truth(tree[1], env, i)
        if test is None:
            return math.nan
        return value(tree[2] if test else tree[3], env, i)
    raise ValueError(op)


def truth(tree, env: dict, i: int):
    op = tree[0]
    if op == "cmp":
        a, b = value(tree[2], env, i), value(tree[3], env, i)
        if not (_finite(a) and _finite(b)):
            return None
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[tree[1]]
    if op == "not":
        inner = truth(tree[1], env, i)
        return None if inner is None else not inner
    if op == "and":
        a, b = truth(tree[1], env, i), truth(tree[2], env, i)
        if a is False or b is False:
            return False
        return None if a is None or b is None else True
    if op == "or":
        a, b = truth(tree[1], env, i), truth(tree[2], env, i)
        if a is True or b is True:
            return True
        return None if a is None or b is None else False
    if op == "label":
        code = env["labels"][tree[2]]
        x = float(env[tree[1]][i])
        return None if math.isnan(x) else x == code
    raise ValueError(op)


def aggregate(fn: str, t, x, active, discrete: bool) -> float:
    """max, min, initial, final, mean, rms, integral of `x` over the samples where `active` holds."""
    ok = [bool(a) and _finite(float(v)) for a, v in zip(active, x)]
    picked = [float(v) for v, keep in zip(x, ok) if keep]
    if not picked:
        return math.nan
    if fn == "max":
        return max(picked)
    if fn == "min":
        return min(picked)
    if fn == "initial":
        return picked[0]
    if fn == "final":
        return picked[-1]
    area = span = 0.0
    for k in range(len(t) - 1):
        dt = float(t[k + 1] - t[k])
        a, b = float(x[k]), float(x[k + 1])
        if discrete:
            if not ok[k]:
                continue
            area += dt * (a * a if fn == "rms" else a)
        else:
            if not (ok[k] and ok[k + 1]):
                continue
            area += dt * ((a * a + b * b + a * b) / 3 if fn == "rms" else (a + b) / 2)
        span += dt
    if fn == "integral":
        return area
    if span <= 0:
        return math.sqrt(sum(v * v for v in picked) / len(picked)) if fn == "rms" else sum(picked) / len(picked)
    return math.sqrt(area / span) if fn == "rms" else area / span


def upper_limit_check(t, x, active, limit: float, *, strict: bool = False, tolerance: float = 0.0) -> dict:
    """An upper-limit requirement on a continuous signal, one sample at a time (docs/requirements.md).

    Returns the verdict (fail, pass or not_applicable, before the gap rule), the margin at the worst sample and the
    violating runs as (start, end, tolerated).
    """
    n = len(t)
    excess = []
    for i in range(n):
        v = float(x[i])
        if not active[i] or not _finite(v):
            excess.append(math.nan)
            continue
        e = v - limit
        excess.append(5e-324 if strict and e == 0 else e)
    finite = [e for e in excess if not math.isnan(e)]
    if not finite:
        return {"verdict": "not_applicable", "margin": None, "runs": []}
    worst = max(finite)
    runs = []
    i = 0
    while i < n:
        if math.isnan(excess[i]) or excess[i] <= 0:
            i += 1
            continue
        first = i
        while i + 1 < n and not math.isnan(excess[i + 1]) and excess[i + 1] > 0:
            i += 1
        last = i
        start = float(t[first])
        if first > 0 and not math.isnan(excess[first - 1]):
            a, b = excess[first - 1], excess[first]
            start = float(t[first - 1]) + (0.0 - a) / (b - a) * float(t[first] - t[first - 1])
        end = float(t[last])
        if last < n - 1 and not math.isnan(excess[last + 1]):
            a, b = excess[last], excess[last + 1]
            end = float(t[last]) + (0.0 - a) / (b - a) * float(t[last + 1] - t[last])
        runs.append((start, end, tolerance > 0 and end - start <= tolerance))
        i += 1
    failed = any(not tolerated for _, _, tolerated in runs)
    return {"verdict": "fail" if failed else "pass", "margin": 0.0 if worst == 5e-324 else -worst, "runs": runs}
