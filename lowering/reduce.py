"""前端可划归性：方程 → 稀疏多项式 / 有理 IR（DESIGN.md §1.5、§7 Phase 1）。

这里只做"把方程写成硬件能接受的规范形"这一步，不解方程、不做硬件映射：

    ODE → jet/一阶化 → 状态增广 → differential-algebraic reduction
        → canonical form（A x + B(x⊗x) + Σ Uφ(Vx) + R(x)）

三个判定：

    polynomial   RHS 是状态的稀疏多项式（Lorenz、Duffing + cos 增广）
    rational     RHS 是多项式之比（HH 的 exp 增广、Bessel 的 1/x）
    rejected     含 Gamma 等 differentially transcendental 函数，或增广域无效

口径（§1.5）：增广只在不变流形上等价。sin/cos 的 s²+c²=1 必须单独验收；
Gamma 是 Hölder 的，没有有限阶代数微分方程，不会因为换个写法就消失。
Bessel 可以精确增广成有理系统，但分母含 1/x，要求 x 远离 0。

不变流形的法向是中性的：d(s²+c²)/dt ≡ 0 恒等式意味着法向偏差 δ = s²+c²−1
满足 δ̇ = 0——线性化在 (s,c) 方向给出零特征值（Jacobian ∂(c·u̇, −s·u̇)/∂(s,c)
在 s²+c²=1 处的法向分量为零）。数值积分里 RK4 的截断让它表现为小漂移，
但连续时间模拟硬件上积分器的 offset 电流会让中性方向按 δ ∝ t 线性（或经
噪声成 Brown 运动）漂离流形。所以 `reduce_system(..., manifold_gamma=γ)`
提供可选的 Baumgarte 流形投影：

    ṡ += −γ·s·(s²+c²−1),   ċ += −γ·c·(s²+c²−1)

δ̇ = −2γδ(s²+c²)，法向从零特征值变成 −2γ。默认关闭：稳定化后的系统只在
流形上等于原系统，且额外引入一个真实的乘法器 + gm 项——Phase 1 的 exactness
声明只对未稳定版本成立。硬件部署必须二选一：打开稳定化，或者调度周期性
S&H 重投影（把 (s,c) 重新归一化到流形上）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

__all__ = [
    "Expr", "C", "V", "add", "mul", "pow", "sub", "div",
    "sin", "cos", "exp", "tanh", "besselj", "gamma",
    "System", "Canonical", "Reduced", "first_order", "reduce_system",
    "PrimitiveCost", "reduce_with_primitives",
    "eval_expr", "eval_system", "complete_env", "NotReducible",
]


class NotReducible(Exception):
    """表达式不在当前 B 的 differential-algebraic 闭包里。"""


# ---------------------------------------------------------------------------
# 表达式
# ---------------------------------------------------------------------------
class Expr:
    """极小的符号表达式：op ∈ {const,var,add,mul,pow,sin,cos,exp,tanh,besselj,gamma}。"""

    __slots__ = ("op", "args")

    def __init__(self, op: str, args):
        self.op = op
        self.args = tuple(args)

    def __eq__(self, other):
        return (isinstance(other, Expr) and self.op == other.op
                and self.args == other.args)

    def __hash__(self):
        return hash((self.op, self.args))

    def __repr__(self):
        return _fmt_expr(self)


def _frac(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool 不是系数")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    raise TypeError("系数必须是 int/float/Fraction")


def C(value) -> Expr:
    return Expr("const", (_frac(value),))


def V(name: str) -> Expr:
    return Expr("var", (name,))


_ZERO = Expr("const", (Fraction(0),))
_ONE = Expr("const", (Fraction(1),))


def _key(e: Expr):
    """确定性排序键；不用 repr（深层表达式会很长）。"""
    op = e.op
    if op == "const":
        f = e.args[0]
        return (0, f.numerator, f.denominator)
    if op == "var":
        return (1, e.args[0])
    if op == "besselj":
        return (2, op, e.args[0], _key(e.args[1]))
    if op == "pow":
        return (2, op, _key(e.args[0]), e.args[1])
    return (2, op, tuple(_key(a) for a in e.args if isinstance(a, Expr)))


def _wrap(x) -> Expr:
    return x if isinstance(x, Expr) else C(x)


def add(*terms) -> Expr:
    flat = []
    acc = Fraction(0)
    stack = [_wrap(t) for t in terms if t is not None]
    while stack:
        t = stack.pop()
        if t.op == "add":
            stack.extend(t.args)
        elif t.op == "const":
            acc += t.args[0]
        else:
            flat.append(t)
    if acc:
        flat.append(Expr("const", (acc,)))
    if not flat:
        return _ZERO
    if len(flat) == 1:
        return flat[0]
    flat.sort(key=_key)
    return Expr("add", tuple(flat))


def mul(*factors) -> Expr:
    flat = []
    acc = Fraction(1)
    stack = [_wrap(f) for f in factors if f is not None]
    while stack:
        f = stack.pop()
        if f.op == "mul":
            stack.extend(f.args)
        elif f.op == "const":
            acc *= f.args[0]
        else:
            flat.append(f)
    if acc == 0:
        return _ZERO
    powers: dict[Expr, int] = {}
    order: list[Expr] = []
    for f in flat:
        if f.op == "pow" and isinstance(f.args[1], int):
            base, n = f.args
        else:
            base, n = f, 1
        if base in powers:
            powers[base] += n
        else:
            powers[base] = n
            order.append(base)
    out = []
    for base in order:
        n = powers[base]
        if n == 0:
            continue
        out.append(base if n == 1 else Expr("pow", (base, n)))
    if acc != 1:
        out.insert(0, Expr("const", (acc,)))
    if not out:
        return _ONE
    if len(out) == 1:
        return out[0]
    out.sort(key=_key)
    return Expr("mul", tuple(out))


def pow(base: Expr, n: int) -> Expr:
    base = _wrap(base)
    if not isinstance(n, int):
        raise TypeError("只支持整数幂")
    if n == 0:
        return _ONE
    if n == 1:
        return base
    if base.op == "const":
        return Expr("const", (base.args[0] ** n,))
    if base.op == "pow" and isinstance(base.args[1], int):
        return pow(base.args[0], base.args[1] * n)
    return Expr("pow", (base, n))


def sub(a: Expr, b: Expr) -> Expr:
    return add(a, mul(C(-1), b))


def div(a: Expr, b: Expr) -> Expr:
    return mul(a, pow(b, -1))


def sin(x: Expr) -> Expr:
    return Expr("sin", (x,))


def cos(x: Expr) -> Expr:
    return Expr("cos", (x,))


def exp(x: Expr) -> Expr:
    return Expr("exp", (x,))


def tanh(x: Expr) -> Expr:
    return Expr("tanh", (x,))


def besselj(n: int, x: Expr) -> Expr:
    return Expr("besselj", (int(n), x))


def gamma(x: Expr) -> Expr:
    return Expr("gamma", (x,))


def _fmt_expr(e: Expr) -> str:
    op = e.op
    if op == "const":
        f = e.args[0]
        return str(f.numerator) if f.denominator == 1 else "%d/%d" % (
            f.numerator, f.denominator)
    if op == "var":
        return e.args[0]
    if op == "add":
        return "(" + " + ".join(_fmt_expr(a) for a in e.args) + ")"
    if op == "mul":
        return "(" + "*".join(_fmt_expr(a) for a in e.args) + ")"
    if op == "pow":
        return "%s^%d" % (_fmt_expr(e.args[0]), e.args[1])
    if op == "besselj":
        return "J_%d(%s)" % (e.args[0], _fmt_expr(e.args[1]))
    if op == "gamma":
        return "Gamma(%s)" % _fmt_expr(e.args[0])
    return "%s(%s)" % (op, ", ".join(_fmt_expr(a) for a in e.args
                                     if isinstance(a, Expr)))


def free_vars(e: Expr) -> set[str]:
    out: set[str] = set()

    def walk(x: Expr):
        if x.op == "var":
            out.add(x.args[0])
        elif x.op != "const":
            for a in x.args:
                if isinstance(a, Expr):
                    walk(a)

    walk(e)
    return out


def _has_op(e: Expr, names: set[str]) -> bool:
    if e.op in names:
        return True
    return any(_has_op(a, names) for a in e.args if isinstance(a, Expr))


def diff(e: Expr, name: str) -> Expr:
    op = e.op
    if op == "const":
        return _ZERO
    if op == "var":
        return _ONE if e.args[0] == name else _ZERO
    if op == "add":
        return add(*(diff(a, name) for a in e.args))
    if op == "mul":
        terms = []
        for i, a in enumerate(e.args):
            d = diff(a, name)
            if d is _ZERO:
                continue
            terms.append(mul(*(e.args[:i] + (d,) + e.args[i + 1:])))
        return add(*terms)
    if op == "pow":
        base, n = e.args
        return mul(C(n), pow(base, n - 1), diff(base, name))
    if op == "sin":
        u = e.args[0]
        return mul(cos(u), diff(u, name))
    if op == "cos":
        u = e.args[0]
        return mul(C(-1), sin(u), diff(u, name))
    if op == "exp":
        u = e.args[0]
        return mul(exp(u), diff(u, name))
    if op == "tanh":
        u = e.args[0]
        return mul(sub(_ONE, pow(tanh(u), 2)), diff(u, name))
    if op == "besselj":
        n, u = e.args
        # dJ_n/du = (J_{n-1} - J_{n+1})/2
        return mul(C(0.5), sub(besselj(n - 1, u), besselj(n + 1, u)),
                   diff(u, name))
    if op == "gamma":
        raise NotReducible("Gamma 的导数仍是 Gamma 的有理式，但没有有限阶代数增广")
    raise NotImplementedError(op)


# ---------------------------------------------------------------------------
# 系统
# ---------------------------------------------------------------------------
@dataclass
class System:
    states: list[str]
    rhs: dict[str, Expr]
    invariants: list[Expr] = field(default_factory=list)
    time: str | None = "t"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(set(self.states)) != len(self.states):
            raise ValueError("状态名重复")
        if set(self.rhs) != set(self.states):
            raise ValueError("states 与 rhs 的键必须一致")


def first_order(name: str, order: int, rhs: Expr, *, time: str | None = "t") -> System:
    """x^(order) = rhs → z = (x, x', …, x^(order−1))，状态名为 name_0 … name_{order−1}。"""
    if order < 1:
        raise ValueError("order 必须 ≥ 1")
    states = ["%s_%d" % (name, i) for i in range(order)]
    eqs = {states[i]: V(states[i + 1]) for i in range(order - 1)}
    eqs[states[-1]] = rhs
    return System(states, eqs, time=time)


# ---------------------------------------------------------------------------
# polynomial / rational 系数
# ---------------------------------------------------------------------------
Poly = dict[tuple[int, ...], Fraction]


def _pconst(c: Fraction, n: int) -> Poly:
    return {} if c == 0 else {(0,) * n: c}


def _padd(a: Poly, b: Poly) -> Poly:
    out = dict(a)
    for m, c in b.items():
        v = out.get(m, Fraction(0)) + c
        if v:
            out[m] = v
        else:
            out.pop(m, None)
    return out


def _pmul(a: Poly, b: Poly) -> Poly:
    if not a or not b:
        return {}
    out: Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            m = tuple(x + y for x, y in zip(m1, m2))
            out[m] = out.get(m, Fraction(0)) + c1 * c2
    return {m: c for m, c in out.items() if c}


def _ppow(a: Poly, n: int, nvars: int) -> Poly:
    if n < 0:
        raise ValueError("_ppow 只接受非负指数")
    out = _pconst(Fraction(1), nvars)
    base = a
    while n:
        if n & 1:
            out = _pmul(out, base)
        base = _pmul(base, base)
        n >>= 1
    return out


def _min_exp(poly: Poly, nvars: int) -> list[int]:
    m = [10 ** 9] * nvars
    for mono in poly:
        for i, v in enumerate(mono):
            if v < m[i]:
                m[i] = v
    return m


def _shift(poly: Poly, s: list[int]) -> Poly:
    if not any(s):
        return poly
    return {tuple(a - b for a, b in zip(mono, s)): c
            for mono, c in poly.items()}


def _cancel_monomial(num: Poly, den: Poly) -> tuple[Poly, Poly]:
    """约掉 num/den 的公 monomial 因子（不是完整 gcd）。"""
    if not num or not den:
        return num, den
    nvars = len(next(iter(num)))
    s = [min(a, b) for a, b in zip(_min_exp(num, nvars), _min_exp(den, nvars))]
    return _shift(num, s), _shift(den, s)


def _poly_frac(e: Expr, idx: dict[str, int], n: int) -> tuple[Poly, Poly]:
    """返回 (numerator, denominator)，都是稀疏多项式。"""
    op = e.op
    if op == "const":
        return _pconst(e.args[0], n), _pconst(Fraction(1), n)
    if op == "var":
        m = [0] * n
        m[idx[e.args[0]]] = 1
        return {tuple(m): Fraction(1)}, _pconst(Fraction(1), n)
    if op == "add":
        # 先按分母分组：同分母的加数先在分子上相加，再跨组合并。
        # 旧实现 den=_pmul(den,ad) 会把共享分母平方（a/x + b/x → /x²），
        # 多付一个 divider 深度。Fraction 算术保持精确。
        groups: dict[frozenset, tuple[Poly, list[Poly]]] = {}
        order: list[frozenset] = []
        for a in e.args:
            an, ad = _poly_frac(a, idx, n)
            k = frozenset(ad.items())
            if k in groups:
                groups[k][1].append(an)
            else:
                groups[k] = (ad, [an])
                order.append(k)
        num: Poly = {}
        den = _pconst(Fraction(1), n)
        for k in order:
            ad, nums = groups[k]
            an = nums[0]
            for extra in nums[1:]:
                an = _padd(an, extra)
            num = _padd(_pmul(num, ad), _pmul(an, den))
            den = _pmul(den, ad)
        return num, den
    if op == "mul":
        num = _pconst(Fraction(1), n)
        den = _pconst(Fraction(1), n)
        for a in e.args:
            an, ad = _poly_frac(a, idx, n)
            num = _pmul(num, an)
            den = _pmul(den, ad)
        return num, den
    if op == "pow":
        base, k = e.args
        bn, bd = _poly_frac(base, idx, n)
        if k >= 0:
            return _ppow(bn, k, n), _ppow(bd, k, n)
        if not bn:
            raise NotReducible("除以零多项式")
        return _ppow(bd, -k, n), _ppow(bn, -k, n)
    raise NotReducible("未增广的特殊函数 %s" % op)


# ---------------------------------------------------------------------------
# 增广
# ---------------------------------------------------------------------------
def _augment(system: System, manifold_gamma: float | None = None):
    rhs = dict(system.rhs)
    states = list(system.states)
    augs: list[dict] = []
    bessel_args: list[Expr] = []
    sin_pairs: dict[Expr, tuple[str, str]] = {}
    exp_map: dict[Expr, str] = {}
    tanh_map: dict[Expr, str] = {}
    bessel_map: dict[tuple[int, Expr], tuple[str, str]] = {}
    counts = {k: 0 for k in ("s", "c", "e", "h", "j", "k")}
    used = set(states)

    def fresh(prefix: str) -> str:
        while True:
            counts[prefix] += 1
            name = "%s%d" % (prefix, counts[prefix])
            if name not in used:
                used.add(name)
                return name

    def total_derivative(u: Expr) -> Expr:
        out = _ZERO
        for name in sorted(free_vars(u)):
            d = rhs.get(name)
            if d is None:
                if system.time is not None and name == system.time:
                    d = _ONE
                else:
                    continue
            out = add(out, mul(diff(u, name), d))
        return out

    def rewrite(e: Expr) -> Expr:
        op = e.op
        if op in ("const", "var"):
            return e
        if op == "gamma":
            raise NotReducible(
                "Gamma 是 differentially transcendental（Hölder）："
                "没有有限阶代数微分方程，不能有限增广")
        if op in ("sin", "cos"):
            u = rewrite(e.args[0])
            if u not in sin_pairs:
                ud = total_derivative(u)
                s, c = fresh("s"), fresh("c")
                states.extend([s, c])
                rhs[s] = mul(V(c), ud)
                rhs[c] = mul(C(-1), V(s), ud)
                sin_pairs[u] = (s, c)
                # 法向中性：d(s²+c²)/dt ≡ 0，偏差 δ̇=0（零特征值方向）。
                # 数值上 δ 只被积分误差驱动；模拟积分器的 offset 电流让它
                # 线性增长。是否加 Baumgarte 稳定化由 manifold_gamma 决定，
                # 在 _augment 末尾统一注入（此时 rhs 已闭合，不会再被
                # rewrite 队列改写）。
                augs.append(dict(
                    kind="sin", arg=u, states=(s, c),
                    invariant=sub(add(mul(V(s), V(s)), mul(V(c), V(c))), _ONE),
                    stabilized=False, gamma=None))
            s, c = sin_pairs[u]
            return V(s) if op == "sin" else V(c)
        if op == "exp":
            u = rewrite(e.args[0])
            if u not in exp_map:
                ud = total_derivative(u)
                name = fresh("e")
                states.append(name)
                rhs[name] = mul(V(name), ud)
                exp_map[u] = name
                augs.append(dict(kind="exp", arg=u, states=(name,),
                                 invariant=None))
            return V(exp_map[u])
        if op == "tanh":
            u = rewrite(e.args[0])
            if u not in tanh_map:
                ud = total_derivative(u)
                name = fresh("h")
                states.append(name)
                rhs[name] = mul(sub(_ONE, pow(V(name), 2)), ud)
                tanh_map[u] = name
                augs.append(dict(kind="tanh", arg=u, states=(name,),
                                 invariant=None))
            return V(tanh_map[u])
        if op == "besselj":
            n, u0 = e.args
            u = rewrite(u0)
            key = (n, u)
            if key not in bessel_map:
                ud = total_derivative(u)
                udd = total_derivative(ud)
                j, k = fresh("j"), fresh("k")
                states.extend([j, k])
                rhs[j] = mul(V(k), ud)
                # u²j'' + uj' + (u²−n²)j = 0
                # j' = k·u̇,  k' = [−k/u − (1−n²/u²)j]·u̇² + k·ü
                rhs[k] = add(
                    mul(add(mul(C(-1), V(k), pow(u, -1)),
                            mul(C(-1), sub(_ONE, mul(C(n * n), pow(u, -2))),
                                V(j))),
                        pow(ud, 2)),
                    mul(V(k), udd))
                bessel_map[key] = (j, k)
                bessel_args.append(u)
                augs.append(dict(kind="besselj", n=n, arg=u, states=(j, k),
                                 invariant=None))
            j, k = bessel_map[key]
            return V(j)
        if op == "add":
            return add(*(rewrite(a) for a in e.args))
        if op == "mul":
            return mul(*(rewrite(a) for a in e.args))
        if op == "pow":
            return pow(rewrite(e.args[0]), e.args[1])
        raise NotImplementedError(op)

    for name in list(system.states):
        rhs[name] = rewrite(rhs[name])
    invariants = [rewrite(inv) for inv in system.invariants]
    invariants += [a["invariant"] for a in augs
                   if a.get("invariant") is not None]

    # rewrite 可能新增状态；新方程里也可能有特殊函数。用工作队列闭合。
    processed = set(system.states)
    i = 0
    while i < len(states):
        name = states[i]
        if name not in processed:
            rhs[name] = rewrite(rhs[name])
            processed.add(name)
        i += 1

    # 非自治系统：把显式时间提升为一个 ramp 状态。
    if system.time is not None:
        used: set[str] = set()
        for e in list(rhs.values()) + invariants:
            used |= free_vars(e)
        if system.time in used and system.time not in states:
            states.append(system.time)
            rhs[system.time] = _ONE

    # Baumgarte 流形投影（可选，默认关）：δ̇ = −2γδ(s²+c²)，把法向的
    # 零特征值变成 −2γ。只在 sin/cos 增广上可用；exp/tanh 的增广状态
    # （e=e^u, h=tanh u）没有显式二次不变流形，漂移由 u 的方程本身
    # 决定，稳定化没有独立的意义。放在 rewrite 队列闭合之后，避免
    # 注入项再被 rewrite 改写。
    if manifold_gamma is not None:
        g = float(manifold_gamma)
        if g <= 0:
            raise ValueError("manifold_gamma 必须为正（None 表示关闭）")
        for a in augs:
            if a["kind"] != "sin" or a.get("invariant") is None:
                continue
            s, c = a["states"]
            inv = a["invariant"]          # s²+c²−1
            rhs[s] = add(rhs[s], mul(C(-g), V(s), inv))
            rhs[c] = add(rhs[c], mul(C(-g), V(c), inv))
            a["stabilized"] = True
            a["gamma"] = g

    return System(states, rhs, invariants, system.time, list(system.notes)), augs, bessel_args


# ---------------------------------------------------------------------------
# canonical form
# ---------------------------------------------------------------------------
@dataclass
class Canonical:
    states: list[str]
    kind: str                       # polynomial | rational
    num: dict[str, Poly]
    den: dict[str, Poly]
    degree: int
    n_monomials: int
    dense_monomials: int
    sparsity: float
    divisions: list[str]
    invariants: list[str]

    def as_dict(self) -> dict:
        def terms(d: dict[str, Poly]):
            out = []
            for state, poly in d.items():
                for mono, coeff in sorted(poly.items(),
                                          key=lambda kv: (sum(kv[0]), kv[0])):
                    out.append(dict(state=state, powers=list(mono),
                                    coeff=float(coeff)))
            return out

        return dict(states=self.states, kind=self.kind, degree=self.degree,
                    n_monomials=self.n_monomials,
                    dense_monomials=self.dense_monomials,
                    sparsity=self.sparsity, divisions=self.divisions,
                    invariants=self.invariants,
                    numerator=terms(self.num), denominator=terms(self.den))


@dataclass
class Reduced:
    system: System | None
    canonical: Canonical | None
    kind: str                       # polynomial | rational | rejected
    level: int                      # 0 exact polynomial; 1 rational/native; 2 rejected
    reason: str | None = None
    augmentations: list[dict] = field(default_factory=list)
    domain: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.kind in ("polynomial", "rational") and self.reason is None

    def as_dict(self) -> dict:
        return dict(
            kind=self.kind, level=self.level, reason=self.reason,
            ok=bool(self.ok),
            states=list(self.system.states) if self.system else [],
            canonical=self.canonical.as_dict() if self.canonical else None,
            domain=self.domain,
            augmentations=[dict(kind=a["kind"], arg=str(a["arg"]),
                                states=list(a["states"]),
                                invariant=(str(a["invariant"])
                                           if a.get("invariant") is not None
                                           else None),
                                n=a.get("n"),
                                stabilized=bool(a.get("stabilized", False)),
                                gamma=a.get("gamma"))
                           for a in self.augmentations])


def _fmt_poly(poly: Poly, states: list[str]) -> str:
    if not poly:
        return "0"
    terms = []
    for mono, coeff in sorted(poly.items(), key=lambda kv: (sum(kv[0]), kv[0])):
        factors = []
        for s, p in zip(states, mono):
            if p == 1:
                factors.append(s)
            elif p > 1:
                factors.append("%s^%d" % (s, p))
        mon = "*".join(factors) if factors else "1"
        terms.append(str(coeff) if mon == "1" else "%s*%s" % (coeff, mon))
    return " + ".join(terms)


def reduce_system(system: System, *, bessel_arg_min: float | None = None,
                  bessel_floor: float = 1e-6,
                  manifold_gamma: float | None = None) -> Reduced:
    """把 System 降到 canonical IR；不满足条件时返回 kind='rejected'。

    manifold_gamma：sin/cos 增广的 Baumgarte 流形投影强度（默认 None=关）。
    开启后 rhs 加 −γ·s·(s²+c²−1)、−γ·c·(s²+c²−1)，augmentation dict 记录
    invariant / stabilized / gamma。关闭时增广是精确的（只差一个不变流形
    上的恒等）；开启时系统只在流形上等于原系统，且多了一个真实的硬件项
    （每对增广状态一个乘法器 + gm）。连续时间模拟部署必须开启稳定化，
    或调度周期性 S&H 重投影：模拟积分器的 offset 电流让中性法向按
    δ ∝ t 线性漂离流形。
    """
    for e in system.rhs.values():
        if _has_op(e, {"gamma"}):
            return Reduced(system, None, "rejected", 2,
                           reason="Gamma 是 differentially transcendental（Hölder）："
                                  "没有有限阶代数微分方程，不能有限增广")

    try:
        aug, augs, bessel_args = _augment(system, manifold_gamma)
        idx = {s: i for i, s in enumerate(aug.states)}
        n = len(aug.states)
        num: dict[str, Poly] = {}
        den: dict[str, Poly] = {}
        for s in aug.states:
            nm, dn = _poly_frac(aug.rhs[s], idx, n)
            nm, dn = _cancel_monomial(nm, dn)
            if not nm:
                nm, dn = {}, _pconst(Fraction(1), n)
            num[s], den[s] = nm, dn
    except NotReducible as exc:
        return Reduced(system, None, "rejected", 2, reason=str(exc),
                       augmentations=[])

    zero = (0,) * n
    polynomial = all(len(d) == 1 and d.get(zero) == 1 for d in den.values())
    kind = "polynomial" if polynomial else "rational"

    degree = 0
    n_mono = 0
    for s in aug.states:
        for mono in num[s]:
            degree = max(degree, sum(mono))
        n_mono += len(num[s])
        if not (len(den[s]) == 1 and den[s].get(zero) == 1):
            n_mono += len(den[s])
    dense = math.comb(n + degree, degree) if n + degree >= 0 else 1
    sparsity = n_mono / float(n * max(dense, 1))
    divisions = ["%s: %s" % (s, _fmt_poly(den[s], aug.states))
                 for s in aug.states
                 if not (len(den[s]) == 1 and den[s].get(zero) == 1)]

    canon = Canonical(
        states=list(aug.states), kind=kind, num=num, den=den,
        degree=degree, n_monomials=n_mono, dense_monomials=dense,
        sparsity=sparsity, divisions=divisions,
        invariants=[str(inv) for inv in aug.invariants])

    domain: dict = {}
    reason = None
    level = 0 if polynomial else 1
    if bessel_args:
        domain = dict(bessel_args=[str(u) for u in bessel_args],
                      bessel_arg_min=bessel_arg_min, bessel_floor=bessel_floor)
        if bessel_arg_min is None:
            reason = ("Bessel 增广需要 Bessel 自变量远离 0 的域证书；"
                      "未给出 bessel_arg_min")
            level = 1
        elif float(bessel_arg_min) <= bessel_floor:
            return Reduced(aug, canon, "rejected", 2,
                           reason="Bessel 自变量下界 %.3g ≤ %.1e：1/x 增广在 x=0 处失效"
                                  % (bessel_arg_min, bessel_floor),
                           augmentations=augs, domain=domain)
    return Reduced(aug, canon, kind, level, reason=reason,
                   augmentations=augs, domain=domain)


# ---------------------------------------------------------------------------
# native 原语成本（增广 vs 原生 tile 的 breakeven 对照）
# ---------------------------------------------------------------------------
@dataclass
class PrimitiveCost:
    """把 exp/tanh 留作原子"原生 tile"（跨导本征律：亚阈 MOS exp、差分对
    tanh）时的 IR 成本，与 reduce_system 的增广路径对照。原子的自变量不求
    导、不进多项式环：它是一次非线性求值，不是 N+1 个状态。"""

    n_states: int
    n_primitives: int
    primitives: list                # ["exp(arg)", ...]
    kind: str                       # polynomial | rational
    degree: int
    n_monomials: int
    dense_monomials: int
    sparsity: float

    def as_dict(self) -> dict:
        return dict(n_states=self.n_states, n_primitives=self.n_primitives,
                    primitives=list(self.primitives), kind=self.kind,
                    degree=self.degree, n_monomials=self.n_monomials,
                    dense_monomials=self.dense_monomials,
                    sparsity=self.sparsity)


def reduce_with_primitives(system: System) -> PrimitiveCost:
    """native_primitives 路径：不改 reduce_system 默认行为，单独计数。

    每个 (去重后的) exp(u)/tanh(u) 原子被替换成一个原子变量 P_k 参与
    polynomial/rational 规范化；P_k 本身的方程 = 一次原生非线性求值，
    不计状态、不计 monomial。degree 沿用 reduce_system 的口径：
    num 恒计，den 仅在非平凡时计，原子变量按 1 次幂计。
    """
    atoms: dict[Expr, str] = {}
    order: list[Expr] = []

    def scan(e: Expr):
        if e.op in ("exp", "tanh") and e not in atoms:
            atoms[e] = ""
            order.append(e)
        for a in e.args:
            if isinstance(a, Expr):
                scan(a)

    for rhs in system.rhs.values():
        scan(rhs)
    names = ["P%d" % (i + 1) for i in range(len(order))]
    for p, nm in zip(order, names):
        atoms[p] = nm

    def sub(e: Expr) -> Expr:
        if e.op in ("exp", "tanh"):
            return V(atoms[e])
        if e.op in ("const", "var"):
            return e
        if e.op == "add":
            return add(*(sub(a) for a in e.args))
        if e.op == "mul":
            return mul(*(sub(a) for a in e.args))
        if e.op == "pow":
            return pow(sub(e.args[0]), e.args[1])
        raise NotImplementedError(e.op)

    idx = {s: i for i, s in enumerate(list(system.states) + names)}
    n = len(idx)
    num: dict[str, Poly] = {}
    den: dict[str, Poly] = {}
    degree = 0
    n_mono = 0
    for s in system.states:
        nm, dn = _poly_frac(sub(system.rhs[s]), idx, n)
        nm, dn = _cancel_monomial(nm, dn)
        if not nm:
            nm, dn = {}, _pconst(Fraction(1), n)
        num[s], den[s] = nm, dn
        for mono in nm:
            degree = max(degree, sum(mono))
        n_mono += len(nm)
        if not (len(dn) == 1 and dn.get((0,) * n) == 1):
            n_mono += len(dn)
            for mono in dn:
                degree = max(degree, sum(mono))
    zero = (0,) * n
    polynomial = all(len(den[s]) == 1 and den[s].get(zero) == 1
                     for s in system.states)
    dense = math.comb(n + degree, degree) if n + degree >= 0 else 1
    return PrimitiveCost(
        n_states=len(system.states), n_primitives=len(order),
        primitives=["%s(%s)" % (p.op, _fmt_expr(p.args[0])) for p in order],
        kind="polynomial" if polynomial else "rational",
        degree=degree, n_monomials=n_mono, dense_monomials=dense,
        sparsity=n_mono / float(n * max(dense, 1)))


# ---------------------------------------------------------------------------
# 数值求值
# ---------------------------------------------------------------------------
def eval_expr(e: Expr, env: dict[str, float]) -> float:
    op = e.op
    if op == "const":
        return float(e.args[0])
    if op == "var":
        return float(env[e.args[0]])
    if op == "add":
        return sum(eval_expr(a, env) for a in e.args)
    if op == "mul":
        out = 1.0
        for a in e.args:
            out *= eval_expr(a, env)
        return out
    if op == "pow":
        return float(eval_expr(e.args[0], env)) ** e.args[1]
    if op == "sin":
        return math.sin(eval_expr(e.args[0], env))
    if op == "cos":
        return math.cos(eval_expr(e.args[0], env))
    if op == "exp":
        return math.exp(eval_expr(e.args[0], env))
    if op == "tanh":
        return math.tanh(eval_expr(e.args[0], env))
    if op == "besselj":
        from scipy.special import jv
        return float(jv(e.args[0], eval_expr(e.args[1], env)))
    if op == "gamma":
        return math.gamma(eval_expr(e.args[0], env))
    raise NotImplementedError(op)


def eval_system(system: System, env: dict[str, float]) -> dict[str, float]:
    return {s: eval_expr(system.rhs[s], env) for s in system.states}


def complete_env(red: Reduced, env: dict[str, float]) -> dict[str, float]:
    """把增广状态的值补进 env（用它们的定义，而不是积分值）。

    增广按创建顺序排列：内层函数的参数先增广，所以单遍即可。
    若 s=sin(u)、c=cos(u)，两者同时补齐；Bessel 补 J_n 与 J_n'。
    """
    out = dict(env)
    for a in red.augmentations:
        u = eval_expr(a["arg"], out)
        kind = a["kind"]
        if kind == "sin":
            out[a["states"][0]] = math.sin(u)
            out[a["states"][1]] = math.cos(u)
        elif kind == "exp":
            out[a["states"][0]] = math.exp(u)
        elif kind == "tanh":
            out[a["states"][0]] = math.tanh(u)
        elif kind == "besselj":
            from scipy.special import jv, jvp
            out[a["states"][0]] = float(jv(a["n"], u))
            out[a["states"][1]] = float(jvp(a["n"], u))
        else:
            raise NotImplementedError(kind)
    return out
