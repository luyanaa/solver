"""无量纲化 + 坐标中心化 + V=Sx 预条件（DESIGN.md §4.0 P0、§10 repo skeleton）。

两个操作，按此顺序施加在 lowering.reduce 的 `System` IR 上：

1.  **center_system**：把工作点平移进坐标。x_i = x̃_i + x0_i 代入每条 rhs，
    状态名保持原名（平移后的系统表示 x̃，offset 记录在返回的 record 里）。
    t0 可选：t = t̃ + t0 同样代入（非自治强迫的相位平移）；time 状态自己的
    方程 ṫ=1 不动——t̃ 仍以单位速率丈量平移后的时间。

2.  **nondimensionalize**：x_i = X_i·x̂_i、t = T0·τ 代入 IR。每条方程

        ẋ_i = f_i(x)  →  x̂_i' = (T0/X_i)·f_i(X·x̂, T0·τ)

    即先把 f_i 的变元重写（var x_j → mul(C(X_j), var x_j)、var t →
    mul(C(T0), var t)），再把整条 rhs 包上 mul(C(T0/X_i), ·)。构造器
    add/mul/pow 自带常数折叠，所以 X=1、T0=1 的状态自动回到恒等。

    **time 语义**（§4.0 口径）：time 状态丈量的是无量纲时间 τ。增强后的
    系统（reduce._augment）给 time 状态 rhs=C(1)；无量纲化后仍是 C(1)——
    新积分变量就是 τ，time 状态跟踪 τ，其他方程里出现的 var t 代表
    t_phys = T0·τ，故替换为 mul(C(T0), var t)。不要把 time 状态当成
    普通坐标去乘 X/T0：时间是自变量本身，不是被缩放的状态。

scaling 是预条件，不是单位换算（DESIGN §4）：V=Sx 不改变特征值/κ，但改变
Γ、瞬态增长和硬件扰动范数。本模块只负责把坐标变成无量纲/居中的；带电压
满摆与频窗的 S_x/alpha 在 lowering.scaling（P3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reduce import (Expr, System, add, C, cos, div, first_order, mul, pow,
                     reduce_system, sub, V)

__all__ = [
    "Normalization", "center_system", "nondimensionalize",
    "pipeline_normalize", "fold",
]


# ---------------------------------------------------------------------------
# 常数折叠 / 重写
# ---------------------------------------------------------------------------
def _subst(e: Expr, var_map: dict[str, Expr]) -> Expr:
    """把 var_map 里的变量替换代入表达式；重建过程经构造器常数折叠。"""
    op = e.op
    if op == "const":
        return C(e.args[0])
    if op == "var":
        return var_map.get(e.args[0], V(e.args[0]))
    if op == "add":
        return add(*(_subst(a, var_map) for a in e.args))
    if op == "mul":
        return mul(*(_subst(a, var_map) for a in e.args))
    if op == "pow":
        return pow(_subst(e.args[0], var_map), e.args[1])
    if op == "besselj":
        return besselj(e.args[0], _subst(e.args[1], var_map))
    if op in ("sin", "cos", "exp", "tanh", "gamma"):
        return Expr(op, (_subst(e.args[0], var_map),))
    raise NotImplementedError(op)


def fold(e: Expr) -> Expr:
    """对纯常数子树做折叠：add(C(2),C(3))→C(5)、mul(C(2),C(3))→C(6)。

    add/mul/pow 构造器本身在"重建"时就做常数合并（acc 累加/累乘），
    所以 fold 只需要走一遍重建；非常数子树结构不变（不展开分配律）。
    """
    return _subst(e, {})


def _rewrite_system(system: System, var_map: dict[str, Expr]) -> System:
    rhs = {s: _subst(system.rhs[s], var_map) for s in system.states}
    invariants = [_subst(inv, var_map) for inv in system.invariants]
    return System(list(system.states), rhs, invariants, system.time,
                  list(system.notes))


# ---------------------------------------------------------------------------
# 1. 中心化
# ---------------------------------------------------------------------------
def center_system(system: System,
                  x0: dict[str, float] | np.ndarray,
                  t0: float | None = None) -> tuple[System, dict]:
    """x_i = x̃_i + x0_i 代入；返回 (平移后的 System, record)。

    record = {"offsets": {state: offset}（仅非零项）, "t0": t0}。
    状态名不变：平移后的系统表示 x̃。time 状态自身不平移（ṫ=1 保留），
    但其他方程里的 var t 代入 t = t̃ + t0。
    """
    states = list(system.states)
    if isinstance(x0, dict):
        offsets = {s: float(x0[s]) for s in states
                   if s in x0 and float(x0[s]) != 0.0}
    else:
        arr = np.asarray(x0, dtype=float).ravel()
        if arr.size != len(states):
            raise ValueError("x0 数组长度必须等于状态数")
        offsets = {s: float(v) for s, v in zip(states, arr) if v != 0.0}

    var_map: dict[str, Expr] = {s: add(V(s), C(off)) for s, off in offsets.items()}

    t_shift = None
    if t0 is not None and system.time is not None:
        t_shift = float(t0)
        # time 状态自己的方程不加 offset（t̃ 仍以速率 1 丈量平移后的时间），
        # 其他方程里的 var t 才代表物理时间 t = t̃ + t0。
        if system.time not in var_map:
            var_map[system.time] = add(V(system.time), C(t_shift))

    out = _rewrite_system(system, var_map)
    record = {"offsets": offsets, "t0": t_shift}
    return out, record


# ---------------------------------------------------------------------------
# 2. 无量纲化
# ---------------------------------------------------------------------------
def _scales_to_array(scales: dict[str, float] | np.ndarray,
                     states: list[str]) -> np.ndarray:
    if isinstance(scales, dict):
        return np.array([float(scales.get(s, 1.0)) for s in states], dtype=float)
    arr = np.asarray(scales, dtype=float).ravel()
    if arr.size != len(states):
        raise ValueError("scales 数组长度必须等于状态数")
    return arr


@dataclass
class Normalization:
    """无量纲化的完整记录：x = X·x̂ + x0，t_phys = T0·τ。"""

    state_names: list[str]
    X: np.ndarray                  # 每状态单位尺度（无量纲坐标的单位）
    T0: float                      # 时间尺度：t_phys = T0 · τ
    dim_groups: dict[str, float] = field(default_factory=dict)
    system_states: list[str] = field(default_factory=list)
    x0: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def physical_to_model(self, x) -> np.ndarray:
        """物理坐标 → 无量纲坐标：x̂ = (x − x0) / X。"""
        x = np.asarray(x, dtype=float)
        return (x - self.x0) / self.X

    def model_to_physical(self, x_hat) -> np.ndarray:
        """无量纲坐标 → 物理坐标：x = X·x̂ + x0。"""
        x_hat = np.asarray(x_hat, dtype=float)
        return self.X * x_hat + self.x0

    def as_dict(self) -> dict:
        return dict(
            state_names=list(self.state_names),
            X=[float(v) for v in self.X],
            T0=float(self.T0),
            dim_groups=dict(self.dim_groups),
            system_states=list(self.system_states),
            x0=[float(v) for v in self.x0],
        )


def nondimensionalize(system: System,
                      scales: dict[str, float] | np.ndarray,
                      t_scale: float = 1.0,
                      dim_groups: dict[str, float] | None = None,
                      ) -> tuple[System, Normalization]:
    """x_i = X_i·x̂_i、t = T0·τ 代入；返回 (无量纲 System, Normalization)。

    time 状态（若在 states 里）的 rhs 保持 C(1)：它丈量 τ 本身；其他方程
    里的 var t 替换为 mul(C(T0), var t)（t_phys = T0·τ）。
    """
    states = list(system.states)
    X = _scales_to_array(scales, states)
    if np.any(X <= 0):
        raise ValueError("单位尺度必须为正")
    T0 = float(t_scale)
    if T0 <= 0:
        raise ValueError("t_scale 必须为正")

    x_map = {s: mul(C(xv), V(s)) for s, xv in zip(states, X) if xv != 1.0}
    # time 不是普通坐标：唯一处理是 t_phys = T0·τ。
    if system.time is not None and T0 != 1.0:
        x_map[system.time] = mul(C(T0), V(system.time))

    out = _rewrite_system(system, x_map)
    # time 状态的方程不包 (T0/X) 因子：它丈量 τ（rhs 通常已是 C(1)）。
    if system.time is not None and system.time in states:
        pass  # var_map 对 rhs[time] 只做了常数替换（C(1) 不变），已正确。

    wrapped: dict[str, Expr] = {}
    for i, s in enumerate(states):
        if system.time is not None and s == system.time:
            wrapped[s] = out.rhs[s]          # rhs[time] = C(1) 保持
        else:
            wrapped[s] = mul(C(T0 / X[i]), out.rhs[s])
    out = System(states, wrapped,
                 [_subst(inv, x_map) for inv in system.invariants],
                 system.time, list(system.notes))

    norm = Normalization(
        state_names=states, X=X.copy(), T0=T0,
        dim_groups=dict(dim_groups) if dim_groups else {},
        system_states=states,
        x0=np.zeros(len(states)),
    )
    return out, norm


# ---------------------------------------------------------------------------
# 组合：center → nondim
# ---------------------------------------------------------------------------
def pipeline_normalize(system: System,
                       x0: dict[str, float] | np.ndarray | None = None,
                       scales: dict[str, float] | np.ndarray | None = None,
                       t_scale: float | None = None,
                       ) -> tuple[System, Normalization]:
    """center_system → nondimensionalize 的组合入口（P0）。

    默认 x0=0（不平移）、X=1、T0=1（恒等）。返回的 Normalization 把
    两步的偏移/尺度合成一份：x = X·x̂ + x0, t_phys = T0·τ。
    """
    states = list(system.states)
    x0_arr = np.zeros(len(states))
    shifted = system
    t0_used = None
    if x0 is not None:
        shifted, record = center_system(system, x0)
        x0_arr = np.array([record["offsets"].get(s, 0.0) for s in states],
                          dtype=float)
        t0_used = record["t0"]

    X = _scales_to_array(scales if scales is not None else {}, states)
    T0 = float(t_scale) if t_scale is not None else 1.0

    normed, norm = nondimensionalize(shifted, X, T0)
    norm.x0 = x0_arr
    if t0_used is not None:
        norm.dim_groups = dict(norm.dim_groups, t0=t0_used)
    return normed, norm


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def selftest() -> dict:
    """快速验证：Duffing 平移+无量纲化的逐点等价；常数 rhs 系统的往返。"""
    import math

    from .reduce import eval_system

    # --- Duffing: x'' = -δ x' - α x - β x³ + γ cos(ω t) ---
    dlt, alpha_c, beta_c, gamma_c, om = 0.3, 1.0, 2.0, 0.5, 1.7
    rhs = add(mul(C(-dlt), V("x_1")),
              mul(C(-alpha_c), V("x_0")),
              mul(C(-beta_c), pow(V("x_0"), 3)),
              mul(C(gamma_c), cos(mul(C(om), V("t")))))
    sys0 = first_order("x", 2, rhs)

    x0 = {"x_0": 0.7}
    X = {"x_0": 2.0, "x_1": 3.0}
    T0 = 0.5
    centered, rec = center_system(sys0, x0)
    assert rec["offsets"] == {"x_0": 0.7}, rec
    assert rec["t0"] is None
    # 中心化逐点检查：f_shift(x̃) == f_orig(x̃ + x0)
    xh = {"x_0": 0.4, "x_1": -0.2, "t": 1.1}
    env_o = {"x_0": 0.4 + 0.7, "x_1": -0.2, "t": 1.1}
    f0 = eval_system(sys0, env_o)
    fc = eval_system(centered, xh)
    assert abs(f0["x_0"] - fc["x_0"]) < 1e-13
    assert abs(f0["x_1"] - fc["x_1"]) < 1e-13

    normed, norm = nondimensionalize(centered, X, T0)
    assert isinstance(norm, Normalization)
    # 逐点检查：f_norm(x̂) == (T0/X_i)·f_center(X·x̂, T0·τ)
    xh = {"x_0": 0.3, "x_1": -0.5, "t": 2.0}
    env_c = {"x_0": X["x_0"] * 0.3, "x_1": X["x_1"] * (-0.5), "t": T0 * 2.0}
    fc = eval_system(centered, env_c)
    fn = eval_system(normed, xh)
    for s, i in (("x_0", 0), ("x_1", 1)):
        want = (T0 / X[s]) * fc[s]
        assert abs(fn[s] - want) < 1e-12 * max(1.0, abs(want)), (s, fn[s], want)

    # pipeline（一步到位，含 x0）与两步结果一致
    piped, pnorm = pipeline_normalize(sys0, x0=x0, scales=X, t_scale=T0)
    assert abs(eval_system(piped, xh)["x_1"] - fn["x_1"]) < 1e-14
    assert np.allclose(pnorm.x0, [0.7, 0.0])
    assert np.allclose(pnorm.X, [2.0, 3.0]) and pnorm.T0 == 0.5
    # physical ↔ model 往返
    xphys = np.array([1.4, -0.9])
    assert np.allclose(pnorm.model_to_physical(pnorm.physical_to_model(xphys)),
                       xphys, atol=1e-14)

    # time 语义之一：增强后 cos(ωt) 被吸收进 (s,c) 对，t 状态不出现——
    # 但 s/c 方程的速率（模型时间单位）必须被 T0 缩放。
    red0 = reduce_system(sys0)
    aug = red0.system
    assert aug.time == "t" and aug.time not in aug.states
    s1, c1 = red0.augmentations[0]["states"]
    aug_n, anorm = nondimensionalize(aug, X, T0)
    # ds1/dt = ω·c1（模型时间）→ ds1/dτ = T0·ω·c1；在 τ=2 数值核对
    fn2 = eval_system(aug_n, {"x_0": 0.3, "x_1": -0.5, "t": 2.0,
                              s1: math.sin(om), c1: math.cos(om)})
    assert abs(fn2[s1] - T0 * om * math.cos(om)) < 1e-12
    # x_1 方程整体也核对：(T0/X)·f_aug(X·x̂, 同一时刻)
    fc2 = eval_system(aug, {"x_0": 2.0 * 0.3, "x_1": 3.0 * (-0.5), "t": 1.0,
                            s1: math.sin(om), c1: math.cos(om)})
    want2 = (T0 / X["x_1"]) * fc2["x_1"]
    assert abs(fn2["x_1"] - want2) < 1e-12 * max(1.0, abs(want2))

    # time 语义之二：time 已是状态（rhs=C(1)）——无量纲化后它丈量 τ，
    # rhs 保持 1，其他方程里的 var t 代表 t_phys = T0·τ。
    sys_t = System(["x", "t"], {"x": mul(C(2), V("t")), "t": C(1)}, time="t")
    nd_t, _ = nondimensionalize(sys_t, {"x": 2.0}, 0.5)
    assert str(nd_t.rhs["t"]) == "1", nd_t.rhs["t"]
    # dx/dt = 2t ⇒ x = t²；x̂ = x/2, τ = 2t ⇒ x̂ = 2τ²，x̂' = 4τ。
    v = eval_system(nd_t, {"x": 3.0, "t": 4.0})
    assert abs(v["x"] - (0.5 / 2.0) * 2 * (0.5 * 4.0)) < 1e-13, v
    assert v["t"] == 1.0

    # --- 常数 rhs 系统：dx=2, dy=y+x，带 x0 平移 ---
    sys2 = System(["x", "y"], {"x": C(2), "y": add(V("y"), V("x"))},
                  time=None)
    sh2, rec2 = center_system(sys2, {"x": 1.0, "y": -0.5})
    assert rec2["offsets"] == {"x": 1.0, "y": -0.5}
    # f_shift(x̃) = (2, (ỹ-0.5)+(x̃+1)) = (2, x̃+ỹ+0.5)
    fv = eval_system(sh2, {"x": 0.3, "y": 0.8})
    assert fv["x"] == 2.0 and abs(fv["y"] - (0.3 + 0.8 + 0.5)) < 1e-14
    # 往返：平移后的系统在 x̃=x−x0 处的值 == 原系统在 x 处的值
    xphys = {"x": 1.3, "y": 0.3}
    f_orig = eval_system(sys2, xphys)
    f_sh = eval_system(sh2, {"x": 1.3 - 1.0, "y": 0.3 + 0.5})
    assert abs(f_orig["x"] - f_sh["x"]) < 1e-14
    assert abs(f_orig["y"] - f_sh["y"]) < 1e-14
    # 无量纲化后同样往返
    nd2, n2 = nondimensionalize(sh2, {"x": 2.0, "y": 4.0}, 1.0)
    n2.x0 = np.array([1.0, -0.5])
    f_nd = eval_system(nd2, {"x": (1.3 - 1.0) / 2.0, "y": (0.3 + 0.5) / 4.0})
    # x̂_i' = (T0/X_i)·f_orig
    assert abs(f_nd["x"] - (1.0 / 2.0) * f_orig["x"]) < 1e-14
    assert abs(f_nd["y"] - (1.0 / 4.0) * f_orig["y"]) < 1e-14

    # fold：纯常数树
    assert str(fold(add(C(2), C(3)))) == "5"
    assert str(fold(mul(C(2), C(3)))) == "6"
    assert str(fold(div(C(1), C(4)))) == "1/4"
    assert str(fold(sub(C(2), C(5)))) == "-3"
    assert str(fold(pow(C(2), 3))) == "8"
    assert str(fold(add(V("x"), C(0)))) == "x"
    assert str(fold(mul(V("x"), C(1)))) == "x"
    assert str(fold(mul(V("x"), C(0)))) == "0"

    # 恒等：X=1、T0=1、无 x0 → rhs 不变
    same, nid = pipeline_normalize(sys2)
    assert np.allclose(nid.X, 1.0) and nid.T0 == 1.0
    assert all(str(same.rhs[s]) == str(sys2.rhs[s]) for s in sys2.states)

    return dict(ok=True, tests=8)
