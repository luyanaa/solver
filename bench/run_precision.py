"""§6.24 精度作为编译优化变量验收：灵敏度位宽分配 / 能耗 / 链路 / 约束。

示例（DESIGN.md §6.24）：5 条边界相对灵敏度 1 / 0.515 / 0.113 / 0.0311 /
0.011，max_bits=16、min_bits=10 ⇒ 16/15/13/11/10 = 65 bit；
统一 max-bit 基线 5×16=80 bit。差分方案在 ε=1e-4 的量化上界内。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.precision import (            # noqa: E402
    allocate_bits, energy_model, optimize_precision, precision_plan,
    uniform_bits,
)

_CHECKS: list[dict] = []
_LINES: list[str] = []


def emit(s: str) -> None:
    _LINES.append(s)
    print(s)


def add_check(name: str, detail, ok: bool) -> None:
    _CHECKS.append(dict(name=name, detail=str(detail), ok=bool(ok)))
    emit("  [%s] %s   %s" % ("PASS" if ok else "FAIL", name, detail))


K5 = [1.0, 0.515, 0.113, 0.0311, 0.011]
EPS = 1e-4


def case_bits():
    emit("[case_bits]  灵敏度位宽分配（示例 16/15/13/11/10）")
    bp = allocate_bits(K5, EPS, max_bits=16, min_bits=10, anchor="max")
    add_check("P1 位宽表 = %s（总 %d bit）；统一 max-bit 基线 %d bit"
              % (bp.bits, bp.total_bits, 5 * 16),
              "65 vs 80", bp.bits == [16, 15, 13, 11, 10]
              and bp.total_bits == 65)
    add_check("P2 量化上界 ΣK_i/2^{b_i+1}=%.3e ≤ ε=%.1e（可行的诚实口径）"
              % (bp.achieved, EPS), "no over-promise",
              bp.feasible and bp.achieved <= EPS)
    ub = uniform_bits(K5, EPS, max_bits=16)
    add_check("P3 均匀对照：满足 ε 的最小均匀 %d×%d=%d bit；与差分等界 "
              "%d×%d=%d bit；max-bit=%d"
              % (ub["uniform_for_eps"], 5, ub["total_for_eps"],
                 ub["uniform_at_equal_bound"], 5,
                 ub["total_at_equal_bound"], ub["total_at_max"]),
              "65 < 70/75/80",
              bp.total_bits < ub["total_for_eps"]
              and bp.total_bits < ub["total_at_max"])
    bp2 = allocate_bits([0.1, 1.0, 10.0], EPS, max_bits=16, min_bits=4,
                        anchor="max")
    add_check("P4 单调性：K=(0.1,1,10) ⇒ b=%s（灵敏度高的位宽不低）"
              % bp2.bits, "monotone", bp2.bits == sorted(bp2.bits))
    bp3 = allocate_bits(K5, 1e-6, max_bits=16, min_bits=10, anchor="max")
    add_check("P5 ε 收紧到 1e-6：achieved=%.2e > ε ⇒ feasible=%s（不许诺）"
              % (bp3.achieved, bp3.feasible), "flag, not promise",
              (not bp3.feasible))
    bp4 = allocate_bits(K5, EPS, max_bits=16, min_bits=10, anchor="budget")
    add_check("P6 budget 锚：b=%s（总 %d），achieved=%.2e ≤ ε"
              % (bp4.bits, bp4.total_bits, bp4.achieved), "solve from ε",
              bp4.feasible and bp4.achieved <= EPS)
    return dict(bits=bp.bits, total=bp.total_bits, achieved=bp.achieved,
                uniform=ub)


def case_energy():
    emit("[case_energy]  能耗与可变精度")
    bp = allocate_bits(K5, EPS, max_bits=16, min_bits=10)
    vr = energy_model(bp.bits, 1.0, variable_resolution=True)
    fx = energy_model(bp.bits, 1.0, variable_resolution=False)
    uni = energy_model([16] * 5, 1.0, variable_resolution=True)
    ratio_vr = vr["e_total"] / uni["e_total"]
    ratio_fx = fx["e_total"] / uni["e_total"]
    add_check("E1 可变精度转换器：差分/统一能耗 = %.3f（省 %.1f×）"
              % (ratio_vr, 1.0 / ratio_vr), "Walden E∝2^b",
              0.2 < ratio_vr < 0.4)
    add_check("E2 固定精度转换器：比值 = %.3f（不省电，只省链路）"
              % ratio_fx, "honest: no phantom saving",
              abs(ratio_fx - 1.0) < 1e-12)
    add_check("E3 位宽和 %.0f%% of 80 ⇒ 采样能耗同比例（%s）"
              % (100 * bp.total_bits / 80,
                 "是" if ratio_vr < 0.4 else "否"),
              "energy tracks bits", vr["e_conv"] < uni["e_conv"])
    return dict(ratio_variable=ratio_vr, ratio_fixed=ratio_fx)


def case_constraints():
    emit("[case_constraints]  约束：deadline / 链路容量 / 多目标")
    T = 1e-3
    p = precision_plan(K5, EPS, fs=1.0 / T, max_bits=16, min_bits=10,
                       link_capacity_bps=70e3)
    add_check("L1 deadline 采样：fs=%.0f Hz，链路负载 %.0f bps ≤ 70 kbps ⇒ "
              "可行=%s" % (p["fs"], p["link_load_bps"], p["feasible"]),
              "65 kbps @ 1 kHz", p["feasible"]
              and abs(p["link_load_bps"] - 65e3) < 1e-6 * 65e3 + 1e-9)
    p2 = precision_plan(K5, EPS, fs=1.0 / T, max_bits=16, min_bits=10,
                        link_capacity_bps=50.0)
    add_check("L2 链路容量 50 kbps < 65 kbit/s ⇒ infeasible（原因：%s）"
              % (p2["reasons"][-1] if p2["reasons"] else "none"),
              "capacity constraint", not p2["feasible"])
    opt = optimize_precision(K5, EPS, T, link_capacity_bps=200e3)
    add_check("L3 最小能耗搜索：fs=%.0f Hz（=1/T）为最优，E=%.3g"
              % (opt["plan"]["fs"], opt["plan"]["energy"]["e_total"]),
              "fs monotone in energy", opt["feasible"]
              and abs(opt["plan"]["fs"] - 1.0 / T) < 1e-12)
    opt2 = optimize_precision(K5, EPS, T, link_capacity_bps=30e3)
    add_check("L4 容量 30 kbps 与 deadline 冲突 ⇒ 无可行点（%s）"
              % opt2["reasons"][0], "no silent promise", not opt2["feasible"])
    opt3 = optimize_precision([1.0] * 5, 1e-6, 2e-3)
    add_check("L5 ε=1e-6、max=16 ⇒ achieved=%.2e，无可行点 ⇒ infeasible"
              % opt3["tried"][0]["energy"],
              "budget vs bit limit", not opt3["feasible"])
    return dict(load=p["link_load_bps"], opt_fs=opt["plan"]["fs"])


def case_quantization():
    emit("[case_quantization]  量化蒙特卡洛 vs 确定性上界")
    bp = allocate_bits(K5, EPS, max_bits=16, min_bits=10)
    rng = np.random.default_rng(11)
    n = 5000
    K = np.array(K5)
    q = 2.0 ** (-(np.array(bp.bits) + 1.0))
    a = K * q
    dJ = rng.uniform(-1, 1, size=(n, 5)) @ a
    add_check("Q1 5000 次量化抽样 max|δJ|=%.3e ≤ 上界 ΣK_i/2^{b_i+1}=%.3e"
              % (np.abs(dJ).max(), bp.achieved), "bound holds",
              np.abs(dJ).max() <= bp.achieved + 1e-15)
    # 独立均匀误差：Var(δJ)=Σa_i²/3，E|δJ|=sqrt(2/π)·sqrt(Σa_i²/3)
    pred_mean = np.sqrt(2.0 / np.pi) * np.sqrt(np.sum(a ** 2) / 3.0)
    add_check("Q2 抽样均值 %.3e ≈ sqrt(2/π)·sqrt(Σa²/3)=%.3e（独立均匀误差）"
              % (np.abs(dJ).mean(), pred_mean), "consistency",
              abs(np.abs(dJ).mean() - pred_mean) / pred_mean < 0.05)
    # 等预算下的位宽再分配：把最小灵敏度边界的 2 bit 挪给最大边界
    alt = list(bp.bits)
    alt[-1] -= 2
    alt[0] += 2
    alt_b = float(np.sum(K / 2.0 ** (np.array(alt) + 1)))
    add_check("Q3 反向再分配（从 K=0.011 挪 2 bit 给 K=1）⇒ 上界 %.3e > %.3e"
              "（灵敏度排序不可颠倒）" % (alt_b, bp.achieved),
              "allocation must follow K", alt_b > bp.achieved)
    return dict(mc_max=float(dJ.max()), bound=bp.achieved,
                alt_bound=alt_b)


def main():
    emit("=" * 72)
    emit("§6.24 精度作为编译优化变量验收")
    emit("=" * 72)
    bt = case_bits()
    emit("")
    en = case_energy()
    emit("")
    lc = case_constraints()
    emit("")
    qz = case_quantization()
    n_pass = sum(1 for c in _CHECKS if c["ok"])
    emit("")
    emit("%d/%d checks passed" % (n_pass, len(_CHECKS)))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "precision_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    import json
    payload = dict(kind="precision",
                   meta=dict(bench="bench.run_precision",
                             section="DESIGN.md §6.24",
                             model="sensitivity-proportional bit allocation"),
                   bits=bt, energy=en, constraints=lc, quantization=qz,
                   checks=_CHECKS)
    with open(os.path.join(out_dir, "precision.json"), "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return n_pass == len(_CHECKS)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
