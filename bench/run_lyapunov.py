"""广义 Lyapunov 边缘模态验收：H2 是输入→输出性质（marginal driven-vs-observable）。

    python3 -m bench.run_lyapunov

P0 修复的验收：被驱动的边缘（零/虚轴）模态若在输出不可观（PBH：
rank([λE−A; C]) < n），它在 H(s) 中的极点被零点对消，不得据此判 σ_y=∞。
旧实现对"边缘块被驱动"一刀切判 ∞：E=I、A=diag(−1,0)、B=[1;1]、C=[1,0]
的噪声驱动积分器错报 σ_y=∞，真值 H(s)=1/(s+1)、σ_y²=0.5。

修复方法（analysis/lyapunov.py `_drop_unobservable_marginal`）：PBH 零空间
收集不可观边缘方向 → 正交剔除 → ω 网格运行时核对 H(s) 剔除前后逐点一致
（对消是恒等式；失败——如 Jordan 链部分可观——回落到诚实 ∞）→ 递归。

诚实负对照必须保持 ∞：
  - 驱动且可观的边缘模态（随机游走，H(s) 真有虚轴/原点极点）；
  - index-1 代数馈通（D_total≠0）与 index-2 nilpotent（非真 ⇒ 脉冲）；
  - sys.D 显式馈通。
交叉验证：σ_y² = (1/π)∫₀^∞|H(jω)|²dω 数值积分（与解析/判定独立的第二条路）。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from analysis import Descriptor, solve_generalized_lyapunov

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_LINES: list[str] = []


def emit(s: str = ""):
    print(s)
    _LINES.append(s)


def hr(title: str):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


def sigma_y2_freq(sys: Descriptor, w_max: float = 1e5, npts: int = 20001) -> float:
    """σ_y² = (1/π)∫₀^∞ |H(jω)|² dω 的数值积分（独立参照，仅限 H2 有限的算例）。"""
    w = np.logspace(-5, np.log10(w_max), npts)
    H = sys.freqresp(w)[:, 0, 0]
    return float(np.trapezoid(np.abs(H) ** 2, w) / np.pi)


def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()

    hr("Lyapunov 边缘模态验收：driven-but-unobservable 极点对消")
    checks: list[dict] = []

    def add(name, passed, got, expected, tol=""):
        checks.append(dict(name=name, got=got, expected=expected,
                           log10_err=0.0, tol=tol, passed=bool(passed), kind="lyap"))

    I2 = np.eye(2)

    # ---- P0 复现例：噪声驱动积分器 + 输出不观测它 ----
    A = np.diag([-1.0, 0.0])
    B = np.array([[1.0], [1.0]])
    sys_repro = Descriptor(I2, A, B, np.array([[1.0, 0.0]]))
    r = solve_generalized_lyapunov(sys_repro)
    add("repro sigma_y2 = 1/2 (was inf)", abs(r.sigma_y2 - 0.5) <= 1e-9,
         "%.12g" % r.sigma_y2, "0.5")
    add("repro h2_finite", r.h2_finite, r.method, "True")
    add("repro method reports PBH drop", "marginal-unobservable" in r.method,
         r.method, "PBH drop")
    add("repro n_marginal_dropped = 1", r.n_marginal_dropped == 1,
         r.n_marginal_dropped, "1")
    # 极点对消的直接证据：H(s) = 1/(s+1)，低频 |H| → 1（不发散）、高频衰减。
    H_lo = float(np.abs(sys_repro.freqresp(1e-3)[0, 0]))
    H_hi = float(np.abs(sys_repro.freqresp(1e3)[0, 0]))
    add("repro |H(j 1e-3)| ~ 1 (no pole at 0)", abs(H_lo - 1.0) <= 1e-4,
         "%.6g" % H_lo, "1")
    add("repro |H(j 1e3)| decays", H_hi < 2e-3, "%.3g" % H_hi, "<2e-3")
    sig_int = sigma_y2_freq(sys_repro)
    add("repro freq-integral sigma_y2 = 1/2", abs(sig_int - 0.5) <= 5e-3,
         "%.6g" % sig_int, "0.5")

    # ---- 负对照 1：同一系统 C=[0,1]（积分器被观测）⇒ 仍 ∞ ----
    r_ctl = solve_generalized_lyapunov(Descriptor(I2, A, B, np.array([[0.0, 1.0]])))
    add("control C=[0,1] is inf", (not r_ctl.h2_finite)
        and (r_ctl.sigma_y2 == float("inf")), r_ctl.method, "inf")
    sys_ctl = Descriptor(I2, A, B, np.array([[0.0, 1.0]]))
    H_ctl_lo = float(np.abs(sys_ctl.freqresp(1e-2)[0, 0]))
    add("control |H(j 1e-2)| ~ 100 (real pole at 0)", abs(H_ctl_lo - 100.0) <= 1.0,
         "%.4g" % H_ctl_lo, "100")

    # ---- 负对照 2：驱动且可观的边缘模态 ⇒ ∞（随机游走）----
    A2 = np.array([[-1.0, 1.0], [0.0, 0.0]])
    r_obs = solve_generalized_lyapunov(
        Descriptor(I2, A2, np.array([[1.0], [1.0]]), np.array([[1.0, 0.0]])))
    add("driven+observable marginal is inf", (not r_obs.h2_finite)
        and (r_obs.sigma_y2 == float("inf")), r_obs.method, "inf")

    # ---- 极点对消的 2 状态精确对照（解析 H(s)）----
    # 例 1：输入同时进入稳定行与边缘行，边缘模态不可观 ⇒ 全对消，H ≡ 0。
    r_zero = solve_generalized_lyapunov(
        Descriptor(I2, A2, np.array([[1.0], [1.0]]), np.array([[1.0, -1.0]])))
    add("full-cancellation sigma_y ~ 0 (analytic H=0)", r_zero.sigma_y2 <= 1e-12,
         "%.3g" % r_zero.sigma_y2, "0")
    add("full-cancellation h2_finite", r_zero.h2_finite, r_zero.method, "True")
    # 例 2：耦合后 H = −1/(s+1)，σ_y² = 1/2（对消后仍有稳定极点的贡献）。
    sys_cpl = Descriptor(I2, A2, np.array([[0.0], [1.0]]), np.array([[1.0, -1.0]]))
    r_cpl = solve_generalized_lyapunov(sys_cpl)
    add("coupled sigma_y2 = 1/2 (analytic H=-1/(s+1))",
        abs(r_cpl.sigma_y2 - 0.5) <= 1e-9, "%.12g" % r_cpl.sigma_y2, "0.5")
    sig_cpl = sigma_y2_freq(sys_cpl)
    add("coupled freq-integral agrees", abs(sig_cpl - 0.5) <= 5e-3,
         "%.6g" % sig_cpl, "0.5")
    # ---- 守恒零模态（对称 pencil）：A=[[-1,1],[1,-1]]，C=[1,-1] ⊥ 零特征向量 ----
    # 解析 H = 1/(s+2) ⇒ σ_y² = 1/4。QZ 路径 + PBH 剔除。
    sys_cons = Descriptor(I2, np.array([[-1.0, 1.0], [1.0, -1.0]]),
                          np.array([[1.0], [0.0]]), np.array([[1.0, -1.0]]))
    r_cons = solve_generalized_lyapunov(sys_cons)
    add("conservation zero-mode sigma_y2 = 1/4", abs(r_cons.sigma_y2 - 0.25) <= 1e-9,
         "%.12g" % r_cons.sigma_y2, "0.25")
    sig_cons = sigma_y2_freq(sys_cons)
    add("conservation freq-integral agrees", abs(sig_cons - 0.25) <= 5e-3,
         "%.6g" % sig_cons, "0.25")

    # ---- 3 状态耦合（边缘模态右特征向量带稳定分量，A 非对称、非正规）----
    # A=[[-1,1,0],[0,0,0],[0,3,-2]]：谱 {-1, 0, -2}；零模态左特征向量 e2 ⇒ B=e2 驱动；
    # 零模态右特征向量 (2,2,3) ⊥ C=[1,-1,0] ⇒ 不可观。解析值由频域积分独立给出。
    sys_3 = Descriptor(np.eye(3), np.array([[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                                            [0.0, 3.0, -2.0]]),
                       np.array([[0.0], [1.0], [0.0]]), np.array([[1.0, -1.0, 0.0]]))
    r_3 = solve_generalized_lyapunov(sys_3)
    sig_3 = sigma_y2_freq(sys_3)
    add("3-state coupled finite (was inf)", r_3.h2_finite
        and np.isfinite(r_3.sigma_y2), "%.6g" % r_3.sigma_y2, "finite")
    add("3-state matches freq integral", abs(r_3.sigma_y2 - sig_3) <= 2e-3 * max(sig_3, 1e-30),
         "%.8g vs %.8g" % (r_3.sigma_y2, sig_3), "equal (2e-3)")
    add("3-state method reports drop", "marginal-unobservable" in r_3.method,
         r_3.method, "PBH drop")

    # ---- 不可观的稳定模态不影响 σ_y（direct 路径，回归守卫）----
    r_su = solve_generalized_lyapunov(
        Descriptor(I2, np.diag([-1.0, -2.0]), np.array([[1.0], [1.0]]),
                   np.array([[1.0, 0.0]])))
    add("stable driven-unobservable sigma_y2 = 1/2", abs(r_su.sigma_y2 - 0.5) <= 1e-9,
         "%.12g" % r_su.sigma_y2, "0.5")

    # ---- 虚轴边缘模态（振荡器）被驱动且可观 ⇒ ∞ ----
    r_osc = solve_generalized_lyapunov(
        Descriptor(I2, np.array([[0.0, 1.0], [-1.0, 0.0]]), np.array([[1.0], [0.0]]),
                   np.array([[1.0, 0.0]])))
    add("oscillator (imaginary marginal, observable) is inf",
        (not r_osc.h2_finite) and (r_osc.sigma_y2 == float("inf")), r_osc.method, "inf")

    # ---- 旧正确行为回归守卫：index-1 / index-2 / sys.D / 未驱动边缘 ----
    E = np.diag([1.0, 0.0])
    Ap = np.array([[-1.0, 1.0], [0.0, 1.0]])
    Bp = np.array([[0.0], [1.0]])
    r_idx1 = solve_generalized_lyapunov(Descriptor(E, Ap, Bp, np.array([[1.0, 0.0]])))
    add("index-1 eliminated sigma_y2 = 1/2 (unchanged)",
        abs(r_idx1.sigma_y2 - 0.5) <= 1e-9 and r_idx1.h2_finite,
        "%.12g (%s)" % (r_idx1.sigma_y2, r_idx1.method), "0.5")
    r_idx1n = solve_generalized_lyapunov(Descriptor(E, Ap, Bp, np.array([[0.0, 1.0]])))
    add("index-1 feedthrough control is inf",
        (not r_idx1n.h2_finite) and (r_idx1n.sigma_y2 == float("inf")),
        r_idx1n.method, "inf")
    r_idx2 = solve_generalized_lyapunov(
        Descriptor(np.array([[0.0, 1.0], [0.0, 0.0]]), np.eye(2),
                   np.array([[0.0], [1.0]]), np.array([[1.0, 0.0]])))
    add("index-2 nilpotent is inf",
        (not r_idx2.h2_finite) and (r_idx2.sigma_y2 == float("inf")),
        r_idx2.method, "inf")
    r_d = solve_generalized_lyapunov(
        Descriptor(I2, -np.eye(2), np.ones((2, 1)), np.ones((1, 2)), np.array([[1.0]])))
    add("sys.D=1 feedthrough is inf",
        (not r_d.h2_finite) and (r_d.sigma_y2 == float("inf")), r_d.method, "inf")
    # 未驱动边缘模态：本来就不判 ∞（B 只进稳定行），行为不变。
    r_und = solve_generalized_lyapunov(
        Descriptor(I2, A, np.array([[1.0], [0.0]]), np.array([[1.0, 0.0]])))
    add("undriven marginal still finite", r_und.h2_finite
        and abs(r_und.sigma_y2 - 0.5) <= 1e-9, "%.12g" % r_und.sigma_y2, "0.5")

    # ---- 汇总 ----
    table = [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
             for c in checks]
    hr("验收对照")
    w0 = max(len(t[0]) for t in table)
    for name, got, exp, ok in table:
        emit("  %s  got=%-46s exp=%-10s %s" % (name.ljust(w0), got, exp, ok))
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))
    emit("\n  读法：PBH 剔除是精确最小实现（运行时 H(s) 核对），不是软化阈值；")
    emit("  驱动+可观 ⇒ ∞、馈通 ⇒ ∞、index-2 ⇒ ∞ 三条诚实负对照不变。")

    with open(os.path.join(OUT, "lyapunov.json"), "w") as fh:
        json.dump(dict(checks=checks,
                       n_pass=len(checks) - n_fail, n_total=len(checks)),
                  fh, indent=2, default=float)
    with open(os.path.join(OUT, "lyapunov_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/lyapunov.json")
    emit("  report → results/lyapunov_report.txt")


if __name__ == "__main__":
    main()
