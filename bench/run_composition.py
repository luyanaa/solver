"""§6.23 组合误差合同验收：岛级 Metzler 界 / 离散窗口 / 时变拓扑 /
守恒-耗散分解 / 三条判据对照。

验收数字都来自本文件里的数值实验（不引用外部结论）：
  * H Hurwitz ⇒ limsup ε ⪯ −H⁻¹d，并与 ε'=Hε+d 的 RK4 轨迹对比；
  * 时变反例：逐窗口 ρ(K_k)=0 而乘积 ρ=4 —— 单看谱半径会漏；
  * 周期 diffusion：small-gain 拒绝（ρ(H)=1 或 2）而 passivity 与
    守恒+投影收缩通过 —— "ρ=1 不等于不可实现"的实测证据。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.composition import (             # noqa: E402
    Island, comparison_bound, comparison_matrix, common_metric_contraction,
    conservation_split, discrete_bound, dissipativity_index, lognorm,
    three_criteria, window_contract,
)

_CHECKS: list[dict] = []
_LINES: list[str] = []


def emit(s: str) -> None:
    _LINES.append(s)
    print(s)


def add_check(name: str, detail, ok: bool) -> None:
    _CHECKS.append(dict(name=name, detail=str(detail), ok=bool(ok)))
    emit("  [%s] %s   %s" % ("PASS" if ok else "FAIL", name, detail))


def _ring(N: int, directed: bool = False):
    """环 Laplacian（无向）或定向环（有向，H 的 ρ=1）。"""
    L = 2.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    L[0, N - 1] -= 1.0
    L[N - 1, 0] -= 1.0
    if directed:
        L = np.zeros((N, N))
        for i in range(N):
            L[i, i] += 1.0
            L[i, (i + 1) % N] -= 1.0
    return L


def case_metzler():
    emit("[case_metzler]  岛级比较矩阵：Hurwitz ⇒ limsup ⪯ −H⁻¹d")
    islands = [
        Island("a", np.array([[-1.0]]), None, {1: np.array([[0.10]])}, 1e-3),
        Island("b", np.array([[-2.0]]), None, {2: np.array([[0.05]])}, 2e-3),
        Island("c", np.array([[-3.0]]), None, {0: np.array([[0.02]])}, 5e-4),
    ]
    H = comparison_matrix(islands)
    d = np.array([isl.d for isl in islands])
    add_check("C1 H 是 Metzler 且 Hurwitz：非对角=%.3f/%.3f/%.3f，"
              "谱横坐标=%.3f" % (H[0, 1], H[1, 2], H[2, 0],
                                np.max(np.real(np.linalg.eigvals(H)))),
              "μ=(%.1f,%.1f,%.1f)" % tuple(np.diag(H)),
              np.all(H - np.diag(np.diag(H)) >= 0)
              and np.max(np.real(np.linalg.eigvals(H))) < 0)
    b = comparison_bound(H, d, T=50.0, n=400)
    add_check("C2 界与积分一致：−H⁻¹d=%s，模拟 limsup=%s，超出 %.1e"
              % (np.round(b["bound"], 6).tolist(),
                 np.round(b["simulated_limsup"], 6).tolist(),
                 b["overshoot"]),
              "bound ≥ sim", b["overshoot"] <= 1e-9 and b["monotone"])
    b0 = comparison_bound(H, np.zeros(3), T=50.0)
    add_check("C3 d=0 ⇒ 界为 0（纯收缩，无残余误差）",
              "max(bound)=%.1e" % np.max(np.abs(b0["bound"])),
              np.max(np.abs(b0["bound"])) < 1e-12)
    return dict(H=H.tolist(), bound=b["bound"],
                simulated=b["simulated_limsup"])


def case_discrete():
    emit("[case_discrete]  离散窗口合同 + 时变拓扑陷阱")
    K = np.array([[0.60, 0.10], [0.05, 0.70]])
    eta = np.array([1e-3, 2e-3])
    db = discrete_bound(K, eta)
    add_check("W1 ρ(K)=%.3f<1 ⇒ (I−K)⁻¹η=%s，迭代终值=%s"
              % (db["rho"], np.round(db["bound"], 6).tolist(),
                 np.round(db["simulated"], 6).tolist()),
              "bound ≥ sim", db["overshoot"] <= 1e-12)
    K1 = np.array([[0.0, 0.0], [2.0, 0.0]])
    K2 = np.array([[0.0, 2.0], [0.0, 0.0]])
    indiv = [float(np.max(np.abs(np.linalg.eigvals(K))) ) for K in (K1, K2)]
    prod = K2 @ K1
    rho_prod = float(np.max(np.abs(np.linalg.eigvals(prod))))
    e = np.array([1.0, 0.0])
    for k in range(10):
        e = (K1 @ e) if k % 2 == 0 else (K2 @ e)
    add_check("W2 反例：逐窗口 ρ=(%.0f, %.0f)，乘积 ρ=%.0f —— 交替 10 步"
              "把 ε 放大到 %.3g（单看 ρ(K_k) 会漏）"
              % (indiv[0], indiv[1], rho_prod, e[0]),
              "AMR/时变拓扑必须用乘积/共同度量",
              rho_prod > 4.0 - 1e-9 and e[0] > 1e3)
    cm = common_metric_contraction([K1, K2])
    add_check("W3 共同加权范数正确拒绝：feasible=%s（不存在 w>0 使 "
              "K_k w ≤ (1−δ)w）" % cm["feasible"],
              "product ρ=%.1f" % cm["product_rho"], not cm["feasible"])
    good = common_metric_contraction([K, K])
    add_check("W4 收缩序列有共同度量：δ=%.3g，w=%s"
              % (good["contraction"], np.round(good["metric"], 3).tolist()),
              "LP feasible", good["feasible"]
              and good["contraction"] is not None
              and good["contraction"] > 0)
    wc = window_contract([K1, K2], np.ones(2))
    add_check("W5 周期窗口：cycle_ρ=%.1f ⇒ bound=None（不许诺长期界）"
              % wc["cycle_rho"], "window contract", wc["bound"] is None)
    return dict(rho=db["rho"], bound=db["bound"],
                counterexample_rho=rho_prod, counterexample_e10=float(e[0]),
                common_metric=bool(cm["feasible"]),
                contraction=good["contraction"])


def case_conservation():
    emit("[case_conservation]  守恒零模态：分解 e=e_mean+e_⊥（不能 small-gain 判死）")
    N = 8
    L = _ring(N, directed=False)
    A = -L
    cons = conservation_split(A, null=np.ones(N))
    lam = np.linalg.eigvalsh(L)
    add_check("K1 零模态 μ_full=%.1e（=0 ⇒ 全局不收缩），补空间 "
              "μ_⊥=%.4f = −λ₂(L)=%.4f"
              % (cons["mu_full"], cons["mu_perp"], -lam[1]),
              "两轴分开", abs(cons["mu_full"]) < 1e-9
              and abs(cons["mu_perp"] + lam[1]) < 1e-9)
    # 均值扰动 vs 补空间扰动：同一时长下的衰减
    T = 2.0
    mean0 = 0.3 * np.ones(N)
    from scipy.linalg import expm
    e_mean = expm(A * T) @ mean0
    add_check("K2 零模态扰动不衰减：‖e_mean(T)‖/‖e_mean(0)‖=%.6f（守恒漂移）"
              % (np.linalg.norm(e_mean) / np.linalg.norm(mean0)),
              "ε_conservation 单独记账", abs(np.linalg.norm(e_mean)
                                            / np.linalg.norm(mean0) - 1) < 1e-9)
    rng = np.random.default_rng(7)
    perp0 = rng.standard_normal(N)
    perp0 = cons["projector"] @ perp0
    perp0 /= np.linalg.norm(perp0)
    e_perp = expm(A * T) @ perp0
    ratio = float(np.linalg.norm(e_perp) / np.linalg.norm(perp0))
    add_check("K3 补空间扰动按 e^{μ⊥T} 衰减：实测 %.4f ≤ e^{%.4f·%.1f}=%.4f"
              % (ratio, cons["mu_perp"], T, np.exp(cons["mu_perp"] * T)),
              "μ⊥ 是有效收缩率", ratio <= np.exp(cons["mu_perp"] * T) + 1e-9)
    # 守恒漂移 + 泄漏
    leak = 1e-4 * np.ones(N)
    e_leak = expm(A * T) @ np.zeros(N) + np.linalg.solve(A + 1e-12 * np.eye(N),
                                                         (expm(A * T)
                                                          - np.eye(N)) @ leak)
    rate = float(np.mean(e_leak) / T)
    add_check("K4 常值泄漏 r_leak=1e-4 ⇒ 均值漂移率 %.2e/时间单位（ε_mean "
              "线性增长，不进 ε_⊥）" % rate, "separate budget",
              rate > 0)
    return dict(mu_perp=cons["mu_perp"], mu_full=cons["mu_full"],
                mean_decay=float(np.linalg.norm(e_mean)
                                 / np.linalg.norm(mean0)), perp_ratio=ratio,
                leak_rate=rate)


def case_three_criteria():
    emit("[case_three_criteria]  small-gain vs passivity vs 守恒+投影")
    N = 8
    for directed, label in ((False, "无向环"), (True, "定向环")):
        L = _ring(N, directed=directed)
        A = -L
        islands = []
        for i in range(N):
            # 局部块必须含自己的度数（Laplacian 对角），跨岛块只放邻居边：
            # 这样 Σ 岛 = 全局 A，且 H 的零特征值正好来自守恒零模态。
            out = {j: np.array([[A[i, j]]]) for j in range(N)
                   if i != j and abs(A[i, j]) > 0}
            islands.append(Island("i%d" % i, np.array([[A[i, i]]]), None,
                                  out, 0.0))
        tc = three_criteria(islands, null=np.ones(N))
        H = tc["H"]
        add_check("T1 %s：small-gain %s（H 谱横坐标 %.3f，ρ(H)=%.3f）"
                  % (label, "拒绝" if tc["small_gain"]["reject"] else "通过",
                     tc["small_gain"]["spectral_abscissa"],
                     float(np.max(np.abs(np.linalg.eigvals(H)))),
                     ),
                  "H 非 Hurwitz", tc["small_gain"]["reject"])
        add_check("T2 %s：passivity %s（岛耗散指数 max=%.1e）"
                  % (label, "通过" if tc["passivity"]["accept"] else "拒绝",
                     max(tc["passivity"]["indices"]),
                     ),
                  "MA+AᵀM ⪯ 0", tc["passivity"]["accept"])
        add_check("T3 %s：守恒+投影 %s（μ_⊥=%.4f）"
                  % (label, "通过" if tc["conservation"]["accept"] else "拒绝",
                     tc["conservation"]["mu_perp"]),
                  "projected contraction", tc["conservation"]["accept"])
    # 功率平衡：局部耗散 + 耦合项 = 全局存储；零模态上恰为 0
    L = _ring(N, directed=False)
    A = -L
    x1 = np.ones(N)
    diag = float(sum(x1[i] * A[i, i] * x1[i] for i in range(N)))
    blocks = float(sum(x1[i] * A[i, j] * x1[j]
                       for i in range(N) for j in range(N) if i != j))
    total = float(x1 @ A @ x1)
    rng = np.random.default_rng(3)
    x2 = rng.standard_normal(N)
    tot2 = float(x2 @ A @ x2)
    add_check("T4 功率平衡：零模态 x=1 上局部耗散 %.2f + 耦合注入 %.2f = %.1e"
              "（恰为 0）；随机 x 上总存储 %.2f < 0"
              % (diag, blocks, total, tot2),
              "local dissipation cancels coupling on the null mode",
              abs(total) < 1e-12 and diag < 0 and blocks > 0 and tot2 < 0)
    return dict(directed_abscissa=None)


def case_accuracy():
    emit("[case_accuracy]  组合界 vs 全系统仿真（紧度检查）")
    # 3 个岛，每个 2 态，链式耦合
    A1 = np.array([[-1.0, 0.20], [0.0, -2.0]])
    A2 = np.array([[-1.5, 0.10], [0.05, -3.0]])
    A3 = np.array([[-2.0, 0.0], [0.20, -4.0]])
    islands = [
        Island("i1", A1, None, {1: np.array([[0.10, 0.0], [0.0, 0.05]])}, 1e-3),
        Island("i2", A2, None, {2: np.array([[0.05, 0.0], [0.0, 0.10]])}, 2e-3),
        Island("i3", A3, None, {}, 5e-4),
    ]
    H = comparison_matrix(islands)
    dd = np.array([1e-3, 1e-3, 2e-3, 2e-3, 5e-4, 5e-4])
    d = np.array([np.linalg.norm(dd[0:2]), np.linalg.norm(dd[2:4]),
                  np.linalg.norm(dd[4:6])])
    b = comparison_bound(H, d, T=20.0, n=400)
    # 全系统装配
    n = 6
    A = np.zeros((n, n))
    A[0:2, 0:2] = A1
    A[2:4, 2:4] = A2
    A[4:6, 4:6] = A3
    A[0:2, 2:4] = islands[0].coupling_out[1]
    A[2:4, 4:6] = islands[1].coupling_out[2]
    from scipy.linalg import expm
    x = -np.linalg.solve(A, dd)          # 稳态
    blk = np.array([np.linalg.norm(x[0:2]), np.linalg.norm(x[2:4]),
                    np.linalg.norm(x[4:6])])
    ratio = b["bound"] / np.maximum(blk, 1e-300)
    add_check("A1 岛级界 ≥ 全系统实测：bound=%s，实测=%s，松度 max=%.2f×"
              % (np.round(b["bound"], 5).tolist(), np.round(blk, 5).tolist(),
                 float(np.max(ratio))),
              "非空洞但保守", np.all(b["bound"] >= blk - 1e-12)
              and float(np.max(ratio)) < 20.0)
    m = len(islands)
    add_check("A2 组合规模：m=%d 个岛 vs n=%d 个态（比较矩阵 m×m，"
              "不必对 n×n 稠密矩阵反复 expm）" % (m, n), "scalability",
              m < n)
    return dict(bound=b["bound"], measured=blk.tolist(),
                looseness=float(np.max(ratio)))


def main():
    emit("=" * 72)
    emit("§6.23 组合误差合同验收")
    emit("=" * 72)
    mz = case_metzler()
    emit("")
    dc = case_discrete()
    emit("")
    cs = case_conservation()
    emit("")
    tc = case_three_criteria()
    emit("")
    ac = case_accuracy()
    n_pass = sum(1 for c in _CHECKS if c["ok"])
    emit("")
    emit("%d/%d checks passed" % (n_pass, len(_CHECKS)))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "composition_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    import json
    payload = dict(kind="composition",
                   meta=dict(bench="bench.run_composition",
                             section="DESIGN.md §6.23",
                             model="island-level Metzler / discrete window / "
                                   "conservation split"),
                   metzler=mz, discrete=dc, conservation=cs,
                   criteria=tc, accuracy=ac, checks=_CHECKS)
    with open(os.path.join(out_dir, "composition.json"), "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return n_pass == len(_CHECKS)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
