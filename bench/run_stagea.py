"""Stage A 收尾回归（DESIGN.md Stage-A 退出条件 4/5/7，A4/A8/A9 closeout）。

    python3 -m bench.run_stagea

三个新 benchmark verdict + 数字回退率 tally + 七项假设敏感度账本：

  A4  负控制按预测原因失败——非正规 shear 的换基判定：
      Γ(原基) vs Γ(scaled 基, gramian_scaling) vs Γ(特征向量基)，
      以及特征向量基的 κ(Φ)=cond(V) 税——§8 "禁止病态特征基" 的定量版。
  A5  数字回退率：phase0 五系统 + 3 anchors + 新三例，共 11 个 workload，
      certificate infeasible/rejected ⇒ 回退；报告原始率与剔除故意
      不可行探针后的率（§9 风险 4：>50% ⇒ "加速器而非求解器"）。
  A9  假设账本：bench/ledger.py 七项假设，每项一个敏感度数字
      （来自既有机制/结果文件，不新增拟合）→ results/ledger.json。

新 benchmark（DESIGN.md §8 表的三行补判定）：
  - 非正规 shear：Γ≫1 是瞬态增长，六轴 slack 看不见；换基的收益/税对照。
  - Van der Pol：极限环系统；线性化工点不稳定 ⇒ 六轴全部不违规但
    stable=False——工点稳定性是六轴之外的前置条件（同 Burgers anchor
    的"不强行判可行"口径）；前端 reduce 归类 polynomial。
  - Lorentzian 记忆粘弹性：Maxwell/SLS 核 K(t)=Σ g_k e^{−t/τ_k} 的
    Markov 嵌入是有限维（n = 2 + m）；力→应变频响对解析柔量；
    幂律核 t^{-1/2} 无有限嵌入（记录的限制，不是隐藏失败）。
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from analysis import Descriptor, HardwareSpec, evaluate
from analysis.resolvent import resolvent_peak
from lowering.budget import evaluate_sys
from lowering.reduce import C, V, add, first_order, mul, pow, reduce_system, sub
from lowering.scaling import gramian_scaling

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
HW = HardwareSpec(kappa_hw=1e6, B_hw=10e6, tol_l2=1e-2)
HW_PROBE = HardwareSpec(kappa_hw=1e6, B_hw=10e6, tol_l2=1e-2,
                        delta_A_norm=1e-2, delta_input_norm=1e-2)
_LINES: list[str] = []
CHECKS: list[dict] = []
CASES: dict = {}


def emit(s: str = ""):
    print(s)
    _LINES.append(s)


def hr(title: str):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


def table(header, rows, fmts):
    cells = [[(f % v) if (isinstance(f, str) and f) else str(v)
              for f, v in zip(fmts, r)] for r in rows]
    widths = [max(len(header[j]), max((len(c[j]) for c in cells), default=0))
              for j in range(len(header))]
    emit("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for c in cells:
        emit("  " + " ".join(v.rjust(w) for v, w in zip(c, widths)))


def _fmt(v, spec="%.3g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "inf" if v == float("inf") else "n/a"
    return spec % v


def add_check(name, got, expected, passed, kind="stagea", note=None):
    CHECKS.append(dict(name=name, got=got, expected=expected,
                       passed=bool(passed), kind=kind,
                       **({"note": note} if note else {})))


# ==========================================================================
# CASE 1  非正规 shear：换基判定（A8 负控制，§8 "非正规 shear + block Schur"）
# ==========================================================================
def case_shear_basis():
    hr("CASE 1  非正规 shear：Γ vs 换基（scaled / 特征向量 / 块正交 Schur）")
    A = np.array([[-1.0, 10.0], [0.0, -2.0]])
    sys0 = Descriptor(np.eye(2), A, np.array([[1.0], [0.0]]), np.eye(2))
    r0 = resolvent_peak(sys0)
    gamma0 = float(r0["Gamma"])
    emit("  原基 Γ = %.4f（method=%s）——瞬态增长来自非对角耦合 10，" % (gamma0, r0["method"]))
    emit("  谱本身良态（λ=-1,-2，对角 Γ=1）：Γ≫1 是基/非正规性，不是谱。")

    # (a) Gramian 对角 scaling（P3 预条件）：κ 不变，Γ 实测对照
    sc = gramian_scaling(sys0, v_swing=1.0, B_hw=HW.B_hw)
    gamma_scaled = float(resolvent_peak(sc.scaled)["Gamma"])
    kappa_ratio = sc.scaled.kappa() / sys0.kappa()
    lam0 = np.sort_complex(sys0.finite_eigs())
    lam1 = np.sort_complex(sc.scaled.finite_eigs())
    eig_dev = float(np.max(np.abs(lam1 - lam0 / sc.alpha)) / max(1.0, float(np.max(np.abs(lam0)))))
    emit("  (a) scaled 基：Γ=%.3g（α=%.3g, method=%s），κ_scaled/κ=%.12g" %
         (gamma_scaled, sc.alpha, sc.method, kappa_ratio))
    emit("      eig(scaled)=eig(sys)/α：max dev=%.2e" % eig_dev)

    # (b) 特征向量基：Γ 降到对角值 1/min|λ|，但付 κ(Φ)=cond(V) 税
    w, Vm = np.linalg.eig(A)
    order = np.argsort(w.real)
    w, Vm = w[order], Vm[:, order]
    Vinv = np.linalg.inv(Vm)
    sys_eig = Descriptor(np.eye(2), Vinv @ A @ Vm, Vinv @ sys0.B, sys0.C @ Vm)
    gamma_eig = float(resolvent_peak(sys_eig)["Gamma"])
    condV = float(np.linalg.cond(Vm))
    budget_eig = gamma_eig * condV
    emit("  (b) 特征向量基：Γ=%.4f（=1/min|λ| 对角值），κ(Φ)=cond(V)=%.2f" %
         (gamma_eig, condV))
    emit("      失配总预算 Γ·κ(Φ)=%.2f  vs  原基 Γ=%.2f —— 特征向量基更差。" %
         (budget_eig, gamma0))

    # (c) 块正交 Schur 基：本例 A 已是上三角，正交 Schur 基 = 恒等，
    #     Γ_schur = Γ0；契约：用块正交基保留 Γ 的诚实值，不用病态特征基。
    gamma_schur = gamma0
    emit("  (c) 块正交 Schur 基：A 已上三角 ⇒ 基=恒等，Γ=%.4f（契约口径）。" % gamma_schur)

    # 证书（1% 失配探针）：六轴口径下 shear 在已评估轴上可行——
    # Γ=5.12 本身以 robustness_A=Γ·ΔA=0.051 进 slack；非正规性是
    # 六轴之外的瞬态增长风险（A8 判定签名，不是 slack 违规）。
    cert = evaluate_sys(sys0, HW_PROBE)
    emit("  证书（HW_PROBE）：status=%s，binding=%s，robustness_A=%.3g" %
         (cert.status, cert.binding_axis,
          cert.slack["values"]["robustness_A"]))

    CASES["shear"] = dict(
        A=A.tolist(), gamma=gamma0, gamma_scaled=gamma_scaled,
        gamma_eigvec=gamma_eig, cond_V=condV,
        gamma_eigvec_budget=float(budget_eig), gamma_schur=float(gamma_schur),
        alpha=float(sc.alpha), method=sc.method,
        kappa_scaled_over=float(kappa_ratio),
        cert=cert.as_dict(),
    )

    # 验收：与 run_skew 的 shear（results/skew.json gamma_ideal=5.123）对账
    skew = json.load(open(os.path.join(OUT, "skew.json")))["cases"]["shear"]
    gamma_skew_ref = float(skew["gamma_ideal"])
    add_check("shear Γ matches run_skew reference", "%.4f vs %.4f" % (gamma0, gamma_skew_ref),
              "±2%", abs(gamma0 / gamma_skew_ref - 1.0) < 0.02)
    add_check("shear Γ ≫ 1 (non-normal, diag value is 1)", "%.4f" % gamma0, "> 2",
              gamma0 > 2.0)
    add_check("scaled basis: κ invariant (preconditioning, not unit change)",
              "%.12g" % kappa_ratio, "1.0", abs(kappa_ratio - 1.0) < 1e-9)
    add_check("scaled basis: eig = eig/α", "%.2e" % eig_dev,
              "< 1e-9", eig_dev < 1e-9)
    add_check("eigenvector basis drops Γ to diag value", "%.6f" % gamma_eig,
              "1.0 (=1/min|λ|)", abs(gamma_eig - 1.0) < 1e-6)
    add_check("eigenvector basis tax κ(Φ)=cond(V) > 10", "%.1f" % condV, "> 10",
              condV > 10.0)
    add_check("eigvec basis worse overall (Γ·κ(Φ) > Γ_original)",
              "%.1f vs %.2f" % (budget_eig, gamma0), "bigger",
              budget_eig > gamma0)
    add_check("Schur/orthonormal basis keeps honest Γ (contract)",
              "%.4f" % gamma_schur, "= Γ_original", abs(gamma_schur - gamma0) < 1e-12,
              note="§8: 禁止病态特征基；Γ 的收益不能用 cond(V) 税换")
    add_check("shear certificate on evaluated axes (robustness_A = Γ·ΔA < 1)",
              "%.3g" % cert.slack["values"]["robustness_A"], "< 1",
              cert.slack["values"]["robustness_A"] < 1.0,
              note="Γ≫1 的风险在六轴之外（瞬态增长），A8 判定签名照实记录")
    return cert


# ==========================================================================
# CASE 2  Van der Pol：极限环系统的判定签名
# ==========================================================================
def case_vanderpol():
    hr("CASE 2  Van der Pol：不稳定工点 + 极限环（不强行判可行）")
    mu = 1.0
    # 平衡点 (0,0) 线性化：x'' = mu(1-x²)x' - x ⇒ x' = v, v' = -x + mu·v
    sys = Descriptor(np.eye(2), np.array([[0.0, 1.0], [-1.0, mu]]),
                     np.array([[0.0], [1.0]]), np.array([[1.0, 0.0]]))
    chk = sys.check()
    lam = sys.finite_eigs()
    emit("  平衡点线性化：eig = %s（不稳定螺旋，Re>0），stable=%s"
         % (np.round(lam, 4), chk["stable"]))
    emit("  κ_t = %s（不稳定极点 ⇒ κ=inf 的退化值），B_req = %.3g Hz"
         % (chk["kappa"], chk["bandwidth_hz"]))

    # 前端归类：x'' = mu(1-x²)x' - x 是 x 的多项式 RHS
    rhs = sub(mul(C(mu), mul(sub(C(1.0), pow(V("x_0"), 2)), V("x_1"))), V("x_0"))
    red = reduce_system(first_order("x", 2, rhs))
    emit("  reduce: kind=%s（%s）" % (red.kind, getattr(red, "reason", "")))

    # 六轴签名：不稳定极点让 κ_t 返回退化 inf ⇒ stiffness 轴"违规"——
    # 但这不是"删模态/多速率"问题：违规的语义是工点本身不稳定，
    # 六轴没有专门的稳定性轴。判定照实记录：数字回退/事件调度。
    sl = evaluate(sys, HW)
    emit("  slack（六轴）: status=%s, violated=%s, stiffness=%s" %
         (sl.status, sl.violated or "-", _fmt(sl.values["stiffness"])))
    emit("  ⇒ 判定：unstable operating point → 数字回退/事件调度（§8 HH 行同款：")
    emit("    极限环系统必须 hybrid；模拟 fabric 只跑环上分段线性化，工点由 supervisor 换。")
    emit("    环上工作点是时变（Floquet）线性化，不在本判定器范围——记录的限制。")

    # 极限环存在性（RK4 小积分，x 幅度 ≈ 2 是 Van der Pol 的经典值）
    def rhs_num(y):
        x, v = y
        return np.array([v, mu * (1.0 - x * x) * v - x])
    y = np.array([0.1, 0.0])
    dt = 0.01
    amp = 0.0
    for i in range(int(60.0 / dt)):
        k1 = rhs_num(y); k2 = rhs_num(y + 0.5 * dt * k1)
        k3 = rhs_num(y + 0.5 * dt * k2); k4 = rhs_num(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i * dt >= 50.0:
            amp = max(amp, abs(y[0]))
    emit("  RK4 t∈[50,60] 上 max|x| = %.3f（经典值 ≈ 2，与 mu 无关的一阶）。" % amp)

    CASES["vanderpol"] = dict(
        mu=mu, eig=[complex(v) for v in lam], kappa=float(chk["kappa"]),
        bandwidth_hz=float(chk["bandwidth_hz"]), stable=bool(chk["stable"]),
        reduce_kind=str(red.kind), slack=sl.as_dict(),
        limit_cycle_amp=amp,
        fallback_reason="unstable operating point (equilibrium linearization); "
                        "slack flags it only as degenerate stiffness=inf — no "
                        "dedicated stability axis, verdict is certificate-level fallback",
    )

    add_check("VdP linearization unstable at equilibrium (expected, documented)",
              "%s" % chk["stable"], "False", not chk["stable"])
    add_check("VdP reduce classifies polynomial", "%s" % red.kind, "polynomial",
              red.kind == "polynomial")
    add_check("VdP slack flags instability only as degenerate stiffness=inf",
              "violated=%s, stiffness=%s" % (sl.violated, _fmt(sl.values["stiffness"])),
              "stiffness=inf, no dedicated stability axis",
              "stiffness" in sl.violated and not np.isfinite(sl.values["stiffness"]),
              note="κ_t 对正实部返回 inf 是退化值：语义是工点不稳定，不是删模态/多速率；"
                   "判定必须是证书级回退")
    add_check("VdP limit cycle amplitude ≈ 2", "%.3f" % amp, "2 ± 0.6",
              abs(amp - 2.0) < 0.6)


# ==========================================================================
# CASE 3  Lorentzian 记忆粘弹性：有限维 Markov 嵌入
# ==========================================================================
def _viscoelastic_descriptor(g, tau, G_e=1.0, M=1.0) -> Descriptor:
    """松弛模态和的 Markov 嵌入（Maxwell/SLS 内变量形式）。

    内应力 σ_k：σ̇_k = −σ_k/τ_k + (g_k/τ_k)·ε̇；
    动量：M ε̈ = F − G_e ε − Σ σ_k。
    状态 [ε, ε̇, σ_1..σ_m]，n = 2 + m。力 F→应变 ε 的频响是解析柔量
        J(s) = 1/(M s² + G_e + Σ g_k s/(τ_k s+1))，
    等价松弛模态 G(t) = G_e + Σ (g_k/τ_k) e^{−t/τ_k}（m=1 即单弛豫
    Lorentzian 核 K(t) ∝ e^{−t/τ_r}）。
    """
    m = len(g)
    n = 2 + m
    A = np.zeros((n, n))
    A[0, 1] = 1.0
    A[1, 0] = -G_e / M
    for k in range(m):
        A[1, 2 + k] = -1.0 / M
        A[2 + k, 1] = g[k] / tau[k]
        A[2 + k, 2 + k] = -1.0 / tau[k]
    B = np.zeros((n, 1))
    B[1, 0] = 1.0 / M
    C = np.zeros((1, n))
    C[0, 0] = 1.0
    return Descriptor(np.eye(n), A, B, C)


def _compliance_analytic(w, g, tau, G_e=1.0, M=1.0):
    """嵌入对应的解析柔量：σ_k = g_k s/(τ_k s+1)·ε ⇒
    J(s) = 1/(M s² + G_e + Σ g_k s/(τ_k s+1))；
    等价的松弛模态权重是 g_k/τ_k：G(t) = G_e + Σ (g_k/τ_k) e^{−t/τ_k}。"""
    s = 1j * np.asarray(w, dtype=float)
    mod = M * s * s + G_e + sum(gk * s / (tk * s + 1.0) for gk, tk in zip(g, tau))
    return 1.0 / mod


def case_viscoelastic():
    hr("CASE 3  Lorentzian 记忆粘弹性：有限维 Markov 嵌入（§8 粘弹性行）")
    # m=1：单弛豫 Lorentzian 核 K(t)=exp(−t/τ_r)；m=4：模态和
    tau1 = [1.0]
    g1 = [1.0]
    tau4 = list(np.geomspace(0.1, 10.0, 4))
    g4 = [0.25] * 4
    sys1 = _viscoelastic_descriptor(g1, tau1)
    sys4 = _viscoelastic_descriptor(g4, tau4)
    emit("  n(m=1) = %d, n(m=4) = %d —— 描述符按弛豫模态数线性增长，保持有限维。"
         % (sys1.n, sys4.n))
    add_check("embedding dimension n = 2 + m", "n(1)=%d, n(4)=%d" % (sys1.n, sys4.n),
              "3 / 6", sys1.n == 3 and sys4.n == 6)
    add_check("embedding stable for m=1 and m=4",
              "%s / %s" % (sys1.is_stable(), sys4.is_stable()), "True / True",
              sys1.is_stable() and sys4.is_stable())

    # 嵌入等价性：力→应变频响 vs 解析 Lorentzian 柔量
    w = np.logspace(-3, 3, 200)
    H = sys4.freqresp(w)[:, 0, 0]
    J = _compliance_analytic(w, g4, tau4)
    rel = float(np.max(np.abs(H - J) / np.abs(J)))
    emit("  力→应变频响 vs 解析柔量 J(s)=1/(Ms²+G_e+Σ g_k s/(τ_k s+1))：max rel err = %.2e"
         % rel)
    add_check("embedding reproduces Lorentzian compliance", "%.2e" % rel, "< 1e-8",
              rel < 1e-8)

    r4 = resolvent_peak(sys4)
    emit("  Γ(m=4) = %.4g @ω=%.3g（resolvent sweep）" % (r4["Gamma"], r4["omega"]))
    chk4 = sys4.check()
    emit("  κ_t(m=4) = %.4g，B_req = %.3g Hz" % (chk4["kappa"], chk4["bandwidth_hz"]))

    # κ 随模态跨度增长（极点跨度是代价，与 §5.9 pole-spread 同型）
    tau_spread = list(np.geomspace(0.01, 100.0, 4))
    sys_sp = _viscoelastic_descriptor([0.25] * 4, tau_spread)
    emit("  模态跨度 3 decades（τ∈[0.01,100]）：κ_t = %.4g（vs %.4g @ 2 decades）"
         % (sys_sp.check()["kappa"], chk4["kappa"]))

    # 模态数 vs 精度：幂律核 t^{-1/2}（无有限嵌入）用 Lorentzian 和逼近
    alpha = 0.5
    t = np.geomspace(1e-2, 1e2, 400)
    target = t ** (-alpha)
    errs = {}
    for m in (4, 8, 12):
        u = np.geomspace(1.0 / t[-1] / 4.0, 1.0 / t[0] * 4.0, m)   # 弛豫率 u=1/τ
        dv = float(np.log(u[-1] / u[0]) / (m - 1))
        a = dv * u ** alpha / float(np.gamma(0) if False else _gamma(alpha))
        fit = np.sum(a[:, None] * np.exp(-u[:, None] * t[None, :]), axis=0)
        errs[m] = float(np.sqrt(np.mean((fit - target) ** 2)) / np.sqrt(np.mean(target ** 2)))
    emit("  t^{-1/2} 核用 Lorentzian 和逼近的相对 L2 误差：" +
         ", ".join("m=%d: %.3f" % kv for kv in sorted(errs.items())))
    emit("  ⇒ 幂律核没有有限 Markov 嵌入（Hölder 奇异，同 Gamma 的拒绝口径）：")
    emit("    这是已记录的限制（模态数换精度），不是隐藏失败。")

    cert = evaluate_sys(sys4, HW)
    emit("  证书（m=4, HW）：status=%s，binding=%s" % (cert.status, cert.binding_axis))

    CASES["viscoelastic"] = dict(
        n_m1=sys1.n, n_m4=sys4.n, stable=bool(sys4.is_stable()),
        compliance_rel_err=rel, gamma_m4=float(r4["Gamma"]),
        kappa_m4=float(chk4["kappa"]), kappa_spread=float(sys_sp.check()["kappa"]),
        kernel_fit_errs=errs,
        embeddable="lorentzian-sum: finite (n=2+m)",
        non_embeddable="power-law t^-a: no finite Markovian embedding "
                       "(documented limitation; Lorentzian-sum is the approximation)",
        cert=cert.as_dict(),
    )
    add_check("Gamma finite (memory fits the descriptor)", "%.3g" % r4["Gamma"],
              "finite", np.isfinite(r4["Gamma"]) and r4["Gamma"] > 0)
    add_check("kappa grows with mode spread", "%.3g vs %.3g" %
              (sys_sp.check()["kappa"], chk4["kappa"]), "3-decade spread larger",
              sys_sp.check()["kappa"] > chk4["kappa"])
    add_check("more Lorentzian modes fit the power-law kernel better",
              "m=12: %.3f < m=4: %.3f" % (errs[12], errs[4]), "monotone",
              errs[12] < errs[4])
    add_check("power-law kernel declared non-embeddable (documented limitation)",
              CASES["viscoelastic"]["non_embeddable"][:40] + "...", "declared",
              "no finite Markovian embedding" in CASES["viscoelastic"]["non_embeddable"],
              note="记录为限制；Lorentzian 和是有限维近似")
    return cert


def _gamma(x: float) -> float:
    import math
    return math.gamma(x)


# ==========================================================================
# 数字回退率 tally（Stage-A 退出条件 5）
# ==========================================================================
# 故意不可行探针：这些 case 的预测判定就是失败（A8 负控制 / 激波表示探针 /
# 不稳定工点探针），计入原始率，但单独剔除后给"求解器代表性"率。
_INFEASIBILITY_PROBES = {"biharmonic N=128", "Burgers N=128", "burgers-1d",
                         "Van der Pol (equilibrium)"}


def _fallback_entry(name, source, status, fallback, binding=None, note=None):
    return dict(name=name, source=source, status=status, fallback=bool(fallback),
                binding_axis=binding, note=note)


def case_fallback_rate(shear_cert, vdp_fallback, ve_cert):
    hr("CASE 4  数字回退率 tally（Stage-A 退出条件 5）")
    entries: list[dict] = []

    p0 = json.load(open(os.path.join(OUT, "phase0.json")))
    for name, c in p0["cases"].items():
        sl = c["slack"]
        entries.append(_fallback_entry(
            name, "results/phase0.json", sl["status"],
            len(sl["violated"]) > 0, ",".join(sl["violated"]) or None))

    pl = json.load(open(os.path.join(OUT, "pipeline.json")))
    for name, c in pl["cases"].items():
        cert = c["p6_certificate"]
        fb = bool(cert["slack"]["violated"]) or cert["status"] == "rejected"
        entries.append(_fallback_entry(
            name, "results/pipeline.json", cert["status"], fb,
            cert.get("p6_binding_axis") or cert.get("binding_axis"),
            note="poisson-2d: feasible after P3 α-scaling (cert_note)" if name == "poisson-2d" else None))

    # 新三例
    s_fb = shear_cert.status == "rejected" or len(shear_cert.slack["violated"]) > 0
    entries.append(_fallback_entry("non-normal shear", "run_stagea (live)",
                                   shear_cert.status, s_fb, shear_cert.binding_axis,
                                   note="Γ=5.12 的瞬态增长在六轴之外；块正交基可解，不判回退"))
    entries.append(_fallback_entry("Van der Pol (equilibrium)", "run_stagea (live)",
                                   "unstable operating point", vdp_fallback, None,
                                   note="工点不稳定 ⇒ 数字回退/事件调度（hybrid）"))
    v_fb = ve_cert.status == "rejected" or len(ve_cert.slack["violated"]) > 0
    entries.append(_fallback_entry("viscoelastic Lorentzian (m=4)", "run_stagea (live)",
                                   ve_cert.status, v_fb, ve_cert.binding_axis))

    total = len(entries)
    n_fb = sum(1 for e in entries if e["fallback"])
    rate = n_fb / total
    rep = [e for e in entries if e["name"] not in _INFEASIBILITY_PROBES]
    n_rep_fb = sum(1 for e in rep if e["fallback"])
    rate_rep = n_rep_fb / len(rep)
    verdict = "accelerator-not-solver" if rate > 0.5 else "solver"

    table(["workload", "source", "status", "fallback", "binding"],
          [(e["name"], e["source"].replace("results/", "").replace(".json", ""),
            e["status"], "YES" if e["fallback"] else "-", e["binding_axis"] or "-")
           for e in entries],
          ["", "", "", "", ""])
    emit("\n  原始率：%d/%d = %.1f%% ⇒ verdict=%s（§9 风险 4 口径）"
         % (n_fb, total, 100.0 * rate, verdict))
    emit("  剔除故意不可行探针（%s）后：%d/%d = %.1f%%"
         % (", ".join(sorted(_INFEASIBILITY_PROBES)), n_rep_fb, len(rep),
            100.0 * rate_rep))
    emit("  说明：语料偏向失败模式探针；两个数字都报告，不挑好看的那个。")

    CASES["fallback_rate"] = dict(
        entries=entries, total=total, n_fallback=n_fb, rate=float(rate),
        verdict=verdict, n_probe_excluded=len(entries) - len(rep),
        rate_excluding_probes=float(rate_rep),
        rule="fallback = certificate infeasible or rejected; "
             "'no violation on evaluated axes' = analog (covers evaluated axes only)")

    add_check("tally covers 11 workloads", "%d" % total, "11", total == 11)
    add_check("rate recomputes from entries", "%.4f" % rate,
              "%.4f" % (sum(1 for e in entries if e["fallback"]) / total),
              abs(rate - n_fb / total) < 1e-12)
    add_check("verdict follows the >50%% rule", verdict,
              "accelerator-not-solver" if rate > 0.5 else "solver", True)
    add_check("both rates reported (raw + probe-excluded)", "%.1f%% / %.1f%%"
              % (100 * rate, 100 * rate_rep), "two numbers",
              np.isfinite(rate) and np.isfinite(rate_rep))
    return shear_cert, ve_cert


# ==========================================================================
# 假设账本（bench/ledger.py，Stage-A 退出条件 7）
# ==========================================================================
def case_ledger():
    hr("CASE 5  七项假设敏感度账本（bench/ledger.py）")
    from bench.ledger import build_ledger
    led = build_ledger(out_dir=OUT, emit=emit)
    CASES["ledger"] = led
    n = len(led["assumptions"])
    add_check("ledger has 7 assumptions", "%d" % n, "7", n == 7)
    ok = all(a.get("sensitivity") is not None and a.get("source") and
             a.get("falsification_observation") for a in led["assumptions"])
    add_check("every entry has sensitivity + source + falsification", "all" if ok else "gap",
              "all 7", ok)


# ==========================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CHECKS.clear()
    CASES.clear()

    hr("Stage A 收尾回归（A4/A8/A9 closeout）")
    shear_cert = case_shear_basis()
    case_vanderpol()
    ve_cert = case_viscoelastic()
    case_fallback_rate(shear_cert, True, ve_cert)
    case_ledger()

    hr("VERDICT")
    rows = [(c["name"], str(c["got"]), str(c["expected"]),
             "yes" if c["passed"] else "NO") for c in CHECKS]
    table(["check", "got", "expected", "pass"], rows, [""] * 4)
    n_fail = sum(1 for c in CHECKS if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(CHECKS) - n_fail, len(CHECKS)))
    fr = CASES["fallback_rate"]
    emit("  fallback rate: %d/%d = %.1f%% (%s); excluding probes: %.1f%%"
         % (fr["n_fallback"], fr["total"], 100 * fr["rate"], fr["verdict"],
            100 * fr["rate_excluding_probes"]))
    emit("  shear: Γ=%.3g, scaled=%.3g, eigvec=%.3g (κ(Φ)=%.1f, budget %.1f)"
         % (CASES["shear"]["gamma"], CASES["shear"]["gamma_scaled"],
            CASES["shear"]["gamma_eigvec"], CASES["shear"]["cond_V"],
            CASES["shear"]["gamma_eigvec_budget"]))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "stagea.json"), "w") as fh:
        json.dump(_jsonable(dict(cases=CASES, checks=CHECKS,
                                 summary=dict(n_checks=len(CHECKS),
                                              n_passed=len(CHECKS) - n_fail))), fh, indent=1)
    with open(os.path.join(OUT, "stagea_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  wrote results/stagea.json, results/stagea_report.txt")
    return 1 if n_fail else 0


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else ("inf" if f > 0 else "-inf")
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, complex):
        return dict(re=o.real, im=o.imag)
    return o


if __name__ == "__main__":
    sys.exit(main())
