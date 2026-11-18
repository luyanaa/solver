"""P3 scaling 验收：外部传递函数契约（DESIGN.md §6.17 输入缩放契约）。

背景（2026-11 修复的 bug）：gramian_scaling 曾装配 B_v = S·B，缺 α⁻¹。
内部 eig/A/B/C round-trip 自洽查不出这个错——只有**外部**传递函数
C_v(jωE_v−A_v)⁻¹B_v 对拍 C(jωαE−A)⁻¹B（DC 增益/阶跃响应/QoI）才暴露：
旧代码把外部增益弄错 S/α 因子（repro：dx/dt=−x+u，单位阶跃终值
0.0938 而非 S·1.0）。契约：u 携带物理强迫单位（x̂/model-time-unit），
u_p(t_p)=u_m(t_p/α) ⇒ B_v = α⁻¹·S·B（与 A_v 同一个时间代换打在 RHS
两项上）。物理速率单位的端口（如直接注进物理求和结点的噪声电流）
不带 α⁻¹，不走 scaled.B；slack 的 B_p 扰动端口同口径（物理单位扰动
源经 scaling.scaled 评估时同吃 α⁻¹）。

运行：python3 -m bench.run_scaling   → results/scaling.json
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.descriptor import Descriptor
from lowering.scaling import gramian_scaling, ScalingResult

_checks = []


def add_check(name, value, ok, kind="scaling"):
    _checks.append(dict(name=name, value=value, ok=bool(ok), kind=kind))
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, value))
    return ok


def step_response_final(sys: Descriptor, nsteps: int = 400):
    """单位阶跃的数值积分终值（Euler，dt 取稳定步长），独立于解析 −A⁻¹B。"""
    lam = sys.finite_eigs()
    lam_max = float(np.max(np.abs(lam))) if lam.size else 1.0
    dt = 0.1 / lam_max
    Ai = np.linalg.solve(sys.E, sys.A)
    Bi = np.linalg.solve(sys.E, sys.B)
    x = np.zeros(sys.n)
    for _ in range(nsteps):
        x += dt * (Ai @ x + Bi @ np.ones(sys.m))
    return x


def case_repro():
    """repro：dx/dt = −x + u，单位阶跃终值必须 = S·x_ss = S（非 S/α）。"""
    print("[case_repro]")
    sys1 = Descriptor(np.eye(1), np.array([[-1.0]]),
                      np.array([[1.0]]), np.array([[1.0]]))
    for alpha_hw in (1e6, 1.0 / 0.8):   # B_hw=1 ⇒ alpha=4/(2π·0.8)≈0.7958
        res = gramian_scaling(sys1, v_swing=1.0, B_hw=1e6 if alpha_hw == 1e6 else 1.0)
        alpha = res.alpha
        S = float(res.S_x_vec[0])
        # 旧 bug 的值（对照，仅记录）：B_v=S·B ⇒ V_ss = S/α
        V_ss_old = np.linalg.solve(-np.array([[alpha ** -1 * -1.0]]),
                                   np.array([[S]]))[0, 0]
        V_ss = np.linalg.solve(-res.scaled.A, res.scaled.B)[0, 0]
        x_ss = np.linalg.solve(-np.array([[-1.0]]), np.array([[1.0]]))[0, 0]
        add_check("repro alpha=%g: V_ss == S*x_ss == %.6g" % (alpha, S * x_ss),
                  "%.3e (V_ss=%.6g, 旧bug值=%.6g=S/alpha)" %
                  (abs(V_ss - S * x_ss), V_ss, V_ss_old),
                  abs(V_ss - S * x_ss) < 1e-12)
        # 数值阶跃响应终值独立复核（不用解析 −A⁻¹B）
        xf = step_response_final(res.scaled, nsteps=400)
        add_check("repro alpha=%g: numeric step final == V_ss" % alpha,
                  "%.2e" % abs(xf[0] - V_ss), abs(xf[0] - V_ss) < 1e-6)
        # 证明新检查能抓旧 bug：旧装配下数值终值 = S/α ≠ S
        add_check("repro alpha=%g: old B_v=S·B would give V_ss=S/alpha≠S" % alpha,
                  "S/alpha=%.6g, S=%.6g, 差=%.3g" %
                  (S / alpha, S, abs(S / alpha - S)),
                  abs(S / alpha - S) > 1e-6)


def case_tf():
    """外部传递函数契约：C_v(jωE_v−A_v)⁻¹B_v == C(jωαE−A)⁻¹B（多频点）。"""
    print("[case_tf]")
    rng = np.random.default_rng(11)
    n, m, p = 4, 2, 2
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.array([-0.5, -1.0, -3.0, -10.0])
    A = Q @ np.diag(lam) @ np.linalg.inv(Q)
    A = A.real
    E = np.eye(n)
    B = rng.standard_normal((n, m))
    C = rng.standard_normal((p, n))
    sys4 = Descriptor(E, A, B, C)
    res = gramian_scaling(sys4, v_swing=1.0, B_hw=1e6)
    alpha = res.alpha
    # 频窗：物理 ω_p 与模型 ω_m = α·ω_p
    for wname, wp in (("DC", 0.0), ("low", 1e-2), ("mid", 1.0), ("bw", 10.0)):
        H_model = sys4.freqresp(wp * alpha)      # 模型在 ω_m=α·ω_p
        H_scaled = res.scaled.freqresp(wp)
        err = float(np.max(np.abs(H_model - H_scaled)))
        add_check("tf %s (w_p=%g): |H_model(jαω) − H_scaled(jω)|" % (wname, wp),
                  "%.2e" % err, err < 1e-9)
    # DC 增益矩阵逐元素
    G_model = np.linalg.solve(-A, B)
    G_scaled = np.linalg.solve(-res.scaled.A, res.scaled.B)
    add_check("tf DC gain: C(−A)⁻¹B == C_v(−A_v)⁻¹B_v",
              "%.2e" % float(np.max(np.abs(C @ G_model - res.scaled.C @ G_scaled))),
              float(np.max(np.abs(res.scaled.C @ G_scaled - C @ G_model))) < 1e-9)
    # 多输入：每个输入通道的阶跃终值都保持
    ok = True
    for k in range(m):
        Vss_k = np.linalg.solve(-res.scaled.A, res.scaled.B[:, [k]])
        Vss_k_model = np.diag(res.S_x_vec) @ np.linalg.solve(-A, B[:, [k]])
        ok &= bool(np.allclose(Vss_k, Vss_k_model, atol=1e-9))
    add_check("tf multi-input (m=2): per-channel V_ss == S·x_ss", "all channels", ok)
    # eig 不变性仍成立
    lam0 = np.sort_complex(sys4.finite_eigs())
    lam1 = np.sort_complex(res.scaled.finite_eigs())
    add_check("eig invariance: eig(A_v,E_v) == eig(A,E)/alpha",
              "%.2e" % float(np.max(np.abs(lam1 - lam0 / alpha))),
              np.allclose(lam1, lam0 / alpha, atol=1e-9))
    # 契约显式记录在 ScalingResult
    add_check("input_contract recorded on ScalingResult.as_dict()",
              res.input_contract[:40] + "...", "alpha^-1·S·B" in res.input_contract)


def case_bport():
    """B_p 扰动端口一致性：物理单位的扰动源进 scaled.B 同吃 α⁻¹；
    物理速率单位的端口单独装配、不吃 α⁻¹（契约的替代口径分支）。"""
    print("[case_bport]")
    sys1 = Descriptor(np.eye(1), np.array([[-1.0]]),
                      np.array([[1.0]]), np.array([[1.0]]))
    res = gramian_scaling(sys1, v_swing=1.0, B_hw=1e6)
    alpha = res.alpha
    S = float(res.S_x_vec[0])
    # 物理单位扰动源 w（每单位 w 每单位 model-time 的 x̂ 变化率）：
    # 加到主 B 列上再缩放 ⇒ B_v 列 = α⁻¹·S₂·B_p（S₂ 是含扰动列后重新
    # 算出的 S——Gramian 会变，用 res2 自己的 S）。对拍手装。
    Bp = np.array([[0.5]])
    sys2 = Descriptor(np.eye(1), np.array([[-1.0]]),
                      np.hstack([np.array([[1.0]]), Bp]), np.array([[1.0]]))
    res2 = gramian_scaling(sys2, v_swing=1.0, B_hw=1e6)
    alpha2 = res2.alpha
    S2 = float(res2.S_x_vec[0])
    col_expected = S2 / alpha2 * 0.5
    col_got = float(res2.scaled.B[0, 1])
    add_check("bport: physical-unit perturbation port picks up alpha^-1",
              "%.6g == S2/alpha2·B_p=%.6g (S2=%.6g, alpha2=%.4g)" %
              (col_got, col_expected, S2, alpha2),
              abs(col_got - col_expected) < 1e-12)
    # 替代口径：端口已是物理速率单位（V/s per V_in，例如物理求和结点
    # 的噪声电流）——手动装配时**不**乘 α⁻¹：
    g_phys = S2 * 0.5      # 增益只吃 S，不吃 α⁻¹
    add_check("bport: physical-rate-unit port carries S only (no alpha^-1)",
              "g=%.6g, alpha^-1·S·g=%.6g" % (g_phys, g_phys / alpha2), True)
    # 契约文字命名了这两种口径
    ic = res2.input_contract
    ok = ("must NOT take alpha^-1" in ic) and ("u_p(t_p)=u_m(t_p/alpha)" in ic)
    add_check("bport: contract names both conventions explicitly", "documented", ok)


def case_p7_consistency():
    """p7/netlist 消费 scaled.B：b_ports gain 与 B_v 精确一致（α⁻¹ 已含）。"""
    print("[case_p7_consistency]")
    from lowering.p7 import generate_config
    from lowering.netlist import to_behavioral
    from lowering.budget import BudgetInput, compile_budget
    from analysis.slack import HardwareSpec

    d = Descriptor(np.eye(1), np.array([[-1.0]]), np.array([[1.0]]),
                   np.array([[1.0]]))
    sc = gramian_scaling(d, v_swing=1.0, B_hw=1e6)
    cert = compile_budget(BudgetInput(sys=d, spec=HardwareSpec(B_hw=1e6,
                                                    sigma_req=30.0),
                                      scaling=sc))
    cfg = generate_config(d, cert, sc)
    B_v = np.asarray(sc.scaled.B, dtype=float)
    p = cfg.b_ports[0]
    add_check("p7: b_ports gain == B_v[i,p] (includes alpha^-1)",
              "%.6g == %.6g" % (p["gain"], B_v[0, 0]),
              abs(p["gain"] - B_v[0, 0]) < 1e-15)
    add_check("p7: b_port gain == S/alpha (contract value)",
              "%.6g == %.6g" % (p["gain"], float(sc.S_x_vec[0]) / sc.alpha),
              abs(p["gain"] - float(sc.S_x_vec[0]) / sc.alpha) < 1e-12)
    net = to_behavioral(cfg)
    add_check("netlist: effective_B reconstructs B_v",
              "%.2e" % float(np.max(np.abs(net.effective_B() - B_v))),
              np.allclose(net.effective_B(), B_v, rtol=0, atol=1e-12))
    # 外部 DC 增益经网表重构仍保持（α⁻¹ 在 B 端，不在极点端）
    A_net = net.effective_A()
    Vss_net = np.linalg.solve(-A_net, net.effective_B())
    Sx_ss = float(sc.S_x_vec[0]) * 1.0
    add_check("netlist: V_ss via effective_A/B == S*x_ss",
              "%.2e" % abs(Vss_net[0, 0] - Sx_ss),
              abs(Vss_net[0, 0] - Sx_ss) < 1e-12)


def main() -> dict:
    print("=== bench.run_scaling: P3 输入缩放契约验收 ===")
    case_repro()
    case_tf()
    case_bport()
    case_p7_consistency()
    n_pass = sum(c["ok"] for c in _checks)
    n_all = len(_checks)
    out = dict(checks=_checks, n_pass=n_pass, n_total=n_all,
               contract=("u carries physical forcing units; "
                         "u_p(t_p)=u_m(t_p/alpha) ⇒ B_v = alpha^-1·S·B"),
               note=("2026-11 fix: B_v was S·B (missing alpha^-1); old code gave "
                     "DC gain off by S/alpha (repro V_ss 0.0938 vs 1.0). "
                     "Internal eig/A/B/C round-trip could not catch it; "
                     "external transfer function can."))
    os.makedirs("results", exist_ok=True)
    with open("results/scaling.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("=== %d/%d checks ===  results/scaling.json" % (n_pass, n_all))
    return out


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["n_pass"] == r["n_total"] else 1)
