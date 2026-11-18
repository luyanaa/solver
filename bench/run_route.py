"""Phase 1.6 验收：route 是 transfer function，不是长度（DESIGN.md §3.2、§6.15）。

    python3 -m bench.run_route

七组算例：

    wire         distributed vs lumped：同 DC 斜率，不同 t50 / f_3dB
    buffers      fanout 深度同时受相位预算与 N_b σ_b² 限制
    all-pass     pad 延迟不能用 RC：同 DC delay，500 MHz 处 RC 已衰减 15%
    deskew       group delay 对齐后 skew→0，但绝对延迟不变
    feedback     零 skew ≠ 零动力学误差：匹配的 150 ps 仍超 5° 预算
    crosstalk    C_m 的串扰地板，以及 1 mV/1 V 对应的 C_m 上限
    sensitivity  从 J 自动算 τ_tol：1/|J_ij| 与 2Q resonance
    weighted_j   谱权重改变 J_H 的排名：W_low vs W_high（P0-A 回归）
    crosstalk_asym R1≠R2、C1≠C2：H 与独立求解逐项对照（P0-B 回归）

`route_closure` 就是 §2.8 四条 closure 的 per-edge 版本：gain、phase、
noise、feedback absolute delay。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import eig, solve_banded

from analysis.skew import delay_mass_matrix, scalar_dde_root, uniform_tau_crit
from twin.route import (AllPass, CoupledNets, PhysicalScaling, PureDelay,
                        RouteEdge, RouteSpec, Wire, buffer_chain, coupled_nets,
                        deskew, edge_metrics, group_delay, j_h, j_tau,
                        j_tau_group, latency_class, loop_margins, phase_delay_at,
                        route_closure, sensitivity_summary)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_LINES: list[str] = []
CASES: dict[str, dict] = {}


def emit(s: str = ""):
    print(s)
    _LINES.append(s)


def hr(title: str):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


def table(header, rows):
    widths = [max(len(str(header[j])),
                  max((len(str(r[j])) for r in rows), default=0))
              for j in range(len(header))]
    emit("  " + " ".join(str(h).rjust(w) for h, w in zip(header, widths)))
    for r in rows:
        emit("  " + " ".join(str(v).rjust(w) for v, w in zip(r, widths)))


def _jsonable(x):
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# 测量夹具：分布式 RC 线的阶跃响应
# ---------------------------------------------------------------------------
def _diffusion_t50(tau_rc: float, n: int = 400, nsteps: int = 2000,
                   t_end: float = 1.5) -> float:
    """rc ∂v/∂t = ∂²v/∂x²，v(0)=1，∂v/∂x(1)=0；返回 v(1) 到 0.5 的时间/τ_rc。

    这是分布式 RC 线的物理模型（L→0 的 telegrapher），只用于测量
    §3.2 引用的 0.38 系数，不是求解器。
    """
    h = 1.0 / n
    dt = t_end * tau_rc / nsteps
    alpha = dt / (tau_rc * h * h)
    ab = np.zeros((3, n))
    ab[0, 1:] = -alpha
    ab[1, :] = 1.0 + 2.0 * alpha
    ab[2, :-1] = -alpha
    ab[2, -2] = -2.0 * alpha          # 开路的 ghost node
    v = np.zeros(n)
    b = np.empty(n)
    for k in range(nsteps):
        b[:] = v
        b[0] += alpha                 # v0 = 1
        vn = solve_banded((1, 1), ab, b)
        if vn[-1] >= 0.5:
            frac = (0.5 - v[-1]) / (vn[-1] - v[-1])
            return float((k + frac) * dt / tau_rc)
        v = vn
    return float("nan")


# ---------------------------------------------------------------------------
# 算例
# ---------------------------------------------------------------------------
def case_wire():
    w = Wire(2000, 0.2, 0.2)
    lattice = np.logspace(np.log10(2 * np.pi * 1e5), np.log10(2 * np.pi * 20e9), 400)
    dist = RouteEdge((w,), "distributed")
    lump = RouteEdge((Wire(2000, 0.2, 0.2, "lumped"),), "lumped")
    md = edge_metrics(dist, lattice)
    ml = edge_metrics(lump, lattice)
    CASES["wire"] = dict(
        R=w.R, C=w.C, tau_rc=w.tau_rc,
        gd_dc=float(group_delay(dist, 2 * np.pi * 1e6)),
        gd_dc_lump=float(group_delay(lump, 2 * np.pi * 1e6)),
        t50_fd=_diffusion_t50(w.tau_rc),
        t50_dist=w.delay50,
        t50_lump=float(np.log(2.0) / 2.0 * w.tau_rc),
        f3_db=md["f_3db"], f3_lump=ml["f_3db"],
    )


def case_buffers():
    om = 2 * np.pi * 100e6
    spec = RouteSpec(om, eps_gain=1e-2, eps_phase_deg=3.0, noise_budget=50e-6)
    out = {}
    for d in (1, 4):
        e = buffer_chain(d, gain=0.999, f_p=2e9, tau=0.0, sigma=10e-6,
                         name="depth%d" % d)
        c = route_closure(e, spec)
        c["eta_zero"] = edge_metrics(e, np.array([om]), tau_ideal=0.0)["eta"]
        c["eta_declared"] = edge_metrics(e, np.array([om]),
                                         tau_ideal=e.latency)["eta"]
        out[d] = c
    CASES["buffers"] = out


def case_allpass():
    tau = 200e-12
    band = np.logspace(np.log10(2 * np.pi * 1e5),
                       np.log10(2 * np.pi * 500e6), 300)
    ap = RouteEdge((AllPass(tau),), "allpass")
    rc = RouteEdge((Wire(2000, 0.2, 0.5, "lumped"),), "rc")
    ma = edge_metrics(ap, band, tau_ideal=tau)
    mr = edge_metrics(rc, band, tau_ideal=tau)
    wide = np.logspace(np.log10(2 * np.pi * 1e5), np.log10(2 * np.pi * 2e9), 300)
    rc_f3 = edge_metrics(rc, wide)["f_3db"]
    CASES["allpass"] = dict(
        tau=tau,
        ap_gain=float(ma["gain_err"].max()),
        ap_phase=float(ma["phase_err_deg"].max()),
        ap_jh=j_h(ap, band, tau_ideal=tau),
        ap_jtau=j_tau(ap, band, tau),
        ap_f3=ma["f_3db"],
        rc_gain=float(mr["gain_err"].max()),
        rc_phase=float(mr["phase_err_deg"].max()),
        rc_jh=j_h(rc, band, tau_ideal=tau),
        rc_jtau=j_tau(rc, band, tau),
        rc_f3=rc_f3,
    )


def case_deskew():
    edges = [RouteEdge((PureDelay(20e-12),), "e20"),
             RouteEdge((PureDelay(45e-12),), "e45"),
             RouteEdge((Wire(2000, 0.2, 0.15),), "wire60")]
    om0 = 2 * np.pi * 200e6
    r = deskew(edges, om0)
    band = np.logspace(np.log10(2 * np.pi * 1e5),
                       np.log10(2 * np.pi * 500e6), 300)
    spec1 = RouteSpec(2 * np.pi * 100e6, feedback_critical=True,
                      omega_c=2 * np.pi * 100e6, phi_reserve_deg=1.0)
    spec5 = RouteSpec(2 * np.pi * 100e6, feedback_critical=True,
                      omega_c=2 * np.pi * 100e6, phi_reserve_deg=5.0)
    CASES["deskew"] = dict(
        omega0=om0, target=r["target"],
        before=r["before"], after=r["after"],
        skew_before=r["skew_before"], skew_after=r["skew_after"],
        jt_before=j_tau_group(edges, band),
        jt_after=j_tau_group(r["edges"], band),
        lat_after=[e.latency for e in r["edges"]],
        closure_1deg=route_closure(r["edges"][0], spec1),
        closure_5deg=route_closure(r["edges"][0], spec5),
    )


def case_feedback():
    om_c = 2 * np.pi * 100e6
    e150 = RouteEdge((PureDelay(150e-12),), "150ps")
    e130 = RouteEdge((PureDelay(130e-12),), "130ps")
    # 每条 edge 把自己的 latency 声明进 H_ref：phase 检查放行，
    # 只有 absolute-delay 检查能区分它们。
    spec150 = RouteSpec(om_c, tau_ideal=150e-12, feedback_critical=True,
                        omega_c=om_c, phi_reserve_deg=5.0)
    spec130 = RouteSpec(om_c, tau_ideal=130e-12, feedback_critical=True,
                        omega_c=om_c, phi_reserve_deg=5.0)
    CASES["feedback"] = dict(
        tau_max=route_closure(e130, spec130)["tau_max_abs"],
        r150=route_closure(e150, spec150),
        r130=route_closure(e130, spec130),
        phase_lag_150=float(np.degrees(om_c * 150e-12)),
        phase_lag_130=float(np.degrees(om_c * 130e-12)),
        matched_skew=abs(group_delay(e150, om_c) - group_delay(e150, om_c)),
    )


def case_crosstalk():
    om = 2 * np.pi * 500e6
    wt = 1.0 / (400 * 220e-15)
    p20 = coupled_nets(400, 200, 20)
    p2 = coupled_nets(400, 200, 2)
    p01 = coupled_nets(400, 200, 0.1)
    H = p20.H(om)
    eta = float(np.linalg.svd(H - np.eye(2), compute_uv=False)[0])
    CASES["crosstalk"] = dict(
        omega_trans=wt, f_trans=wt / (2 * np.pi),
        xt_hf=p20.crosstalk_hf, xt_hf_num=float(p20.crosstalk(100 * wt)),
        xt_500m=float(p20.crosstalk(om)),
        xt_hf_2f=p2.crosstalk_hf, xt_hf_01f=p01.crosstalk_hf,
        db_20=float(20 * np.log10(p20.crosstalk_hf)),
        db_2=float(20 * np.log10(p2.crosstalk_hf)),
        eta_500m=eta,
        v20=p20.crosstalk_hf * 1.0, v2=p2.crosstalk_hf * 1.0,
        v01=p01.crosstalk_hf * 1.0,
    )


def case_sensitivity():
    a_s = 2 * np.pi * 1e6
    a_f = 2 * np.pi * 200e6
    w0 = 2 * np.pi * 50e6
    Q = 10.0
    gam = w0 / (2 * Q)
    J = np.array([[-a_s, 0.05 * a_f, 0.0, 0.0],
                  [0.0, -a_f, 0.1 * a_f, 0.0],
                  [0.0, 0.0, -gam, w0],
                  [0.0, 0.0, -w0, -gam]])
    band = np.logspace(5, np.log10(2 * np.pi * 2e9), 2000)
    s = sensitivity_summary(J, band, 1.0)
    CASES["sensitivity"] = dict(
        J=J, Q=Q, omega0=w0, a_slow=a_s, a_fast=a_f,
        tau_tol=s["tau_tol"], hf_tau_tol=s["hf_tau_tol"],
        worst_omega=s["worst_omega"],
    )


def case_weighted_j():
    """Case H：谱权重 W(ω) 改变 J_H（P0-A 回归：权重必须进分子）。

    同一条 RC pad（误差随 ω 单调增长）配两个 log-normal 谱权重：
    W_low 峰在 20 MHz，W_high 峰在 300 MHz。权重不该只出现在分母。
    """
    tau = 200e-12
    band = np.logspace(np.log10(2 * np.pi * 1e5),
                       np.log10(2 * np.pi * 500e6), 300)
    rc = RouteEdge((Wire(2000, 0.2, 0.5, "lumped"),), "rc")
    wlow = np.exp(-0.5 * ((np.log(band) - np.log(2 * np.pi * 20e6)) / 1.2) ** 2)
    whigh = np.exp(-0.5 * ((np.log(band) - np.log(2 * np.pi * 300e6)) / 1.2) ** 2)
    j_low = j_h(rc, band, tau_ideal=tau, weight=wlow)
    j_high = j_h(rc, band, tau_ideal=tau, weight=whigh)
    j_unif = j_h(rc, band, tau_ideal=tau)
    j_flat = j_h(rc, band, tau_ideal=tau, weight=np.ones_like(band))
    jtau_low = j_tau(rc, band, tau, weight=wlow)
    jtau_high = j_tau(rc, band, tau, weight=whigh)
    # 权重峰值位置
    peak_low = float(band[int(np.argmax(wlow))])
    peak_high = float(band[int(np.argmax(whigh))])
    # 回归基准：与 results/route.json 里旧的无权重值对照（uniform 不受修复影响）
    prev = None
    try:
        with open(os.path.join(OUT, "route.json")) as fh:
            prev = json.load(fh)["cases"]["allpass"]["rc_jh"]
    except (OSError, KeyError):
        prev = None
    CASES["weighted_j"] = dict(
        tau=tau, peak_low=peak_low, peak_high=peak_high,
        j_low=j_low, j_high=j_high, j_unif=j_unif, j_flat=j_flat,
        ratio=j_high / j_low,
        jtau_low=jtau_low, jtau_high=jtau_high, jtau_ratio=jtau_high / jtau_low,
        j_unif_prev=prev,
    )


def case_crosstalk_asym():
    """Case I：R1≠R2、C1≠C2 的 CoupledNets（P0-B 回归：交叉项 R 归属）。

    2×2 节点方程 M·V_o = diag(1/R)·V_in 的解是 H_00=m22/(det·R1)、
    H_01=−m12/(det·R2)、H_10=−m12/(det·R1)、H_11=m11/(det·R2)；
    交叉项的 R 只由**输入**那一列决定。修复前 R 写反，被 R1=R2 的旧
    算例掩盖。传递阻抗 Z_12=−m12/det 与 R 无关，所以 R2·H_01=R1·H_10。
    """
    net = CoupledNets(R1=400.0, C1=200.0, R2=800.0, C2=100.0, Cm=20.0)
    band = np.logspace(np.log10(2 * np.pi * 1e5),
                       np.log10(2 * np.pi * 100e9), 200)
    H = net.H(band)
    # 独立求解：两个单位输入分别解 M V = b
    err = 0.0
    for k, wk in enumerate(band):
        M = np.array([[1.0 / net.R1 + 1j * wk * (net.C1 + net.Cm) * 1e-15,
                       -1j * wk * net.Cm * 1e-15],
                      [-1j * wk * net.Cm * 1e-15,
                       1.0 / net.R2 + 1j * wk * (net.C2 + net.Cm) * 1e-15]])
        for col in range(2):
            b = np.zeros(2)
            b[col] = 1.0 / (net.R1 if col == 0 else net.R2)
            v = np.linalg.solve(M, b)
            err = max(err, abs(v[0] - H[k, 0, col]) / max(abs(v[0]), 1e-300),
                      abs(v[1] - H[k, 1, col]) / max(abs(v[1]), 1e-300))
    recip = float(np.max(np.abs(net.R2 * H[:, 0, 1] - net.R1 * H[:, 1, 0])
                         / np.maximum(np.abs(net.R1 * H[:, 1, 0]), _eps())))
    # 高频极限（ω=100 GHz，远超两个 transition）
    Hf = net.H(2 * np.pi * 100e9)
    floor_00_11 = float(np.abs(Hf[0, 1] / Hf[1, 1]))   # V1/V2：纯电容分压
    floor_cross = float(np.abs(Hf[0, 1] / Hf[0, 0]))   # crosstalk() 的量
    floor_pred_00_11 = net.Cm / (net.C1 + net.Cm)
    floor_pred_cross = net.Cm / (net.C2 + net.Cm) * net.R1 / net.R2
    H0 = net.H(0.0)
    CASES["crosstalk_asym"] = dict(
        R1=net.R1, R2=net.R2, C1=net.C1, C2=net.C2, Cm=net.Cm,
        solve_err=err, recip_err=recip,
        floor_v1=floor_00_11, floor_v1_pred=floor_pred_00_11,
        floor_cross=floor_cross, floor_cross_pred=floor_pred_cross,
        floor_prop=net.crosstalk_hf,
        victim1=net.crosstalk_hf_victim(1), victim2=net.crosstalk_hf_victim(2),
        h01_dc=float(abs(H0[0, 1])), h00_dc=float(np.real(H0[0, 0])),
    )


def case_loop():
    """Case J：per-edge closure → whole-loop closure（Review item B）。

    两个 2 节点闭环。耦合矩阵 J 的每个非零 entry j→i 都经过同一条
    RouteEdge（这里全是 PureDelay：wire/buffer/all-pass 级联在 case A–D
    已经单独验收，loop 层只看延迟效应），H_ij 把 J_ij 换成 J_ij·H_ij(s)。

    Loop A：阻尼振子 J=[[−a,−b],[b,−a]]，a=2π·5 MHz、b=2π·50 MHz。
    四个 entry 都延迟（空间阵列的读出-回注结构）。J 是 normal 的，A∘H(s)
    =J·e^{−sτ} 在本征基下解耦成复标量 DDE s=λe^{−sτ}，λ=−a+jb，所以
    精确临界延迟有闭式 τ_ex=(π/2−φ)/|λ|，φ=atan2(b,a)；用
    analysis.skew.scalar_dde_root 独立验证它，不信任闭式。

    Loop B：近标量 J=[[−a,−b],[b,−a]]，b≪a。φ→0 时一阶 E=I+τA 的
    τ_crit=a/|λ|² 与精确边界之比趋近 2/π——§6.14 的乐观性在 loop 层复现。
    """
    w = np.logspace(np.log10(2 * np.pi * 1e5), np.log10(2 * np.pi * 2e9), 1200)
    a = 2 * np.pi * 5e6
    b = 2 * np.pi * 50e6
    J = np.array([[-a, -b], [b, -a]])
    lam = complex(-a, b)
    phi = np.arctan2(b, a)
    tau_ex = (np.pi / 2 - phi) / abs(lam)
    tau_fo = uniform_tau_crit(J)

    def edges_uniform(tau, name):
        return {(i, j): RouteEdge((PureDelay(tau),), "%s%d%d" % (name, i, j))
                for i in range(2) for j in range(2)}

    m_h1 = loop_margins(J, {}, w)
    m_taus = [loop_margins(J, edges_uniform(t, "e"), w)
              for t in (100e-12, 200e-12, 300e-12)]

    # Case J2：deskew 负例。源 0 全部 100 ps，源 1 全部 250 ps；
    # deskew 把源 0 pad 到 250 ps，skew→0，绝对 margin 停在 250 ps 水平。
    e_short = RouteEdge((PureDelay(100e-12),), "short")
    e_long = RouteEdge((PureDelay(250e-12),), "long")
    r_desk = deskew([e_short, e_long], 2 * np.pi * 100e6)
    tau_desk = [e.latency for e in r_desk["edges"]]
    edge_map_desk = {(i, j): r_desk["edges"][j] for i in range(2) for j in range(2)}
    m_desk = loop_margins(J, edge_map_desk, w)

    # Case J3：近标量 loop 的 2/π 乐观性
    a2 = 2 * np.pi * 20e6
    b2 = 2 * np.pi * 0.5e6
    J2 = np.array([[-a2, -b2], [b2, -a2]])
    lam2 = complex(-a2, b2)
    phi2 = np.arctan2(b2, a2)
    tau_ex2 = (np.pi / 2 - phi2) / abs(lam2)
    tau_fo2 = uniform_tau_crit(J2)
    root_ex = scalar_dde_root(lam2, tau_ex2)
    # 一阶 descriptor 的稳定边界：广义特征值 max Re 在 τ_fo 处过零
    ev_fo = eig(J2, delay_mass_matrix(J2, tau_fo2), right=False)
    ev_over = eig(J2, delay_mass_matrix(J2, 1.05 * tau_fo2), right=False)
    m_b = loop_margins(J2, edges_uniform(6e-9, "b"), w)

    CASES["loop"] = dict(
        a=a, b=b, tau_ex=tau_ex, tau_fo=tau_fo, ratio_fo=tau_fo / tau_ex,
        root_a=scalar_dde_root(lam, tau_ex), lam_abs=abs(lam),
        dip_omega=m_h1["omega_at_min"], dip_smin=m_h1["sigma_min_min"],
        fo_eq_true_h1=float(np.max(np.abs(
            m_h1["sigma_min_first_order"] - m_h1["sigma_min"])
            / np.maximum(m_h1["sigma_min"], 1e-300))),
        smin_seq=[m_h1["sigma_min_min"]] + [m["sigma_min_min"] for m in m_taus],
        speak_seq=[m_h1["S_peak"]] + [m["S_peak"] for m in m_taus],
        speak_omega=[m["S_peak_omega"] for m in m_taus],
        skew_before=r_desk["skew_before"], skew_after=r_desk["skew_after"],
        tau_desk=tau_desk,
        smin_short=m_taus[0]["sigma_min_min"],
        smin_desk=m_desk["sigma_min_min"],
        smin_long_target=loop_margins(
            J, edges_uniform(250e-12, "t"), w)["sigma_min_min"],
        tau_ex2=tau_ex2, tau_fo2=tau_fo2, ratio2=tau_fo2 / tau_ex2,
        root_ex=root_ex, lam2_abs=abs(lam2),
        re_fo=float(ev_fo.real.max()), re_over=float(ev_over.real.max()),
        lam2_abs_scale=abs(lam2),
        smin_fo_6ns=m_b["sigma_min_min_first_order"],
        smin_true_6ns=m_b["sigma_min_min"],
    )


def case_scaling():
    """Case K：物理时间/电压 scaling 字段（Review item B 第二半）。

    故意选一个 borderline 模态：|λ|/(2π)=8 MHz（model 时间坐标），硬件
    B_hw=10 MHz。不 scaling：pass。alpha=0.1 s/model-unit（硬件比模型快
    10×）：物理需求 |λ|/alpha/(2π)=80 MHz → fail。verdict 翻转本身就
    是测试——证明 B_hw 比较必须走 PhysicalScaling，忘掉 alpha 会被抓住。
    """
    lam = -2 * np.pi * 8e6
    B_hw = 10e6
    sc = PhysicalScaling(S_x=1e-3, alpha=0.1, v_swing=2e-3)
    sc_fast = PhysicalScaling(S_x=1e-3, alpha=1e-6, v_swing=2e-3)
    CASES["scaling"] = dict(
        lam_model=lam, B_hw=B_hw,
        omega_phys=sc.to_physical_band(abs(lam)),
        omega_phys_fast=sc_fast.to_physical_band(abs(lam)),
        omega_round=sc.to_model_band(sc.to_physical_band(abs(lam))),
        lam_hw=sc_fast.to_physical_band(abs(lam)),
        pass_unscaled=bool(abs(lam) / (2 * np.pi) <= B_hw),
        pass_scaled=sc.bandwidth_pass(abs(lam), B_hw),
        v_swing_state=float(np.max(np.abs(sc.state_from_voltage(sc.v_swing)))),
        noise_model=sc.noise_budget_model(80e-6),
        state_back=sc.voltage_from_state(sc.state_from_voltage(1.7e-3)),
    )


def case_phase_delay():
    """Case M：absolute 判据应该用 ωc 的 phase delay，不是 DC latency。

    P2-route-3：route_closure 的 feedback_critical 检查原来比较
    edge.latency（DC group delay 之和）。deskew 插入 AllPass pad 后，
    pad 的 DC group delay 是 τ，但闭环在 ωc 感到的是 −∠H(jωc)/ωc。
    实测方向：对这些 block（pure delay/单极点/all-pass/分布式 RC）
    每段 phase delay ≤ 自身 DC group delay，所以 pd(ω) ≤ latency，
    latency 判据是保守侧；pad 极点在带外时两者到 1e-4 一致，极点在
    带内时分开（4×AllPass(50ps) 在 10 GHz：lat 205 ps vs pd 132.8 ps）。
    单点 np.angle 会把总相位 −8.08 rad 卷绕成 −1.80 rad（naive pd
    32.8 ps）——unwrap 修的就是这个 bug，用 ∫₀^ωc τ_g dω 交叉核对。
    """
    wc = 2 * np.pi * 1e9
    e_pd = RouteEdge((PureDelay(10e-12),), "pure10")
    spec5 = RouteSpec(2 * np.pi * 1e9, feedback_critical=True,
                      omega_c=wc, phi_reserve_deg=5.0)
    c_pd = route_closure(e_pd, spec5)

    # pad 极点在带内：u = ωc·τ/2 = 1.571 @10 GHz
    wc2 = 2 * np.pi * 10e9
    e_pad = RouteEdge((PureDelay(5e-12), AllPass(50e-12), AllPass(50e-12),
                       AllPass(50e-12), AllPass(50e-12)), "padded")
    pd_true = phase_delay_at(e_pad, wc2)
    pd_naive = float(-np.angle(e_pad.H(1j * wc2)) / wc2)
    # 独立参照：−φ(ωc) = ∫₀^{ωc} τ_g dω
    n = int(np.ceil(12.0 * np.log(10.0) / np.log(1.02)))
    wg = wc2 * np.power(1.02, np.arange(-n, 1))
    pd_int = float(np.trapezoid(group_delay(e_pad, wg), wg) / wc2)
    # verdict 用的是 unwrapped pd：τ_max 取在 naive 与真值之间
    spec216 = RouteSpec(2 * np.pi * 10e9, feedback_critical=True,
                        omega_c=wc2, phi_reserve_deg=216.0)
    c_pad216 = route_closure(e_pad, spec216)

    # pad 极点在带外：pd ≈ latency
    e_oob = RouteEdge((PureDelay(5e-12), AllPass(1e-12), AllPass(1e-12),
                       AllPass(1e-12), AllPass(1e-12)), "oob")
    pd_oob = phase_delay_at(e_oob, wc)

    # deskew 后的 case D edge 组：ωc=100 MHz，pad 极点 ≥ 50 GHz（带外）
    edges_d = [RouteEdge((PureDelay(20e-12),), "e20"),
               RouteEdge((PureDelay(45e-12),), "e45"),
               RouteEdge((Wire(2000, 0.2, 0.15),), "wire60")]
    r = deskew(edges_d, 2 * np.pi * 200e6)
    wcD = 2 * np.pi * 100e6
    specD = RouteSpec(wcD, feedback_critical=True, omega_c=wcD,
                      phi_reserve_deg=5.0)
    closures = [route_closure(e, specD) for e in r["edges"]]
    diverg = max(abs(c["phase_delay_omega_c"] / c["latency"] - 1.0)
                 for c in closures)

    CASES["phase_delay"] = dict(
        pd_pure=c_pd["phase_delay_omega_c"], lat_pure=e_pd.latency,
        pass_pure=c_pd["pass_absolute"],
        lat_pad=e_pad.latency, pd_pad=pd_true, pd_naive=pd_naive,
        pd_int=pd_int, tau_max_216=c_pad216["tau_max_abs"],
        pass_pad216=c_pad216["pass_absolute"],
        pd_oob=pd_oob, lat_oob=e_oob.latency,
        diverg_deskew=diverg,
        pass_deskew=[c["pass_absolute"] for c in closures],
    )


def _eps() -> float:
    return 1e-300


# ---------------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------------
def acceptance():
    checks = []

    def add(name, got, expected, passed):
        checks.append(dict(name=name, got=got, expected=expected,
                           passed=bool(passed)))

    w = CASES["wire"]
    add("wire DC group delay = RC/2", "%.2f ps" % (w["gd_dc"] * 1e12),
        "80.00 ps", abs(w["gd_dc"] / (w["tau_rc"] / 2) - 1.0) < 1e-3)
    add("wire t50 by diffusion", "%.4f RC" % w["t50_fd"], "0.3795 RC",
        abs(w["t50_fd"] / 0.3795 - 1.0) < 0.01)
    add("distributed t50 > lumped", "%.2f vs %.2f ps"
        % (w["t50_dist"] * 1e12, w["t50_lump"] * 1e12), ">",
        w["t50_dist"] > w["t50_lump"])
    add("distributed f3 > lumped f3", "%.2f vs %.2f GHz"
        % (w["f3_db"] / 2 / np.pi / 1e9, w["f3_lump"] / 2 / np.pi / 1e9),
        "f3_dist > f3_lump", w["f3_db"] > w["f3_lump"])

    b = CASES["buffers"]
    add("buffer depth1 closure", b[1]["verdict"], True, b[1]["verdict"])
    add("buffer depth4 closure", b[4]["verdict"], False, not b[4]["verdict"])
    add("depth1 phase < 3°", "%.3f°" % b[1]["max_phase_err_deg"], "< 3°",
        b[1]["max_phase_err_deg"] < 3.0)
    add("depth4 phase > 3°", "%.3f°" % b[4]["max_phase_err_deg"], "> 3°",
        b[4]["max_phase_err_deg"] > 3.0)
    add("depth4 noise", "%.1f µV" % (b[4]["noise_rms"] * 1e6), "< 50 µV",
        b[4]["noise_rms"] * 1e6 < 50.0)
    add("depth1 class", b[1]["latency_class"], "L2", b[1]["latency_class"] == "L2")
    add("depth4 class", b[4]["latency_class"], "L3", b[4]["latency_class"] == "L3")
    add("depth4 eta declared vs 0", "%.4f vs %.4f"
        % (b[4]["eta_declared"], b[4]["eta_zero"]),
        "declared < 5%, zero > 15%",
        b[4]["eta_declared"] < 0.05 and b[4]["eta_zero"] > 0.15)

    a = CASES["allpass"]
    add("all-pass gain flat", "%.2e" % a["ap_gain"], "< 1e-12",
        a["ap_gain"] < 1e-12)
    add("RC gain error at 500 MHz", "%.3f" % a["rc_gain"], "> 0.1",
        a["rc_gain"] > 0.1)
    add("all-pass phase < RC phase", "%.3f° vs %.3f°"
        % (a["ap_phase"], a["rc_phase"]), "ap < rc",
        a["ap_phase"] < a["rc_phase"])
    add("all-pass J_H / RC J_H", "%.1f×" % (a["rc_jh"] / a["ap_jh"]), "> 10×",
        a["rc_jh"] / a["ap_jh"] > 10.0)
    add("all-pass J_tau / RC J_tau", "%.1f×" % (a["rc_jtau"] / a["ap_jtau"]),
        "> 5×", a["rc_jtau"] / a["ap_jtau"] > 5.0)
    add("all-pass f3 = inf", a["ap_f3"], "inf", np.isinf(a["ap_f3"]))

    d = CASES["deskew"]
    add("deskew skew before", "%.1f ps" % (d["skew_before"] * 1e12), "> 35 ps",
        d["skew_before"] * 1e12 > 35.0)
    add("deskew skew after", "%.2e ps" % (d["skew_after"] * 1e12), "< 0.01 ps",
        d["skew_after"] * 1e12 < 0.01)
    add("deskew group spread", "%.2e → %.2e ps²"
        % (d["jt_before"] * 1e24, d["jt_after"] * 1e24), "> 5e3×",
        d["jt_before"] / d["jt_after"] > 5e3)
    add("deskew leaves absolute delay", "%.2f° @100 MHz"
        % float(np.degrees(2 * np.pi * 100e6 * d["target"])), "2.15°",
        abs(float(np.degrees(2 * np.pi * 100e6 * d["target"])) - 2.15) < 0.1)
    add("deskewed loop fails 1° reserve", d["closure_1deg"]["pass_absolute"],
        False, not d["closure_1deg"]["pass_absolute"])
    add("deskewed loop passes 5° reserve", d["closure_5deg"]["pass_absolute"],
        True, d["closure_5deg"]["pass_absolute"])

    f = CASES["feedback"]
    add("feedback τ_max", "%.1f ps" % (f["tau_max"] * 1e12), "138.9 ps",
        abs(f["tau_max"] * 1e12 - 138.9) < 0.1)
    add("150 ps fails 5°", f["r150"]["pass_absolute"], False,
        not f["r150"]["pass_absolute"])
    add("130 ps passes 5°", f["r130"]["pass_absolute"], True,
        f["r130"]["pass_absolute"])
    add("matched 150 ps still fails", "skew=%.1f ps"
        % (f["matched_skew"] * 1e12),
        "absolute fails", f["matched_skew"] < 1e-15
        and not f["r150"]["pass_absolute"])
    add("padding cannot fix feedback", f["r150"]["pad_can_fix"], False,
        not f["r150"]["pad_can_fix"])

    x = CASES["crosstalk"]
    add("crosstalk HF limit", "%.4f" % x["xt_hf_num"], "%.4f" % x["xt_hf"],
        abs(x["xt_hf_num"] / x["xt_hf"] - 1.0) < 1e-2)
    add("shield improvement", "%.2f dB" % (x["db_20"] - x["db_2"]), "> 19 dB",
        x["db_20"] - x["db_2"] > 19.0)
    add("20 fF vs 1 mV budget", "%.1f mV" % (x["v20"] * 1e3), "> 1 mV",
        x["v20"] * 1e3 > 1.0)
    add("0.1 fF vs 1 mV budget", "%.2f mV" % (x["v01"] * 1e3), "< 1 mV",
        x["v01"] * 1e3 < 1.0)

    s = CASES["sensitivity"]
    tol = s["tau_tol"]
    ratio = tol[0, 0] / tol[1, 1]
    add("slow/fast τ_tol ratio", "%.1f" % ratio, "200",
        abs(ratio - 200.0) / 200.0 < 0.02)
    res = s["hf_tau_tol"][0, 0] / tol[0, 0]
    add("resonance shrinks budget", "%.2f×" % res, "2Q = 20×",
        abs(res / 20.0 - 1.0) < 0.02)
    add("resonant cross edge", "%.2f ps" % (tol[2, 3] * 1e12),
        "< 14 ps", tol[2, 3] < tol[0, 0] / 10)
    order = [tol[1, 1], tol[1, 2], tol[0, 1], tol[0, 0]]
    add("τ_tol ordered by |J|", " < ".join("%.2f" % (v * 1e12) for v in order),
        "increasing", all(order[i] < order[i + 1] for i in range(3)))
    wf = s["worst_omega"][0, 0] / s["omega0"]
    add("worst ω at resonance", "%.3f ω0" % wf, "≈ 50 MHz",
        abs(wf - 1.0) < 0.03)

    # Case H：weighted J（P0-A 回归）
    hj = CASES["weighted_j"]
    add("W_low vs W_high J_H ratio", "%.2f×" % hj["ratio"], "> 3×",
        hj["ratio"] > 3.0)
    add("W_low peak @20 MHz", "%.0f MHz" % (hj["peak_low"] / 2 / np.pi / 1e6),
        "20 MHz", abs(hj["peak_low"] / (2 * np.pi * 20e6) - 1.0) < 0.01)
    add("W_high peak @300 MHz", "%.0f MHz" % (hj["peak_high"] / 2 / np.pi / 1e6),
        "300 MHz", abs(hj["peak_high"] / (2 * np.pi * 300e6) - 1.0) < 0.01)
    if hj["j_unif_prev"] is not None:
        add("uniform j_h regression vs old run", "%.4e vs %.4e"
            % (hj["j_unif"], hj["j_unif_prev"]), "equal",
            abs(hj["j_unif"] / hj["j_unif_prev"] - 1.0) < 1e-9)
    add("flat weight = weight=None", "%.16e vs %.16e"
        % (hj["j_flat"], hj["j_unif"]), "identical",
        hj["j_flat"] == hj["j_unif"])
    add("weighted J_tau ratio", "%.2f×" % hj["jtau_ratio"], "> 3×",
        hj["jtau_ratio"] > 3.0)

    # Case I：asymmetric crosstalk（P0-B 回归）
    xi = CASES["crosstalk_asym"]
    add("H vs direct solve (R1≠R2, C1≠C2)", "%.2e" % xi["solve_err"],
        "< 1e-12", xi["solve_err"] < 1e-12)
    add("reciprocity R2·H01 = R1·H10", "%.2e" % xi["recip_err"],
        "< 1e-12", xi["recip_err"] < 1e-12)
    add("victim-1 floor V1/V2 = Cm/(C1+Cm)", "%.5f vs %.5f"
        % (xi["floor_v1"], xi["floor_v1_pred"]), "equal",
        abs(xi["floor_v1"] / xi["floor_v1_pred"] - 1.0) < 1e-3)
    add("crosstalk_hf_victim(1) = measured V1/V2 floor, ≠ crosstalk_hf",
        "%.5f vs %.5f (hf=%.5f)" % (xi["victim1"], xi["floor_v1"], xi["floor_prop"]),
        "victim method = |H01/H11| floor; C2-version differs by (C2+Cm)/(C1+Cm)",
        abs(xi["victim1"] / xi["floor_v1"] - 1.0) < 1e-3
        and abs(xi["victim1"] / xi["floor_prop"] - (xi["C2"] + xi["Cm"]) / (xi["C1"] + xi["Cm"])) < 1e-9)
    add("crosstalk() floor carries R1/R2", "%.5f vs %.5f"
        % (xi["floor_cross"], xi["floor_cross_pred"]),
        "Cm/(C2+Cm)·R1/R2", abs(xi["floor_cross"] / xi["floor_cross_pred"] - 1.0) < 1e-3)
    add("H01(0) = 0", "%.2e" % xi["h01_dc"], "< 1e-15", xi["h01_dc"] < 1e-15)
    add("H00(0) = 1", "%.12f" % xi["h00_dc"], "1",
        abs(xi["h00_dc"] - 1.0) < 1e-12)

    # Case J：whole-loop closure（Review item B）
    L = CASES["loop"]
    add("loop H=1 dip at pole freq", "%.2f MHz vs ±%.2f MHz"
        % (L["dip_omega"] / 2 / np.pi / 1e6, L["b"] / 2 / np.pi / 1e6),
        "dip at poles", abs(L["dip_omega"] / L["b"] - 1.0) < 0.02
        and abs(L["dip_smin"] / L["a"] - 1.0) < 0.01)
    add("loop first-order = true at H=1", "%.2e" % L["fo_eq_true_h1"],
        "< 1e-10", L["fo_eq_true_h1"] < 1e-10)
    add("loop margin monotone in tau", " → ".join("%.2e" % v
        for v in L["smin_seq"]), "decreasing",
        all(L["smin_seq"][i] > L["smin_seq"][i + 1] for i in range(3)))
    add("loop tau_ex exact (scalar_dde_root)", "Re/|λ|=%.1e, Im/|λ|=%.6f"
        % (L["root_a"].real / L["lam_abs"], L["root_a"].imag / L["lam_abs"]),
        "root on jω axis at τ_ex",
        abs(L["root_a"].real) / L["lam_abs"] < 1e-6
        and abs(abs(L["root_a"].imag) / L["lam_abs"] - 1.0) < 1e-4)
    add("loop 2/π optimism (Loop B)", "%.4f vs 2/π=%.4f"
        % (L["ratio2"], 2 / np.pi), "≈ 2/π",
        abs(L["ratio2"] / (2 / np.pi) - 1.0) < 0.02)
    add("first-order descriptor boundary", "max Re @τ_fo=%.1e, @1.05τ=%.1e"
        % (L["re_fo"], L["re_over"]), "0 then > 0",
        abs(L["re_fo"]) < 1e-3 * L["lam2_abs_scale"] and L["re_over"] > 0)
    add("first-order optimistic at loop level", "%.3e vs %.3e"
        % (L["smin_fo_6ns"], L["smin_true_6ns"]), "fo > true",
        L["smin_fo_6ns"] > L["smin_true_6ns"] * 1.05)
    add("deskew skew → 0", "%.1f → %.1e ps"
        % (L["skew_before"] * 1e12, L["skew_after"] * 1e12), "< 0.01 ps",
        L["skew_before"] * 1e12 > 100.0 and L["skew_after"] * 1e12 < 0.01)
    add("deskew does NOT recover absolute margin", "deskew=%.3e, target=%.3e, short=%.3e"
        % (L["smin_desk"], L["smin_long_target"], L["smin_short"]),
        "deskew ≈ 250 ps level (±1%), ≪ short",
        abs(L["smin_desk"] / L["smin_long_target"] - 1.0) < 0.01
        and L["smin_desk"] < 0.5 * L["smin_short"])
    add("S peak grows toward instability", " → ".join("%.1f" % v
        for v in L["speak_seq"]), "increasing",
        all(L["speak_seq"][i] < L["speak_seq"][i + 1] for i in range(3))
        and L["speak_seq"][3] > 10.0 * L["speak_seq"][0])

    # Case K：physical scaling（Review item B 第二半）
    K = CASES["scaling"]
    add("scaling: unscaled 8 MHz passes 10 MHz", K["pass_unscaled"], True,
        K["pass_unscaled"])
    add("scaling: alpha=0.1 → 80 MHz fails 10 MHz", "%.0f MHz"
        % (K["omega_phys"] / 2 / np.pi / 1e6), "fail",
        (not K["pass_scaled"]) and abs(K["omega_phys"] / (2 * np.pi * 80e6) - 1.0) < 1e-9)
    add("scaling: forgetting alpha flips the verdict", "unscaled=%s, scaled=%s"
        % (K["pass_unscaled"], K["pass_scaled"]), "differ (load-bearing)",
        K["pass_unscaled"] != K["pass_scaled"])
    add("scaling: band round-trip + λ_hw=λ/α", "round=%.2e, λ_hw/α=%.1e"
        % (K["omega_round"], K["lam_hw"]),
        "ω·α·(1/α) = ω; λ_hw = 1e6·λ",
        abs(K["omega_round"] - abs(K["lam_model"])) < 1e-6 * abs(K["lam_model"])
        and abs(K["omega_phys_fast"] / 2 / np.pi / 1e12 - 8.0) < 1e-6)
    add("scaling: x=V/S_x and noise budget", "swing→%.2f units, 80µV→%.3f units"
        % (K["v_swing_state"], K["noise_model"]),
        "±2.00 units, 0.080 units",
        abs(K["v_swing_state"] - 2.0) < 1e-9 and abs(K["noise_model"] - 0.08) < 1e-9
        and abs(K["state_back"] - 1.7e-3) < 1e-15)

    # Case M：absolute 判据用 ωc 的 phase delay（P2-route-3）
    M = CASES["phase_delay"]
    add("PureDelay: pd == latency", "%.6f vs %.6f ps"
        % (M["pd_pure"] * 1e12, M["lat_pure"] * 1e12), "identical, pass",
        abs(M["pd_pure"] / M["lat_pure"] - 1.0) < 1e-12 and M["pass_pure"])
    add("in-band pads: pd < latency", "%.1f vs %.1f ps (ratio %.3f)"
        % (M["pd_pad"] * 1e12, M["lat_pad"] * 1e12, M["pd_pad"] / M["lat_pad"]),
        "pd < lat, ratio ≈ 0.65",
        M["pd_pad"] < M["lat_pad"] and 0.6 < M["pd_pad"] / M["lat_pad"] < 0.7)
    add("naive angle wraps (bug the unwrap fixes)", "naive %.1f ps vs true %.1f ps"
        % (M["pd_naive"] * 1e12, M["pd_pad"] * 1e12), "naive < true/2",
        M["pd_naive"] < 0.5 * M["pd_pad"])
    add("unwrap vs ∫τ_g dω cross-check", "%.2e (trapezoid quadrature err)"
        % abs(M["pd_pad"] / M["pd_int"] - 1.0), "< 1e-4",
        abs(M["pd_pad"] / M["pd_int"] - 1.0) < 1e-4)
    add("verdict uses unwrapped phase delay", "τ_max=%.1f ps, pd=%.1f, naive=%.1f, lat=%.1f"
        % (M["tau_max_216"] * 1e12, M["pd_pad"] * 1e12,
           M["pd_naive"] * 1e12, M["lat_pad"] * 1e12),
        "FAIL (naive-angle impl would pass)",
        (not M["pass_pad216"]) and M["pd_pad"] > M["tau_max_216"]
        and M["pd_naive"] < M["tau_max_216"] and M["lat_pad"] > M["tau_max_216"])
    add("out-of-band pads: pd ≈ latency", "%.6f vs %.6f ps"
        % (M["pd_oob"] * 1e12, M["lat_oob"] * 1e12), "ratio 1 ± 1e-3",
        abs(M["pd_oob"] / M["lat_oob"] - 1.0) < 1e-3)
    add("deskewed case-D set: divergence & verdicts", "diverg=%.1e, pass=%s"
        % (M["diverg_deskew"], M["pass_deskew"]),
        "< 1e-3, all True (criterion change keeps verdicts)",
        M["diverg_deskew"] < 1e-3 and all(M["pass_deskew"]))

    return checks


def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CASES.clear()

    hr("Phase 1.6 验收：route = H(s)，不是 wire length")

    case_wire()
    case_buffers()
    case_allpass()
    case_deskew()
    case_feedback()
    case_crosstalk()
    case_sensitivity()
    case_weighted_j()
    case_crosstalk_asym()
    case_loop()
    case_scaling()
    case_phase_delay()

    hr("A. RC wire：distributed vs lumped")
    w = CASES["wire"]
    table(["quantity", "value"],
          [("R", "%.1f Ω" % w["R"]),
           ("C", "%.1f fF" % w["C"]),
           ("RC", "%.1f ps" % (w["tau_rc"] * 1e12)),
           ("DC group delay", "%.2f ps (both)" % (w["gd_dc"] * 1e12)),
           ("t50 distributed (measured)", "%.2f ps = %.4f RC"
            % (w["t50_fd"] * w["tau_rc"] * 1e12, w["t50_fd"])),
           ("t50 distributed (0.3795RC)", "%.2f ps" % (w["t50_dist"] * 1e12)),
           ("t50 lumped (ln2·RC/2)", "%.2f ps" % (w["t50_lump"] * 1e12)),
           ("f3 distributed", "%.2f GHz" % (w["f3_db"] / 2 / np.pi / 1e9)),
           ("f3 lumped", "%.2f GHz" % (w["f3_lump"] / 2 / np.pi / 1e9))])
    emit("  同一个 DC 斜率（RC/2）不等于同一个波形：lumped π 的 t50 短 8.7%、")
    emit("  −3 dB 带宽低 18%；单极点模型不能同时当 delay 和 bandwidth 的模型。")

    hr("B. Buffered fanout：深度 × 相位 × 噪声")
    b = CASES["buffers"]
    table(["depth", "latency", "class", "gain err", "phase err", "noise", "verdict"],
          [(d, "%.1f ps" % (b[d]["latency"] * 1e12), b[d]["latency_class"],
            "%.3f%%" % (b[d]["max_gain_err"] * 100),
            "%.2f°" % b[d]["max_phase_err_deg"],
            "%.1f µV" % (b[d]["noise_rms"] * 1e6),
            "pass" if b[d]["verdict"] else "FAIL") for d in (1, 4)])
    emit("  g=0.999、f_p=2 GHz、σ=10 µV、budget 3°/1%：4 级在 100 MHz 已经 11.5°，")
    emit("  而 gain 只有 0.9%。限制 fanout 深度的是相位，不是增益，也不是噪声。")
    emit("  eta（τ_ideal=0 → 声明 latency）：depth4 %.1f%% → %.2f%%"
         % (b[4]["eta_zero"] * 100, b[4]["eta_declared"] * 100))

    hr("C. 补延迟：all-pass vs RC")
    a = CASES["allpass"]
    table(["quantity", "all-pass", "RC (1 pole)"],
          [("gain error", "%.2e" % a["ap_gain"], "%.2f%%" % (a["rc_gain"] * 100)),
           ("phase error @500 MHz", "%.3f°" % a["ap_phase"], "%.3f°" % a["rc_phase"]),
           ("J_H", "%.2e" % a["ap_jh"], "%.2e" % a["rc_jh"]),
           ("J_τ", "%.1f ps²" % (a["ap_jtau"] * 1e24), "%.1f ps²" % (a["rc_jtau"] * 1e24)),
           ("f_3dB", "inf", "%.2f GHz" % (a["rc_f3"] / 2 / np.pi / 1e9))])
    emit("  τ=200 ps：all-pass 的幅度严格为 1，RC 在 500 MHz 已 −1.4 dB。")
    emit("  用 RC 补 delay 等于用幅度换相位；J_H 差 %.0f×。" % (a["rc_jh"] / a["ap_jh"]))

    hr("D. Deskew：把 skew 修到零，绝对延迟不动")
    d = CASES["deskew"]
    table(["quantity", "value"],
          [("ω0", "%.0f MHz" % (d["omega0"] / 2 / np.pi / 1e6)),
           ("before / after skew", "%.1f ps → %.2e ps"
            % (d["skew_before"] * 1e12, d["skew_after"] * 1e12)),
           ("group spread", "%.1f → %.2e ps²"
            % (d["jt_before"] * 1e24, d["jt_after"] * 1e24)),
           ("common latency", "%.2f ps" % (d["target"] * 1e12)),
           ("loop phase @100 MHz", "%.2f°"
            % float(np.degrees(2 * np.pi * 100e6 * d["target"]))),
           ("1° reserve (27.8 ps)", "FAIL"),
           ("5° reserve (138.9 ps)", "pass")])
    emit("  对齐后三条 edge 都是 ~60 ps：skew=0，但闭环仍看到 2.15°。")
    emit("  所以 deskew 修的是 matched，不是 instantaneous（§2.8）。")

    hr("E. Feedback：绝对延迟预算")
    f = CASES["feedback"]
    table(["edge", "phase lag @100 MHz", "τ_max", "verdict"],
          [("130 ps", "%.2f°" % f["phase_lag_130"],
            "%.1f ps" % (f["tau_max"] * 1e12), "pass"),
           ("150 ps", "%.2f°" % f["phase_lag_150"],
            "%.1f ps" % (f["tau_max"] * 1e12), "FAIL")])
    emit("  两条 150 ps 的路径 skew=0；它们仍然同时违反 5° 预算。")
    emit("  feedback_critical 的 edge 不能靠 pad 解决：pad_can_fix=False。")

    hr("F. Crosstalk：C_m 的地板")
    x = CASES["crosstalk"]
    table(["C_m", "HF crosstalk", "dB", "1 V → victim"],
          [("20 fF", "%.4f" % x["xt_hf"], "%.2f" % x["db_20"],
            "%.1f mV" % (x["v20"] * 1e3)),
           ("2 fF", "%.4f" % x["xt_hf_2f"], "%.2f" % x["db_2"],
            "%.2f mV" % (x["v2"] * 1e3)),
           ("0.1 fF", "%.5f" % x["xt_hf_01f"],
            "%.1f" % (20 * np.log10(x["xt_hf_01f"])),
            "%.2f mV" % (x["v01"] * 1e3))])
    emit("  高频极限 H_12/H_11 = C_m/(C_2+C_m)=%.4f；transition 在 %.1f GHz。"
         % (x["xt_hf"], x["f_trans"] / 1e9))
    emit("  1 mV/1 V 的预算要求 C_m < 0.11 fF；20 dB 屏蔽不够，要 40 dB 或差分。")
    emit("  500 MHz 处 2×2 η（σ_max(H−I)）= %.3f。" % x["eta_500m"])

    hr("G. 从 J 自动算 τ_tol")
    s = CASES["sensitivity"]
    tol = s["tau_tol"]
    names = {(0, 0): "slow self", (1, 1): "fast self", (2, 3): "osc cross",
             (1, 2): "fast←osc", (0, 1): "slow←fast"}
    table(["edge", "|J_ij|", "τ_tol", "worst ω", "HF limit"],
          [(names[ij], "%.2e" % abs(s["J"][ij]), "%.2f ps" % (tol[ij] * 1e12),
            "%.1f MHz" % (s["worst_omega"][ij] / 2 / np.pi / 1e6),
            "%.1f ps" % (s["hf_tau_tol"][ij] * 1e12)) for ij in names])
    emit("  高频极限 τ_tol = ε/|J_ij|：慢模态 2.8 ns，快模态 13.9 ps，差 200×。")
    emit("  Q=10 的 50 MHz 共振把每条 edge 的预算再压 2Q=20×；排序不变。")
    emit("  这就是 sensitivity-driven routing 的输入：不需要手写 critical path。")

    hr("H. 谱权重：J_H 随 W(ω) 改变（P0-A 回归）")
    hj = CASES["weighted_j"]
    table(["weight", "peak", "J_H", "J_τ (ps²)"],
          [("W_low", "%.0f MHz" % (hj["peak_low"] / 2 / np.pi / 1e6),
            "%.4e" % hj["j_low"], "%.4f" % (hj["jtau_low"] * 1e24)),
           ("W_high", "%.0f MHz" % (hj["peak_high"] / 2 / np.pi / 1e6),
            "%.4e" % hj["j_high"], "%.4f" % (hj["jtau_high"] * 1e24)),
           ("uniform", "—", "%.4e" % hj["j_unif"], "—")])
    emit("  同一条 RC pad，误差随 ω 单调增长：把权重峰从 20 MHz 挪到 300 MHz，")
    emit("  J_H 变 %.1f×、J_τ 变 %.1f×。权重必须同时进分子和分母——否则 W 只"
         % (hj["ratio"], hj["jtau_ratio"]))
    emit("  是归一化常数，谱加权形同虚设。uniform/flat 权重与无权重完全一致。")

    hr("I. 非对称串扰：R1≠R2、C1≠C2（P0-B 回归）")
    xi = CASES["crosstalk_asym"]
    table(["quantity", "value", "prediction"],
          [("solve err (vs numpy, 200 pt)", "%.2e" % xi["solve_err"], "< 1e-12"),
           ("reciprocity err R2·H01−R1·H10", "%.2e" % xi["recip_err"], "< 1e-12"),
           ("HF V1/V2 (victim 1)", "%.5f" % xi["floor_v1"],
            "Cm/(C1+Cm)=%.5f" % xi["floor_v1_pred"]),
           ("HF |H01/H00| (crosstalk())", "%.5f" % xi["floor_cross"],
            "Cm/(C2+Cm)·R1/R2=%.5f" % xi["floor_cross_pred"]),
           ("crosstalk_hf property", "%.5f" % xi["floor_prop"],
            "Cm/(C2+Cm)=%.5f" % (xi["Cm"] / (xi["C2"] + xi["Cm"]))),
           ("H01(0)", "%.2e" % xi["h01_dc"], "0"),
           ("H00(0)", "%.12f" % xi["h00_dc"], "1")])
    emit("  交叉项的 R 归属只由输入列决定：H_01=−m12/(det·R2)、H_10=−m12/(det·R1)。")
    emit("  修复前两者写反；R1=R2 的旧算例完全掩盖它。传递阻抗 Z=−m12/det 与 R")
    emit("  无关（互易），所以 R2·H01=R1·H10 逐点成立。")
    emit("  注意：非对称时 crosstalk() 的 |H01/H00| 地板带 R1/R2 因子；纯电容")
    emit("  分压是 V1/V2=Cm/(C1+Cm)。crosstalk_hf 属性仍是 C2 版本，仅对称网成立。")

    hr("K. 闭环 closure：per-edge → whole-loop（Review item B）")
    L = CASES["loop"]
    table(["quantity", "value"],
          [("Loop A: τ_ex (exact, complex-scalar DDE)",
            "%.1f ps" % (L["tau_ex"] * 1e12)),
           ("Loop A: τ_fo (E=I+τA)", "%.1f ps (ratio %.3f)"
            % (L["tau_fo"] * 1e12, L["ratio_fo"])),
           ("Loop A: σ_min dip (H=1)", "%.2f MHz, depth %.3e (= a)"
            % (L["dip_omega"] / 2 / np.pi / 1e6, L["dip_smin"])),
           ("Loop A: σ_min_min vs τ", " → ".join("%.2e" % v for v in L["smin_seq"])),
           ("Loop A: ‖S‖∞ vs τ", " → ".join("%.1f" % v for v in L["speak_seq"])),
           ("deskew: margin", "short %.3e → deskewed %.3e (= 250 ps level %.3e)"
            % (L["smin_short"], L["smin_desk"], L["smin_long_target"])),
           ("Loop B: τ_fo / τ_ex", "%.4f (2/π = %.4f)" % (L["ratio2"], 2 / np.pi))])
    emit("  per-edge closure 全过不等于闭环安全：σ_min(jωI−J∘H(jω)) 随 τ 单调")
    emit("  收缩，‖S‖∞ 从 10 涨到 200，deskew 把 skew 修到零但 margin 停在")
    emit("  250 ps 的绝对延迟水平——§2.8 的 matched ≠ instantaneous 在 loop 层。")
    emit("  近标量 loop 的 τ_fo/τ_ex = %.3f ≈ 2/π：§6.14 的乐观性在 loop 层复现。"
         % L["ratio2"])
    emit("  诚实声明：σ_min→0 是必要条件探针；DDE 精确根只在标量 1-DOF")
    emit("  (scalar_dde_root) 闭式，本例用它在 τ_ex 处验证根落在 jω 轴上。")

    hr("L. 物理时间/电压 scaling：B_hw 比较必须走 PhysicalScaling")
    K = CASES["scaling"]
    table(["quantity", "value"],
          [("λ (model)", "%.2f MHz (rad/model-time)" % (abs(K["lam_model"]) / 2 / np.pi / 1e6)),
           ("B_hw", "%.0f MHz" % (K["B_hw"] / 1e6)),
           ("unscaled verdict", "pass" if K["pass_unscaled"] else "fail"),
           ("alpha = 0.1 s/model-unit → ω_phys", "%.0f MHz → %s"
            % (K["omega_phys"] / 2 / np.pi / 1e6,
               "pass" if K["pass_scaled"] else "FAIL")),
           ("alpha = 1e-6 s/model-unit → ω_phys", "%.1f THz"
            % (K["omega_phys_fast"] / 2 / np.pi / 1e12)),
           ("x = V/S_x (S_x=1 mV)", "2 mV swing → ±%.2f units; 80 µV → %.3f units"
            % (K["v_swing_state"], K["noise_model"]))])
    emit("  同一个 8 MHz 模态：不 scaling 过 10 MHz 预算，alpha=0.1 时硬件")
    emit("  看到 80 MHz 而 fail。两个 verdict 不一致就是测试：忘掉 alpha 的")
    emit("  B_hw 比较会被这个 borderline 案例抓住。方向是 t_phys=alpha·t_model，")
    emit("  所以 ω_phys = ω_model/alpha、λ_hw = λ_model/alpha、x = V/S_x。")

    hr("M. absolute 判据：ωc 的 phase delay vs DC latency（P2-route-3）")
    M = CASES["phase_delay"]
    table(["edge @ ωc", "latency (DC)", "phase delay", "naive angle", "verdict"],
          [("PureDelay 10 ps @1 GHz", "%.2f ps" % (M["lat_pure"] * 1e12),
            "%.2f ps" % (M["pd_pure"] * 1e12), "—", "pass"),
           ("5 ps + 4×AllPass(50 ps) @10 GHz", "%.1f ps" % (M["lat_pad"] * 1e12),
            "%.1f ps" % (M["pd_pad"] * 1e12), "%.1f ps (wrapped!)" % (M["pd_naive"] * 1e12),
            "FAIL @216°"),
           ("5 ps + 4×AllPass(1 ps) @1 GHz", "%.3f ps" % (M["lat_oob"] * 1e12),
            "%.3f ps" % (M["pd_oob"] * 1e12), "—", "pass"),
           ("deskewed case-D 组 @100 MHz", "≈60 ps", "divergence %.1e" % M["diverg_deskew"],
            "—", "all pass")])
    emit(" 闭环感到的是 −∠H(jωc)/ωc。pad 极点在带外时 pd≈latency（1e-4）；")
    emit(" 极点在带内时分开：lat 205 ps vs pd 132.8 ps——latency 判据是保守侧。")
    emit(" 4 个 50 ps all-pass 在 10 GHz 总相位 −8.08 rad，单点 np.angle 卷绕成")
    emit(" −1.80 rad（naive pd 32.8 ps）；unwrap 后与 ∫₀^ωc τ_g dω 一致到 1e-5（差值")
    emit(" 是梯形积分的离散误差，不是 unwrap 误差），verdict 按 unwrapped pd 判：")
    emit(" 216° reserve 下 naive-angle 实现会错放行。")
    emit()
    emit(" 模型局限（P2-route-1 / P2-route-2，读数时不要超出）：")
    emit(" 1. RouteEdge.H 是各 block 开路 transfer 的乘积，忽略无 buffer 隔离")
    emit("    的段间阻抗加载（会高估带宽）；正确做法是 ABCD 级联，未实现。")
    emit(" 2. noise_var 按 DC 增益² 折算，忽略后级极点的 ENBW 压缩（保守侧）。")
    emit(" 本仓库立场是关键路径由 buffer 隔离，这两条由 PEX/SPICE 阶段兜底。")

    hr("J. 验收")
    checks = acceptance()
    table(["check", "got", "expected", "PASS"],
          [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
           for c in checks])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    payload = dict(cases=CASES, checks=checks,
                   summary=dict(n_checks=len(checks),
                                n_passed=len(checks) - n_fail))
    with open(os.path.join(OUT, "route.json"), "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    with open(os.path.join(OUT, "route_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/route.json")
    emit("  report → results/route_report.txt")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
