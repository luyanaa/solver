"""三个 anchor benchmark problem bundle（Stage A5 step 3）。

每个 bundle = PDESpec → discretize → (descriptor, reference, run_twin)。
reference 只放解析/高分辨率参考与"预测失败轴"；run_twin() 返回**测量值**
（孪生不需要读证书 —— 证书对比是 pipeline bench 的职责，见
bench/run_pipeline.py）。

  diffusion-1d  ν=1e-3, N=64 Fourier —— 可行（κ=4096 ≪ κ_hw=1e6），
                expected_axis=None。参考：模态精确衰减。
  poisson-2d    16×16 FD 约束网格，Dirichlet（接地环软 Dirichlet，O(h²)），
                c_node=1e-12，expected_axis=None —— P6 原始证书判
                infeasible/bandwidth（model-time bandwidth=λ_min(G)/c_node
                ≈1.2e11 rad/s ≫ B_hw）是**正确**判决；α=1.578e6 的时间缩放把
                物理带宽折回窗内（sc.bandwidth_pass=True），即 §6.15 的
                verdict-flip —— run_pipeline 对"原始 bandwidth 违规但 α 可
                移除"记 feasible-after-P3-scaling 并显式入账，不算 raw 可行、
                也不是评 sc.scaled（slack 带宽轴尚无 α 入口）。
                robustness_A=∞（G 守恒零模 → Γ_proj）是第二条真实违规，失配
                误差经 §3.4 漂移动预算进 floor_bound。
                参考：解析 u = sin(πx)sin(πy)/(2π²) + G⁻¹I 直接解。
  burgers-1d    N=128 FV，线性化状态 = T=0.3 的 pre-shock 精确解
                （1/π≈0.318 未成激波），expected_axis="representation"
                （run_phase0 已确立 Burgers 绑定表示轴）。
                T=0.5 的参考是**后激波**时刻：inviscid_burgers_exact 的
                特征线在 T≥1/π 多值，不可用；改用 run_transform case_weno
                的机制 —— 高分辨率 WENO5（n_ref=512）收敛参考，块平均到
                N=128 网格。§6.16 的 shock 位置 anchor（0.7480，n=256）随
                reference 一并记录。

符号约定（poisson）：G·u = h²·sin(πx)sin(πy)（推导：G·V 在节点 i 是
Σ g(V_i−V_j) ≈ −h²ΔV，−Δu = +sin·sin）。已对解析解数值验证（见
_discrete_poisson_rhs 的 self-check）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis import Descriptor, shock_sensor
from bench.run_phase0 import inviscid_burgers_exact  # 参考解，非求解器
from bench.run_twin_experiments import burgers  # 允许的求解器位置
from lowering.discretize import PDESpec, discretize, linearize_rhs
from twin.transform import (SigmaTracker, blend_face_flux, central_face_flux,
                            conservation_drift, muscl_face_flux,
                            rhs_from_face_flux, smoothness_indicator)

PROBLEMS = ("diffusion-1d", "poisson-2d", "burgers-1d")


@dataclass
class ProblemBundle:
    name: str
    expected_axis: str | None       # 预测绑定 slack 轴（None = 可行）
    descriptor: object              # analysis.Descriptor
    system: object | None           # 无符号 System（FV rhs 是数值的，符号
                                    # AST 即 PDESpec 本身；见 build_burgers）
    grid: tuple
    spec_meta: dict
    reference: dict
    run_twin: object                # callable() -> 测量值 dict


def _rk4(u, dt, rhs):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ===========================================================================
# diffusion-1d
# ===========================================================================
def build_diffusion_1d():
    spec = PDESpec(name="diffusion-1d", operator="diffusion", bc="periodic",
                   domain=((0.0, 1.0),), coeff=1e-3, n=(64,), basis="fourier")
    d = discretize(spec)
    n, nu = 64, 1e-3
    T = 1.0
    # û(0) = e1（规范扰动端口，B=e1）
    u0 = np.zeros(n)
    u0[0] = 1.0

    def reference_u(t):
        k = np.arange(1, n + 1, dtype=float)
        return np.exp(-nu * k * k * t) * u0

    # RK4 步长：h = 1e-3 × 最快时间常数 (ν·k_max²)⁻¹ → 截断 ~h⁴(νk_max²)⁵，
    # 远小于任何预测 σ_y 尺度。
    h = 1e-3 / (nu * n * n)
    nsteps = int(np.ceil(T / h))

    def run_twin():
        u = u0.copy()
        dt = T / nsteps
        lam = -nu * np.arange(1, n + 1, dtype=float) ** 2
        for _ in range(nsteps):
            u = _rk4(u, dt, lambda v: lam * v)
        u_end = u
        det_err = float(np.max(np.abs(u_end - reference_u(T))))
        # 硬件级白噪声注入：B = e1 单位白噪声。u_k>1 从零初值恒为零，C=I
        # 输出 y = u_1 是唯一的受驱分量 —— 对线性 OU 用精确 AR(1) 更新
        # （无漂移离散化误差），跑 4096 条独立链估计平稳输出 std，
        # 对照 Lyapunov σ_y² = 1/(2ν)。
        h_noise = 0.25                      # λ_max·h = 1.02 < 2（AR(1) 稳定）
        n_chains = 4096
        burn_steps = int(5.0 * (1.0 / nu) / h_noise)   # 5×最慢时间常数
        samp_steps = int(2000.0 / h_noise)              # 2×τ_corr 采样窗
        rng = np.random.default_rng(7)
        a1 = 1.0 - nu * h_noise
        x = rng.standard_normal(n_chains)               # 任意初值，burn 后遗忘
        sq = np.sqrt(h_noise)
        s1 = 0.0
        s2 = 0.0
        m = 0
        for _ in range(burn_steps):
            x = a1 * x + sq * rng.standard_normal(n_chains)
        for _ in range(samp_steps):
            x = a1 * x + sq * rng.standard_normal(n_chains)
            s1 += x.sum()
            s2 += float(x @ x)
            m += n_chains
        meas_sigma = float(np.sqrt(s2 / m - (s1 / m) ** 2))
        return dict(
            det_err=det_err, meas_sigma=meas_sigma,
            u_end=u_end, u_exact=reference_u(T),
            dt_rk4=float(T / nsteps), h_noise=h_noise,
            t_noise=float(h_noise * (burn_steps + samp_steps)),
            n_chains=n_chains,
        )

    return ProblemBundle(
        name="diffusion-1d", expected_axis=None,
        descriptor=d.descriptor, system=None, grid=d.grid,
        spec_meta={**{k: v for k, v in d.meta.items() if k != "mesh"},
                   "T": T, "u0": "e1"},
        reference=dict(u=reference_u, T=T, u0=u0.copy(),
                       kappa_analytic=float(n * n),
                       hinf_analytic=float(1.0 / nu),
                       sigma_y_analytic=float(np.sqrt(1.0 / (2.0 * nu)))),
        run_twin=run_twin,
    )


# ===========================================================================
# poisson-2d
# ===========================================================================
def build_poisson_2d():
    nx = ny = 16
    spec = PDESpec(name="poisson-2d", operator="poisson", bc="dirichlet",
                   domain=((0.0, 1.0), (0.0, 1.0)), coeff=1.0,
                   n=(nx, ny), basis="fd")
    d = discretize(spec)
    mesh = d.meta["mesh"]
    x, y = d.grid
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y) / (2.0 * np.pi ** 2)
    h, hy = d.meta["h"], d.meta["hy"]
    i_rhs = (np.sin(np.pi * X) * np.sin(np.pi * Y) * h * hy).ravel()
    # self-check：离散 RHS 对解析解的残差是软 Dirichlet 的 O(h²) 一致性误差
    resid_analytic = float(np.max(np.abs(mesh.G @ u_exact.ravel() - i_rhs)))
    assert resid_analytic < 3.0 * h * h, \
        f"poisson self-check failed: ||G u_analytic − I||∞ = {resid_analytic}"
    v_direct = np.linalg.solve(mesh.G, i_rhs)
    c_node = 1e-12
    from twin.constraint import relax_time as _rt, mismatch_floor as _mf
    tau_pred = _rt(mesh, c_node)
    floor_bound = _mf(mesh, i_rhs, lam=0.0, eps_dev=0.01,
                      seed=None, n_draws=1)["bound_inf"]
    descriptor = None
    try:
        from twin.constraint import kcl_relaxation_descriptor
        descriptor = kcl_relaxation_descriptor(mesh, c_node)
    except ImportError:                     # pragma: no cover
        pass
    from twin.constraint import (mismatch_floor, relax_time, settle,
                                 settle_with_c)

    def run_twin():
        # (a) 松弛到平稳 vs 直接解：t_end = 40τ_relax（e^{-40} ≈ 4e-18 →
        # 残余瞬变在机器精度以下）
        tau = relax_time(mesh, c_node)
        t_end = 40.0 * tau
        t = np.linspace(0.0, t_end, 200)
        V = settle_with_c(mesh, c_node, i_rhs, t, v0=np.zeros(mesh.n))
        v_star = V[-1]
        steady_err = float(np.max(np.abs(v_star - v_direct)))
        # (c) τ 拟合：指数尾斜率，拟合窗 [2τ, 10τ]（> 2τ 后 λ_min 模态主导，
        #     < 40τ 则未触及 roundoff 地板）。tail = e^{−λ_min t/C}
        lam_min = float(np.linalg.eigvalsh(0.5 * (mesh.G + mesh.G.T))[0])
        i1 = int(np.searchsorted(t, 2.0 * tau))
        i2 = int(np.searchsorted(t, 10.0 * tau))
        tail = np.max(np.abs(V[i1:i2 + 1] - v_star), axis=1)
        # tail(t) = A·e^{−λ_min·t/C} ⇒ C/λ_min = −Δt / ln(ratio)
        tau_fit = float(-(t[i2] - t[i1]) / np.log(tail[-1] / tail[0]))
        # (b) 失配底噪：一次固定 seed 抽样（确定性 bound 由 pipeline 对照）
        mf = mismatch_floor(mesh, i_rhs, lam=0.0, eps_dev=0.01, seed=1234)
        # (d) KCL 残差
        resid = float(np.max(np.abs(mesh.G @ v_star - i_rhs)))
        return dict(
            steady_err=steady_err, floor_meas=mf["floor_inf"],
            tau_fit=tau_fit, residual=resid,
            tau_relax_analytic=float(tau),
            resid_analytic=resid_analytic,
            v_star=v_star, v_direct=v_direct,
        )

    return ProblemBundle(
        name="poisson-2d", expected_axis=None,
        descriptor=descriptor, system=None, grid=d.grid,
        spec_meta={**{k: v for k, v in d.meta.items() if k != "mesh"},
                   "c_node": c_node,
                   "rhs": "h*hy*sin(pi x)sin(pi y)",
                   "analytic": "sin(pi x)sin(pi y)/(2 pi^2)"},
        reference=dict(u=u_exact, i_rhs=i_rhs, v_direct=v_direct, h=h,
                       resid_analytic=resid_analytic,
                       tau_pred=tau_pred, floor_bound=floor_bound),
        run_twin=run_twin,
    )


# ===========================================================================
# burgers-1d
# ===========================================================================
def build_burgers_1d():
    n = 128
    T_lin = 0.3
    T_run = 0.5
    spec = PDESpec(name="burgers-1d", operator="burgers", bc="periodic",
                   domain=((0.0, 1.0),), n=(n,), basis="fv",
                   meta=dict(alpha=1.6, linearize_at=("exact", T_lin)))
    d = discretize(spec)
    dx = d.meta["dx"]
    x = (np.arange(n) + 0.5) * dx                       # run_phase0/run_transform 的网格
    rhs_func = d.rhs_func
    u_lin = inviscid_burgers_exact(x, T_lin)            # pre-shock（T<1/π）
    J = linearize_rhs(rhs_func, u_lin, eps=1e-6)
    B = np.sin(2.0 * np.pi * x)
    B = B / np.linalg.norm(B)                           # 零均值：不驱动守恒模态
    descriptor = Descriptor(np.eye(n), J, B, np.eye(n))
    reg = shock_sensor(u_lin)
    # reg_w：WENO5 捕捉解的谱尾 —— "capturing hides the tail" 的对照证据
    u_w_lin = burgers(inviscid_burgers_exact(x, 0.0), T_lin, "weno5",
                      cfl=0.2, alpha=1.6)
    reg_w = shock_sensor(u_w_lin)
    # 初值：u0 = 0.5+0.5 sin(2πx) 点值 —— 与 run_phase0.build_burgers 和
    # run_transform.case_weno（§6.16 shock anchor 0.7480 的出处）完全一致，
    # 这样 run_twin 的 l2/shock_x 与既有记录可直接对照。
    u0 = 0.5 + 0.5 * np.sin(2.0 * np.pi * x)

    def _shock_x(u, dx):
        i = int(np.argmax(np.abs(np.diff(u))))
        return (i + 0.5) * dx

    # 后激波参考：高分辨率 WENO5（run_transform case_weno 的机制——u_ref =
    # 细网格 WENO5 收敛解），块平均 4→1 到 N=128 的单元平均。
    n_ref, block = 512, 4
    dx_ref = 1.0 / n_ref
    x_ref = (np.arange(n_ref) + 0.5) * dx_ref
    u0_ref = 0.5 + 0.5 * np.sin(2.0 * np.pi * x_ref)
    u_ref_fine = burgers(u0_ref, T_run, "weno5", cfl=0.2, alpha=1.6)
    u_ref = u_ref_fine.reshape(n, block).mean(axis=1)
    shock_x_exact = _shock_x(u_ref_fine, dx_ref)
    flux = lambda v: 0.5 * v * v

    def run_twin():
        cfl = 0.2
        u_w = burgers(u0, T_run, "weno5", cfl=cfl, alpha=1.6)
        u_f = burgers(u0, T_run, "first", cfl=cfl, alpha=1.6)
        # adaptive σ-blend：run_transform case_weno 的驱动模式，n=128
        dt = cfl * dx / max(float(np.max(np.abs(u0))), 1e-12)
        nsteps = int(np.ceil(T_run / dt))
        dt = T_run / nsteps
        tracker = SigmaTracker(beta0=0.15, k=16.0, hysteresis=0.08,
                               max_rate=400.0, release_rate=10.0)
        u_ad = u0.copy()
        E = float(np.sum(0.5 * u_ad * u_ad) * dx)
        dE_max = -np.inf
        dE_min = np.inf
        for _ in range(nsteps):
            sig = tracker.update(smoothness_indicator(u_ad), dt)

            def rhs_adapt(v, sig=sig):
                a = max(float(np.max(np.abs(v))), 1e-12)
                Fs = central_face_flux(v, flux)
                Fw = muscl_face_flux(v, flux, a)
                return rhs_from_face_flux(blend_face_flux(Fs, Fw, sig), dx)

            u_ad = _rk4(u_ad, dt, rhs_adapt)

            E_new = float(np.sum(0.5 * u_ad * u_ad) * dx)
            dE = E_new - E
            dE_max = max(dE_max, dE)
            dE_min = min(dE_min, dE)
            E = E_new
        entropy_total = float(E - float(np.sum(0.5 * u0 * u0) * dx))
        # 判据（与 run_transform case_weno 的验收同式）：每步 ΔE ≤ 1e-10。
        # 追踪净熵变化作为额外测量值（周期域熵不等式 → 总变分必须 < 0）。
        entropy_sign_ok = bool(dE_max <= 1e-10 and entropy_total < 0.0)
        l2 = lambda u: float(np.sqrt(np.mean((u - u_ref) ** 2)))
        mass_w = conservation_drift(u_w, u0, dx)
        mass_b = conservation_drift(u_ad, u0, dx)
        return dict(
            l2_weno5=l2(u_w), l2_first=l2(u_f), l2_blend=l2(u_ad),
            mass_drift=float(mass_b), conservation=float(mass_w),
            shock_x_blend=_shock_x(u_ad, dx), shock_x_weno5=_shock_x(u_w, dx),
            shock_x_exact=shock_x_exact,
            entropy_dE_max=float(dE_max), entropy_dE_min=float(dE_min),
            entropy_total=entropy_total,
            entropy_sign_ok=entropy_sign_ok,
            u_weno5=u_w, u_blend=u_ad, u_first=u_f,
            nsteps=nsteps, dt=float(dt),
        )

    return ProblemBundle(
        name="burgers-1d", expected_axis="representation",
        descriptor=descriptor, system=None, grid=(x,),
        spec_meta={**{k: v for k, v in d.meta.items()},
                   "cfl": 0.2,
                   "rhs_func": rhs_func,       # FV rhs 是数值的；符号
                   "linearize_at": dict(state="inviscid_burgers_exact",
                                        T=T_lin),  # AST 即 PDESpec
                   "u0": "0.5+0.5 sin(2 pi x) point values "
                         "(run_phase0/run_transform convention)",
                   "reg": dict(p=float(reg["p"]), E_tail=float(reg["E_tail"])),
                   "reg_w": dict(p=float(reg_w["p"]),
                                 E_tail=float(reg_w["E_tail"]))},
        reference=dict(
            u=u_ref, u_fine=u_ref_fine, x_fine=x_ref, n_ref=n_ref,
            t=T_run, u_lin=u_lin, T_lin=T_lin,
            shock_x_exact=shock_x_exact, shock_anchor_s616=0.7480,
            reg=dict(p=float(reg["p"]), E_tail=float(reg["E_tail"])),
            B=B, note="T=0.5 post-shock: exact reference is the "
                      "high-resolution WENO5 convergent reference "
                      "(run_transform case_weno mechanism), not "
                      "inviscid_burgers_exact (multi-valued for T ≥ 1/π)"),
        run_twin=run_twin,
    )


_BUILDERS = {
    "diffusion-1d": build_diffusion_1d,
    "poisson-2d": build_poisson_2d,
    "burgers-1d": build_burgers_1d,
}


def build_problem(name: str) -> ProblemBundle:
    if name not in _BUILDERS:
        raise KeyError(f"unknown problem {name!r}; known: {PROBLEMS}")
    return _BUILDERS[name]()
