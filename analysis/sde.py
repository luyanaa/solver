"""SDE 线性代数层（DESIGN.md §1.3、§1.4、§6.11、§10 计划的 sde.py）。

三件事，全部纯线性代数，不做积分、不做 Monte Carlo（那是 bench/run_sde.py 的事）：

    linear_sde_covariance(A, Gs, Q)   二阶矩 Lyapunov 方程 + 均方稳定（§1.3）
    ito_to_stratonovich / ...         修正漂移，编译期合同（§1.4）
    noise_realizability(spec)         噪声合同的带宽数（§1.3 + §6.11 的 6.4%）

约定（与 lyapunov.py 同一种自检哲学）：每个求解都带 residual，必须 ~1e-12；
每个解析常数都在 docstring 里可推导。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

__all__ = [
    "SdeResult", "linear_sde_covariance",
    "multiplicative_correction", "ito_to_stratonovich",
    "stratonovich_to_ito", "linear_correction", "hardware_drift",
    "wong_zakai_factor",
    "NoiseSpec", "NoiseRealizability", "noise_realizability",
    "ou_variance_deficit",
]

# 二阶矩向量化是 n² 的稠密 Sylvester，n≤32（DESIGN.md §1.3 写明的上限）。
_N_MAX = 32


# ---------------------------------------------------------------------------
# 1. 线性乘性二阶矩（§1.3）
# ---------------------------------------------------------------------------

@dataclass
class SdeResult:
    """linear_sde_covariance 的返回。

    mean_square_stable 只认证二阶矩（方差），不认证几乎必然稳定性——
    §1.3 明确"均方稳定 ≠ 样本稳定"。
    """

    P: np.ndarray                 # E[x xᵀ]
    sigma_y2: float               # tr(C P Cᵀ)，未给 C 时为 NaN
    mean_square_stable: bool
    stability_margin: float       # max Re eig(L)，<0 即均方稳定
    residual: float
    rel_residual: float
    method: str


def _vec(P: np.ndarray) -> np.ndarray:
    """列堆叠 vec（Fortran 序），与 kron 约定 I⊗A / A⊗I 配套。"""
    return P.reshape(-1, order="F")


def _unvec(p: np.ndarray, n: int) -> np.ndarray:
    return p.reshape((n, n), order="F")


def sylvester_operator(A: np.ndarray, Gs: list[np.ndarray]) -> np.ndarray:
    """L = I⊗A + A⊗I + Σ_k G_k⊗G_k，作用在 vec(P) 上（列堆叠）。

    推导：dP/dt = A P + P Aᵀ + Σ G_k P G_kᵀ + W，列堆叠恒等式
    vec(A P) = (I⊗A) vec(P)、vec(P Aᵀ) = (A⊗I) vec(P)、
    vec(G P Gᵀ) = (G⊗G) vec(P)。标量情形 A=[μ]、G=[σ] 时 L = 2μ+σ²，
    均方稳定 ⇔ 2μ+σ² < 0——GBM 的解析判据，vec 约定由它钉死。
    """
    n = A.shape[0]
    I = np.eye(n)
    L = np.kron(I, A) + np.kron(A, I)
    for G in Gs:
        L = L + np.kron(G, G)
    return L


def linear_sde_covariance(
    A: np.ndarray,
    Gs: list[np.ndarray] | tuple[np.ndarray, ...] = (),
    Q: np.ndarray | None = None,
    C: np.ndarray | None = None,
) -> SdeResult:
    """解二阶矩方程 A P + P Aᵀ + Σ_k G_k P G_kᵀ + W = 0（DESIGN.md §1.3）。

    Q 是加性噪声的协方差 W = B Q Bᵀ（与 solve_generalized_lyapunov 的
    sys.B @ Q @ sys.B.T 同一个量）；直接传 W。G_k(x) = G_k·x 为线性乘性，
    G_k 是常数矩阵。向量化成 n²×n² 稠密方程用 scipy.linalg.solve，
    n≤32（§1.3 写明的可解上限）。Gs=[] 时严格退化为普通 Lyapunov——
    验收时与 solve_generalized_lyapunov 对拍。
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")
    n = A.shape[0]
    if n > _N_MAX:
        raise ValueError(f"n={n} 超过 §1.3 的 n≤32 上限")
    Gs = [np.asarray(G, dtype=float) for G in Gs]
    for G in Gs:
        if G.shape != A.shape:
            raise ValueError("G_k 必须与 A 同形")
    if Q is None:
        Q = np.zeros((n, n))
    W = np.asarray(Q, dtype=float)
    if W.shape != (n, n):
        raise ValueError("Q 形状应为 (n, n)")

    L = sylvester_operator(A, Gs)
    eigs = linalg.eigvals(L)
    margin = float(np.max(np.real(eigs)))
    stable = bool(margin < 0.0)

    # 非均方稳定（margin ≥ 0）时二阶矩方程无有界解：P 记 NaN，不硬解。
    if not stable:
        return SdeResult(np.full((n, n), np.nan), float("nan"), False,
                         margin, float("nan"), float("nan"),
                         "unstable-no-solve")

    b = -_vec(W)
    p = linalg.solve(L, b)
    P = _unvec(p, n)
    P = 0.5 * (P + P.T)

    # 自检：与 coupling.py 的 mass budget 同种角色，必须 ~1e-12。
    R = L @ _vec(P) - b
    num = float(np.linalg.norm(R))
    den = float(np.linalg.norm(L @ _vec(P))) + float(np.linalg.norm(b))
    rel = num / max(den, 1e-300)

    sigma_y2 = float(np.real(np.trace(np.asarray(C) @ P @ np.asarray(C).T))) \
        if C is not None else float("nan")
    return SdeResult(P, sigma_y2, stable, margin, num, rel, "sylvester-n2")


# ---------------------------------------------------------------------------
# 2. Itô / Stratonovich 修正漂移（§1.4）
# ---------------------------------------------------------------------------

def multiplicative_correction(
    f, G, x: np.ndarray, eps: float | None = None
) -> np.ndarray:
    """½ Σ_{j,k} G_jk(x) ∂G_ik/∂x_j，数值 Jacobian 版（§1.4 修正项）。

    G 可以是：
      * list/tuple of (n,n) 常数矩阵 —— 线性乘性 G_k(x)=G_k·x，走解析路径；
      * 单个 callable，返回 (n, m) 矩阵 G(x)，列为各 Wiener 通道。

    ∂G/∂x 用中心差分，步长 eps = 1e-6 · max(1, |x_j|)（相对双精度最优
    步长的量级：h ~ eps_machine^(1/3) · scale ≈ 6e-6·scale，取 1e-6 折中
    截断与舍入）。解析常数：f_S,i = f_I,i − ½ Σ G_jk ∂G_ik/∂x_j，
    即本函数返回的量前面带负号进入 f_S、带正号进入 f_I。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if isinstance(G, np.ndarray) and G.ndim == 2 and not callable(G):
        G = [G]  # 单个 (n,n) 常数矩阵 = 一个线性乘性通道
    if callable(G) and not isinstance(G, (list, tuple)):
        Gx = np.asarray(G(x), dtype=float)
        if Gx.shape[0] != n:
            raise ValueError("G(x) 的行数必须等于状态维数")
        m = Gx.shape[1]
        corr = np.zeros(n)
        if eps is None:
            eps = 1e-6
        for j in range(n):
            h = eps * max(1.0, abs(x[j]))
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            dGj = (np.asarray(G(xp), dtype=float)
                   - np.asarray(G(xm), dtype=float)) / (2.0 * h)
            # corr_i += ½ Σ_k G_jk · ∂G_ik/∂x_j
            corr += 0.5 * (Gx[j, :] @ dGj.T)
        return corr

    # 线性乘性：G_k(x) = G_k·x ⇒ ∂G_ik/∂x_j = (G_k)_ij、G_jk(x) = (G_k x)_j，
    # 于是 corr = ½ Σ_k G_k (G_k x) = ½ Σ_k G_k² x——是状态的函数，不是常数；
    # 写进 drift 就是 A_S = A − ½ Σ_k G_k²（§1.4）。解析式，无差分误差。
    corr = np.zeros(n)
    for Gk in G:
        Gk = np.asarray(Gk, dtype=float)
        if Gk.shape != (n, n):
            raise ValueError("线性 G_k 必须是 (n, n)")
        corr += 0.5 * (Gk @ (Gk @ x))
    return corr


def _apply_drift(f, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    val = f(x)
    return np.asarray(val, dtype=float)


def ito_to_stratonovich(f, G, x: np.ndarray) -> np.ndarray:
    """模型 Itô、硬件 Stratonovich：f_S = f_I − ½ Σ G ∂G（§1.4 方向写死）。"""
    x = np.asarray(x, dtype=float)
    return _apply_drift(f, x) - multiplicative_correction(f, G, x)


def stratonovich_to_ito(f, G, x: np.ndarray) -> np.ndarray:
    """模型 Stratonovich、硬件 Itô：f_I = f_S + ½ Σ G ∂G（§1.4 方向写死）。"""
    x = np.asarray(x, dtype=float)
    return _apply_drift(f, x) + multiplicative_correction(f, G, x)


def linear_correction(A: np.ndarray, Gs) -> tuple[np.ndarray, np.ndarray]:
    """线性乘性 G_k(x)=G_k·x 的解析修正（§1.4）：A_S = A − ½ Σ_k G_k²。

    注意矩阵平方，不是 G Gᵀ：§1.4 写的是 G_k²。返回 (A_S, A_I)，
    其中 A_I = A + ½ Σ G_k²（反向合同用）。
    """
    A = np.asarray(A, dtype=float)
    half = np.zeros_like(A)
    for G in Gs:
        G = np.asarray(G, dtype=float)
        if G.shape != A.shape:
            raise ValueError("G_k 必须与 A 同形")
        half = half + 0.5 * (G @ G)
    return A - half, A + half


def hardware_drift(
    model_interp: str,
    hardware_limit: str,
    f,
    G,
    x: np.ndarray,
) -> np.ndarray:
    """按 §1.4 的编译期合同加减修正漂移。

    model_interp ∈ {ito, stratonovich}：用户模型声明的解释；
    hardware_limit ∈ {stratonovich, ito}：电路实际的噪声极限。
    两者相同 ⇒ 漂移不变；ito 模型 + strato 硬件 ⇒ 减；strato 模型 +
    ito 硬件 ⇒ 加。未知字符串直接 raise——合同字段不允许静默默认。
    """
    if model_interp not in ("ito", "stratonovich"):
        raise ValueError(f"未知 model_interp: {model_interp!r}")
    if hardware_limit not in ("ito", "stratonovich"):
        raise ValueError(f"未知 hardware_limit: {hardware_limit!r}")
    if model_interp == hardware_limit:
        return _apply_drift(f, x)
    if model_interp == "ito":
        return ito_to_stratonovich(f, G, x)
    return stratonovich_to_ito(f, G, x)


# ---------------------------------------------------------------------------
# 3. 广义 Wong–Zakai 因子（延迟反馈回路下的乘性噪声漂移）
# ---------------------------------------------------------------------------

def wong_zakai_factor(tau_d: float, tau_c: float) -> float:
    """广义 Wong–Zakai 漂移因子 θ(χ) = 1/(2(1+τ_d/τ_c))，χ = τ_d/τ_c。

    经典 Wong–Zakai：乘性噪声 G(x)ξ(t) 的有色源经 Stratonovich 解释
    进入 Itô 时，漂移修正是 ½·G·∂G/∂x——系数固定 ½。这里的推广是：
    当乘性通道经过一个延迟为 τ_d 的硬件回路（乘法器输出在反馈/注入
    前被延迟 τ_d），噪声与状态响应的重叠记忆缩短，修正系数变为

        θ(χ) = G·∂G/∂x / (2(1+τ_d/τ_c))。

    极限（写死约定，避免经典歧义）：
      * χ = 0（τ_d ≪ τ_c，回路瞬时）：θ = ½，严格回到经典
        Stratonovich 修正。
      * χ → ∞（τ_d ≫ τ_c，纯白噪声 + 延迟回路）：θ → 0，即 Itô——
        延迟把噪声与响应完全去相关，修正消失。
      * τ_d = 0 且 τ_c → 0 同时取极限（白且瞬时）是经典教科书里的
        歧义极限：Stratonovich/Itô 语义取决于离散化约定，不在本函数
        的单一自变量里裁定。本合同约定：先取 χ = τ_d/τ_c（噪声
        相关性固定的语义），τ_c→0 沿 χ=0 走 Stratonovich ½。
        需要纯 Itô 语义的场合必须在 §1.4 合同里显式声明 hardware_limit
        而不是依赖这个极限顺序。

    返回的是无量纲系数 θ；漂移修正 = θ·G·∂G/∂x。测点：
    bench/run_sde.py 的 case F（一阶极点代替纯延迟的诚实模型，
    χ ∈ {0, 0.25, 1, 4, 16} 实测曲线 vs 本式）。
    """
    if tau_d < 0.0:
        raise ValueError("tau_d 必须非负")
    if tau_c < 0.0:
        raise ValueError("tau_c 必须非负")
    if tau_c == 0.0:
        if tau_d == 0.0:
            # 白且瞬时：经典歧义极限，按上面的约定给 Stratonovich ½。
            return 0.5
        # τ_c = 0 且 τ_d > 0：χ → ∞ 分支，Itô。
        return 0.0
    return 1.0 / (2.0 * (1.0 + tau_d / tau_c))


# ---------------------------------------------------------------------------
# 4. 噪声可实现性（§1.3 噪声合同 + §6.11 的 6.4%）
# ---------------------------------------------------------------------------

def ou_variance_deficit(k: float, shape: str = "flat") -> float:
    """带宽受限噪声源驱动一阶系统的方差亏损，相对白噪声极限 D/λ。

    推导（可复核）。系统 ẋ = −λx + ξ，λ 为状态极点（rad/s）。
    B_req = λ/(2π) 是该一阶系统的 3 dB 带宽（Hz），即工作负载的
    "required bandwidth"；B_n = k·B_req。

    * shape="flat"（§1.3 合同里 B_noise 的定义就是"PSD 平坦带宽"，
      也是 §6.11 钉死 6.4% 的约定）：ξ 的单边 PSD 在 [0, B_n] 内平为
      Φ₀、带外为零。白噪声极限要求 Var(∞) = D/λ，而定出 Φ₀ = 2D/(π·λ/(2π))·
      ——直接积分更清楚：
          Var(B_n) = ∫₀^{B_n} Φ₀ / (λ² + (2πf)²) df
                   = (Φ₀/(2πλ)) · arctan(2πB_n/λ)
                   = (Φ₀/(2πλ)) · arctan(k)。
      白极限 Var(∞) = Φ₀/(4λ)，令其等于 D/λ 得 Φ₀ = 4D，于是
          Var(k) = (D/λ) · (2/π)·arctan(k)，  deficit(k) = 1 − (2/π)arctan(k)。
      检查：k=10 → 1 − (2/π)·arctan(10) = 6.348%；k=100 → 0.626%。
      这正是 §6.11 "100×≈0%、10×≈6.4%" 的出处；极限 B_n→∞ 严格回到 D/λ。

    * shape="lorentzian"（一阶低通滤波后的物理噪声源，PSD ∝ 1/(1+(f/B_n)²)，
      3 dB 点在 B_n）：等价于 OU 噪声源（相关时间 τ_c = 1/(2πB_n)）驱动
      OU 系统，精确方差 Var = (D/λ)/(1 + λτ_c) = (D/λ)·k/(k+1)，
      deficit(k) = 1/(k+1)：k=10 → 9.09%，k=100 → 0.99%。

    两种形状在 k=10 处差 2.7 个百分点；§6.11 的 6.4% 只与 flat 一致，
    所以默认 flat。若合同写明噪声源是一阶滤波的，用 lorentzian 并把
    差异记进 warnings。
    """
    if shape not in ("flat", "lorentzian"):
        raise ValueError(f"未知 shape: {shape!r}")
    if shape == "flat":
        return 1.0 - (2.0 / math.pi) * math.atan(k)
    return 1.0 / (1.0 + k)


@dataclass
class NoiseSpec:
    """§1.3 噪声合同的分析子集（缺一项就补默认并记 warning）。"""

    S_xi: float | None = None      # 单边 PSD 平台值（x²/Hz）
    tau_c: float | None = None     # 噪源相关时间 (s)
    B_noise: float | None = None   # PSD 平坦带宽 (Hz)
    B_req: float | None = None     # 工作负载要求带宽 (Hz)；一阶系统 = λ/(2π)
    D: float | None = None         # 白噪声强度（Var = D/λ 的 D）
    intensity_range: tuple[float, float] | None = None  # 可标定的 D 范围
    shape: str = "flat"            # flat | lorentzian，见 ou_variance_deficit


@dataclass
class NoiseRealizability:
    ratio: float                       # B_noise / B_req
    passes_100x: bool                  # §1.3 硬要求：≥100
    passes_10x: bool                   # 边缘带：≥10（警告，不通过）
    variance_deficit_at_ratio: float   # 本比值下的 OU 方差亏损
    deficit_at_10x: float
    deficit_at_100x: float
    tau_c_vs_state: float              # τ_c · λ（无量纲）；λ = 2π·B_req
    realized_D_range: tuple[float, float]
    warnings: list[str] = field(default_factory=list)


def noise_realizability(spec: NoiseSpec | dict) -> NoiseRealizability:
    """检查噪声合同的带宽数（§1.3 表格 + §6.11 单独验收）。

    B_noise/B_req ≥ 100 才算通过；10–100 之间进警告带；<10 直接失败
    级 warning。方差亏损按 ou_variance_deficit 的推导；D 的可实现
    范围按 D_eff = D·(1 − deficit) 缩放（带限源把实际注入状态的强度
    按同一因子压低——这与 Var = (D/λ)(1−deficit) 是同一句话）。
    τ_c·λ ≪ 1 才允许白噪声近似与 Itô 语义（§1.4：有限 τ_c 给 O(τ_c) 偏差）。
    """
    if isinstance(spec, dict):
        spec = NoiseSpec(**spec)
    warnings: list[str] = []

    B_noise = spec.B_noise
    B_req = spec.B_req
    if B_req is None or B_req <= 0:
        raise ValueError("B_req 必须为正（§1.3 合同缺项不能编译）")
    lam = 2.0 * math.pi * B_req  # 一阶系统的状态极点

    if B_noise is None:
        ratio = float("nan")
        passes_100x = passes_10x = False
        warnings.append("B_noise 缺失：无法判定 ≥100·B_req（§1.3）")
        deficit = float("nan")
    else:
        ratio = float(B_noise / B_req)
        passes_100x = ratio >= 100.0
        passes_10x = ratio >= 10.0
        deficit = ou_variance_deficit(ratio, spec.shape)
        if not passes_10x:
            warnings.append(
                f"B_noise/B_req = {ratio:.3g} < 10：方差亏损 "
                f"{100.0 * deficit:.1f}%，白噪声近似失效")
        elif not passes_100x:
            warnings.append(
                f"B_noise/B_req = {ratio:.3g} 在 10–100 警告带：§1.3 要求 ≥100")

    d10 = ou_variance_deficit(10.0, spec.shape)
    d100 = ou_variance_deficit(100.0, spec.shape)

    tau_c_vs = float("nan")
    if spec.tau_c is not None:
        tau_c_vs = float(spec.tau_c * lam)
        if tau_c_vs > 0.01:
            warnings.append(
                f"tau_c·λ = {tau_c_vs:.3g}：§1.4 的 O(τ_c) Itô/Stratonovich "
                "偏差不可忽略，合同必须显式写解释")

    if spec.D is None and spec.intensity_range is None:
        warnings.append("D / intensity_range 缺失：无法给出可实现强度范围")
        realized = (float("nan"), float("nan"))
    else:
        lo, hi = spec.intensity_range or (spec.D, spec.D)
        realized = (lo * (1.0 - deficit), hi * (1.0 - deficit))

    if spec.S_xi is None and B_noise is not None:
        warnings.append("S_xi 缺失：PSD 平台值未标定（§1.3 标定残差项）")

    return NoiseRealizability(
        ratio=ratio, passes_100x=passes_100x, passes_10x=passes_10x,
        variance_deficit_at_ratio=deficit,
        deficit_at_10x=d10, deficit_at_100x=d100,
        tau_c_vs_state=tau_c_vs, realized_D_range=realized,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 自检（repo 惯例：每个解析数都能在 __main__ 里对拍）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}")
        return bool(cond)

    # (a) vec 约定：L @ vec(P) == vec(A P + P Aᵀ + Σ G P Gᵀ)，随机矩阵。
    n = 4
    A = rng.standard_normal((n, n))
    Gs = [rng.standard_normal((n, n)) for _ in range(2)]
    P = rng.standard_normal((n, n))
    L = sylvester_operator(A, Gs)
    rhs = A @ P + P @ A.T + sum(G @ P @ G.T for G in Gs)
    e = np.linalg.norm(L @ _vec(P) - _vec(rhs))
    ok &= check("vec convention (kron)", e < 1e-12, f"err={e:.2e}")

    # (b) 标量 GBM：L = 2μ+σ²，稳定 ⇔ 2μ+σ²<0（§1.4 / §6.11 解析）。
    mu, sig = -1.0, 1.0
    r = linear_sde_covariance(np.array([[mu]]), [np.array([[sig]])],
                              np.array([[0.0]]))
    ok &= check("scalar GBM L=2mu+sigma^2",
                abs(r.stability_margin - (2 * mu + sig**2)) < 1e-12,
                f"margin={r.stability_margin:.6f} expect={2*mu+sig**2:.6f}")
    ok &= check("scalar GBM stable flag", r.mean_square_stable)
    ok &= check("scalar GBM residual", r.rel_residual < 1e-10,
                f"rel={r.rel_residual:.2e}")
    # P = −W/(2μ+σ²)。另取 2μ+σ²=0 的临界点：不稳定分支返回 NaN，不硬解。
    r2 = linear_sde_covariance(np.array([[mu]]), [np.array([[sig]])],
                               np.array([[1.0]]))
    ok &= check("scalar GBM P=−W/L",
                abs(r2.P[0, 0] - 1.0 / -(2 * mu + sig**2)) < 1e-10,
                f"P={r2.P[0, 0]:.6f}")
    r_crit = linear_sde_covariance(np.array([[-0.5]]), [np.array([[1.0]])],
                                   np.array([[1.0]]))
    ok &= check("critical 2mu+sigma^2=0 ⇒ NaN P, unstable",
                (not r_crit.mean_square_stable)
                and abs(r_crit.stability_margin) < 1e-12
                and np.isnan(r_crit.P[0, 0]),
                f"margin={r_crit.stability_margin:.2e}")

    # (c) OU 加性：Gs=[] 退化为普通 Lyapunov，与 solve_generalized_lyapunov 对拍。
    from .descriptor import Descriptor
    from .lyapunov import solve_generalized_lyapunov
    lam, D = 2.0, 3.0
    r3 = linear_sde_covariance(np.array([[-lam]]), [], np.array([[2 * D]]),
                               C=np.array([[1.0]]))
    sysd = Descriptor(np.eye(1), np.array([[-lam]]), B=np.array([[math.sqrt(2 * D)]]),
                      C=np.array([[1.0]]))
    r4 = solve_generalized_lyapunov(sysd, np.eye(1))
    ok &= check("OU Var=D/lambda", abs(r3.P[0, 0] - D / lam) < 1e-12,
                f"P={r3.P[0,0]:.10f} expect={D/lam:.10f}")
    ok &= check("OU == generalized Lyapunov",
                abs(r3.P[0, 0] - r4.P[0, 0]) < 1e-10,
                f"diff={abs(r3.P[0,0]-r4.P[0,0]):.2e}")

    # (d) 2×2 线性乘性：Gs=[] 时矩阵结果与 scipy solve_lyapunov 一致。
    A2 = np.array([[-1.0, 0.3], [0.1, -2.0]])
    W2 = np.eye(2)
    r5 = linear_sde_covariance(A2, [], W2)
    P_ref = linalg.solve_lyapunov(A2, -W2)
    ok &= check("2x2 additive == solve_lyapunov",
                np.linalg.norm(r5.P - P_ref) < 1e-10,
                f"diff={np.linalg.norm(r5.P - P_ref):.2e}")

    # (e) 线性乘性解析修正：标量 A_S = μ−σ²/2（§1.4 表格）。
    A_S, A_I = linear_correction(np.array([[mu]]), [np.array([[sig]])])
    ok &= check("A_S = mu−sigma^2/2", abs(A_S[0, 0] - (mu - sig**2 / 2)) < 1e-14,
                f"A_S={A_S[0,0]:.6f}")
    ok &= check("A_I = mu+sigma^2/2", abs(A_I[0, 0] - (mu + sig**2 / 2)) < 1e-14,
                f"A_I={A_I[0,0]:.6f}")
    # 数值 Jacobian 版在同一个线性 G 上应与解析一致。
    Gm = np.array([[sig]])
    corr_num = multiplicative_correction(
        lambda x: mu * np.asarray(x, dtype=float),
        lambda x: np.array([[sig * float(np.asarray(x).ravel()[0])]]),
        np.array([1.0]))
    ok &= check("numeric correction == sigma^2/2",
                abs(corr_num[0] - sig**2 / 2) < 1e-8,
                f"corr={corr_num[0]:.8f}")

    # (f) hardware_drift 合同：同 → 不变；ito+strato → 减；strato+ito → 加。
    f = lambda x: mu * np.asarray(x)
    x0 = np.array([1.0])
    ok &= check("hardware same drift",
                np.allclose(hardware_drift("ito", "ito", f, Gm, x0), f(x0)))
    ok &= check("ito model + strato hw ⇒ f_S",
                np.allclose(hardware_drift("ito", "stratonovich", f, Gm, x0),
                            f(x0) - sig**2 / 2))
    ok &= check("strato model + ito hw ⇒ f_I",
                np.allclose(hardware_drift("stratonovich", "ito", f, Gm, x0),
                            f(x0) + sig**2 / 2))
    try:
        hardware_drift("heisenberg", "ito", f, Gm, x0)
        ok &= check("unknown interpretation raises", False)
    except ValueError:
        ok &= check("unknown interpretation raises", True)

    # (g) §6.11 的 6.4%：flat 约定下 deficit(10×)=6.345%、deficit(100×)=0.637%。
    d10 = ou_variance_deficit(10.0)
    d100 = ou_variance_deficit(100.0)
    ok &= check("deficit(10x)=6.345%", abs(d10 - 0.06345102) < 1e-7,
                f"deficit={100*d10:.4f}%")
    ok &= check("deficit(100x)=0.637%", abs(d100 - 0.0063657) < 1e-5,
                f"deficit={100*d100:.4f}%")
    # lorentzian 对照：9.09% / 0.99%（docstring 里的推导）。
    ok &= check("lorentzian deficit(10x)=9.09%",
                abs(ou_variance_deficit(10.0, "lorentzian") - 1 / 11) < 1e-12)

    # (g2) 广义 Wong–Zakai 因子的极限与单调性（解析，逐条可推导）。
    ok &= check("wz chi=0 → 1/2", wong_zakai_factor(0.0, 0.04) == 0.5)
    ok &= check("wz chi=1 → 1/4", abs(wong_zakai_factor(0.04, 0.04) - 0.25) < 1e-15)
    ok &= check("wz chi=16 → 1/34",
                abs(wong_zakai_factor(0.64, 0.04) - 1.0 / 34.0) < 1e-15)
    ok &= check("wz white+delay → Itô (0)", wong_zakai_factor(1.0, 0.0) == 0.0)
    ok &= check("wz white+instant → ambiguous, 1/2 by convention",
                wong_zakai_factor(0.0, 0.0) == 0.5)
    ok &= check("wz monotone in chi",
                all(wong_zakai_factor(k * 0.04, 0.04)
                    > wong_zakai_factor((k + 1) * 0.04, 0.04)
                    for k in range(20)))

    # (h) noise_realizability 端到端。
    spec = NoiseSpec(S_xi=1e-6, tau_c=1e-11, B_noise=1e9, B_req=1e7, D=1.0)
    nr = noise_realizability(spec)
    ok &= check("realizability 100x pass", nr.passes_100x and not nr.warnings,
                f"ratio={nr.ratio:.0f} deficit={100*nr.variance_deficit_at_ratio:.3f}%")
    spec2 = NoiseSpec(B_noise=5e7, B_req=1e7, D=1.0)
    nr2 = noise_realizability(spec2)
    ok &= check("realizability <10x warns",
                (not nr2.passes_10x) and nr2.warnings,
                f"deficit={100*nr2.variance_deficit_at_ratio:.1f}%")

    print("ALL OK" if ok else "SOME CHECKS FAILED")
