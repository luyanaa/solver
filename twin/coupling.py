"""spectral ↔ local 接口守恒对照实验（DESIGN.md §6.4）。

被检验的命题
------------
1. 单通量源 + 电流镜像  ⇒ 质量漂移在机器精度（守恒是拓扑性质）
2. 两侧各自算一遍        ⇒ 漂移 ~ ε·F（ε = 电流镜增益误差），且随 ε 线性
3. 提高阶数 p            ⇒ 漂移不变（泄漏是拓扑误差，不是截断误差）
4. 交换延迟本身不破坏守恒  ⇒ 只有"不一致"才泄漏
5. 不做 trace 交换（只知对侧平均值）⇒ 漂移 O(h)，最坏情形

结构：周期域 [0,1]，左半 [0,xc] 用单结点高阶 nodal-DG（模拟"谱/高阶 smooth 岛"），
右半 [xc,1] 用有限体积（模拟"local flux 岛"）。两个接口 A(x=xc) 与 B(x=0≡1)。

质量账本（连续形式）
    dM/dt = (F_A^fv − F_A^spec) − (F_B^fv − F_B^spec)
单源时两项恒等 ⇒ dM/dt = 0。这个恒等式就是本实验的判据：我们同时记录
实测漂移与按上式积分的预测漂移，两者必须一致（否则代码有隐藏泄漏）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "lgl", "SpectralElement", "FVRegion", "InterfaceModel", "CoupledSystem",
    "rusanov", "minmod", "vanleer", "central",
]


# --------------------------------------------------------------------------
# 基函数
# --------------------------------------------------------------------------
def lgl(p: int):
    """Legendre–Gauss–Lobatto 节点/权重/微分矩阵（[-1,1]）。"""
    if p == 0:
        return np.array([0.0]), np.array([2.0]), np.zeros((1, 1))
    c = np.zeros(p + 1)
    c[p] = 1.0
    r = np.polynomial.legendre.legroots(np.polynomial.legendre.legder(c))
    xi = np.concatenate(([-1.0], np.sort(np.asarray(r, float).real), [1.0]))
    P = np.polynomial.legendre.legval(xi, c)
    w = 2.0 / (p * (p + 1) * P ** 2)
    D = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            if i != j:
                D[i, j] = P[i] / (P[j] * (xi[i] - xi[j]))
    D[np.diag_indices(p + 1)] = -D.sum(axis=1)
    return xi, w, D


# --------------------------------------------------------------------------
# 通量与限制器
# --------------------------------------------------------------------------
def _f(u):
    return 0.5 * u * u


def rusanov(uL, uR, alpha: float = 2.0):
    """Burgers 的局部 Lax–Friedrichs 通量。alpha 应 >= max|u|。"""
    return 0.5 * (_f(uL) + _f(uR)) - 0.5 * alpha * (uR - uL)


def minmod(u, h):
    n = u.size
    s = np.zeros(n)
    if n < 3:
        return s
    dl = (u[1:-1] - u[:-2])
    dr = (u[2:] - u[1:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = dl / h
        sr = dr / h
    mm = np.where(sl * sr > 0,
                  np.sign(sl) * np.minimum(np.abs(sl), np.abs(sr)), 0.0)
    s[1:-1] = mm * h
    return s


def vanleer(u, h):
    n = u.size
    s = np.zeros(n)
    if n < 3:
        return s
    sl = (u[1:-1] - u[:-2]) / h
    sr = (u[2:] - u[1:-1]) / h
    den = sl + sr
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(np.abs(den) > 1e-15, 2.0 * sl * sr / den, 0.0)
    v = np.where(sl * sr > 0, v, 0.0)
    s[1:-1] = v * h
    return s


def central(u, h):
    s = np.zeros(u.size)
    if u.size >= 3:
        s[1:-1] = 0.5 * (u[2:] - u[:-2])
    return s


# --------------------------------------------------------------------------
# 两个"岛"
# --------------------------------------------------------------------------
class SpectralElement:
    """单结点高阶 nodal-DG（LGL 集中质量），模拟谱/高阶 smooth 岛。"""

    def __init__(self, a: float, b: float, p: int):
        self.a, self.b, self.p = a, b, p
        xi, w, D = lgl(p)
        self.xi, self.w, self.D = xi, w, D
        self.x = a + 0.5 * (xi + 1.0) * (b - a)
        self.Dx = D * (2.0 / (b - a))
        self.wx = w * 0.5 * (b - a)
        self.n = p + 1
        # 重心插值权重
        bw = np.ones(self.n)
        for j in range(self.n):
            for k in range(self.n):
                if k != j:
                    bw[j] /= (self.x[j] - self.x[k])
        self.bw = bw

    def interpolate(self, u, xq):
        """重心 Lagrange 插值（把谱岛的连续场读出来）。"""
        u = np.asarray(u, float)
        xq = np.clip(np.asarray(xq, float), self.a, self.b)
        d = xq[..., None] - self.x[None, :]
        hit = np.abs(d) < 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self.bw[None, :] / np.where(hit, 1.0, d)
        t = np.where(hit, 0.0, t)
        num = (t * u[None, :]).sum(axis=-1)
        den = t.sum(axis=-1)
        out = np.where(den != 0.0, num / np.where(den == 0.0, 1.0, den), 0.0)
        any_hit = hit.any(axis=-1)
        if np.any(any_hit):
            j = np.argmax(hit, axis=-1)
            out = np.where(any_hit, u[j], out)
        return out

    def mass(self, u) -> float:
        return float(np.dot(self.wx, u))

    def rhs(self, u, f_left: float, f_right: float) -> np.ndarray:
        r = -(self.Dx @ _f(u))
        r[0] += (f_left - _f(u[0])) / self.wx[0]
        r[-1] -= (f_right - _f(u[-1])) / self.wx[-1]
        return r


class FVRegion:
    """单元中心有限体积（MUSCL + Rusanov）。"""

    def __init__(self, a: float, b: float, n: int):
        self.a, self.b, self.n = a, b, n
        self.h = (b - a) / n
        self.xc = a + (np.arange(n) + 0.5) * self.h

    def mass(self, u) -> float:
        return float(np.sum(u) * self.h)

    def traces(self, u, limiter):
        s = limiter(np.asarray(u, float), self.h)
        return u - 0.5 * s, u + 0.5 * s      # (左面值, 右面值)

    def rhs(self, u, f_left: float, f_right: float, limiter, alpha: float):
        uL, uR = self.traces(u, limiter)
        F = np.empty(self.n + 1)
        F[0] = f_left
        F[-1] = f_right
        if self.n > 1:
            F[1:-1] = rusanov(uR[:-1], uL[1:], alpha)
        return -(np.diff(F)) / self.h


# --------------------------------------------------------------------------
# 接口模型
# --------------------------------------------------------------------------
@dataclass
class InterfaceModel:
    single_source: bool = True     # 单通量源 + 镜像
    mismatch: float = 0.0          # 电流镜增益误差（相对），只在不单源时生效
    extrapolate: bool = False      # 不做 trace 交换，只用对侧单元平均值
    stale: int = 0                 # trace 交换延迟（步，sample/hold：精确 k 个时间步）
    # --- 真实器件（twin.spice 提取值）---
    mode: str = "ideal"            # ideal | gm
    lam: float = 0.0               # 1/V，gds/ID
    eps_dev: float = 0.0           # 单管随机失配（相对电流），一次抽取
    v_per_u: float = 0.1           # 状态 u 到节点电压的标度 V/u
    shared_ref: bool = True        # True=单源共享参考节点；False=两条独立镜像
    eps_sys: float = 0.0           # 独立镜像时的系统 ratio 误差（如 0.022）


@dataclass
class CoupledSystem:
    p: int = 8
    n_cells: int = 32
    xc: float = 0.5
    limiter: str = "minmod"
    alpha: float = 1.6
    model: InterfaceModel = field(default_factory=InterfaceModel)

    def __post_init__(self):
        self.spec = SpectralElement(0.0, self.xc, self.p)
        self.fv = FVRegion(self.xc, 1.0, self.n_cells)
        self.n = self.spec.n + self.fv.n
        self._lim = {"minmod": minmod, "vanleer": vanleer, "central": central}[self.limiter]
        self._hist = []
        self.dmdt_sum = 0.0
        self.leak_max = 0.0     # 单接口 |ΔF| 的历史最大，等价于"漏出去的电流"

    # ---- 状态打包 ----
    def pack(self, us, uf):
        return np.concatenate([np.asarray(us, float), np.asarray(uf, float)])

    def unpack(self, u):
        return u[:self.spec.n], u[self.spec.n:]

    def mass(self, u) -> float:
        us, uf = self.unpack(u)
        return self.spec.mass(us) + self.fv.mass(uf)

    # ---- 接口通量 ----
    def interface_fluxes(self, u):
        us, uf = self.unpack(u)
        uL_fv, uR_fv = self.fv.traces(uf, self._lim)

        a_spec_L = us[-1]                      # A 的左迹（谱岛）
        a_spec_R = us[0]                       # B 的右迹（谱岛）
        a_fv_R = uL_fv[0]                      # A 的右迹（局部岛首单元左面）
        a_fv_L = uR_fv[-1]                     # B 的左迹（局部岛末单元右面）

        # 延迟：两侧都用同一组"旧"迹，因此延迟本身不破坏守恒。
        # interface_fluxes 是 (u, _hist) 的纯函数：只读历史，不推进历史。
        # 历史由 step() 在每个被接受的时间步推进一次（采样本步初值），
        # 因此 stale=k 精确表示 k 个物理时间步的 sample/hold 延迟
        # （分段常数延迟耦合，对应 L4 latch 类；连续 DDE 插值是未来扩展）。
        if self.model.stale > 0:
            idx = max(0, len(self._hist) - self.model.stale)
            if self._hist:
                a_spec_L, a_fv_R, a_fv_L, a_spec_R = self._hist[idx]

        if self.model.extrapolate:
            # 只知对侧"平均值"，没有 trace
            F_A_spec = rusanov(a_spec_L, float(np.mean(uf)), self.alpha)
            F_A_fv = rusanov(float(np.mean(us)), a_fv_R, self.alpha)
            F_B_spec = rusanov(float(np.mean(uf)), a_spec_R, self.alpha)
            F_B_fv = rusanov(a_fv_L, float(np.mean(us)), self.alpha)
        else:
            F_A = rusanov(a_spec_L, a_fv_R, self.alpha)
            F_B = rusanov(a_fv_L, a_spec_R, self.alpha)
            if self.model.mode == "gm":
                # 每侧重注入 = F*(1 + eps_dev) + F*lam*(V_node - V_ref)
                # 共享参考时系统项抵消，只剩 lam*dV 与随机失配之差
                m = self.model
                dvA = m.v_per_u * (a_fv_R - a_spec_L)
                dvB = m.v_per_u * (a_spec_R - a_fv_L)
                dA = F_A * (m.lam * dvA + m.eps_dev)
                dB = F_B * (m.lam * dvB - m.eps_dev)
                if not m.shared_ref:
                    dA = dA + F_A * m.eps_sys
                    dB = dB - F_B * m.eps_sys
                F_A_spec, F_A_fv = F_A, F_A + dA
                F_B_spec, F_B_fv = F_B, F_B + dB
            elif self.model.single_source:
                F_A_spec = F_A_fv = F_A
                F_B_spec = F_B_fv = F_B
            else:
                F_A_spec, F_A_fv = F_A, (1.0 + self.model.mismatch) * F_A
                F_B_spec, F_B_fv = F_B, (1.0 + self.model.mismatch) * F_B
        self.leak_max = max(self.leak_max,
                            abs(F_A_fv - F_A_spec),
                            abs(F_B_fv - F_B_spec))
        return F_A_spec, F_A_fv, F_B_spec, F_B_fv

    def rhs(self, u):
        F_A_spec, F_A_fv, F_B_spec, F_B_fv = self.interface_fluxes(u)
        us, uf = self.unpack(u)
        du_s = self.spec.rhs(us, F_B_spec, F_A_spec)
        du_f = self.fv.rhs(uf, F_A_fv, F_B_fv, self._lim, self.alpha)
        dmdt = (F_A_fv - F_A_spec) - (F_B_fv - F_B_spec)
        return np.concatenate([du_s, du_f]), dmdt

    # ---- RK4 ----
    def step(self, dt: float):
        u = self.u
        # 采样本步初值的接口迹，推进历史恰好一次（每被接受的时间步）。
        if self.model.stale > 0:
            us, uf = self.unpack(u)
            uL_fv, uR_fv = self.fv.traces(uf, self._lim)
            self._hist.append((us[-1], uL_fv[0], uR_fv[-1], us[0]))
            if len(self._hist) > self.model.stale:
                del self._hist[:len(self._hist) - self.model.stale]
        k1, m1 = self.rhs(u)
        k2, m2 = self.rhs(u + 0.5 * dt * k1)
        k3, m3 = self.rhs(u + 0.5 * dt * k2)
        k4, m4 = self.rhs(u + dt * k3)
        self.u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.dmdt_sum += (dt / 6.0) * (m1 + 2 * m2 + 2 * m3 + m4)

    # ---- 驱动 ----
    def run(self, u0, T: float, dt: float, record_every: int | None = None):
        self.u = np.array(u0, float)
        if self.u.size != self.n:
            raise ValueError("初值尺寸 %d ≠ 系统尺寸 %d（p=%d, cells=%d）"
                             % (self.u.size, self.n, self.p, self.n_cells))
        self.dmdt_sum = 0.0
        self.leak_max = 0.0
        self._hist = []
        M0 = self.mass(self.u)
        nsteps = int(np.ceil(T / dt))
        hist = dict(t=[0.0], M=[M0], u=[self.u.copy()])
        for k in range(nsteps):
            self.step(dt)
            if record_every and (k + 1) % record_every == 0:
                hist["t"].append((k + 1) * dt)
                hist["M"].append(self.mass(self.u))
                hist["u"].append(self.u.copy())
        hist["t"].append(nsteps * dt)
        hist["M"].append(self.mass(self.u))
        hist["u"].append(self.u.copy())
        return dict(
            M0=M0, M1=self.mass(self.u),
            drift=(self.mass(self.u) - M0) / M0,
            drift_pred=self.dmdt_sum / M0, leak_max=self.leak_max,
            hist=hist, u=self.u.copy(), dt=dt,
        )

    def smooth_ic(self, amp: float = 0.3, mean: float = 1.0):
        x = np.concatenate([self.spec.x, self.fv.xc])
        return mean + amp * np.sin(2.0 * np.pi * x)
