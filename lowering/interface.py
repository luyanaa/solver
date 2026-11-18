"""接口 pass：单通量源合约 + fitting/capturing selector（DESIGN.md §5.5/§6.4/§6.5）。

两条职责，都是**判定 + 合约**，不是新求解器：

1. **单通量源合约**（§6.4，`twin/coupling.py` 的 `InterfaceModel`
   `single_source=True` 语义）：任何跨表示接口必须是"单通量源 +
   电流镜像"，禁止两边各算一遍。守恒账本与 coupling.py 同一恒等式：

       dM/dt = (F_A_fv − F_A_spec) − (F_B_fv − F_B_spec)

   单源 ⇒ F_A_spec=F_A_fv、F_B_spec=F_B_fv ⇒ dM/dt=0 精确成立
   （拓扑性质，不依赖两侧公式一致）。实测对照：single-source
   漂移在机器精度，mismatch=1e-2 的独立镜像泄漏 ≈ ε·|F|·2（§6.4d）。

2. **fitting vs capturing**（§6.5 selector）：检测到激波/奇异性后，
   决定把奇异区搬到 flux tile（capturing，付 ν_num）还是把基重配
   （fitting，x_s 升为状态，付重配/remesh 成本）。§6.5 的两个数：

       M_int = 界面数        T_int = 界面寿命 / T_obs

   静态状态给不出 T_int（时序属于 supervisor），本模块用成本比较：
   capture 成本 = Rusanov 局部耗散 ν_num = ½·α_loc·dx·σ_face，
   fit 成本 = remesh 单价 × M_int。ν_num 超过阈值 ⇒ 翻到 fitting。
   光滑状态 ⇒ capture 免费（无特殊接口）。

传感器复用 `twin/transform.py` 的 `smoothness_indicator`
（相对 ε floor，1% 场幅）——避免 §6.16 实测的 0/0 饱和陷阱。
阈值迟滞/速率限制（`SigmaTracker`）属于时序回路，不在静态判定里。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twin.transform import smoothness_indicator, sigma_from_indicator

__all__ = ["InterfaceContract", "InterfaceResult", "choose_interface",
           "mass_budget"]

_SHOCK_FACE = 0.5          # σ_f 阈值（sigmoid β0=0.5 对应 sensor≈0.5）
_REMESH_UNIT = 0.25        # 每次 remesh/重构的默认相对单价（capture 单位）


# ---------------------------------------------------------------------------
# 合约
# ---------------------------------------------------------------------------
@dataclass
class InterfaceContract:
    """单通量源 + 电流镜像的接口合约（§6.4）。

    mass_budget 与 `twin/coupling.py:rhs` 的 dM/dt 恒等式逐字一致；
    `check()` 断言单源语义下两侧通量相等 ⇒ 账本为零。
    """

    single_source: bool = True
    source_side: str = "spec"        # 计算侧（镜像 = 电流镜复制到对侧）
    mirror_side: str = "fv"
    mirror: str = ("one flux value computed once, mirrored to both sides "
                   "(current mirror); two independent copies are forbidden")

    def mass_budget(self, f_a_spec: float, f_a_fv: float,
                    f_b_spec: float, f_b_fv: float) -> float:
        """dM/dt = (F_A_fv − F_A_spec) − (F_B_fv − F_B_spec)（coupling.py 同式）。"""
        return (f_a_fv - f_a_spec) - (f_b_fv - f_b_spec)

    def check(self, f_a_spec: float, f_a_fv: float,
              f_b_spec: float, f_b_fv: float) -> float:
        """单源下的泄漏：必须为零（否则违反合约语义，不是数值误差）。"""
        if not self.single_source:
            raise ValueError("合约必须 single_source=True（§6.4 禁止两边各算一遍）")
        leak = self.mass_budget(f_a_spec, f_a_fv, f_b_spec, f_b_fv)
        if abs(leak) > 0.0:
            raise AssertionError("single-source 合约下 dM/dt ≠ 0：镜像失效")
        return leak


def mass_budget(f_a_spec: float, f_a_fv: float,
                f_b_spec: float, f_b_fv: float) -> float:
    """§6.4 守恒账本（耦合系统质量漂移的预测式；独立函数便于对账）。"""
    return (f_a_fv - f_a_spec) - (f_b_fv - f_b_spec)


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
@dataclass
class InterfaceResult:
    """choose_interface 的输出：判定 + 单源合约 + 成本账。"""

    mode: str                        # "capture" | "fit"
    singular: bool                   # 检测到激波/奇异区
    n_interfaces: int                # M_int：连续奇异面段的段数
    shared_flux_index: int | None    # 单源面 i+1/2（F[i] 为 i+1/2 面通量）
    shared_flux_value: float | None
    nu_num_expected: float           # capturing 成本（Rusanov 局部耗散模型）
    remesh_cost: float               # fitting 成本 = 单价 × M_int
    reason: str                      # 一行理由
    contract: InterfaceContract = field(default_factory=InterfaceContract)
    sigma_max: float = 0.0           # 两侧平均后的最大面 σ_f

    def as_dict(self) -> dict:
        return dict(
            mode=self.mode, singular=self.singular,
            n_interfaces=self.n_interfaces,
            shared_flux_index=self.shared_flux_index,
            shared_flux_value=self.shared_flux_value,
            nu_num_expected=self.nu_num_expected,
            remesh_cost=self.remesh_cost, reason=self.reason,
            sigma_max=self.sigma_max,
            single_source=self.contract.single_source,
        )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def choose_interface(state_or_sigma, detector=None, flux=None,
                     dx: float = 1.0, remesh_cost: float = _REMESH_UNIT,
                     alpha: float | None = None) -> InterfaceResult:
    """决定 capturing / fitting，并返回单通量源合约。

    state_or_sigma: 周期网格上的状态场，或已经算好的 σ ∈ [0,1] 场
                    （detector 给出时也接受 detector 的直接输出）。
    detector:       可选 u → σ 场 的检测器；缺省用
                    twin.transform.smoothness_indicator（1% 相对 floor）
                    + sigma_from_indicator。
    flux:           f(u)（缺省 Burgers f=½u²）；α_loc = max|f'| 用
                    差分近似（Burgers ⇒ α_loc ≈ max|u| 近邻域）。
    dx:             网格间距（ν_num = ½·α_loc·dx·σ_f 的空间尺度）。
    remesh_cost:    每次 remesh/重构的相对单价（fit 成本 = 单价 × M_int）；
                    翻转阈值——调大它把判定推向 capturing。

    判定规则（§6.5 selector 的静态版；T_int/T_obs 属于时序回路，
    由 supervisor 携带，见 reason 字段）：
      光滑（σ_f < 0.5 无段）⇒ capture，无特殊接口，ν_num=0；
      奇异 ⇒ cost_fit = remesh_cost·M_int vs cost_capture = ν_num，
      fit 更便宜 ⇒ fit，否则 capture。两种模式的合约都是单源。
    """
    u = np.asarray(state_or_sigma, dtype=float).ravel()
    if detector is not None:
        sigma = np.asarray(detector(u), dtype=float).ravel()
    else:
        s = smoothness_indicator(u)          # 相对 ε floor（§6.16 陷阱）
        sigma = sigma_from_indicator(s)
    if flux is None:
        flux = lambda v: 0.5 * v * v         # Burgers

    # 面 σ（cell → face 平均，与 blend_face_flux 同款）
    sf = 0.5 * (sigma + np.roll(sigma, -1))
    shock_faces = sf >= _SHOCK_FACE
    n_int = _count_runs(shock_faces)
    sigma_max = float(np.max(sf, initial=0.0))

    contract = InterfaceContract()
    if n_int == 0:
        return InterfaceResult(
            mode="capture", singular=False, n_interfaces=0,
            shared_flux_index=None, shared_flux_value=None,
            nu_num_expected=0.0, remesh_cost=0.0,
            reason="smooth: no special interface (capture is free; "
                   "single-source contract still attached for cross-representation use)",
            contract=contract, sigma_max=sigma_max)

    # 奇异：最坏面 = 单源位置
    i = int(np.argmax(sf))
    u_l, u_r = u[i], u[(i + 1) % u.size]
    if alpha is None:
        # α_loc：Rusanov 波速，用 |f'| 的邻域差分近似（Burgers ⇒ ≈max|u|）
        f = flux(u)
        h = 1e-6 * max(1.0, float(np.max(np.abs(u))))
        fp = (flux(u + h) - flux(u - h)) / (2.0 * h)
        alpha_loc = float(np.max(np.abs(fp)))
    else:
        alpha_loc = float(abs(alpha))
    nu_num = 0.5 * alpha_loc * float(dx) * sigma_max
    cost_fit = remesh_cost * n_int
    cost_capture = nu_num
    jump = float(u_r - u_l)
    f_val = 0.5 * (float(flux(np.array([u_l]))[0])
                   + float(flux(np.array([u_r]))[0])) \
        - 0.5 * alpha_loc * jump

    if cost_fit < cost_capture:
        mode = "fit"
        reason = ("shock: remesh cost %.3g < nu_num %.3g -> move x_s onto an "
                  "explicit singular state (T_int check is the supervisor's job)"
                  % (cost_fit, cost_capture))
    else:
        mode = "capture"
        reason = ("shock: nu_num %.3g <= remesh cost %.3g -> keep the fixed grid "
                  "and smear on a flux tile (paying nu_num)"
                  % (cost_capture, cost_fit))

    return InterfaceResult(
        mode=mode, singular=True, n_interfaces=n_int,
        shared_flux_index=i, shared_flux_value=float(f_val),
        nu_num_expected=nu_num, remesh_cost=cost_fit,
        reason=reason, contract=contract, sigma_max=sigma_max)


def _count_runs(mask: np.ndarray) -> int:
    """周期布尔场的连续段数（全 True ⇒ 1 段；全 False ⇒ 0）。"""
    if not mask.any():
        return 0
    if mask.all():
        return 1
    rolled = np.roll(mask.astype(int), 1)
    return int(np.sum((mask.astype(int) - rolled) == 1))
