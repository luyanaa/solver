"""校准 FSM 与温漂数字孪生更新（DESIGN.md §C3 / Stage B1）。

校准在 DESIGN.md 里是一等合同，不是事后补丁（§C3）：

    * foreground（一次性）：已知激励（known stimulus）下测 offset/gain/
    非线性，存 correction。测完就验证（VERIFY 态）：correction 作用于
    **独立**的 held-out 激励+读数（验证点集 ≠ 拟合点集），残差以**测量
    噪声底**收敛——不承诺低于噪声底，那是撒谎（§6.16 Gram 标定口径：
    标定后 isometry 3.46e-15 是因为注入的扰动是确定性的；真实芯片上
    残差 = 噪声底）。VERIFY 用拟合同一点集只能算训练残差，抓不住
    过拟合/漂移/修错——没有 held-out 集时 FSM 会在 history 里如实记录
    independent=False（训练残差，降级口径），不冒充独立验证。
  * background（持续运行）：跟踪慢漂移（温度/老化），replica/reference
    锁定；相邻两次重标之间的残差 = drift_rate × update_interval + 噪声底。
    这与 A7 的 `gm_c_calibrate` 语义一致（residual = drift/(1+T) +
    meas floor；这里把有限环路增益项并入最坏情形 drift_rate×interval）。
    update_interval 变长 ⇒ 残差变大：这是诚实的权衡，不隐藏。
  * 温漂孪生更新：`drift_update(params, delta_T, tempco)` 把这颗芯片的
    实测 tempco 应用到孪生参数上——compiler 针对的是**当前这颗芯片的
    calibrated digital twin**，而不是 TT_25C.lib（DESIGN.md §4 B1 行）。

依赖方向：本模块不 import analysis/ 或 lowering/——校准机器只认数值
参数字典，编译器（lowering.p7）反过来消费 TwinState。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["CalibrationResult", "CalibratedTwin", "CalibrationFSM",
           "calibrate_foreground", "calibrate_background", "drift_update"]


# ---------------------------------------------------------------------------
# 前台标定：已知激励 → correction → 残差 = 噪声底
# ---------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    """一次标定的记录。residual_ppm ≥ meas_noise_ppm（合同，不是巧合）。"""

    correction: dict            # {"offset": .., "gain": ..}（作用在 measured 上）
    residual_ppm: float         # 标定后的残差（分数 ppm = 1e6 × 相对误差）
    method: str                 # "offset" | "gain+offset" | ...
    meas_noise_ppm: float
    n_points: int = 0
    raw_rms_ppm: float = float("nan")   # 拟合残差（未加底）——诊断用

    def as_dict(self) -> dict:
        return dict(correction=dict(self.correction),
                    residual_ppm=float(self.residual_ppm),
                    method=self.method,
                    meas_noise_ppm=float(self.meas_noise_ppm),
                    n_points=int(self.n_points),
                    raw_rms_ppm=float(self.raw_rms_ppm))


def _fit_gain_offset(meas, ref):
    """最小二乘 fit  ref ≈ gain·meas + offset（ stimulated 已知真值 ref）。"""
    n = len(meas)
    sx = sum(meas)
    sxx = sum(m * m for m in meas)
    sy = sum(ref)
    sxy = sum(m * r for m, r in zip(meas, ref))
    det = n * sxx - sx * sx
    if abs(det) < 1e-30:
        # 退化激励（单点）：只解 offset，gain 固定 1
        return 1.0, (sy - sx) / n
    gain = (n * sxy - sx * sy) / det
    offset = (sy - gain * sx) / n
    return gain, offset


def calibrate_foreground(measured, target, meas_noise_ppm: float = 100.0,
                         ) -> CalibrationResult:
    """前台标定：对已知激励（target 为真值）拟合 offset/gain correction。

    `measured`/`target`：等长标量序列（激励扫描的读数 vs 已知真值）。
    `meas_noise_ppm`：本次测量的相对噪声底（分数，1e-4 = 100 ppm）。

    语义合同（与 §6.16 / A7 一致）：
      * correction 用带噪读数拟合 ⇒ 修正后残差 ≈ 噪声底，**不承诺低于
        meas_noise_ppm**；
      * 若拟合残差（raw）碰巧低于噪声底，报告值仍取 max(raw, floor)——
        因为下一次测量的读数就是带噪的，"零残差"只在仿真里存在。
    """
    m = [float(v) for v in (measured if _is_seq(measured) else [measured])]
    t = [float(v) for v in (target if _is_seq(target) else [target])]
    if len(m) != len(t) or not m:
        raise ValueError("measured/target 必须等长且非空")
    floor = float(meas_noise_ppm)
    if floor < 0:
        raise ValueError("meas_noise_ppm 必须 >= 0")

    gain, offset = _fit_gain_offset(m, t)
    # 拟合残差（相对参考幅值）
    scale = max(abs(v) for v in t) or 1.0
    raw = 1e6 * math.sqrt(sum((gain * mi + off - ti) ** 2
                              for mi, ti, off
                              in zip(m, t, [offset] * len(m))) / len(m)) / scale
    residual = max(raw, floor)          # 合同：不低于噪声底
    return CalibrationResult(
        correction={"offset": offset, "gain": gain},
        residual_ppm=residual,
        method="gain+offset" if len(m) > 1 and abs(gain - 1.0) > 0 else "offset",
        meas_noise_ppm=floor,
        n_points=len(m),
        raw_rms_ppm=raw,
    )


def _is_seq(v) -> bool:
    return hasattr(v, "__len__") and not isinstance(v, (str, bytes))


# ---------------------------------------------------------------------------
# 后台标定：漂移跟踪 + 重标调度
# ---------------------------------------------------------------------------
def calibrate_background(drift_model, update_interval, meas_noise_ppm=100.0,
                         loop_gain: float = 1e4) -> dict:
    """后台标定稳态：相邻两次重标之间的残差与下次标定时刻。

    `drift_model`：{"drift_rate_ppm_per_unit": r, "tempco_ppm_per_K": ..}
      r 是被跟踪量（gm/C、Gram 元素）随温度/时间的最坏漂移速率；
      temperature drift 通过 tempco × delta_T 摊进来（调用方折算）。
    `update_interval`：两次 background 标定之间的间隔（同单位的时间/ΔK）。
    `loop_gain`：校准环路的有限增益（A7 语义：drift/(1+T)）。

    返回 {"residual_ppm", "next_cal_time", "rate_ppm_per_unit"}，其中
        residual_ppm = drift_rate·interval/(1+loop_gain) + meas_noise_ppm
    即"最坏间隔内漂移的未修部分 + 测量底"。update_interval↑ ⇒ residual↑
    （线性），这是诚实的调度权衡：后台速率由 residual_budget 反解。
    """
    rate = float(drift_model.get("drift_rate_ppm_per_unit",
                                 drift_model.get("tempco_ppm_per_K", 0.0)))
    interval = float(update_interval)
    floor = float(meas_noise_ppm)
    if rate < 0 or interval < 0 or floor < 0 or loop_gain <= 0:
        raise ValueError("rate/interval/floor 必须 >= 0，loop_gain > 0")
    trimmed = rate * interval / (1.0 + loop_gain)
    return {
        "residual_ppm": trimmed + floor,
        "next_cal_time": interval,
        "rate_ppm_per_unit": rate,
        "meas_noise_ppm": floor,
        "loop_gain": float(loop_gain),
    }


def update_interval_for_budget(drift_rate_ppm_per_unit: float,
                               residual_budget_ppm: float,
                               meas_noise_ppm: float,
                               loop_gain: float = 1e4) -> float:
    """给定残差预算反解最大允许的重标间隔（超预算 ⇒ 速率不够，须加密）。"""
    if drift_rate_ppm_per_unit < 0 or residual_budget_ppm < 0:
        raise ValueError("rate/budget 必须 >= 0")
    headroom = residual_budget_ppm - float(meas_noise_ppm)
    if headroom <= 0:
        return 0.0        # 噪声底已吃光预算：只能前台标定
    return headroom * (1.0 + loop_gain) / drift_rate_ppm_per_unit


# ---------------------------------------------------------------------------
# 温漂孪生更新
# ---------------------------------------------------------------------------
def drift_update(params: dict, delta_T: float, tempco: float) -> tuple[dict, float]:
    """把温度增量应用到孪生参数：p ← p·(1 + tempco·ΔT)。

    `params`：{"gm_over_c": .., "alpha": .., ...}（只更新标量数值字段）。
    `tempco`：分数每 K（1e-4 = 100 ppm/K）。返回 (new_params, drift_ppm)，
    drift_ppm = |tempco·ΔT|·1e6 是本次漂移量——调用方把它投进后台残差
    公式（calibrate_background 的 rate·interval 恰是它的逐段版本）。
    """
    if not params:
        raise ValueError("params 为空")
    frac = float(tempco) * float(delta_T)
    out = {}
    for k, v in params.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = type(v)(v * (1.0 + frac))
        else:
            out[k] = v
    return out, abs(frac) * 1e6


# ---------------------------------------------------------------------------
# CalibratedTwin：当前这颗芯片的实测参数（compiler 的输入）
# ---------------------------------------------------------------------------
@dataclass
class CalibratedTwin:
    """这颗芯片的标定后参数集（不是 TT_25C.lib 的名义值）。

    params 的键与 lowering.p7 的 ElementConfig 字段对齐
    （"gm_over_c"、"alpha"、"S_x"……）；每项带来源
    （foreground/background/nominal）与最近一次标定的残差。
    """

    params: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)     # key -> "foreground"/"background"/"nominal"
    residuals_ppm: dict = field(default_factory=dict)
    rates: dict = field(default_factory=dict)       # key -> drift rate (ppm/unit)
    last_cal: dict = field(default_factory=dict)    # key -> 更新序号/时刻
    counter: int = 0

    def apply_foreground(self, key: str, result: CalibrationResult) -> None:
        """把前台 correction 折进孪生参数：p ← correction(p)。"""
        self.counter += 1
        c = result.correction
        v = float(self.params.get(key, 0.0))
        self.params[key] = v * float(c.get("gain", 1.0)) + float(c.get("offset", 0.0))
        self.sources[key] = "foreground"
        self.residuals_ppm[key] = result.residual_ppm
        self.last_cal[key] = self.counter

    def apply_background(self, key: str, bg: dict) -> None:
        """后台结果：残差记账 + 漂移速率入账（供 P7 排程反解）。"""
        self.counter += 1
        self.sources[key] = "background"
        self.residuals_ppm[key] = float(bg["residual_ppm"])
        if "rate_ppm_per_unit" in bg:
            self.rates[key] = float(bg["rate_ppm_per_unit"])
        self.last_cal[key] = self.counter

    def apply_drift(self, key: str, delta_T: float, tempco: float) -> float:
        """温漂：更新该键并返回本次漂移量（ppm）——进残差账。"""
        self.counter += 1
        sub = {key: self.params.get(key, 0.0)}
        new, drift_ppm = drift_update(sub, delta_T, tempco)
        self.params[key] = new[key]
        self.last_cal[key] = self.counter
        return drift_ppm

    def as_dict(self) -> dict:
        return dict(params={k: float(v) for k, v in self.params.items()},
                    sources=dict(self.sources),
                    residuals_ppm={k: float(v) for k, v in self.residuals_ppm.items()},
                    rates={k: float(v) for k, v in self.rates.items()},
                    last_cal=dict(self.last_cal), counter=self.counter)


# ---------------------------------------------------------------------------
# CalibrationFSM：IDLE → STIMULATE → MEASURE → CORRECT → VERIFY
# ---------------------------------------------------------------------------
@dataclass
class CalibrationFSM:
    """前台标定的显式状态机（§C3 顺序的机械版）。

    用法：`fsm = CalibrationFSM(meas_noise_ppm=…)`；反复 `fsm.advance(...)`
    直到 state == "IDLE"（一个完整循环结束）。VERIFY 态用一次带噪复测
    验证残差，验证失败 ⇒ 回到 STIMULATE（最多 retries 次）。
    """

    meas_noise_ppm: float = 100.0
    state: str = "IDLE"
    retries: int = 0
    max_retries: int = 2
    result: CalibrationResult | None = None
    _stimulus: list = field(default_factory=list)
    _reading: list = field(default_factory=list)
    _verify_stimulus: list = field(default_factory=list)   # held-out 验证集
    _verify_reading: list = field(default_factory=list)
    history: list = field(default_factory=list)

    def advance(self, stimulus=None, reading=None,
                verify_stimulus=None, verify_reading=None) -> str:
        """推进一步；每步给当前态需要的输入，返回新态名。

        VERIFY 态的合同（P1-CALIB）：verify_stimulus/verify_reading 是
        **held-out** 验证点集（与拟合用的 _stimulus/_reading 不同点），
        correction 作用于验证读数、残差必须 ≈ 噪声底量级；过拟合的
        correction（训练点上残差≈底、held-out 上放大）在这里被抓住。
        不提供验证集 ⇒ 回退到拟合点集本身，history 记录
        independent=False——那是训练残差，不是独立验证，不冒充。
        """
        s = self.state
        if s == "IDLE":
            self._stimulus, self._reading = [], []
            self.state = "STIMULATE"
        elif s == "STIMULATE":
            if stimulus is not None:
                self._stimulus.extend(_aslist(stimulus))
            self.state = "MEASURE"
        elif s == "MEASURE":
            if reading is not None:
                self._reading.extend(_aslist(reading))
            self.state = "CORRECT"
        elif s == "CORRECT":
            self.result = calibrate_foreground(
                self._reading, self._stimulus, self.meas_noise_ppm)
            self.state = "VERIFY"
        elif s == "VERIFY":
            if verify_stimulus is not None:
                self._verify_stimulus = _aslist(verify_stimulus)
            if verify_reading is not None:
                self._verify_reading = _aslist(verify_reading)
            independent = bool(self._verify_stimulus) and bool(self._verify_reading)
            # 独立验证优先；没有 held-out 集时退回拟合点集（训练残差）
            v_stim = self._verify_stimulus if independent else self._stimulus
            v_read = self._verify_reading if independent else self._reading
            r = self.result
            corrected = [r.correction["gain"] * mi + r.correction["offset"]
                         for mi in v_read]
            scale = max(abs(v) for v in v_stim) or 1.0
            err_ppm = 1e6 * max(abs(ci - ti) / scale
                                for ci, ti in zip(corrected, v_stim))
            ok = err_ppm <= max(r.residual_ppm, self.meas_noise_ppm) * 10.0
            self.history.append({"state": "VERIFY", "err_ppm": err_ppm,
                                 "independent": bool(independent),
                                 "n_points": len(v_stim),
                                 "passed": bool(ok)})
            if ok or self.retries >= self.max_retries:
                self.state = "IDLE"
                self.retries = 0
            else:
                self.retries += 1
                self.state = "STIMULATE"
        else:
            raise ValueError(f"未知状态 {s!r}")
        return self.state

    def run_foreground(self, stimulus, reading,
                       verify_stimulus=None, verify_reading=None) -> CalibrationResult:
        """驱动完整一个循环（便捷入口），返回标定结果。

        verify_stimulus/verify_reading：held-out 验证点集（P1-CALIB）。
        缺省 None ⇒ VERIFY 退回训练残差（history 记 independent=False）。
        """
        self.advance()                          # IDLE → STIMULATE
        self.advance(stimulus=stimulus)         # STIMULATE → MEASURE
        self.advance(reading=reading)           # MEASURE → CORRECT
        self.advance(verify_stimulus=verify_stimulus,
                     verify_reading=verify_reading)   # CORRECT → VERIFY
        while self.state != "IDLE":
            self.advance(verify_stimulus=verify_stimulus,
                         verify_reading=verify_reading)  # 重试后复验
        return self.result


def _aslist(v):
    return list(v) if _is_seq(v) else [v]
