"""Config → 行为级网表（P7 的最后一个输出侧 pass，DESIGN.md §4 P7）。

`to_behavioral(config)` 把 lowering.p7 的 Config 编成**结构化的行为级
primitive 列表**——每个积分器带自己的极点（a_ii，物理 rad/s）、每条
coeff_edge 带物理增益、B 输入/C 抽头带增益、噪声源带 σ_scale。这不是
ngspice 网表（真实 SPICE 是 P2/B2 的事），而是 macromodel/求解器可
直接消费的中间形式。

它存在的目的是让 round-trip 检验成为可能：从网表**重构**有效 A 矩阵
（对角 = 极点，非对角 = 边增益），其特征值必须 ≈ scaling.scaled 的
特征值。这证明 config 真的编码了动力学，而不只是一张速度 spec——
只有 a_ii/边增益都进 config，重构才可能成立。

依赖方向：netlist.py 只消费 Config（lowering.p7），不 import twin/、
analysis/——重构出的矩阵由调用方（bench）拿去对拍。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BehavioralIntegrator", "BehavioralEdge", "BehavioralPort",
           "BehavioralTap", "BehavioralNetlist", "to_behavioral",
           "attach_nonidealities"]


@dataclass
class BehavioralIntegrator:
    """一个 gm/C 积分器：dot V_i = pole·V_i + Σ_edges(gain·V_j) + Σ_ports(...)"""

    name: str
    pole: float            # 带符号有效自系数 a_ii（rad/s，物理时间轴）
    target_gm_over_c: float
    C_bank: float
    C_code: int
    gm_code: int
    s_x: float | None
    alpha: float
    noise_sigma_scale: float
    cal_method: str
    flagged: bool = False
    flag_reason: str = ""

    def as_dict(self) -> dict:
        return dict(name=self.name, pole=float(self.pole),
                    target_gm_over_c=float(self.target_gm_over_c),
                    C_bank=float(self.C_bank), C_code=int(self.C_code),
                    gm_code=int(self.gm_code),
                    s_x=(float(self.s_x) if self.s_x is not None else None),
                    alpha=float(self.alpha),
                    noise_sigma_scale=float(self.noise_sigma_scale),
                    cal_method=self.cal_method, flagged=bool(self.flagged),
                    flag_reason=self.flag_reason)


@dataclass
class BehavioralEdge:
    """一条状态耦合边 j → i：gm 把 V_j 注入积分器 i 的求和结点。"""

    src: str
    dst: str
    gain: float            # A_v[i,j]（物理 rad/s per V_j）
    tile: str              # "mvm-tile" | "flux-edge"
    route_contract: int | None   # Config.route_contracts 的契约序号

    def as_dict(self) -> dict:
        return dict(src=self.src, dst=self.dst, gain=float(self.gain),
                    tile=self.tile, route_contract=self.route_contract)


@dataclass
class BehavioralPort:
    """一条输入端口边 p → i：B_v[i,p]（物理 V/s per V_in）。

    B_v = α⁻¹·S·B（输入缩放契约，lowering/scaling.py docstring）：
    gain 已含 α⁻¹，对应物理驱动 u_p(t_p)=u_m(t_p/α)。
    """

    input: int
    dst: str
    gain: float

    def as_dict(self) -> dict:
        return dict(input=int(self.input), dst=self.dst, gain=float(self.gain))


@dataclass
class BehavioralTap:
    """一条输出抽头 i → q：C_v[q,i]。"""

    tap: int
    src: str
    gain: float

    def as_dict(self) -> dict:
        return dict(tap=int(self.tap), src=self.src, gain=float(self.gain))


@dataclass
class BehavioralNetlist:
    """行为级网表：可重估（effective_A/eigs）、可 JSON 序列化。"""

    name: str
    alpha: float
    integrators: list                      # [BehavioralIntegrator]
    edges: list                            # [BehavioralEdge]
    ports: list                            # [BehavioralPort]
    taps: list                             # [BehavioralTap]
    source: dict = field(default_factory=dict)
    # B2 非理想性附件（lowering 不生成它，只承载）：元素名 → 参数 dict，
    # 由 twin.macromodel.b2_nonideal_map 之类的外部 pass 生成后挂上。
    # 空 dict == 理想默认（extraction=None 路径），effective_A() 不受影响。
    nonideal: dict = field(default_factory=dict)

    # ---- 重构有效动力学（round-trip 的核心）----
    def effective_A(self) -> np.ndarray:
        """从极点 + 边增益重构 A_v（物理时间轴）。

        对角元 = 每个积分器自己的 pole；非对角元 = 边增益（按元素名
        索引，因此顺序无关）。重构是**精确装配**，不是拟合。
        """
        names = [itg.name for itg in self.integrators]
        idx = {n: k for k, n in enumerate(names)}
        A = np.zeros((len(names), len(names)))
        for itg in self.integrators:
            A[idx[itg.name], idx[itg.name]] = itg.pole
        for e in self.edges:
            A[idx[e.dst], idx[e.src]] += e.gain
        return A

    def effective_B(self) -> np.ndarray:
        """从端口边重构 B_v（按元素名索引）。"""
        names = [itg.name for itg in self.integrators]
        idx = {n: k for k, n in enumerate(names)}
        n_in = max((p.input for p in self.ports), default=-1) + 1
        B = np.zeros((len(names), n_in))
        for p in self.ports:
            B[idx[p.dst], p.input] += p.gain
        return B

    def effective_C(self) -> np.ndarray:
        """从抽头重构 C_v（按元素名索引）。"""
        names = [itg.name for itg in self.integrators]
        idx = {n: k for k, n in enumerate(names)}
        n_out = max((t.tap for t in self.taps), default=-1) + 1
        C = np.zeros((n_out, len(names)))
        for t in self.taps:
            C[t.tap, idx[t.src]] += t.gain
        return C

    def eigs(self) -> np.ndarray:
        return np.linalg.eigvals(self.effective_A())

    def as_dict(self) -> dict:
        return dict(
            name=self.name, alpha=float(self.alpha),
            integrators=[i.as_dict() for i in self.integrators],
            edges=[e.as_dict() for e in self.edges],
            ports=[p.as_dict() for p in self.ports],
            taps=[t.as_dict() for t in self.taps],
            source=dict(self.source),
            nonideal={k: dict(v) for k, v in self.nonideal.items()})


def to_behavioral(config, name: str = "behavioral") -> BehavioralNetlist:
    """Config → BehavioralNetlist（纯转写：每个数都来自 config 本身）。

    积分器取 primitive=="integrator-gm_c" 的 ElementConfig；pole 用
    a_ii（config 缺失时回退 −target_gm_over_c 的符号约定会撒谎，故
    缺失 ⇒ pole=0.0 并在 source 里记录 n_missing_pole——round-trip
    会因此失败，这是想要的失败方式，不是静默回退）。
    """
    ints, missing = [], 0
    for el in config.elements:
        if el.primitive != "integrator-gm_c":
            continue
        if el.a_ii is None:
            missing += 1
        ints.append(BehavioralIntegrator(
            name=el.name, pole=(float(el.a_ii) if el.a_ii is not None else 0.0),
            target_gm_over_c=float(el.target_gm_over_c),
            C_bank=float(el.C_bank), C_code=int(el.C_code),
            gm_code=int(el.gm_code), s_x=el.s_x, alpha=float(el.alpha or 0.0),
            noise_sigma_scale=float(el.noise_setting.get("sigma_scale", 1.0)),
            cal_method=el.cal_method, flagged=bool(el.flagged),
            flag_reason=el.flag_reason))
    edges = [BehavioralEdge(src=e["src"], dst=e["dst"], gain=float(e["gain"]),
                            tile=e.get("tile", "mvm-tile"),
                            route_contract=e.get("route_contract"))
             for e in config.coeff_edges]
    ports = [BehavioralPort(input=int(p["input"]), dst=p["dst"],
                            gain=float(p["gain"]))
             for p in config.b_ports]
    taps = [BehavioralTap(tap=int(t["tap"]), src=t["src"], gain=float(t["gain"]))
            for t in config.c_taps]
    src = dict(n_route_contracts=len(config.route_contracts),
               n_missing_pole=missing,
               cert_status=config.source.get("cert_status"))
    return BehavioralNetlist(name=name, alpha=float(config.source.get("alpha", 0.0)),
                             integrators=ints, edges=edges, ports=ports,
                             taps=taps, source=src)


def attach_nonidealities(netlist: BehavioralNetlist, mapping: dict
                         ) -> BehavioralNetlist:
    """把 B2 非理想性参数挂到网表上（B2 macromodel pass，DESIGN.md §6.20）。

    `mapping`：元素名 → 参数 dict（由 twin.macromodel.b2_nonideal_map 之类
    的外部 pass 从 results/b2_extract.json 生成——本模块不 import twin/，
    依赖方向不变）。校验：未知元素名 ⇒ 显式报错，不静默丢弃；重复挂载
    也报错（非理想性是实测事实，覆盖 = 撒谎）。

    理想默认：不调用本函数时 netlist.nonideal == {}，effective_A()/round-trip
    完全不受影响（run_p7_netlist 的旧验收照旧成立）。
    """
    known = {itg.name for itg in netlist.integrators}
    unknown = sorted(set(map(str, mapping)) - known)
    if unknown:
        raise ValueError("attach_nonidealities: 未知元素名 %s（integrators: %s）"
                         % (unknown, sorted(known)))
    if netlist.nonideal:
        raise ValueError("attach_nonidealities: nonideal 已挂载（%s），"
                         "不允许覆盖" % sorted(netlist.nonideal))
    netlist.nonideal = {k: dict(v) for k, v in mapping.items()}
    return netlist
