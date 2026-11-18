# 参考文献与阅读状态

这个目录是论文入口，不是综述。每条只记三件事：

1. 它解决了什么（可验证的 claim / 数字）；
2. 对 `DESIGN.md` 的哪一节有用；
3. 还缺什么（也就是我们的 research gap）。

`ref/pdfs/` 只放**开放获取**的副本；付费论文只记 DOI/链接，不在这里存 PDF。
阅读状态按深度标注：**全文**、**摘要+引言**、**摘要**、**仅元数据**。

---

## A. FPAA physical synthesis / analog CAD

### A1. Baskaya, Reddy, Lim, Anderson — Placement for Large-Scale Floating-Gate FPAAs
IEEE TVLSI 14(8):906–910, Aug. 2006. DOI [10.1109/TVLSI.2006.878477](https://doi.org/10.1109/TVLSI.2006.878477)
本地全文：[baskaya2006_tvlsi_placement.pdf](pdfs/baskaya2006_tvlsi_placement.pdf)　状态：**全文**

- 目标器件是 12×12 = 144 个 CAB 的 floating-gate FPAA；routing 只有 local / vertical / horizontal crossbar 三类 wire，global wire 不分段，一根线只能承载一个 net。
- 关键物理事实：**每个 routing switch 都给信号路径加一个 pole**；ZVTC 给出的主极点约 `1/[R(2C_w+C_l)]`。所以 parasitic 是**沿路径累积**的，不是一段线长。
- clustering 的目标不是面积，而是：最小化 switch 总数 + 让同一组 IO 路径的 switch 数**平衡**；placement 再把 cluster 映射到 CAB，继续优化 switch 使用与平衡。
- 实现细节：k-means 按初始路径长度分组，然后给短路径补 switch 做平衡；cell 排序用 MHEC（modified hyper-edge coarsening）或 static timing analysis 的 slack；邻接管理用 binary heap，clustering `O(n log n)`。
- **对我们的意义**：这就是 “analog CTS” 的早期版本——它已经在做 path balancing，但 balance 的对象是 **switch 数**（标量寄生代理），不是 `H_ij(s)` / group delay / 相位。`DESIGN.md §3.2、§6.15`。

### A2. Baskaya, Anderson, Lim — Net-Sensitivity-Based Optimization of Large-Scale FPAA Placement and Routing
IEEE TCAS-II 56(7):565–569, Jul. 2009. DOI [10.1109/TCSII.2009.2023351](https://doi.org/10.1109/TCSII.2009.2023351)
本地全文：[baskaya2009_tcasii_netsens.pdf](pdfs/baskaya2009_tcasii_netsens.pdf)　状态：**全文**

- 这是与 “给不同 net 定义不同敏感性” 最直接的一篇。它把 **net 对 interconnect parasitic 的 sensitivity** 做成 `3×n` 的 net-sensitivity matrix（3 种 wire type × n 个 net），而不是只用最短线/switch 数。
- sensitivity 的提取是**仿真驱动**的：以 “所有 net 都用最小寄生 wire” 为 reference，逐个把 net 换到另一种 wire type，比较用户关心的性能指标偏差；`2n+1` 次 SPICE 仿真即可填满矩阵。
- 每个 metric 有三种 cost 形式：equalize / maximize / minimize，统一写成 `c_ijt = |X_i − x_ijt| / max(1,|X_i|)`（maximize 用 `x_ijt−X_i`，minimize 用 `X_i−x_ijt`）；多 metric 加权成 `nc_jt`。
- 优化器是 simulated annealing，目标函数：

      cost = α·num_sw + β·Σ_j (nc_j + num_col_j)

  其中 `num_col_j` 惩罚跨列（type-3 global wire）。benchmark 是 Butterworth / Chebyshev / inverse Chebyshev / Bessel / elliptic 的 gm-C 低通，最高 8 阶。
- 结果（相对 initial placement）：fc error −50%、passband gain error −90%、ripple −91%、rolloff +9%；**只最小化 switch 数反而会让某些电路性能变差**（inverse Chebyshev 除外，它比较 robust）。
- **对我们的意义**：这证明 “parasitic-aware” 已经进化到 “net-sensitivity-aware”，但 sensitivity 仍然是 **scalar 性能指标（fc/gain/ripple/rolloff）对 wire type 的偏差**。我们的 `S_τ,ij = ω|J_ij|·‖(jωI−J)⁻¹‖`、`H_ij(s)`、`S_n` 是把同一思想从 “filter 指标” 推广到 “目标动力系统”。`DESIGN.md §3.2、§6.15`。

### A3. Hasler, Twigg — An OTA-based Large-Scale FPAA for Faster On-Chip Communication and Computation
IEEE ISCAS 2007. DOI [10.1109/ISCAS.2007.378305](https://doi.org/10.1109/ISCAS.2007.378305)
状态：**摘要/TLDR**（IEEE 付费；Semantic Scholar 记录）

- 直接讨论 large-scale FPAA 的**高频 routing / routing-fabric speed limitation**：用 transconductance amplifier（OTA）做多级 routing 与 computation，而不是只靠传统 switch matrix。
- TLDR 的措辞是 “multiple levels of OTA based routing and computation … combined with more traditional switch matrices”。
- **对我们的意义**：它支持 `DESIGN.md §3.2` 的 “state 用 voltage 保存、通信尽量用 current/differential” 以及 “buffer 是阻抗隔离而不是 memory”。它做的是电路层的 speed，不是 dynamical sensitivity；这两条可以叠加。

### A4. Wang, Vrudhula — Behavioral Synthesis of Field Programmable Analog Array Circuits
ACM TODAES 7(4):563–604, Oct. 2002. DOI [10.1145/605440.605445](https://doi.org/10.1145/605440.605445)
状态：**摘要**（ACM 付费）

- 经典流程：behavioral-level analog description → **function decomposition** → **macrocell synthesis** → placement/routing → post-placement simulation；论文重点是前三步。
- function decomposition 把高阶 system function 分解成一组低阶 function，并用搜索选择分解；这与 `lowering/reduce.py` 的 “高阶 ODE/DAE → canonical form / 辅助状态” 在**结构**上同类（都是 decomposition），但对象不同：从摘要看它处理的是 system function / 传递函数层面，是否覆盖非线性动力学需要读全文确认。
- 它已经把 parasitic-induced performance degradation 作为 P&R cost 的一部分。
- **对我们的意义**：`DESIGN.md §1.5、§6.13、§7` 的 “可划归性 + primitive basis” 路线可以直接接上；差别是我们要把线性 system-function synthesis 扩到 nonlinear ODE/DAE/SDE/PDE，并且把 `H_ij(s)` 作为 contract 而不是事后仿真。

### A5. George, Kim, Shah, Hasler, Collins, Adil, Wunderlich, Nease, Ramakrishnan — A Programmable and Configurable Mixed-Mode FPAA SoC
IEEE TVLSI 24(6):2253–2261, Jun. 2016. DOI [10.1109/TVLSI.2015.2504119](https://doi.org/10.1109/TVLSI.2015.2504119)
本地全文：[george2016_tvlsi_fpaa_soc.pdf](pdfs/george2016_tvlsi_fpaa_soc.pdf)　状态：**摘要+引言**

- FG-based SoC FPAA：模拟/数字可配置块 **interdigitated**（A/D/A/D…）排在同一条 routing grid 上，旁边集成 16-bit open-source MSP430、DAC/ADC、memory-mapped I/O。
- 结果：command-word classifier 23 µW；参数密度比非 FG 方案高约 600,000×；routing fabric 支持 partial rapid reconfiguration。
- **对我们的意义**：这正是 “analog fabric + digital control/calibration island” 的硅上先例（`DESIGN.md` 的两级 config、digital quiet 协议）。但它的 digital 是通用 µP + 工具链，不是 “global time contract + per-edge latency class + calibration FSM” 的 compiler 目标；我们的 digital block 要承担的是 **配置/校准/instrumentation**，而不是通用计算。

---

## B. Waveform relaxation / multirate

### B1. Lelarasmee, Ruehli, Sangiovanni-Vincentelli — The Waveform Relaxation Method for Time-Domain Analysis of Large Scale Integrated Circuits
IEEE TCAD 1(3):131–145, Jul. 1982. DOI [10.1109/TCAD.1982.1270004](https://doi.org/10.1109/TCAD.1982.1270004)
本地技术报告：[lelarasmee1981_wr_techreport.pdf](pdfs/lelarasmee1981_wr_techreport.pdf)（UCB/ERL M81/75, 26 Jun 1981）　状态：**摘要+引言**

- 标准 SPICE 类模拟器的三件套是：stiffly stable implicit integration → Newton → sparse Gaussian；WR 则把非线性 DAE **分解成若干子系统**，每个子系统在**整个时间区间**上求解，子系统之间迭代（Gauss–Jacobi / Gauss–Seidel over waveforms）。
- 论文给出 WR 的收敛条件，并证明在 MOS 数字电路中这些条件通常很宽松；理论+计算实验都表明方法高效可靠。
- **对我们的意义**：这是 “global dynamical system → local continuous-time islands → waveform exchange” 的数学基础（`DESIGN.md §2.8` 的时间域 hierarchy、`§4` 的 multirate、`§7` 的 regional/global boundary）。当某条 coupling 无法满足 `ω_max τ ≪ 1`，正确的出口不是继续 pad，而是把它划成一个显式的 WR/multirate 边界。

### B2. Lelarasmee, Sangiovanni-Vincentelli — RELAX: A New Circuit Simulator for Large Scale MOS Integrated Circuits
会议版：ACM/IEEE DAC 1982, pp. 682–690. DOI [10.1109/DAC.1982.1585570](https://doi.org/10.1109/DAC.1982.1585570)
期刊版：Computer-Aided Design 15(5):262–270, Sep. 1983. DOI [10.1016/0010-4485(83)90014-3](https://doi.org/10.1016/0010-4485(83)90014-3)
状态：**摘要片段**（付费；CrossRef 元数据已核对）

- RELAX 是 WR 的 MOS 数字电路实现：按方程分解电路，利用 MOS 电路的单向性；论文声称在通常条件下有保证收敛，并给出相对 SPICE2 的加速。
- 与 B1 的分工：B1 是方法与收敛性，B2 是第一个大规模 MOS 实现与工程验证。

### B3. Hayashi, Asai — Numerical Stability and Multirate Effect in Waveform Relaxation Algorithm with Under Relaxation Technique
IEICE Trans. Fundamentals E75-A(6):685–690, Jun. 1992.
状态：**摘要**（付费）

- 把 **under-relaxation** 和 **virtual state formulation (VSF)** 引入 WR，显式利用 multirate behavior；实现 relaxation-based simulator DESIRE2，对 MOS 电路做瞬态分析。
- 结论：可以在不损失 waveform accuracy 的前提下降低总仿真时间。
- **对我们的意义**：它处理的是 “不同子系统时间尺度不同” 时的**数值稳定性**；对应我们 “每个 region 可以有自己的 `A_t`，但 boundary 必须显式 sample/resynchronize” 的规则。不要把它误读成 “模拟硬件可以直接多速率而不付同步误差”。

---

## C. Delay 进入动力学

### C1. Jiménez, Campos-Cantón, Ontañón-García — Electronic Circuit Emulating a First-order Time-delay Differential Equation
Rev. Mex. Fís. E 19(1):010202, 2022. DOI [10.31349/RevMexFisE.19.010202](https://doi.org/10.31349/RevMexFisE.19.010202)
本地全文：[jimenez2022_rmfe_delay.pdf](pdfs/jimenez2022_rmfe_delay.pdf)　状态：**全文（方法+实验）**

- 研究对象就是标量 DDE：`ẋ(t) = a·x(t−τ)`，初值给在 `[−τ,0]` 上的 `g(t)`；特征方程有**无穷多个根**。
- 随 `aτ` 变化出现三种响应：一阶无延迟型、二阶型、**harmonic oscillator**。电路由 integrator + phase-shift + gain 三个 block 组成，用运放/RC 实现，Simulink 与实验对照。
- **对我们的意义**：这是 “routing delay 不是 timing error，而是改变系统” 的最干净演示（`DESIGN.md §2.8、§6.14`）。它也说明为什么 `twin/route.py` 里 `feedback_critical` 必须单独检查绝对延迟：matched skew 不会把 DDE 变回 ODE。

---

## D. GPAC / 微分代数 / 连续时间复杂度

### D1. Shannon — Mathematical Theory of the Differential Analyzer
J. Math. Phys. 20(1–4):337–354, 1941. DOI [10.1002/sapm1941201337](https://doi.org/10.1002/sapm1941201337)
状态：**元数据+摘要**

- 把 differential analyzer 抽象成由 adder / multiplier / integrator / constant 组成的网络，研究它究竟能生成什么函数。这个模型后来被称为 **GPAC**；其可生成函数是 **differentially algebraic** 的（Pour-El、Lipshitz–Rubel 的后续工作）。
- **对我们的意义**：`DESIGN.md §1.5` 的 primitive basis 不是凭空选的：`{+, ×, ∫, const}` 就是 GPAC 的现代 CMOS 版本；sin/exp/tanh 可以通过辅助状态消掉（代价是 invariant manifold 的维护）。

### D2. Bournez, Graça, Pouly — Computing with Polynomial Ordinary Differential Equations
J. Complexity 36:106–140, 2016. DOI [10.1016/j.jco.2016.05.002](https://doi.org/10.1016/j.jco.2016.05.002)
本地副本（arXiv 版）：[bournez2016_computing_poly_ode.pdf](pdfs/bournez2016_computing_poly_ode.pdf)　状态：**摘要+引言**

- 摘要的核心论证：Shannon 的 GPAC 生成函数必须满足 polynomial DAE；Euler Γ、Riemann ζ 不满足任何 polynomial DAE，所以过去常被用来论证 GPAC 弱于数字计算。但在更现代的（非实时生成）计算定义下，GPAC 实际与 Turing machine 等价（Bournez et al. 2007）。
- 本文做两件事：证明上述计算定义的**鲁棒性**（多种自然变体给同一可计算性）；再推进到时间/空间复杂度，证明多种自然复杂度变体在复杂度层也等价——因此 GPAC / polynomial ODE 有一个稳健的时间复杂度概念，polynomial ODE 可以作为编程模型。
- **对我们的意义**：`lowering/reduce.py` 的 “rational / polynomial canonical form” 与 `DESIGN.md §1.5` 的 Level 0/1 可划归性，直接落在这条线上。注意：**可表达 ≠ 稳定可实现**；physical constraints 在 D5 之后才进来。

### D3. Bournez, Graça, Pouly — On the Functions Generated by the General Purpose Analog Computer
Information and Computation 257:34–57, Dec. 2017. DOI [10.1016/j.ic.2017.09.015](https://doi.org/10.1016/j.ic.2017.09.015)
本地副本：[bournez2017_gpac_functions.pdf](pdfs/bournez2017_gpac_functions.pdf)　状态：**摘要**

- 把 GPAC 从一元函数扩展到**多变量** generable functions；引入 “生成一个函数需要多少 resource/space” 的概念，把 GPAC-generable functions 分层成真正的子类。
- 证明该模型对算术运算、composition、ODE solving 的**稳定性**，并把每一步的 resource 记账；逼近结果从 Shannon 的 compact domain 加强到 **unbounded domain 上的一致逼近**。
- 关键区别：GPAC-generable 函数是 analytic 的，但仍能一致逼近很广泛的一类（连续甚至不连续）非解析函数。也就是说 **exact representability ≠ arbitrarily accurate approximation**。
- **对我们的意义**：对 silicon solver，正确的问题不是 “Heaviside 是否精确可实现”，而是 “是否存在 `H_ε` 满足 workload 的 error budget”。这与有限带宽、有限 swing 的硬件天然一致，也是 `DESIGN.md §1.5` 的 Level 1（ε-可划归）而不是 Level 0 的哲学依据。

### D4. Rubel — A Universal Differential Equation
Bull. Amer. Math. Soc. 4(3):345–349, 1981. DOI [10.1090/S0273-0979-1981-14910-7](https://doi.org/10.1090/S0273-0979-1981-14910-7)
状态：**元数据+评论**（AMS PDF 403）

- 构造了一个固定的四阶非线性代数微分方程，其解族可以任意精度逼近任意连续函数（universal differential equation）。
- 概念意义：**非常有限的 differential structure → 极丰富的 function space**。但不能直接推出 “做一个 Rubel circuit 就能稳定编程任意 ODE”：存在性 ≠ 稳定可编程表示，更不等于 noise-robust / efficient computation。
- **对我们的意义**：这正好是我们不想重复的跳跃。`DESIGN.md` 的 slack 向量与 `twin/route.py` 的 closure 就是在 “存在性” 和 “可编程可实现” 之间加的可计算判据。

### D5. Bournez, Graça, Pouly — Polynomial Time Corresponds to Solutions of Polynomial ODEs of Polynomial Length
ICALP 2016, LIPIcs 55, article 109, 15 pp. DOI [10.4230/LIPIcs.ICALP.2016.109](https://doi.org/10.4230/LIPIcs.ICALP.2016.109)
本地全文：[bournez2016_icalp_poly_length.pdf](pdfs/bournez2016_icalp_poly_length.pdf)　状态：**摘要+结论**

- 用 polynomial ODE 的 **trajectory length** 刻画 PTIME：多项式时间计算 ↔ 多项式长度的 polynomial ODE 解。这给出了一个不依赖离散机器的复杂度定义。
- **对我们的意义**：analog computer 的 wall-clock time 不是好的 complexity measure；`t → 10^12 t` 在物理上会撞上 `f_T、g_m/C、noise、energy、delay`。真正接近 intrinsic effort 的是 trajectory length + 带宽/SNR/能耗/通信。`DESIGN.md §0、§2.2` 的 `σ²τ = αkT/g_m` 和 `A_t` 约束可以视为这条线的物理版本。

---

## E. Research gap：从 net sensitivity 到 dynamical-sensitivity-aware physical synthesis

已有工作已经做过：

    parasitic-aware FPAA P&R          （Baskaya 2006）
    net-sensitivity-aware P&R          （Baskaya 2009）
    OTA/current-mode routing           （Hasler–Twigg 2007）
    behavioral decomposition + P&R     （Wang–Vrudhula 2002）
    analog/digital SoC FPAA            （George 2016）

但它们 sensitivity 的对象仍然是 **scalar 性能指标对 wire type / switch 数的偏差**（fc、gain、ripple、rolloff、switch count）。我们想往前推一步：

    J(x,t) = ∂F/∂x
      → resolvent / eigenmode / bandwidth
      → S_τ,ij(ω), S_g,ij(ω), S_n,ij(ω)
      → P&R contract {ΔG, Δφ, τ, S_n}_ij

也就是 **dynamical-sensitivity-aware physical synthesis**：不是 “这个 net 对寄生敏感”，而是 “这条 edge 对目标动力系统的相位/增益/噪声有多敏感”。

这条线在仓库里的落点：

| 已有 | 位置 | 还缺 |
|---|---|---|
| κ、Γ、H∞、Lyapunov | `analysis/descriptor.py`、`hinf.py`、`lyapunov.py` | 把 `J(x,t)` 沿轨迹的 resolvent 峰值接进 P&R cost |
| `E = I + A∘T`、`τ_crit`、uniform phase budget | `analysis/skew.py` | 非均匀 τ 的稳定性边界；per-edge `τ_tol` 已由 `route.py` 给出，但还没接进 descriptor |
| `H_ij(s)`、`η_route`、`J_H/J_τ`、`route_closure` | `twin/route.py` | P&R 的 cost 函数、latency-class 约束、WR 边界 |
| polynomial/rational 可划归 | `lowering/reduce.py` | 与 GPAC 的 Level 0/1 合同、误差证书 |
| multirate 的时间语义 | `DESIGN.md §2.8、§4` | 一个最小 WR/两岛 partition 的可运行 demo |

**如果 coupling 无法满足 instantaneous contract**，不要继续 pad；按 B1/B3 的思路把系统 partition 成 multirate continuous-time islands，boundary 显式做 waveform exchange / resampling。这正是 `DESIGN.md §2.8` 的 `L4: sampled/global` 与 §4 的 multirate solver。

**下一步（按依赖排序）**

1. 把 `twin/route.py` 的 `edge_sensitivity` 从 `S_τ` 扩成 `S_g = |J_ij|·‖(jωI−J)⁻¹‖`（gain）与 `S_n`（噪声注入），保持同样的 `τ_tol = ε/S` 形式。
2. 给 P&R 一个 contract-driven cost：
   `J = Σ_edges w_ij [ (ΔG/ε_g)² + (Δφ/ε_φ)² + (τ/τ_max)² + (S_n/S_budget)² ]`，
   并对 `feedback_critical` 的 edge 禁止用 pad 修。
3. 用两个 ODE 子系统做一个最小 WR/multirate 验证：monolithic vs 两岛 + 显式 delay，比较 `κ`、误差、通信量。
4. 再把 GPAC 的 “physical constraints” 写成 `DESIGN.md §7` 的 compiler 前置条件（swing、B、S_n>0、τ>0、P<∞）。

---

## F. 阅读状态与本地文件

| 文献 | 状态 | 本地副本 |
|---|---|---|
| Baskaya 2006 (TVLSI) | 全文 | `pdfs/baskaya2006_tvlsi_placement.pdf` |
| Baskaya 2009 (TCAS-II) | 全文 | `pdfs/baskaya2009_tcasii_netsens.pdf` |
| Hasler–Twigg 2007 (ISCAS) | 摘要/TLDR | —（付费） |
| Wang–Vrudhula 2002 (TODAES) | 摘要 | —（付费） |
| George et al. 2016 (TVLSI) | 摘要+引言 | `pdfs/george2016_tvlsi_fpaa_soc.pdf` |
| Lelarasmee et al. 1982 (TCAD) | 摘要+引言 | `pdfs/lelarasmee1981_wr_techreport.pdf`（技术报告） |
| RELAX 1982 (DAC) | 摘要片段 | —（付费） |
| Hayashi–Asai 1992 (IEICE) | 摘要 | —（付费） |
| Jiménez et al. 2022 (RMF E) | 全文（方法+实验） | `pdfs/jimenez2022_rmfe_delay.pdf` |
| Shannon 1941 (J. Math. Phys.) | 元数据+摘要 | —（付费） |
| Bournez et al. 2016 (J. Complexity) | 摘要+引言 | `pdfs/bournez2016_computing_poly_ode.pdf`（arXiv 版） |
| Bournez et al. 2017 (Inf. Comput.) | 摘要 | `pdfs/bournez2017_gpac_functions.pdf` |
| Rubel 1981 (Bull. AMS) | 元数据+评论 | —（AMS 403） |
| Bournez et al. 2016 (ICALP) | 摘要+结论 | `pdfs/bournez2016_icalp_poly_length.pdf` |

> 注：`pdfs/` 里的副本只用于本地阅读；引用时以 DOI/期刊版本为准。付费论文没有绕过访问限制。

---

## G. Heterogeneous substrates：MEMS / 微流控 / 光子

这一节与 A–D 不同：下面列出的是**阅读目标（reading targets）**，不是已核对文献。元数据可能有误，**不要直接引用**；等真正读到 PDF/摘要后再升级成 A–F 的格式。

对应 `DESIGN.md §3.7`。核心问题不是“这些技术能不能做计算”，而是：

    对每个 operator，哪个 substrate 的 constitutive law 本身就等于它？
    跨域的 transducer 要付多少 gain / noise / bandwidth / energy？

### G1 MEMS：二阶 ODE 与时间基准

要读的三类：

1. **高 Q 谐振器与滤波器**——`ω0=√(k/m)`、`Q=mω0/c`、`τ_settle~2Q/ω0`，以及 Q 随温度/应力/封装的漂移。
   目标：Bannon–Clark–Nguyen, “High-Q HF microelectromechanical filters,” IEEE JSSC 35(4), 2000；Nguyen, “Frequency-selective MEMS for miniaturized low-power communication devices,” IEEE Trans. MTT 47(8), 1999。
2. **非线性 MEMS**——Duffing `αx³`、参数激励、mode coupling、bistability、pull-in。
   目标：Lifshitz–Cross 关于 nanomechanical/micromechanical resonator 非线性动力学的综述章节；Senturia, *Microsystem Design*（Kluwer, 2001）的 pull-in/stiction 部分。
3. **接口**——静电驱动 `F∝V²`、电容读出 `ΔC/C`、thermo-mechanical noise `S_x=4kT·c`。
   目标：同上两本的 actuator/sensor 章节。

要问的问题：Q 能否在片上调？`ω_ref` 的温漂能否被 characterize 到优于 `g_m/C` 的漂移？pull-in margin 与驱动幅度的关系是什么？

### G2 微流控：graph Laplacian 与 chemical reaction network

要读的两类：

1. **fluid network = RC network**——`C_f·Ṗ = −L_R P + Q`，`P↔V, Q↔I, R_f↔R`；扩散、输运、reaction–diffusion；蒸发/气泡/吸附等不可逆漂移。
   目标：Whitesides, “The origins and the future of microfluidics,” Nature 442, 2006；Squires–Quake, “Microfluidics: Fluid physics at the nanoliter scale,” Rev. Mod. Phys. 77, 2005。
2. **chemical reaction network = polynomial ODE**——mass action `v=k∏c^ν`、`ċ=S·v(c)`、steady state / deficiency。
   目标：Horn–Jackson, “General mass action kinetics,” Arch. Rational Mech. Anal. 47, 1972；Feinberg 的 chemical reaction network theory 讲义；Epstein–Pojman, *An Introduction to Nonlinear Chemical Dynamics*（Oxford, 1998）。

要问的问题：哪些 reaction 的 `S`、`v` 能在片上稳定实现？`θ` 的可辨识性需要多少测量通道？慢漂移的时间尺度是否落在 background calibration 的窗口内？

### G3 光子：communication 与 linear transform

要读的三类：

1. **WDM / 光互连**——一条 waveguide 承载 N 个 λ；physical wire count < logical channel count；laser power、loss、PD 噪声。
   目标：Sun et al., “Single-chip microprocessor that communicates directly using light,” Nature 528, 2015。
2. **光 MVM / weight bank**——ring/MZI 调 weight，`I_j=Σ_i w_ji P_i`；相干 vs 非相干。
   目标：Shen et al., “Deep learning with coherent nanophotonic circuits,” Nature Photonics 11, 2017；Tait et al., “Neuromorphic photonic networks using silicon photonic weight banks,” Scientific Reports 7, 2017。
3. **器件非理想性**——ring 热漂移、波长校准、modulator linearity、shot noise。
   目标：Bogaerts et al., “Silicon microring resonators,” Laser & Photonics Reviews 6, 2012；Marpaung–Yao–Capmany, “Integrated microwave photonics,” Nature Photonics 13, 2019。

要问的问题：在目标 `S_n`/linearity 下，每个 λ 的功率预算还剩多少？热漂移的闭环校准带宽够不够？光 MVM 的权重是准静态还是可在线更新？

### G4 共同的一张表（要填的数）

| substrate | 要实测/查到 | 进哪个 slack 轴 |
|---|---|---|
| MEMS | `ω0, Q, α, S_x(f), τ_settle, pull-in margin` | κ_hw、噪声、鲁棒 |
| 微流控 | `R_f, C_f, τ, 漂移率, 可辨识性` | κ_hw、校准、表示 |
| 光子 | `loss, P_laser, shot/PD noise, thermal drift, linearity` | 噪声、带宽、路由 |
| transducer | `gain, noise, bandwidth, energy, 非线性` | 每次跨域都进 |

这张表填不完，`substrate.py` 就不该存在（对应 `DESIGN.md §10` 的 `substrate.py` 是待实现，不是已实现）。

---

## H. Adaptive representation / wavelet / adjoint：reading targets

与 §G 同样性质：**未核对，不要直接引用**。对应 `DESIGN.md §5.7–§5.9`、Phase 4.5。

要回答的问题不是“WENO 怎么实现”，而是：

    哪个算子在做 regularity detection？
    哪个在做 representation 选择？
    哪个在做守恒通量？
    forward / inverse / adjoint 三者中，哪两个可以共用硬件？

### H1 WENO 与其“尺度选择”解释

- Liu–Osher–Chan, “Weighted essentially non-oscillatory schemes,” J. Comput. Phys. 115:200–212, 1994。
- Jiang–Shu, “Efficient implementation of weighted ENO schemes,” J. Comput. Phys. 126:202–228, 1996。
- Shu, “Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws,” 1998 综述（Lecture Notes in Mathematics / ICASE report）。
- Borges–Carmona–Costa–Don, “Improvement of weighted essentially non-oscillatory schemes with a new smoothness indicator”（WENO-Z），J. Comput. Phys. 227:3191–3211, 2008。

要问：`β_k` 的定义与 stencil 选择能否分离？哪些部分必须是数字的（权重、判决），哪些可以在 analog 并行算（候选通量）？

### H2 小波、lifting 与 biorthogonal

- Daubechies, “Orthonormal bases of compactly supported wavelets,” Comm. Pure Appl. Math. 41:909–996, 1988。
- Mallat, “A theory for multiresolution signal decomposition: the wavelet representation,” IEEE TPAMI 11:674–693, 1989。
- Cohen–Daubechies–Feauveau, “Biorthogonal bases of compactly supported wavelets,” Comm. Pure Appl. Math. 45:485–560, 1992。
- Sweldens, “The lifting scheme: a construction of second generation wavelets,” SIAM J. Math. Anal. 29(2):511–546, 1998。

要问：lifting 的 predict/update 是否都能写成可转置 edge？undecimated 版本的 redundancy 与 state 预算如何权衡？biorthogonal 时 `W_inv` 与 `W_star` 的差是否可测？

### H3 Adjoint / 反问题

- Giles–Pierce, “An introduction to the adjoint approach to design,” Flow, Turbulence and Combustion 65:393–415, 2000。
- Plessix, “A review of the adjoint-state method for computing the gradient of a functional with geophysical applications,” Geophys. J. Int. 167:495–503, 2006。

要问：时间反演需要多少 checkpoint？事件/reset 是否让轨迹不可逆？transposeable edge 的边界在哪里？

### H4 小波预条件

- Beylkin–Coifman–Rokhlin, “Fast wavelet transforms and numerical algorithms I,” Comm. Pure Appl. Math. 44:141–183, 1991。

要问：`‖∂_x^m ψ_{j,k}‖~2^{mj}` 的尺度结构能否直接映射成 per-scale `C_j` / `g_m/C`？预条件后 `κ` 不变（§0(A)），但 pole 跨度与噪声如何重新分配？
