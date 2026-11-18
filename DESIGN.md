# 连续时间模拟/混合计算 fabric —— 可落地方案

> 目标：把"这方程能不能跑"从口号变成一个**可计算的可行性判定**，并给出对应的硬件分层、编译器 pass 结构和分阶段验证路线。

---

## 0. 先把设计钉死的三个结论

**(A) 刚度不可能靠缩放或换基消除。**
时间缩放 t→at 把全部 λ 乘同一个数，相似变换 TJT⁻¹ 保持谱不变，因此

    κ_t = max|Re λ| / min|Re λ|

在任何 (T, A_t) 下不变。硬件的频率窗宽度 κ_hw = ω_max/ω_min 是工艺常数（τ_min ~ 寄生极限 ≈ ns，τ_max ~ 漏电/观测窗 ≈ ms），取

    κ_hw ≈ 10^5 ~ 10^6   (100–120 dB)

于是可行性就是一条**宽度必须装进窗口**的判据：κ_t ≤ κ_hw；时间缩放只负责平移窗口中心：

    A_t = sqrt( ω_hw,max · ω_hw,min / (ω_max · ω_min) )

超出宽度的唯一出路是**删模态**（QSSA/CSP/ILDM）或**多速率**（付同步误差），不是"更好的积分器"。

**(B) 单节点精度–速度守恒律。**

    σ² = α·kT/C ,  τ = C/g_m   ⟹   σ²τ = α·kT/g_m

α ≈ 0.4–2（理想单极点 OTA 为 (2/π)γ）。**C 只在 σ 和 τ 之间分配，乘积由 g_m（即功耗）决定。** 换算成动态范围（V_swing = 1 V）：

    DR ≈ 85 dB + 10·log10(C / pF)

即"每 10 倍电容换 10 dB"。这条式子直接替代"逐级 SNR budget"。

**(C) 一切扰动走同一个 resolvent。**
噪声、失配、开关电荷注入、路由寄生、事件抖动（δx = ẋ·δt）、量化，全部一阶等价于在 (E,A) 上注入 B_p·p(t)，其到观测的传递都是

    H(s) = C (sE − A)⁻¹ B_p

所以三个泛函就是全部判据：

| 泛函 | 数学 | 管什么 |
|---|---|---|
| ‖H‖₂² = tr(C P Cᵀ) | 广义 Lyapunov：E P Aᵀ + A P Eᵀ + B_p Q B_pᵀ = 0 | 方差 / SNR |
| ‖H‖∞ | Hamiltonian 二分 | 最坏情况、非正规放大、闭环 small-gain |
| κ_t / 谱 | 特征值 | 频率窗 |

**SNR、switch matrix、stiffness 不是三个问题，是同一个 (E,A) 的三个范数。**

---

## 1. 统一形式：硬件坐标下的 descriptor 系统

所有层（integrator / MVM / 变换 / stiff bank / flux / 事件）都归一到

    E(θ)·ẋ = A(θ)·x + B(θ)·u + B_p(θ)·p(t) ,   y = C x
    θ = ( T,        A_t,       σ,          c,        w )
         basis    time-scale  splitting   routing   sizing

E 奇异不是缺陷而是特性：代数约束（QSSA 后的快流形、理想运放虚地、KCL 求和节点）天然是奇异 E——这正是"低阻节点做电流求和"的形式化。

### 1.1 扰动端口 B_p 清单（compiler 的输入规范）

| 源 | 注入位置 | 强度 | 备注 |
|---|---|---|---|
| 热噪声 | 每个积分器状态 | 输入折算 4kTγ/g_m；输出 σ² = α kT/C_i | 对角 Q |
| 1/f | 同上，**增广状态** | Σ Lorentzian 拟合 | 增广后仍在 Lyapunov 框架内 |
| 失配 | ΔA 静态 | Pelgrom σ ∝ 1/√(WL) | 结构化，非范数 |
| 开关电荷注入 | 每次切换 | δQ/C，与 C 成反比 | 事件相关，非平稳 |
| 路由寄生 | ΔE, ΔA（每 config） | PEX 给 Y_c(s) = G_c + sC_c | 见 §3.2 |
| 事件抖动 | ẋ·δt | δt ~ 比较器噪声/延迟 | 自动被 resolvent 放大 |
| 采样 | kT/C + 孔径抖动 | 2πf·σ_j·A | 仅在有 S&H 处 |

> 因为 1/f 与抖动都能写成增广的 LTI 端口，**"非白噪声"不破坏 H2 框架**——这是全案能用一套线性代数做预算的前提。

### 1.2 可行性不是布尔值，是 slack 向量

编译输出**六个无量纲余量**（不是一维"难度分"）；其中鲁棒轴拆成失配与接口两路：

    slack = ( σ_y/σ_req ,  B_req/B_hw ,  κ_t/κ_hw ,
              Γ_proj‖ΔA‖ ,  ‖H‖∞‖Δu‖ ,  η_route ,  E_tail/tol_L2² )
              噪声         带宽         刚度
              鲁棒(失配)    鲁棒(接口)    路由      表示

- **Γ_proj** 是 QZ 投影掉守恒零模态后的 resolvent 峰值；全 Γ 在 s=0 处有零模态极点，
  只作诊断，不进 slack——否则 Burgers 这类守恒系统会被误判为"无限脆弱"。
- **‖H‖∞‖Δu‖** 是接口扰动的小增益；它与失配轴是两个对象，不能用一个 Γ 代替。
- **E_tail_req = tol_L2²**：tol_L2=1% 的 L2 表示误差对应 1e-4 的谱尾能量占比。

任一 > 1 即不可行，且**告诉你该拉哪根杠杆**：κ 超 → 删模态；Γ_proj‖ΔA‖ 超 → 降 Q / 校准 / 换基；‖H‖∞‖Δu‖ 超 → 衰减接口扰动或降闭环增益；E_tail 超 → 换局部表示；σ_y 超 → 加 C（但只加在 ‖H_i‖₂ 大的节点上）。

---

## 2. 五条尺度律（它们决定架构，不是审美）

### 2.1 精度–速度–功耗

单节点，B = 10 MHz，V_DD = 1 V，g_m/I_D = 15：

| σ_rms | C | g_m | 单节点功耗 | 1000 节点 |
|---|---|---|---|---|
| 1 mV | 2.8 fF | 0.17 µS | 12 nW | 12 µW |
| 100 µV | 0.28 pF | 17 µS | 1.2 µW | 1.2 mW |
| 10 µV | 28 pF | 1.7 mS | 115 µW | 115 mW |

**结论：功耗与面积都不是瓶颈**（1000 节点 @ 10 µV @ 10 MHz ⇒ ~0.1 W、~3 mm² 电容）。真正稀缺的是 κ_hw、路由寄生和校准漂移。资源该投在**拓扑与校准**，而不是"更省电的 OTA"。

### 2.2 加速的代价

B_hw ≈ A_t·B_model，且白噪声下 σ² ∝ B ⇒ **加速 A_t 倍，噪声功率涨 A_t 倍**。又由 §2.1，σ²τ 固定 ⇒ 加速不改变 σ²τ，只改变 τ 落在窗口哪一端。真正的约束是 §0(A) 的窗口宽度。

### 2.3 κ 的相似不变性（对直觉的修正）

κ_t 在 (T, A_t) 下不变，所以：

- **"用专用 stiff 线性块实现"不降低 κ**。它降低的是"实现成本"（每模态一条便宜的一阶支路，而不是让通用积分器跨 8 个数量级），但芯片仍要能产生 τ ∈ [10 ns, 1 ms]。
- 能改 κ 的只有两种：**删模态**（快模态被奴役且稳定 ⇒ g(x,y)=0 ⇒ 从谱里消失）、**多速率**（付同步误差，§4.3）。

### 2.4 λ_k = −νk²：spectral 对模拟比对数字更重要

    κ_diffusion = N² ,   κ_biharmonic = N⁴

N=64：4×10³（轻松）；N=1024：10⁶（贴边）；biharmonic N=128：2.7×10⁸（**必然超窗**）。

第一条硬规则：**扩散类优先 spectral；双调和/高阶算子在中等分辨率下就必须降维或混合**。反过来，spectral 基里每个模态是独立的一阶支路 g_m/C = νk²，这是模拟硬件最擅长的事——前提是 k² 的跨度装得进窗口。

### 2.5 路由容差 ∝ 1/Q

二阶节极点灵敏度 ∂λ/∂p ~ Q。开关/Ron/C_par 的相对误差 2% ⇒

    δω₀/ω₀ ~ Q · 2%     ⇒     误差 ~ 1/Q 级

**规则：Q ≲ 50 才允许"免校准"；更高 Q 必须 trimming/校准，或分解为一阶节。** 这就是"2% 的 switch network 在高 Q 环里造成大 pole shift"的定量版，也是 §1.2 中 Γ_proj‖ΔA‖ < 1 的工程化形式。

---

## 3. 分层 fabric：8 tile + 1 supervisor

不做"很多 op-amp + 一个巨大 crossbar"。理由不是审美，而是 §2.5：crossbar 寄生随规模线性增长，容差随 Q 放大，两者相乘不可控。

| # | Tile | 语义 | 误差模型 | 硬约束 |
|---|---|---|---|---|
| T1 | **State** | C·V̇ = I，V 域高阻存储 | α kT/C + 漏电 | 高阻节点**只**接本地 C |
| T2 | **Algebra** | 电流域 SUM/MUL/NL（Gilbert, tanh） | g_m 失配、有限摆幅 | 只在低阻节点 |
| T3 | **Operator** | 可编程 MVM（T, T⁻¹, dense A） | 权值失配 ∝ 1/√(WL) | 规模 ≤ 256，超了拆 |
| T4 | **Spectral/Stiff** | 对角极点阵 ż_k = −λ_k z_k + (·)_k | 每支路独立 kT/C | τ 覆盖 1e-8…1e-3 |
| T5 | **Flux** | cell C + 非线性 flux + limiter | 单调性丧失、g_m 失配 | KCL 天然守恒 |
| T6 | **Event** | 比较器 + 事件锁存 + 复位 + S&H | 抖动 ẋ·δt、亚稳态 | 迟滞防 Zeno |
| T7 | **Route** | 局部电流网格 + 缓冲主干 + 开关阵 | Y_c(s) = G_c + sC_c | 跨 island 只走电流 |
| T8 | **Noise** | 物理噪声源 / 可重放 PRNG+DAC | 用于 SDE 与鲁棒性测试 | Q 可控 |
| S | **Supervisor** | 数字：配置、换基、校准、时序 | — | 不在模拟信号路径 |

### 3.1 两条不可违反的接口契约

1. **阻抗域标注**：每个网络必须声明 current（低阻，可长距离、可求和）或 voltage（高阻，仅本地 C）。跨 island 一律 current。
2. **每个 tile 必须给出 (E,A,B,C) 以及 ‖H‖₂、‖H‖∞ 两个数**，否则不能进 layout。这条把"模拟设计"和"编译器"绑在同一套数学上。

### 3.2 开关矩阵怎么算（问题 2 的答案）

对每个 routing config c：

1. PEX → Y_c(s) = G_c + sC_c（必要时 +s⁻¹L⁻¹ 或频变 macromodel）
2. 组装 E_c·ẋ = A_c·x + Bu ⇒ H_c(s) = C(sE_c − A_c)⁻¹B
3. 与 ideal 比：η_route = sup_ω ‖H_real − H_ideal‖₂/‖H_ideal‖₂，加群延迟误差 τ_g、串扰 ‖H_ij‖∞/‖H_ii‖∞
4. 闭环判据：‖G_loop·Δ_route‖∞ < 1，Δ_route 用**结构化界**（每个开关的 R_on、C 区间），用 μ 而非范数，否则过度保守
5. 成本函数保留 φ / X_talk 项，但**权重由 ∂σ_y/∂p（敏感度）给出**，不是手调

R_on 与寄生 C 的冲突（做宽降 R_on、升 C）就体现在 G_c 与 C_c 同时进 Y_c——优化的是 ‖H_c‖∞，不是 R_on。

---

## 4. 编译器：7 个 pass

    equation
      → P0 离散化/结构化        (M,K,f) 或 (L,N) 分裂
      → P1 谱与正则性分析        λ, κ_t, Γ, 谱系数衰减率 p
      → P2 表示选择              basis T / splitting / 局部表示
      → P3 尺度对齐              A_t, 频窗检查
      → P4 扰动预算              H2 per source, C 按敏感度分配
      → P5 布局与路由            config c, Y_c, Δ, small-gain
      → P6 可行性证书            slack 向量 / 不可行原因
      → P7 配置生成 + 校准计划

### 4.1 P2 决策树（"落地"的核心）

对谱聚类后的每个模态簇：

    if  κ_cluster ≤ κ_hw·(1−m):              → 直接实现（优先对角/spectral 支路）
    elif 快模态稳定 && 有谱间隙 && 被奴役:    → QSSA/CSP 降为代数约束（E 奇异化）
    elif 快模态只在局部活跃:                  → 空间局部 island（边界层专用快 tile）
    elif 非正规放大主导（Γ 大）:              → 换基做对称化；不行则交数字协处理器
    else:                                    → 多速率 + 同步误差预算（§4.3）

注意"换基"排在"多速率"之前：基变换能同时改稀疏度与条件数（§5.1），通常比多速率便宜。

### 4.2 积分变换：空间 vs 时间必须分开（问题 3a 的答案）

| | 空间 u(x,t)→û(k,t) | 时间 x(t)→X(ω) |
|---|---|---|
| 因果性 | 当前时刻可算 | 需要未来信息，**非因果** |
| 硬件 | 一次 MVM（T3） | filter bank / resonator bank / STFT |
| 实现 | T, T⁻¹ 可编程 | ż_k = (−α_k + iω_k) z_k + x |

卷积核同理：K(s) ≈ d + Σ_m r_m/(s − p_m) ⇒ 每极点一个辅助状态

    ż_m = p_m z_m + x ,    y = d·x + Σ_m r_m z_m

**memory kernel → 辅助动力学状态**，包括分数阶/粘弹性核。这条让"非马尔可夫"问题重新变成有限维 (E,A)，从而回到同一套范数。

### 4.3 多速率/分裂的合法性条件

两 island 时间坐标不同（t_A = 10³ t_B）**不能直接连线**。合法形式只有两种：

1. **waveform relaxation**：迭代到收敛，收敛率由 ‖G_AB·G_BA‖ 的谱半径给出——可算，进 slack 向量。
2. **S&H 同步**：每次交换付 kT/C + 孔径抖动，误差进 B_p；必须验证 ‖G_loop·Δ_sync‖∞ < 1。

否则不严格。写进 compiler 合法性检查，不接受"差不多"。

---

## 5. 表示层：basis 是一等公民，且是双向杠杆

### 5.1 换基同时改稀疏度和条件数（问题 2+3 的统一回答）

    z = T x :   ż = T A T⁻¹ z + T·N(T⁻¹ z)

- ‖T‖·‖T⁻¹‖ 直接乘在硬件扰动上：δz = (sI − TAT⁻¹)⁻¹ · T·δA·T⁻¹·z
- 且 Γ(TAT⁻¹) 可以远小于 Γ(A)（非正规 → 对称化）

所以目标不是"最稀疏的 T"，而是

    T* = argmin_T [ op density(TAT⁻¹) + log‖T‖‖T⁻¹‖ + state density ]
                     硬件量              条件数税

**这是"开关矩阵对频域影响"的表示层版本**：Γ 是同一个量，只是扰动从"寄生"换成了"换基矩阵的有限精度"。

### 5.2 pseudo-spectral 回路

    û --T⁻¹--> u --N--> N(u) --T--> N̂(u)

对应硬件：**spectral state ↔ physical nonlinear island**。非线性局部算子（u³ 在 Fourier 域变卷积）必须回物理空间算——这不是软件技巧，而是拓扑必然：T3(MVM) 与 T2(NL) 之间一定有一次往返。

### 5.3 激波/跳变：三条硬事实

1. 全局谱基遇跳变必然 Gibbs（过冲 ~9%），且指数收敛退化为代数收敛——**不是实现问题**。
2. shock sensor 用谱尾：拟合 log|a_k| vs log k，指数衰减 → 光滑；代数衰减率 p → 正则性下降；同时看 E_tail = Σ_{k>N/2}|a_k|² / Σ|a_k|²。阈值取 **E_tail_req = tol_L2²**（tol_L2=1% ⇒ 1e-4），它是验收参数，不是自然常数。**必须区分"未解析的陡梯度"和"真激波"**（前者加分辨率，后者换表示）。
3. wavelet 也不是自动无 Gibbs；基与重构必须显式选，且切换要有**迟滞**，否则在阈值附近抖动，产生开关电荷注入噪声。

### 5.4 守恒律：flux tile 是最漂亮的模拟原语

    C_i · V̇_i = I_{i−1/2} − I_{i+1/2}

这就是**电容梯上的 KCL**。Σ_i V̇_i = 0 自动成立（内部通量相消）——离散守恒在模拟域是"免费"的，在数字域反而要小心。

limiter 用饱和跨导（tanh 型光滑 minmod）实现简单，但**丧失严格 TVD**：compiler 必须用数值孪生验证单调性/极值原理，不能声称。这是明确的工程取舍，写进规格。

### 5.5 事件：连续动力学 + 离散事件机

    ẋ = f(x), g(x) < 0  直到  g = 0  ⇒  x⁺ = R(x⁻)

必须是一等原语，不要用多项式逼近跳变。精度由抖动给出：δx = ẋ·δt，所以

    |ẋ|·δt_cmp < ε   ⇒   δt_cmp < ε / max|ẋ|

比较器噪声+延迟 ~ns ⇒ 对 |ẋ| ~ 10⁶ 的状态，相对精度 10⁻³ 是极限。这个数直接决定哪些 hybrid 系统能上、哪些不能。

---

## 6. 奇异区：QoI 才是指标，formal order 不是

> 结论先行：不连续附近降阶不是"精度损失"，而是用局部精度换守恒/熵/稳定性。真正的代价**不是降阶本身，而是耗散区的宽度**——而这个宽度对航空和海洋可以用**同一个数**衡量。

### 6.1 用一个数统一航空与海洋：等效数值黏性 ν_num

稳态 1D 激波（Burgers 型）的黏性剖面：

    u(x) = s − (Δu/2)·tanh( Δu·(x − x_s) / (4ν) )

因此 10–90% 的激波宽度（以 cell 计）与等效黏性的关系是

    W_d = 8.79 · ν_num / (Δu · Δx)   ⟺   ν_num = W_d · Δu · Δx / 8.79

这个反演式让 compiler 从**实测剖面**直接拿到 ν_num，不需要知道里面用了什么 limiter。于是"几阶"这个量被彻底绕开了。

- **航空**：ν_num ↑ ⇒ shock smear ⇒ x_s 偏移、pressure recovery 变差 ⇒ wave drag 错。更敏感的是 shock–BL interaction 的 separation onset、buffet onset、壁面热流；此时 C_L 可能只差 1%，但敏感量已经完全不对。
- **海洋**：ν_num ↑ ⇒ 虚假跨密度面混合。用 Osborn–Cox 关系从 tracer 方差反演

      κ_num = −(1/2)·(d⟨c'²⟩/dt) / |∇⟨c⟩|²

  红线：κ_num ≳ 0.1·κ_diapycnal（≈1e−6 m²/s）时，长期环流（heat/salt transport、PV、层结、斜压不稳定）不可信。

⇒ **验收指标固定为五元组 (W_d, ν_num 或 κ_num, 守恒漂移, x_s(t) 轨迹, smooth 区 L2)**，阶数不进验收。

### 6.2 模拟 fabric 里没有"cell 数"这个旋钮

数字 CFD 靠加密网格锐化激波。fabric 里 cell = 物理 tile，"加密" = 单位长度放更多 flux tile = 面积。于是 AMR 退化成**物理资源的动态重分配**，而重分配要切换开关 ⇒ 电荷注入（§1.1）。

量级：每次重构 δQ ~ fC 级。1 fC 注入 1 pF 节点 ⇒ δV = 1 mV。若重构率 f_sw = 1 MHz，等效输入噪声

    √S ≈ √( 2·δV² / f_sw ) ≈ 1.4 µV/√Hz

而 100 pF / 100 µS 积分器自身热噪声约 10 nV/√Hz（4kTγ/g_m，γ=2/3）。**重构噪声是热噪声的 ~140 倍**，会一次性吃掉 §0(B) 给的全部预算；相对 §6.9 的器件噪声 7.9 / 4.7 nV/√Hz 是 180× / 300×。

若把注入节点做到 ~30 pF（δV = 33 µV），√S ≈ 45 nV/√Hz —— 这是后文引用的乐观值，比 1 pF 情形低 31×，相对器件噪声仍是 5.7× / 9.6×。**节点电容是这里唯一能换噪声的杠杆，倍数必须连同 C 一起引用。**

三条对策，必须选一条并把结果写进 B_p：

    (a) 静态分配 + 足够宽缓冲带（x_s 在带内移动多个时间常数，不切换）
    (b) 环状可寻址 flux 阵列：tile 常开，只移动"活跃窗口指针"，零拓扑重构 ⇒ 零注入   ← 推荐
    (c) 软路由（只改输入权重，不改拓扑），代价是 MVM 端口数与权值失配

### 6.3 保正性：连续时间 FV 的结构性优势

    C_i · V̇_i = F_{i−1/2} − F_{i+1/2}

没有时间步 ⇒ 没有 CFL。若 F 单调（upwind）且满足单边 Lipschitz 条件，则不变区**前向不变**：保正/保单调是结构性的，不是条件性的。

这是相对数字 WENO 的真实优势——数字要 CFL + limiter 才勉强保正，模拟只要保证 F 的形状。但两条必须诚实：

- tanh 型光滑 limiter **不是 TVD**，只是前向不变。验收只能验"不变区 + 不产生新极值"，不能声称 TVD。
- 验证方法是检查不变区边界上向量场的指向（一次性与步长无关），不是扫 CFL。

### 6.4 唯一的结构性泄漏点：spectral ↔ local 接口

flux tile 内部守恒是免费的（KCL）；但全局谱表示与局部 tile 的耦合**不自动守恒**——这是整个架构里唯一会系统性丢守恒量的地方。

数字做法是接口用同一个数值通量、两侧一致。模拟可以做得更硬：

    F_{i+1/2} 由单一 g_m 单元算一次  →  +I 注入左 cell，−I 注入右 cell（电流镜像）

守恒由拓扑保证，不依赖两侧公式一致、不依赖数值一致性检查。规则：**任何跨表示接口必须是"单通量源 + 电流镜像"**，禁止"两边各自算一遍"。这是 analog 版的 consistent numerical flux，而且更便宜。

### 6.5 explicit singular state：与航空巡航场景高度匹配

把界面位置升为状态：

    ẋ_s = s(x_s, u⁻, u⁺)      s 由 Rankine–Hugoniot 从两侧局部状态给出

两侧继续用 smooth 表示；拓扑变化（生成/合并/消失/激波-激波相交）交给 event tile + reset map。

为什么对模拟特别合适：

- 一个移动不连续本质只有 O(1) 自由度，Eulerian 均匀网格却要 O(N) 高频模态——纯浪费。
- **x_s 是连续状态，"激波穿过 cell 边界"这个数字伪影根本不存在**（数字里它是守恒误差和降阶的来源之一）。

compiler 的 selector 用两个数决定 fitting 还是 capturing：

    M_int = 界面数        T_int = 界面寿命 / T_obs

    M_int 小 且 T_int/T_obs ≳ 0.5            →  shock fitting（x_s 作状态）
    否则（相交、Mach 反射、破碎、湍流）        →  退回 local flux tile（capturing）

典型：跨声速机翼巡航 = 每侧翼面 1–2 道、长寿命、x_s(y) 空间光滑 ⇒ 强匹配。高超声速 bow shock = 驻定单道，但后方要保 T/p/ρ 高阶 ⇒ 正好是"sharp shock + high-order post-shock"结构。

### 6.6 检测必须超前（模拟没有 rewind）

数字可以 checkpoint、可以拒绝重算；模拟已经物理实现，**混合一旦发生不可逆**：Gibbs 过冲穿过非线性 EOS 或饱和元件之后，信息已经丢了。

所以 basis switching 的触发器不能是"检测到过冲"，而必须是**预测**：

    滑动窗拟合谱尾指数 p(t)：|a_k| ~ k^(−p(t))
    dp/dt < −阈值   →   预测 t + τ_lead 处穿越   →   提前切换

τ_lead = 重构延迟（开关建立 + supervisor 判决，µs–ms 量级）。这条改写了 §5.3 的 sensor 规格：**它是速率检测器，不是阈值检测器**，且必须带迟滞——否则在阈值附近抖动，又变回 §6.2 的注入噪声。

### 6.7 资源分配律：把"1–5%"变成预算

误差两项方向相反：

    E_smooth ~ C₁·e^(−αN)          谱截断，N 个模态
    E_sing   ~ C₂·f(W_d)           局部层宽度，n_local 个 tile
    N_state  = (1−s)·n_smooth + s·n_local ,   s = 奇异区面积占比

存在最优 (N, n_local)。**Phase 2/3 的任务是实测这条曲线，不是假设它。**

降级判据：若 s ≳ 10%，局部 tile 总面积超过全局 tile，"用 1% 的奇异区省下全局高阶"的论证失效，机器退化成一个更慢的 FV 求解器。这个阈值必须在 Phase 0 就算出来。

### 6.8 对照实验结果（twin/qoi.py + twin/coupling.py，可复跑）

    python3 -m bench.run_twin_experiments      # 约 3 min，输出 results/report.txt + json + png

**(a) ν_num 反演自检**（解析 Taylor 剖面 + 白噪声）

| ν_true | 无噪声 rel.err | 1% 噪声 rel.err | W_d [cell] |
|---|---|---|---|
| 2e-3 | 9e-15 | 1.0% | 1.12 |
| 1e-2 | 4e-16 | 0.04% | 5.58 |
| 8e-2 | 2e-16 | 0.4% | 44.7 |

反演链（W_d → ν_num）在 1.1–45 cell 的宽度范围内都闭合；W_d → 1 cell 时到达网格分辨率下限，此时 ν_num 只可作上界。

**(b) 同一激波上，名义阶数不预测分辨率**（N=128，Burgers，T=0.3）

| scheme | 名义 | W_d [cell] | ν_num | L1 |
|---|---|---|---|---|
| first-order | 1 | 7.3 | 5.8e-3 | 1.2e-2 |
| minmod | 2 | 9.5 | 7.5e-3 | 2.2e-2 |
| van Leer | 2 | 9.5 | 7.4e-3 | 2.2e-2 |
| 5-point WENO | 5 | 4.5 | 3.5e-3 | 4.6e-4 |

**二阶 limiter 在这里比一阶还宽**。原因不是 bug，而是两个已知效应：limiter 在光滑极值处截断（L1 退化到 O(h)），以及"重构＋Riemann 耗散"的耦合——名义阶数完全不能替代 W_d 这个实测值。这正是 §6.1 要用 ν_num 而不是 order 的理由。

诚实标注：本原型的 WENO 用固定 ε=1e-6，在低幅低分辨率下 smoothness indicator 与 ε 量级不匹配会掉阶；它只被当作"一个压缩性很强的 5 点重构"，不宣称已验证 5 阶。要做阶数认证，需 ε ∝ h² 并重测。

**(c) 锋面宽度增长 → κ_num**

| case | 参考值 | κ_est | 比值 |
|---|---|---|---|
| 纯扩散 κ=1e-4 | 1e-4 | 1.05e-4 | 1.05 |
| 纯扩散 κ=3e-4 | 3e-4 | 3.06e-4 | 1.02 |
| 半离散迎风，CFL=0.4 | 4.88e-4 | 4.94e-4 | 1.01 |
| 半离散迎风，CFL=0.8 | 4.88e-4 | 4.94e-4 | 1.01 |

两处值得记住：κ 估计器误差 2–5%，可以直接当 QoI 用；**半离散迎风的数值扩散是 ½·a·h，与 CFL 无关**（(1−CFL) 因子只属于显式欧拉的时间离散），所以两个 CFL 给出同一个 κ_num——这也顺带验证了时间积分器已收敛。红线换算：κ_num < 1e-6 m²/s ⇒ 一阶迎风需要 h < 4e-5 m（a=0.1 m/s），这就是海洋模型必须用高阶传输的原因。

**(d) 接口守恒：单通量源 vs 两侧各算一遍**（周期域，T=0.2）

| 模式 | drift | 预测 drift | max\|ΔF\| |
|---|---|---|---|
| 单源（镜像） | ≤1e-15（舍入） | 0 | 0 |
| 单源 + 5 步延迟 | 0.0 | 0 | 0 |
| 两侧各算，ε=1e-2 | 6.87e-4 | 6.87e-4 | 8.5e-3 |
| 两侧各算，ε=5e-2 | 3.57e-3 | 3.57e-3 | 4.2e-2 |
| 无 trace 交换 | -1.88e-2 | -1.88e-2 | 2.9e-1 |

四条结论，全部可复现：

1. **守恒是拓扑性质，不是精度性质。** 单源 ≤1e-15（舍入），两侧各算 6.9e-4（ε=1%）——相差 **12 个数量级**；无 trace 交换再差 27 倍。
2. **漂移率可预测**：drift vs ε 的 log–log 斜率 = **1.004**，即 drift = ε·|F|·接口数·T。这一项应当直接进 B_p 预算，而不是当"实现细节"。
3. **延迟本身不破坏守恒**（单源+延迟 5 步 drift = 0）。所以流水线/流水级可以随便加；破坏守恒的是"两个独立的通量电路"。
4. **提高阶数不修守恒**：p 从 2 到 12，谱岛 L2 降 70×（6.0e-2 → 8.5e-4），drift 只从 6.61e-4 变到 6.87e-4（+4%）。泄漏不是截断误差。

对硬件的直接后果（已写入 §3.1 契约）：
- 跨表示接口在物理上必须是**同一个求和节点**，两侧看到的必须是同一个电流；
- 允许延迟，禁止两套乘法器（"两边各自算一遍"）；
- 不要用"提高阶数"来修守恒问题。

### 6.9 真实 g_m 接口：PDK 提取与重跑守恒实验

把 §6.4 的"理想单源+镜像"换成晶体管级模型。方法：ngspice + 真实 PDK，二极管接法电流镜，扫 I_in 得 (I_out, V_gs)，扫 V_out 得 g_ds，再推 λ = g_ds/I_D。脚本 twin/spice/extract.py。

| PDK / 器件 | I_D | g_m/I_D | λ = g_ds/I_D | r_o | I_out/I_in | 折算输入噪声 |
|---|---|---|---|---|---|---|
| sky130 nfet_01v8 (10/1) | 10 µA | 17.4 /V | 0.081 /V | 1.21 MΩ | 1.0222 | 7.9 nV/√Hz |
| gf180mcu nfet_03v3 (10/1) | 10 µA | 13.3 /V | 0.034 /V | 2.84 MΩ | 1.0328 | 9.0 nV/√Hz |
| IHP SG13G2 sg13_lv_nmos (10/0.13) | 10 µA | 22.4 /V | 0.970 /V | 65 kΩ | 1.5860 | 5.6 nV/√Hz |
| IHP SG13G2 npn13G2 | 10 µA | 37.4 /V | 0.028 /V | 3.61 MΩ | 0.9990 | 4.7 nV/√Hz |

> 噪声列不是 ngspice `.noise` 的实测值，而是由提取到的 g_m 按 4kTγ/g_m 折算（MOS γ=2/3，BJT γ=1/2，T=300 K）。extract.py 只做 op/dc；要改成实测，需加 `.noise` 分析并重跑。IHP MOS 的 λ 与 58.6% 镜像误差说明短沟道器件不适合做接口镜像，与 HBT 的 2.8% 形成对照。

接回 twin/coupling.py 的 InterfaceModel(mode="gm") 后，重跑 §6.8(d) 同一算例：

| 接口 | drift | max\|ΔF\| |
|---|---|---|
| 理想镜像 | ≤1e-15（舍入） | 0 |
| IHP HBT，仅 λ 项（共享参考） | −6.6e-6 | 5.1e-5 |
| IHP HBT，两条独立镜像（0.1% 系统误差） | −2.2e-4 | 8.6e-4 |
| gf180，仅 λ 项（共享参考） | −8.1e-6 | 6.3e-5 |
| gf180，两条独立镜像（含 3.3% 系统误差） | 7.1e-3 | 2.8e-2 |
| sky130，仅 λ 项（共享参考） | −1.9e-5 | 1.5e-4 |
| sky130，两条独立镜像（含 2.2% 系统误差） | 4.8e-3 | 1.9e-2 |
| IHP MOS，仅 λ 项（共享参考） | −2.4e-4 | 1.9e-3 |
| IHP MOS，两条独立镜像（含 58.6% 系统误差） | 0.179 | 0.523 |

三条结论：

1. **共享参考节点把系统误差整个删掉。** sky130 从 4.8e-3 降到 1.9e-5（250×），HBT 从 2.2e-4 降到 6.6e-6（34×），gf180 从 7.1e-3 降到 8.1e-6（880×），IHP MOS 从 0.179 降到 2.4e-4（740×）。这是"单通量源"在真实器件下的确切价值——不是"零泄漏"，而是**把可预测的系统误差换成不可预测的随机项与 λ 项**。indep 行的 eps_sys 取实测 ratio−1（含 λ·ΔV 项），所以这些倍数是上界。
2. **残余泄漏 = λ·ΔV·F，且被四个工艺的实测比例确认。** 共享参考下按 λ 升序：HBT 6.59e-6 (λ=0.0277) → gf180 8.10e-6 (0.0341) → sky130 1.93e-5 (0.0810) → IHP MOS 2.43e-4 (0.970)；相邻 drift 比值 1.23 / 2.39 / 12.6 对 λ 比值 1.23 / 2.38 / 11.97。这不是巧合，是把"漏电流 ∝ (1/r_o)·节点电压差"这条机制钉死了。
3. **理想镜像不可达。** 最好的实现（HBT + 共享参考）仍比理想差约 10¹⁰ 倍。10⁻¹⁶ 是双精度舍入极限，10⁻⁵ 是器件极限。设计目标应定在后者，不能拿前者做验收。

噪声：由 g_m 折算的输入热噪声 sky130 7.9 nV/√Hz、gf180 9.0、IHP MOS 5.6、HBT 4.7（非 .noise 实测，见上表注）。§6.2 的乐观重构注入 45 nV/√Hz（30 pF 节点）相对它们是 5.7× / 5.0× / 8.0× / 9.6×；1 pF 节点情形是 1.4 µV/√Hz，即 ~180× / 156× / 250× / 300×。两种口径都支持"重构噪声主导"，但引用时必须带节点电容。

**HBT vs MOS（回答"IHP SiGe 是否值得重点建模"）**：同一电流下 HBT 的 λ 小 2.9×、r_o 大 3×、g_m 大 2×、镜像系统误差小 20×（0.1% vs sky130 的 2.2%）。把四个工艺放在一起，接口层排序是 **HBT ≫ gf180 > sky130 ≫ 短沟道 MOS**：HBT 的 λ 比 IHP MOS 小 35×，镜像系统误差小 ~590×。这解释了为什么接口层应优先落在 BiCMOS/RF 工艺，而数字 supervisor 留在低成本 CMOS。代价是偏置电路（V_BE 温度系数 −2 mV/K）必须进校准回路。

**原先两个失败的工艺已解决**（都是 netlist 引入方式，不是 PDK 模型问题）：
- gf180mcu：`sm141064.ngspice` 的 noia/noib 表达式引用 `fnoicor`，而 `design.ngspice` 才给它默认值。extract 直接 include 模型文件，需补 `.param fnoicor=0`；
- ihp-sg13g2 MOS：必须 include `cornerMOSlv.lib`（它定义 `sg13_lv_nmos` 子电路并 include parm 文件），只 include parm 文件会报 "could not find a valid modelname"；PSP103 走 OSDI，子进程需设 `PDK_ROOT`/`PDK`。
两处修好后四个 PDK 均提取成功；sky130 仍不得设 PDK_ROOT（见 AGENTS.md）。

### 6.10 Phase 0 判定器：五个验收算例（可复跑）

    python3 -m bench.run_phase0        # ~2 min；23/23 checks passed

| 算例 | n | κ_t | κ/κ_hw | Γ | Γ_proj | σ_y | B_req | ‖H‖∞ | 方法 | 违规轴 |
|---|---|---|---|---|---|---|---|---|---|---|
| diffusion N=64 | 64 | 4.10e3 | 0.004 | 1.00e3 | 1.00e3 | 22.4 | 0.65 Hz | 1.00e3 | Hamiltonian | — |
| diffusion N=1024 | 1024 | 1.05e6 | 1.049 | 1.00e3 | 1.00e3 | 22.4 | 167 Hz | 1.00e3 | 扫频 | stiffness |
| biharmonic N=128 | 128 | 2.68e8 | 268 | 1.00e8 | 1.00e8 | 7.07e3 | 0.43 Hz | 1.04e8 | Hamiltonian | stiffness |
| Burgers N=128 | 128 | 2.22e3 | 0.002 | ∞* | 36.8 | 3.63 | 27 Hz | 6.13 | 扫频 | representation |
| Robertson t=1e4 | 3 | 6.22e7 | 62.2 | 1.71e4 | 6.19e3 | 0.190 | 1.43 kHz | 22.3 | 扫频 | stiffness, robustness_A |

\* Burgers 的全 Γ=∞ 是守恒零模态在 s=0 的极点；QZ 投影掉它后 Γ_proj=36.8（127 个稳定模态）。Robertson 的 1.71e4 也是慢模态的低频尖峰，投影后 6.19e3。**全 Γ 只作诊断，进 slack 的是 Γ_proj**——否则守恒系统会被误判为无限脆弱。

四条验收：

1. **κ 与解析式一致（±0.5 decade）**：diffusion κ=N²（4096 / 1.048e6）、biharmonic κ=N⁴（2.684e8），三条全中，log10 误差 0。
2. **H∞ 与解析值一致（±3 dB）**：diffusion 1/ν=1.00e3、biharmonic 1.00e8；Hamiltonian 与扫频的相对差 9.5e-7 / 3.8%（均 <3 dB）。n>256 只用扫频——2n×2n 稠密特征值不划算。
3. **σ_y 与解析值一致（±3 dB）**：diffusion √(1/2ν)=22.36、biharmonic 7.07e3；广义 Lyapunov 的相对残差 0（direct 路径）与 6.4e-14（Burgers 的 QZ 路径，127 个稳定模态 + 1 个守恒零模态）。
4. **判定落在正确的轴**：diffusion N=64 无违规；N=1024 在 κ_hw=1e6 下 1.049×（5% 越界，§2.4 的"贴边"）；biharmonic → stiffness；Burgers → representation；Robertson → stiffness，且 robustness_A 同量级。23/23 全过（原 19 条 + 鲁棒轴有限性 2 条 + 投影前后 H∞ 扫频一致性 2 条）。

**鲁棒轴探针**（1% 失配/接口预算，只加在 Burgers/Robertson 上）：Burgers 的
robustness_A = 36.8×1% = 0.37、robustness_io = 6.13×1% = 0.061，两轴都有限——鲁棒轴不是它的失败轴。
Robertson 的 robustness_A = 6.19e3×1% = 61.9（与 κ/κ_hw=62.2 同量级，慢模态是同一根杠杆）、
robustness_io = 22.3×1% = 0.22。其余三个算例没有失配模型，这两轴记 n/a（未知），不是 0。

三个口径决策，已经写进 `analysis/`：

- **κ 只去掉数值零模态**（|λ| < 1e-12·max|λ|），物理慢模态保留。Robertson 的 κ 随观测窗变化：t=1e-3 时 ~5e3、t=1e2 时 4.7e5、t=1e4 时 6.2e7、t=1e6 时 5.1e9；10¹¹ 是速率常数比，不是 Jacobian 谱（§8 已改）。
- **鲁棒轴是两个轴**：Γ_proj‖ΔA‖（失配，离不稳定多远）与 ‖H‖∞‖Δu‖（接口，small-gain）。全 Γ 含零模态极点，只进诊断不进 slack；1% 预算只是投影验收的探针，没有失配模型时记 NaN。
- **表示阈值 = tol_L2²**（默认 1% ⇒ 1e-4）。**Burgers 的表示轴必须用物理状态算**：精确解 p=1.44、E_tail=2.6e-4（representation=2.57）；WENO 捕捉解 p=7.28、E_tail=4.4e-5（representation=0.44）。捕捉格式用 ν_num 换掉了谱尾，所以它不能用来证明"表示轴可行"。

---

## 7. 分阶段落地（每阶段都有可证伪的验收）

### Phase 0：数字孪生 + 可行性判定器（无硬件）—— 判定器已验收
- descriptor + 广义 Lyapunov（QZ）+ Hamiltonian 二分：已实现于 `analysis/`，见 §6.10
- 5 个验收算例 + 四张图：已跑通（`bench/run_phase0.py`）；§8 的 20 个 benchmark 仍待补齐
- **验收**：diffusion(N=64/1024)、biharmonic(N=128)、Burgers、Robertson 的判定与解析对照 23/23 通过（±3 dB / ±0.5 decade）

### Phase 1：单 island（4 积分器 + 1 小 MVM + 比较器）
- **验收**：实测 σ² = α kT/C 与 σ²τ = α kT/g_m 各在 ±3 dB 内；实测 Q 容差规则（Q = 10/50/200 三档，验证 pole shift ∝ Q·Δ）

### Phase 2：spectral tile（8 模态扩散）
- **验收**：τ_k ∝ 1/k² 实现误差 < 5%；预测 κ 与实际一致；与数字参考解 L2 误差符合 σ_y 预测；实测 §6.7 的 E_smooth(N) 曲线（决定 N 该买多少）

### Phase 3：flux tile（1D Burgers 含激波）—— 按 §6.1 五元组验收
- 守恒：Σ V_i 与 Σ V_i·V_i 漂移 < 1e-4
- **W_d → ν_num 反演**：实测剖面拟合 tanh，与预测值一致（±20%）
- x_s(t) 轨迹误差（wave drag 的代理量）
- **接口守恒对照实验**：单通量源+镜像 vs 两边各算一遍，泄漏量应差 2 个数量级以上
- limiter 前向不变性（检查边界向量场；可能失败，失败要给出失效边界）

### Phase 4：分层路由 + 换基 + 事件
- **验收**：η_route 与 ‖G_loop·Δ‖∞ 的预测-实测误差；一次真实 basis switching（smooth↔shock）的切换瞬态与噪声代价；实测重构注入噪声是否落在 §6.2 的 45 nV/√Hz 量级

### Phase 5：explicit singular state（shock fitting）
- 1D nozzle 移动激波 + 跨声速翼型截面：x_s(t) 与 RH 解析解对比
- 拓扑事件：生成 / 合并 / 消失各触发一次，验证 reset map 与事件后重初始化（这里最容易崩）
- 同 QoI 对比 Phase 3 的 capturing：验证"M_int 小、T_int 长时，fitting 用更少状态达到同等 W_d"

---

## 8. Benchmark 集（每个针对一条尺度律或一条 QoI）

| 系统 | 针对 | 预期判定 |
|---|---|---|
| 1D diffusion（spectral） | κ = N² | N ≤ 1024 可行 |
| biharmonic / KS | κ = N⁴ | N = 128 不可行 → 必须降维 |
| Burgers（激波） | Gibbs + flux tile | 需 basis switching |
| 线性对流 | 离散守恒 | flux tile 必须 |
| Robertson | κ ~ 10⁶–10⁹（Jacobian 谱，随 T_obs）/ 10¹¹（速率常数比） | 必须 QSSA/多速率 |
| 非正规 shear | Γ ≫ 1 | 换基或数字回退 |
| Van der Pol（stiff 版） | κ + 极限环 | 边界 |
| HH 神经元 | 事件原语 | 必须 hybrid |
| 带核的粘弹性 | Lorentzian 展开 | 辅助状态数 vs 精度 |
| 分数阶扩散 | s^α 有理逼近 | 同上 |
| 跨声速翼型（Euler, NACA0012, M≈0.8） | x_s、ν_num、wave drag | fitting 优先，否则窄 capturing |
| 高超声速 bow shock（驻定） | post-shock T/p/ρ 的 L2 | sharp shock + 高阶后方 |
| 浅水 tidal bore | bore speed、质量/动量守恒 | flux tile；κ_num 要求松 |
| 温盐锋面 / 密度流 | κ_num、front thickness | 必须保正 + 超前换基 |
| Rayleigh–Taylor / 界面破碎 | M_int、T_int 判据 | capturing，fitting 会崩 |

---

## 9. 明确不做 + 已知风险

**不做**
- >256×256 的全 crossbar（寄生与容差不可控）
- 纯模拟（必须有数字 supervisor 做配置/校准/换基）
- "能表达任意 ODE"的宣称，只做 region 判定
- "两块 analog 各自跑不同时间再直连"

**风险（按杀伤力排序）**
1. κ_hw 估计过于乐观：实际受低频 1/f 与漏电双重挤压，τ_max 可能只有 100 µs ⇒ κ_hw 降到 10⁴，直接砍掉一半 PDE
2. 校准漂移：温度使 θ(t) 慢变，ΔA 时变 ⇒ H2/H∞ 需周期性重算；校准开销是真实成本
3. α 被低估：真实 OTA 的 γ_eff 与 excess noise 可达 5–10×，SNR 预算系统性偏乐观
4. **降级判定**：若 benchmark 中 >50% 需要数字回退，这台机器就是"模拟加速器"而非"模拟求解器"。可接受的降级，但必须在 Phase 0 就承认，而不是 Phase 4 才发现
5. **模拟无 rewind**：换基若滞后于 Gibbs 出现，误差不可逆（过冲穿过非线性 EOS/饱和元件后信息已丢）；τ_lead 预算必须实测，不能假设
6. tanh limiter 非 TVD：若目标应用要求严格 TVD（如某些 ocean front 诊断），此 tile 方案不成立，需换 limiter 拓扑
7. s ≳ 10% 时架构优势消失（§6.7）：这决定机器只对"奇异区局部化"的问题有意义

---

## 10. 仓库骨架（✅ = 已实现）

    grid/solver/
      analysis/
        ✅ descriptor.py  # E,A,B,C 容器 + 广义特征值 + 频响 + κ/带宽
        ✅ lyapunov.py    # 广义 Lyapunov (QZ) → P, tr(C P Cᵀ)，含残差自检
        ✅ hinf.py        # Hamiltonian 二分 + 扫频交叉验证
        ✅ resolvent.py   # Γ 与 Γ_proj（QZ 投影零模态）；pseudospectra 待补
        ✅ regularity.py  # 谱系数衰减拟合、E_tail、shock sensor
        ✅ slack.py       # 六轴 slack 向量（鲁棒拆失配/接口）+ 不可行原因
      lowering/
        split.py          # F = L + N，CSP/QSSA
        basis.py          # T 的候选集 + 条件数税
        scaling.py        # A_t 公式 + 频窗检查
        budget.py         # 按 ‖H_i‖₂ 分配 C_i
        interface.py      # §6.4 单通量源+镜像；§6.5 fitting/capturing selector
      twin/
        ✅ spice/extract.py  # 真实 PDK 提取：ngspice 电流镜 → λ, g_m/I_D, r_o, ratio
        noise.py          # 热 / 1f(Lorentzian 增广) / 抖动 / 电荷注入(含重构注入)
        route.py          # Y_c(s) → descriptor → η_route
        ✅ coupling.py     # spectral↔local 接口守恒：单通量源 + 电流镜像
        ✅ qoi.py          # W_d → ν_num；Osborn–Cox → κ_num；x_s(t) 对比
      bench/
        ✅ run_twin_experiments.py  # §6.8 五组对照（3 min）
        ✅ run_phase0.py            # Phase 0 五算例验收（~2 min）
        diffusion.py biharmonic.py burgers.py robertson.py shear.py hh.py
        airfoil.py bore.py front.py   # 20 个 benchmark 待补；当前模型在 run_phase0.py 内
      results/                    # report.txt / twin_experiments.json /
                                  # device_extract.json / phase0.json /
                                  # phase0_report.txt / png / 提取日志

关键 API（与实现一致）：

    sys  = Descriptor(E, A, B, C, D)
    ly   = solve_generalized_lyapunov(sys, Q)  # ly.P, ly.sigma_y, ly.rel_residual
    hi   = hinf_norm(sys)                      # hi["value"], hi["method"]
    res  = peak_with_projection(sys)           # res["Gamma"], res["Gamma_proj"]
    sl   = evaluate(sys, HardwareSpec(...),
                    gamma_proj=res["Gamma_proj"], hinf=hi["value"])
                                               # sl.values, sl.violated, sl.status

---

## 一句话总结

不是"模拟电路能不能表示方程"，而是：**给定 (E,A)、工艺窗口 κ_hw 和扰动端口 B_p，是否存在 θ = (T, A_t, σ, c, w) 使 ‖H‖₂ ≤ σ_req、‖GΔ‖∞ < 1、κ_t ≤ κ_hw。** 三个数，一个判定，一张 slack 向量；越界的哪一轴，决定 compiler 拉哪根杠杆。

而奇异区不是"降阶"，是**把 QoI 从点值阶数换成 (W_d/ν_num、守恒漂移、x_s(t))**：95–99% 的流场保持高阶，1–5% 的自由度交给一个狭窄、严格守恒、拓扑上不可能泄漏的 flux 层。WENO 式的降阶因此从"精度损失"变成**资源局部化机制**。
