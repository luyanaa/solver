# 连续时间模拟/混合计算 fabric —— 可落地方案

> 目标：把"这方程能不能跑"从口号变成一个**可计算的可行性判定**，并给出对应的硬件分层、编译器 pass 结构和分阶段验证路线。

---

## 0. 先把设计钉死的四个结论

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

注：以上对**加性**扰动严格；乘性噪声 G(x)dW 的端口依赖状态，resolvent 只在加性/线性乘性时成立，边界见 §1.3。

**(D) 噪声是可编程资源，但它和精度共用同一个 C。**
SDE 的扩散系数不是免费的。注入电流噪声（单边 PSD S_I）给电容充电：

    D = S_I / (2C²)        [V²/s，按 D = σ²/2 的定义]

而同一个 C 又给出 §0(B) 的 σ² = αkT/C。想独立调 D，必须用专用 noise injector + 增益/滤波，但增益误差直接乘在 D 上，必须标定。白噪声近似还要求噪声带宽覆盖状态带宽：一阶系统要 B_noise ≥ 100·B_req 才能把方差误差压到 1% 以下。**“模拟解 SDE”不是把噪声接到电容上就完了：噪声源、电容、带宽、标定是一组耦合约束。**

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
  **使用前提是 ΔA 保守**：ΔA_conservative 保持守恒不变子空间（通量列和结构）不变，
  零模态仍在机器精度内。一旦 ΔA 是泄漏型（两侧注入电流不等，列和≠0，如电流镜失配
  映射到单条通量边），零模态被激发成 +6e-3 量级的慢漂移/失稳，Γ_proj 完全看不见
  （`analysis/slack.py` docstring 有实测反例）。泄漏型必须单独算质量/电荷漂移率
  （coupling.py 的 dM/dt 账本）或有限时间增益，不得用 Γ_proj‖ΔA‖。
  验证方法：检查 `sum(dA, axis=0)`（列和）是否 ~0。
- **‖H‖∞‖Δu‖** 是接口扰动的小增益；它与失配轴是两个对象，不能用一个 Γ 代替。
- **E_tail_req = tol_L2²**：tol_L2=1% 的 L2 表示误差对应 1e-4 的谱尾能量占比。

任一 > 1 即不可行，且**告诉你该拉哪根杠杆**：κ 超 → 删模态；Γ_proj‖ΔA‖ 超 → 降 Q / 校准 / 换基；‖H‖∞‖Δu‖ 超 → 衰减接口扰动或降闭环增益；E_tail 超 → 换局部表示；σ_y 超 → 加 C（但只加在 ‖H_i‖₂ 大的节点上）。

注：SDE 的随机可实现性（噪声带宽、τ_c、D 范围、读出）不塞进这六轴，而是在 P6 作为独立的**随机证书**。它们不是可交易的余量，而是编译前置条件（§4.4）。

**噪声轴不是单节点 SNR。** 对 ẋ=f(x)+B_n n(t) 在轨迹附近线性化：

    δẋ = J(t) δx + B_n n(t),   J = ∂f/∂x
    Ṗ = J P + P Jᵀ + B_n Q B_nᵀ
    P_y = C P Cᵀ

要优化的是 observable 的误差 `tr(WP)`（W=CᵀC 时就是 `tr(CPCᵀ)`），不是“这个 OTA 60 dB”。LTI 时解代数 Lyapunov；时变时沿轨迹积分，或用冻结时间近似并附有效性检查。这把 SNR、stiffness、non-normal 放大连到同一个 P 上：J 的慢模/非正规模决定噪声被放大多少，带宽进入 Q 和 B_n。

### 1.3 SDE 的 descriptor 扩展：drift / diffusion 两套图

**能。** 加性 SDE 与 §1 的 ODE 是同一套 descriptor，只是多一个常数 B_p；乘性 SDE 多一个状态依赖的扩散图。真正需要新增的是噪声合同、随机语义和随机证书。

把 §1 的 descriptor 扩成

    E dx = F(x,u,θ) dt + Σ_k G_k(x,u,θ) dW_k ,   y = H(x,u,θ)

加性噪声是 G_k 与 x 无关的特例；线性乘性噪声是 G_k(x)=G_k x；一般乘性噪声是 G_k(x)。硬件侧仍然是

    C_i dV_i/dt = I_{f,i} + I_{noise,i}

所以 **SDE 不需要先离散成 Euler–Maruyama**：电容就是连续时间积分器。但“不离散”不等于“没有误差”：噪声的有限带宽、τ_c、相关矩阵和读出采样是新误差源。模拟的三条真实优势是：连续时间积分没有步长误差、N 条轨迹可以在同一阵列上并行、噪声是物理的而不是伪随机数。这条路线本身不新——可编程模拟计算机/FPAA 已经做过 ODE/PDE/SDE；本方案的区别是把噪声、刚度和表示放进同一套可计算判定（slack + 随机证书），而不是依赖手工配置。

**可重构性在哪里**：硬件不随方程变，变的是配置——T3 的权值矩阵（A、B、C）、T7 的开关、T8 的 PSD/τ_c、T6 的阈值、T1/T4 的电容阵列。compiler 把五张图映射到这组配置；所以“方程很多种”是 compiler 的问题，不是 tile 的问题。

compiler 的输入是五张图，而不是一个字符串公式：

    STATE       x0: C0, range; x1: C1, range; ...
    DRIFT       dx0 = add(mul(k0,x0), tanh(x1)); ...
    DIFFUSION   g0 = mul(sigma0, noise0); g1 = mul(sigma1, x1, noise1); ...
    NOISE       noise0: PSD, tau_c, rho_ij, mode=physical|replay
    OBSERVATION y0 = x0; ...

普通 ODE 只是 DIFFUSION 为空；Langevin 是 F=−∇V；Neural SDE 只是把 F/G 换成 MVM→NL→MVM。五张图最终都降到同一组 (E,A,B,C) 加 B_p 端口；对加性/线性化系统，§1.2 的 slack 仍然适用；乘性非线性部分由下面的分析边界和 P6 随机证书处理。

**噪声合同**（缺一项就不能编译）：

| 字段 | 含义 | 为什么必须写 |
|---|---|---|
| S_ξ(f) | 单边/双边 PSD 要注明 | 差 2×，直接错 3 dB |
| τ_c | 相关时间 | 白噪声近似与 Itô/Stratonovich 的判据 |
| ρ_ij | 噪声源相关矩阵 | 共享电源/地的串扰 |
| B_noise | PSD 平坦带宽 | 一阶系统要 ≥100·B_req |
| mode | physical / replay | 采样 vs CRN |
| 量化/滤波 | replay 的 DAC 位数、重构滤波器 | 决定 replay 的有效 τ_c |
| 标定 | 增益、PSD、ρ 的残差 | D 是乘法误差 |

**分析边界**（写进 `analysis/`，不要过度承诺）：

- **加性**（G 与 x 无关）：现有 Lyapunov 精确；有色噪声用增广状态，仍是 LTI。
- **线性乘性** G_k(x)=G_k x：二阶矩精确，

      dP/dt = A P + P Aᵀ + Σ_k G_k P G_kᵀ + Q

  向量化后是 n² 的 Sylvester 方程，n≤32 可解。**均方稳定 ≠ 样本稳定**：二阶矩认证的是方差，不是几乎必然稳定性。
- **非线性乘性**：二阶矩不闭合，只能 Monte Carlo / 统计线性化；H∞ 不能直接宣称。

**PDE 的噪声多一条约束**：Q-Wiener 的协方差必须 trace-class。1D 空间白噪声通常可以；2D/3D 的空间白噪声 Σσ_k²=∞，必须给相关长度/谱。离散化时噪声幅度随 cell 体积缩放，否则网格越细噪声越强。这条进 P0.5。

### 1.4 Itô / Stratonovich 是编译期合同，不是实现细节

对 dx = f dt + G dW，两种解释的 drift 差一项：

    f_S,i = f_I,i − ½ Σ_{j,k} G_{jk} ∂G_{ik}/∂x_j

所以 compiler 必须显式记录两个字段：

    interpretation ∈ {ito, stratonovich}      用户模型
    hardware_limit ∈ {stratonovich, ito}      电路实际极限

两者不同时，P0.5 加/减修正漂移。方向要写死：

- 模型 Itô、硬件 Stratonovich ⇒ 硬件漂移取 f_S = f_I − ½ Σ G ∂G；
- 模型 Stratonovich、硬件 Itô ⇒ 硬件漂移取 f_I = f_S + ½ Σ G ∂G。

加性噪声（∂G/∂x=0）两项相同；乘性噪声不同。线性乘性 G_k(x)=G_k x 时修正退化为

    A_S = A − ½ Σ_k G_k²

**不能默认“模拟 = Stratonovich”。** Wong–Zakai 说的是 τ_c→0 的光滑噪声逼近极限；真实电路 τ_c 有限、还有 delay，偏差是 O(τ_c)，甚至会出现 Stratonovich→Itô transition。合同里写 τ_c，验收里测偏差。

可证伪的标量测试：dx = μx dt + σx dW。

    Itô:          E[x(t)] = x₀ e^{μt}
    Stratonovich: E[x(t)] = x₀ e^{(μ+σ²/2)t}

两条曲线在 σ²t ~ 1 时明显分开，这是 §6.11 的 GBM 验收。第一版只承诺标量/对角、两两可交换的 G；非交换 G 的 Lévy area 与多维修正留到后面。

### 1.5 可划归性：高阶方程 → canonical form

**先消阶数，再谈函数空间。** n 维 r 阶方程 x^(r)=F(t,x,…,x^(r−1)) 令 z=(x,x′,…,x^(r−1))，即得

    ż = [z₂, z₃, …, F(t,z)]

阶数只乘状态数（n·r 个积分器），不增加 primitive 类型。隐式 G(t,x,…,x^(r))=0：det ∂G/∂x^(r) ≠ 0 时由隐函数定理局部解成显式；否则是 DAE，进 descriptor 的奇异 E。**index > 1 的 DAE 必须先做 index reduction**，否则 §6.10 的 Lyapunov/H∞ 需要额外的 index-1/正则性条件，不能直接套用。

**“可划归”不是“能被多项式逼近”，而是三个等级：**

| 等级 | 定义 | 误差度量 | compiler 行为 |
|---|---|---|---|
| Level 0 精确有限 | 存在有限状态增广，右侧由 B={+,×,INT,φ_i} 构造，且在不变流形上恒等 | 0（符号/高精度验证） | 接受，生成 exact 配置 |
| Level 1 ε-可划归 | ∃E~∈Closure(B)，Cost(E~)≤N_max，在 (K,T,Ω,O) 上 d_O<ε | QoI 误差，不是逐点状态 | 接受，附误差证书 |
| Level 2 统计可划归 | 逐点轨迹不可比（混沌/长期），但不变测度/谱/相关函数在 ε 内 | 统计距离 | 只承诺统计量，不承诺轨迹 |

Level 1 的合同写成

    R = (B, K, T, Ω, O, ε, N_max, d_O)

E 可划归 ⇔ ∃E~∈Closure(B)：Cost(E~)≤N_max 且 d_O(E,E~;K,T,Ω)<ε。没有 O 和 d_O 的“可划归”没有工程意义；硬件只在 K×[0,T]×Ω 上正确。

**B 的 differential-algebraic 字典：**

| primitive | 增广 | 关系 | 代价/注意 |
|---|---|---|---|
| sin/cos | 2 状态 | ṡ=c·v, ċ=−s·v, s²+c²=1 | 二次；必须维护不变流形 |
| exp | 1 状态 | ẏ=y·ẋ | 线性×；y>0 要 limiter |
| tanh | 1 状态 | ẏ=(1−y²)ẋ | 二次；|y|<1 要 limiter |
| Bessel | 2 状态（+ x 的导数） | 线性 ODE，系数含 1/x；在 t 域还需 ẋ, ẍ | 有理，不是多项式；x 需远离 0 |
| Gamma | — | Hölder：无有限阶代数微分方程 | 不可有限增广；只能原生块/有界近似 |

所以“sin/cos/exp/tanh 可以消掉”是对的，但**只在不变流形上**：初值不满足 s²+c²=1 或硬件漂移出流形，得到的是另一个系统。compiler 必须选择：原生 NL 块，还是增广状态 + 投影/反馈。判据是

    Cost(增广) + Cost(不变流形维护) < Cost(原生块)，且条件数可接受

否则原生块更便宜。例：单摆 θ̈+γθ̇+sinθ=0 用增广是 4 状态 [θ,ω,s,c]、ω̇=−γω−s、s²+c²=1；用原生 sin 块只有 2 状态。两者都“对”，但代价不同。

**canonical form（结构化，不是完整多项式展开）：**

    F(x) = A x + B(x⊗x) + Σ_r U_r φ_r(V_r x) + R(x)
           linear   low-rank quadratic   few special funcs   sparse residual

标量二次 xᵀBx 且 B=Σ λ_r u_r u_rᵀ 时，代价从 O(n²) 降到 O(Rn)；向量值二次是 3-tensor，低秩分解要逐分量做，不能假装成一个矩阵。

**误差不是 ‖F−F~‖∞。** 对 Lipschitz L 的向量场，Grönwall 给

    ‖x(t)−x~(t)‖ ≲ e^{Lt}‖x₀−x~₀‖ + (ε_f/L)(e^{Lt}−1)

所以 ε_f≪1 本身没有意义，真正的是 (ε_f,L,T)。LT≫1 或混沌时，改测 attractor / stationary distribution / spectrum / correlation / Lyapunov spectrum（Level 2）。这就是 §6.1 的 QoI 指标在函数层的版本。

---

## 2. 七条尺度律（它们决定架构，不是审美）

### 2.1 精度–速度–功耗

单节点，B = 10 MHz，V_DD = 1 V，g_m/I_D = 15：

| σ_rms | C | g_m | 单节点功耗 | 1000 节点 |
|---|---|---|---|---|
| 1 mV | 2.8 fF | 0.17 µS | 12 nW | 12 µW |
| 100 µV | 0.28 pF | 17 µS | 1.2 µW | 1.2 mW |
| 10 µV | 28 pF | 1.7 mS | 115 µW | 115 mW |

**结论：功耗与面积都不是瓶颈**（1000 节点 @ 10 µV @ 10 MHz ⇒ ~0.1 W、~3 mm² 电容）。真正稀缺的是 κ_hw、路由寄生和校准漂移。资源该投在**拓扑与校准**，而不是"更省电的 OTA"。

**工艺包络不是一个 f_T 数。** 每种工艺给

    P_tech = (V_swing, B_accurate, S_n(f), DR, g_m/I_D, f_T, f_max, R_sw, C_sw)

三层频率要分开：f_signal ≪ f_block ≪ f_T, f_max。能保证模拟精度的是 f_signal，不是“晶体管还能工作”的频率。compiler 不问“f_T=10 GHz 能不能做 1 GHz solver”，而问“这个映射在所需误差下需要多少 B/SNR/P”。

精度–带宽–能量没有免费项：σ²τ = αkT/g_m，σ² ∝ B，σ² = kT/C。想提高 SNR 就加 C；要保持 ω_c=g_m/C 就加 g_m；加 g_m 就加功耗。

### 2.2 加速的代价

B_hw ≈ A_t·B_model，且白噪声下 σ² ∝ B ⇒ **加速 A_t 倍，噪声功率涨 A_t 倍**。又由 §2.1，σ²τ 固定 ⇒ 加速不改变 σ²τ，只改变 τ 落在窗口哪一端。真正的约束是 §0(A) 的窗口宽度。

不要套通信容量 `BT log(1+SNR)`：模拟 ODE 的约束是状态协方差/observable 误差，不是比特率。

### 2.3 κ 的相似不变性（对直觉的修正）

κ_t 在 (T, A_t) 下不变，所以：

- **"用专用 stiff 线性块实现"不降低 κ**。它降低的是"实现成本"（每模态一条便宜的一阶支路，而不是让通用积分器跨 8 个数量级），但芯片仍要能产生 τ ∈ [10 ns, 1 ms]。
- 能改 κ 的只有两种：**删模态**（快模态被奴役且稳定 ⇒ g(x,y)=0 ⇒ 从谱里消失）、**多速率**（付同步误差，§4.3）。

工程上还要把 κ_hw 换成"可用"带宽比：f_max,usable = min(GBW_OTA, 寄生极点)，f_min,usable = max(漏电流, 1/f 转角, 漂移)。数学 κ_t 与 κ_t,usable 的差就是 §9 风险 1；compiler 用后者做判定，不用前者安慰自己。

另外，κ 是相似不变的，但 **Γ/瞬态不是**；§4.0 的 V=Sx 预条件可以在不改谱的前提下降低 Γ。

### 2.4 λ_k = −νk²：spectral 对模拟比对数字更重要

    κ_diffusion = N² ,   κ_biharmonic = N⁴

N=64：4×10³（轻松）；N=1024：10⁶（贴边）；biharmonic N=128：2.7×10⁸（**必然超窗**）。

第一条硬规则：**扩散类优先 spectral；双调和/高阶算子在中等分辨率下就必须降维或混合**。反过来，spectral 基里每个模态是独立的一阶支路 g_m/C = νk²，这是模拟硬件最擅长的事——前提是 k² 的跨度装得进窗口。

### 2.5 路由容差 ∝ 1/Q

二阶节极点灵敏度 ∂λ/∂p ~ Q。开关/Ron/C_par 的相对误差 2% ⇒

    δω₀/ω₀ ~ Q · 2%     ⇒     误差 ~ 1/Q 级

**规则：Q ≲ 50 才允许"免校准"；更高 Q 必须 trimming/校准，或分解为一阶节。** 这就是"2% 的 switch network 在高 Q 环里造成大 pole shift"的定量版，也是 §1.2 中 Γ_proj‖ΔA‖ < 1 的工程化形式。

### 2.6 随机精度–速度–功耗

把 §0(B) 的精度律和 SDE 的扩散系数放在一起：

    σ² = αkT/C ,   D = S_I/(2C²)   （S_I 为单边电流 PSD，D = σ²/2）

同一个 C 同时决定“状态精度”和“扩散强度”。专用 noise injector 可以解耦，但：

- 增益误差 ε_g 直接给 D 乘 (1+ε_g)²；**D 的标定精度就是分布精度**。
- 本征热噪声给出底噪 D_int；有效 D = D_inj + D_int，注入必须显著高于底噪。
- 白噪声近似要求 B_noise ≥ 100·B_req：一阶系统的方差误差是 (2/π)arctan(f_c/B)，f_c=10B 时已经低 6.4%。
- replay DAC 的更新率 f_s 决定它只能做 f_s/2 以内的噪声；10 MHz 系统要 1 GHz DAC，功耗/面积不便宜。**物理热噪声做采样，replay 做 CRN/梯度**，不是冗余。

数量级（300 K，R=1 kΩ）：

| C | σ_th = √(kT/C) | D = S_I/(2C²) |
|---|---|---|
| 1 pF | 64 µV | 8.3 V²/s |
| 1 fF | 2.0 mV | 8.3e6 V²/s |

OU 的稳态方差是 D/λ，必须同时满足“落在状态摆幅内”和“高于 kT/C 底噪”。这两个不等式给出 D 的可用区间；超了就换 C、换 injector 或降带宽。

### 2.7 表示资源律：monomial 数量与 low-rank 二次

n 个状态、总 degree p 的稠密多项式有

    N_mono = C(n+p, p)

n=100、p=4 就是 4.6e6 项，已经超过任何 crossbar。Stone–Weierstrass 只说“能逼近”，不说“能高效实现”。所以 canonical form 必须是结构化的：

- 线性项 A x：T3 MVM，O(n²) 权值（或稀疏/低秩）。
- 标量二次 xᵀBx，B 低秩 R：Σ λ_r (u_rᵀx)²，O(Rn) 乘法器。
- 特殊函数 φ_r(V_r x)：T2 NL，每个一块。
- 稀疏残差 R(x)：只给真正需要的项。

**状态增广不是免费的**：每个增广状态 = 1 个积分器 + 1 份噪声/失调/漂移/校准。用 §2.1 的 σ²τ 和 §6.7 的资源律给增广定价；能原生 NL 就不增广，能低秩就不稠密。

**Carleman 截断是一种 closure，不是无损表示。** 对多项式系统，monomial 提升在无限维时精确；截断后可能改变稳定性，且 monomial 数按 N_mono 爆炸。它和 Fourier 投影、QSSA 一样，必须附 closure 误差，不能当作“精确线性化”。

### 2.8 Analog timing closure：四条，不是一条

数字 STA 只问 t_comb < T_clk；这台机器至少要问四个：

| 条件 | 数学 | 违反后的物理 |
|---|---|---|
| 器件带宽 | f_dyn/f_device ≪ 1 | OTA 的 GBW/寄生极点进入动力学 |
| 模拟 skew | ω_max Δτ_route ≪ 1 | wire 从 interconnect 变成 DDE；一阶等价于 E=I+A∘T（§6.14、§6.15 的 per-edge closure） |
| 约束时序 | τ_constraint ≪ τ_state,min | Poisson/KCL 松弛跟不上，elliptic 变 parabolic |
| 噪声 | P_hw ≺ P_error,budget | 相关噪声不能靠 ensemble 平均消掉 |

第一条给出 f_signal ≪ f_block ≪ f_T,f_max（§2.1）。第二条把 A_t 与芯片几何绑死：A_t 越大，ω_max 越大，允许的 τ 越小；1° 相位预算在 100 MHz 只有 27.8 ps（§6.14）。第三条是 T10 的 timing：ε_c=τ_constraint/τ_state,min，稳态约束被硬件松弛成 C_y ẏ=−Ay+b，必须保证 ε_c 足够小。第四条不是“加 C”：correlated noise 的协方差是低秩的，不能按 1/√N 平均；global reference 上的噪声会变成全芯片相关噪声。

---

## 3. 分层 fabric：10 tile + 1 supervisor

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
| T8 | **Noise** | 物理热噪声 / 可重放 PRNG+DAC；PSD、τ_c、ρ 可编程 | 1/f、带宽、量化、相关 | B_noise ≥ 100·B_req |
| T9 | **Statistics** | 片上均值/协方差/直方图/似然累加 | 乘法器失配、积分泄漏 | 只读 O(p²) 统计量 |
| T10 | **Constraint** | 稀疏代数约束 A(x)y=b(x)：Poisson/KCL/压力投影的物理松弛网 | 松弛速度、条件数、残差 | 共享的场/约束资源 |
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

写成一项：

    C_route = w_R R + w_C C + w_φ Δφ + w_X X_talk + w_N N_noise

权重由当前 dynamical graph 的敏感度 ∂σ_y/∂p 和 Γ 给出；wire length 只是其中一项。

R_on 与寄生 C 的冲突（做宽降 R_on、升 C）就体现在 G_c 与 C_c 同时进 Y_c——优化的是 ‖H_c‖∞，不是 R_on。

**configuration switch 只做准静态配置；被模拟的开关必须是 computational primitive。** 不能在每次仿真开关动作时去开关 routing transmission gate——那会把 R_on、C_off、charge injection、clock feedthrough 混进模型。理想开关用有限斜率 G(V_c)；真实 MOS 用 compact-model 的有限带宽；拓扑事件用 T6。

**信号分四级，fast state 只允许前两级。** local / neighbor（同一或相邻 island）走直接电流；regional 走 buffered differential/current-mode trunk；global 只放 slow control、parameter、reference、constraint result。configuration 走 digital distribution + local DAC，不做 global analog parameter voltage——否则一条 global reference 的噪声会变成全芯片相关噪声，ensemble averaging 消不掉。

**buffer 不是 memory，是阻抗隔离 + 能量补充。** 连续时间机器里所有状态一直同时存在，不需要“存住再发”。buffer 把 state capacitor 看到的总负载从 C_s+ΣC_in+C_wire 降到 C_s+C_in,buf；后面的大负载由 buffer 驱动。代价是每个 buffer 加 G(s)、offset、THD、噪声；fanout tree 的深度受 N_b σ_b² 限制，不能像数字 clock tree 那样无限加。

**长线优先在 source 侧做 contribution reduction，不要 broadcast state。** 传统 crossbar 把 x_j 广播给所有 sink；更好的做法是在 source 附近先算 I_ij=g_ij V_j，按 region 求和后再走线，到 destination 直接进 summing node。communication 的对象从 global state 变成 locally reduced contribution，fanout tree 自然变浅。

**每条 edge 的 closure 不是“长度 < L”，而是在 0<ω<ω_dyn,max 内**

    H_ij(s) = H_wire(s) · ∏_k H_buf,k(s) ≈ g_ij e^{−sτ_ij}

同时满足 |g_ij−g_ij^ideal|<ε_g、|φ_ij−φ_ij^ideal|<ε_φ、S_n,route<S_n,budget。这是 §2.8 第二条的 per-edge 版本；`C_route` 的 φ 项就是从这个条件来的。

`twin/route.py` 把这条写成可算的：wire 默认用分布式 1/cosh√(sRC)（不是一个极点），buffer 用 g/(1+s/ω_p)，pad 用 all-pass (1−sτ/2)/(1+sτ/2)；`route_closure` 一次返回 gain/phase/noise/feedback-absolute 四个 pass、η_route、J_H、J_τ 与 latency class。§6.15 是它的 38 项验收。

每条 edge 还带一个 `transposeable` 标志：forward 用 `I_j += g·V_i`，adjoint 用 `I_i += g·V_j`，`A↔A^T` 只是重新配置 switch matrix（§5.8）。比较器、事件、reset 这类不可转置元件必须显式标出，adjoint 在段边界处切换策略，不能假装整条路径都可逆。

### 3.3 T8/T9：噪声合同与反问题读出

T8 不是“一个噪声源”，是一个 bank：

- **物理模**：独立电阻/OTA 热噪声，PSD 由温度、R、增益决定；带宽由寄生/放大器决定，可到 GHz。
- **replay 模**：PRNG → DAC → 重构滤波器；seed、位数、更新率、滤波器全部进配置。
- **相关矩阵**：用 T3 的混合矩阵从 m 个独立源合成 ρ_ij；混合矩阵失配进 Δ，不假设理想。
- **标定**：PSD、τ_c、ρ 三项都要在片上调；标定残差直接进 P6 的随机证书。

T9 是反问题的关键：没有它，N 条轨迹的原始波形会把 ADC/链路打爆。

- 累加：时间平均、协方差、互相关、直方图（比较器 bank）、Gaussian log-likelihood。
- 读出：每个 ensemble 只回传 O(p²) 个数，数字端做 EnKF/SMC/优化。
- 约束：T9 的乘法器失配与积分泄漏进 Δ；长时间积分要复位/斩波，否则 1/f 与漏电会污染统计量。

噪声路由也有合同：T7 的寄生电容/电阻会改变噪声 PSD，共享电源/地会产生 ρ_ij。跨 island 的噪声走差分电流对，不走单端电压。

整机层次（不是新架构，是把 §3 的 tile 按 SDE 语义重新标注）：

    digital host
      ├─ equation compiler (P0–P7)
      ├─ placer / router
      └─ optimizer / calibration / seed
              │
    ─────────────────────────────────────
    Analog dynamical fabric
      State T1/T4 → Drift T2/T3 → Σ → ∫ → State
                        ↑                    │
      Diffusion T2/T3 ← Noise T8            │
      Constraint T10 (Poisson/KCL) ↔ State/Flux
      Event T6 (comparator/reset) ──────────┘
      Observation → T9 → ADC
      Route T7（配置、寄生、相关）

### 3.4 约束 fabric：Poisson / KCL / 压力投影

三类 workload 共享的不是“ODE 积分器”，而是一个稀疏代数约束：

    A(x) y = b(x)

- **MD 的 Coulomb**：∇·(σ∇φ)=ρ；电阻/gm mesh 的节点电压就是离散 Poisson 解。
- **NS 的压力投影**：∇²p = −ρ∇·[(u·∇)u]；同一块 mesh，源项来自 advection。
- **SPICE 的 KCL**：Σ I=0；没有独立动态状态的节点就是 algebraic variables。

硬件不是数字求解器，而是快速松弛：

    C_y ẏ = −A y + b,    τ_relax ~ C_y/λ_min(A)

这个式子只对 SPD/对称部分正定的 A 给出单调松弛；非对称 KCL 要用 λ_min((A+Aᵀ)/2) 或谱横坐标，且必须检查反馈稳定性——电阻 mesh 的 M-矩阵性质不能直接搬到有源 gm 网络。稳态就是约束解；残差、条件数、松弛速度进 P6。动态状态在 T1/T4，代数节点在 T10；两者通过电流/电压接口耦合。T10 是可共享资源：MD 的 field、NS 的 pressure、SPICE 的 KCL 用同一块可编程稀疏 mesh，只是源项和边界条件不同。

**约束 fabric 的 timing 条件。** 真实模型要求 A y=b 是 instantaneous；硬件给的是

    τ_c ẏ + A y = b

必须保证

    ε_c = τ_constraint / τ_state,min ≪ 1

否则 MD 出现假 retardation，NS 的 pressure projection 传播不动，SPICE 的 KCL 没收敛。τ_constraint 由 λ_min((A+Aᵀ)/2) 和 C_y 决定；这是 §2.8 的第三条，也是 T10 与 T1 之间唯一的硬时序契约。

**ε_c 的分母是外部时标，不是约束自己的极点（P0 finding 修复，`results/constraint_timing.json`）。** τ_state,min 是**外部受约束动力学**的最快时标（NS 的 τ_fluid,min、MD 的粒子时标），必须由 workload 显式声明（`eps_c(mesh, c_node, external_state_timescale)` / `solve_constraint(..., external_state_timescale=)`）。从约束子系统自身的极点推 `τ_state_min = 1/max|λ|` 是无意义的——约束网络对自身最快模态的比值不回答"约束是否在状态需要它之前收敛"（poisson-2d 上该比值实测 1.17e8，是一个假 FAIL）。纯 Poisson 稳态没有外部动态 ⇒ **无 ε_c 可算**：判决改为 (solve_deadline 达到? residual < tol?) 的事实（verdict PASS/FAIL on convergence，或无 deadline 时 verdict="STEADY" 只报 τ_relax+残差），不伪造比值、也不标 infeasible。ε_c FAIL 必须进 P6 证书（`Certificate.constraint` 字段，`evaluate_sys(..., constraint=)`）：FAIL ⇒ status="rejected"，纯稳态事实 ⇒ 证书保持"no violation on evaluated axes"并把 deadline/residual 事实记进 constraint 项。

| workload | 最大困难 | 主要 fabric |
|---|---|---|
| MD | moving particles + long-range force | 局部 particle T1/T2 + T10 Poisson |
| NS | multiscale cascade + incompressibility | 局部守恒 T5 + 多分辨率 + T10 |
| SPICE | DAE + extreme stiffness + events | T1/T4 + T10 KCL + T6 event |

**守恒律做进拓扑，不靠匹配**：MD 的 pair force 成对产生；CFD 的 flux 从 cell A 流出等量流入 B；KCL 的节点电流自动求和。这样 conservation 不受 transistor mismatch 一阶破坏。

**顶层架构的修正**：核心不是“analog integrator array”，而是 **continuous state fabric + sparse constraint fabric**；积分器只是其中一半。

### 3.5 器件与电压等级：不要让一个 MOS 同时解决速度、噪声、摆幅

| class | 典型器件 | 用途 | 代价 |
|---|---|---|---|
| A0 ±0.2–0.4 V | LV RF MOS / HBT | 高速、低噪声 g_m、MVM、buffer | 摆幅小，SNR 靠 C |
| A1 ±1–2 V | 3.3/5 V MOS + MIM | state、积分器、regional driver | 速度中等 |
| A2 5–20 V | HV MOS/LDMOS | 大摆幅、global driver、特殊 constraint/接口 | C_gd/C_oss/R_on 大，f_T 低 |

三条规则：

- **HV 器件不放在 RF node 上**：500 fF 的 C_off 在 2 GHz 是 159 Ω，已经不是“关掉的开关”。
- **RF/fast domain** 只见 LV/MV MOS、HBT、Schottky、MIM、low-C switch；**HV domain** 只见 LDMOS、thick-oxide、大二极管；中间用 buffer/transconductor/level converter。
- **HV 的优势是摆幅**：V_FS 从 0.5 V 到 3 V，kT/C 的 SNR 提高 15.6 dB；同 SNR 下 C 降 36 倍。劣势是面积和寄生；所以 HV 只做“大尺度”的 global/constraint/actuator，不做高密度 state。

**工艺选择**：单一 RF CMOS 或单一 BCD 都不是最优；最有趣的是 RF-capable BCD/HVCMOS/SOI mixed-signal（XT011/XH035 这一类），因为它允许把速度、噪声、摆幅、隔离、global driver 分给不同器件。SOI/DTI 的价值不只是耐压，而是把 noisy high-swing domain 与 low-noise analog dynamics 隔开。HBT 的 g_m/I_C=1/V_T≈38.7 V⁻¹ 适合做 line driver/buffer，不适合存 state；MOS/MIM 存 state；CMOS digital 做配置/校准。

### 3.6 校准：校准方程，不是晶体管

目标不是 g_m=1.000000 mS，而是

    当前硬件实现的无量纲动力学 ≈ 目标无量纲动力学

误差拆成四项：**common-mode scale**（全局 g_m/C 漂移，等价于改 A_t）、**low-rank systematic**（空间梯度/温度）、**local mismatch**（Pelgrom）、**random noise**（不可校，只能表征）。前两项只需更新 scaling；第三项局部校准；第四项进 P6 的随机证书。

**primitive behavioral model**：每个 INT/MUL/route/NL 保存实测 H(s)、offset、THD、噪声 PSD，而不是 nominal transistor 参数。compiler 针对“当前这颗芯片的 calibrated digital twin”编译，而不是 TT_25C.lib。

**三个时间尺度**：foreground（上电扫 gain/offset/pole/NL/noise）、background（pilot + lock-in，ω_p 放在 workload 能量低的位置）、run-time residual（replica + 物理残差）。不要在运行中的计算节点上塞 pilot；用 replica tile（每 32–64 tile 一个）或 shadow block。

**物理残差是免费 reference**：MD 的 ΔH、NS 的 ‖∇·u‖、SPICE 的 ΣI、KVL 的 ΣV 都是 R(x)=0。在线最小化 ‖R(x)‖²+λL_pilot 就是 self-supervised calibration。ratio accuracy 比 absolute accuracy 重要：unit-element arrays + ratiometric 设计。

**噪声不能校掉**：估计 S_n(f)、Q、ρ_ij；SDE 的注入是 Q_inj=Q_target−Q_hw≥0。若 Q_hw>Q_target，只能加 C、降带宽、加 bias、改器件或换 placement。

**placement 也可以适应硬件**：温度梯度下把高 SNR state 迁到噪声低的 tile，把不敏感中间量放热 tile；这比强行把硬件调回去更便宜。

### 3.7 Heterogeneous substrates：state 是 generalized physical coordinate

T1–T10 默认 state 是电压。这是实现选择，不是数学要求。把 §1 的 descriptor 放宽到

    E_q·q̇ = F_q(q) + Σ_r C_qr·z_r ,    q ∈ { V, I, x_mech, P_fluid, c_chem, a_optical }

`q` 是广义坐标，`E_q` 是该域的容量（C、m、C_f、…），`F_q` 是该域的 constitutive law（I=g(V)、−kx−cẋ、−L_R P、S·v(c)、…），`C_qr` 是跨域 transducer。三件事随之改变：

1. **κ_hw 变成 per-domain**。电子窗约 1e-8–1e-3 s，MEMS 约 1e-6–1e-1 s，微流控约 1e-3–1e2 s，光子约 1e-12–1e-9 s（但没有 storage）。一个系统可以同时用多个窗，前提是跨域边界被显式 sample 或锁相；这不是"两块 analog 不同速直连"（§9 不做项），而是有合同的 multirate 边界。
2. **transducer 是一等公民**。每次跨域都要付 gain、nonlinearity、noise、bandwidth、energy，并且它自己就是一个 §1.2 的扰动端口。
3. **compiler 的算子集从 `{+,×,∫}` 变成 `{E_CMOS, E_MEMS, E_fluid, E_optical}`**。判据仍是 (error, energy, bandwidth, area, communication) 上的 slack 向量，只是每轴按 substrate 重算。

状态（A7，`results/substrate.json`，40/40）：MEMS 的 `ω0=√(k/m)`、`Q=mω0/c` 精确公式 + 小幅阻尼解析响应 RK4 命中（4.9e-11）；Duffing 一阶微扰骨架线 `ω(A)≈ω0(1+3k3A²/8k)` 与微幅 RK4 测频差 <1%（A² 缩放比 3.99≈4，k3 符号双向可辨）；gm/C 用 ω_ref 锁定后残差 ≈ 测量噪声底（10 ppm）而非温漂（1000 ppm），残差对 drift 100× 变化不动——参考自身漂移（§9 风险33）须单独入账；photonic 4×4 MVM：零噪机器精度、shot 项严格 ∝1/√N（能量份额探测使其与 M,x 无关）、漂移残差地板与光子数无关（Monte-Carlo 与解析 quad 相加一致 ±20%）。流体扩散算例（1D fluidic diffusion 孪生）与真实 transducer 测量数据仍是 Phase 8 TODO。

| substrate | 最自然的 primitive | 数学形式 | 不擅长 |
|---|---|---|---|
| CMOS 积分器 | 一阶 ODE / 积分 | C·V̇ = I | 高 Q、长延迟、二阶机械模态 |
| CMOS 非线性 | 乘法 / 饱和 / 比较 | I=g(V)、tanh | 精确线性、大动态范围 |
| MEMS resonator | 二阶 ODE / 共振 | mẍ+cẋ+kx=F | 速度、可编程性、pull-in |
| 电阻 / gm mesh | graph Laplacian / 约束 | GV=I | 动态、非对称耦合 |
| 微流控 | 输运 / 扩散 / 守恒 | C_f·Ṗ = −L_R P + Q | 速度（ms–min） |
| 化学反应网络 | 多项式非线性 ODE | ċ = S·v(c) | 可控性、可观测性、寿命 |
| 光子 | 线性变换 / 卷积 / 传输 | y=Wx、e^{−jβL} | 非线性、存储、local 短连 |

同一个 PDE 可以按算子拆成四路：

    ∂_t u = L[u] + N[u] + C[u] + η
    L → RC/MEMS/photonic 物理算子；N → CMOS 非线性；
    C → elliptic/constraint fabric；η → 标定过的物理噪声

这不是把数字算法逐条翻译成电路，而是把每个算子放到本身就服从同类定律的介质上。优势来自"介质天然做的事"，不是"低精度换低功耗"。

#### MEMS：二阶 ODE 是免费的

    mẍ + cẋ + kx = F(t),    ω0 = √(k/m),    Q = mω0/c

单模态就是一个二阶 dynamical primitive。不需要 OTA+C 去"模拟"二阶系统——机械结构自己就在解这个方程。耦合后

    MẌ + CẊ + KX = Bu

`M,K,C` 编码矩阵，generalized eigenproblem `Kφ=ω²Mφ` 直接是模态基准。Duffing `+αx³`、参数激励、mode coupling、bistability 都是免费的物理非线性。

三条工程约束：

- **高 Q 是双刃**。Q=10³–10⁵ 给窄带、低相位噪声、长衰减，也意味着 τ_settle ~ 2Q/ω0 很长，且 Q 随温度/应力/封装漂移。高-Q pole 用 gm-C 实现会被噪声和 dynamic range 限制，MEMS 反而天然适合——前提是它能被标定。
- **接口是非线性的**。静电驱动 `F ∝ V²`，电容读出 `ΔC/C`。`V↔x` 不是理想 g_m：要么差分线性化，要么进 behavioral model 标定。
- **可编程性差**。M/K 固定，compiler 只能选 bank 里的哪个 resonator、调耦合/增益/反馈。MEMS 是 fixed-function accelerator，不是通用 tile。

**时间基准**：`ω0=√(k/m)` 可以作为片上 dynamical reference。它比测温+LUT 更直接，因为测的是同类的物理时间常数。用法是把绝对精度转成 ratio accuracy：

    g_m/C = [ (g_m/C) / ω_ref ] · ω_ref

括号里是两次频率计数之比，`ω_ref` 的温漂被 characterize 后进 §3.6 的 common-mode scale。它不是绝对时钟，也不消除温漂；它只是把"测温度再查表"换成"测另一个时间常数"。

#### 微流控：同一类 graph Laplacian

chamber + channel：

    C_f·dP_i/dt = Σ_j Q_ji − Σ_j Q_ij ,    Q_ij = (P_i − P_j)/R_ij
    ⇒  C_f·Ṗ = −L_R·P + Q

与 `C·V̇ = −G·V + I` 同构：`P↔V, Q↔I, R_f↔R`。这不是类比，是同一类 graph Laplacian dynamics。扩散、输运、reaction-diffusion 天然可做。

加入化学（mass action `v = k∏c^ν`）：

    ċ = S·v(c)

`S` 是化学计量矩阵，`v` 是非线性——直接给出乘法，也就是 GPAC primitive（§1.5）的湿件版本。

但时间尺度是 ms–min，比电子慢 6–9 个数量级。所以定位不是快 solver，而是

    fast electronic solver  ↔  slow physical/chemical system

电子部分实时估计 `θ̂`，控制 valve/heater/electrode/pump，读 concentration/fluorescence/impedance/electrochemical signal。真实系统 `ċ=f(c,θ)`，片上模型 `ċ̂=f(ĉ,θ̂)`，误差 `e=c−ĉ` 在线更新 `θ̂`。整个芯片变成 experiment + model + parameter inference 的闭环；这比"analog 比 GPU 快多少"更独特。

代价：蒸发、气泡、堵塞、表面吸附、温度/pH 漂移、生物污染。它们是慢漂移，正好进 §3.6 的 background calibration 和 §1.2 的 slack；但不能假设"流体网络 = 理想 RC"。

#### 光子：只买 global dense 与 linear transform

    V_i → modulator → λ_i → waveguide → photodiode → I_j

`I_PD ∝ P_opt`，直接注入 `C_j·V̇_j`。这与 §3.2 的 "voltage state → current communication" 一致，而且 PD 输出天然是电流。

WDM 让一条 waveguide 承载 N 个 λ：

    物理线数 < 逻辑通道数

ring/MZI 调 weight 给

    I_j = Σ_i w_ji·P_i    ⇒    y = W x

MVM 恰好是 `ẋ = Ax + N(x)` 里最贵的部分。

但它不是免费午餐：laser power、shot noise（σ²∝P）、PD 噪声、modulator linearity、ring thermal drift（~0.1 nm/K 量级）、波长校准、optical loss、EO conversion overhead。对 100 µm 的 local link 光基本没有意义。规则：

    local → electrical
    regional → electrical / current-mode
    global dense → optical / WDM

与 §2.8 的 latency hierarchy 一致。光子买的是"不随 L² 增长的带宽"，不是"更快的晶体管"。

#### 优先级与 benchmark 梯度

研究价值排序：

1. **CMOS FPAA + MEMS**：接口最自然（g_m 驱动、C 读出），MEMS 本身就是 dynamical element；先做 resonator bank + CMOS programmable coupling/readout。
2. **CMOS FPAA + photonics**：解决 large-scale connectivity / MVM / transform；如果 state 数真的上规模，这项最终可能比 MEMS 更重要。
3. **CMOS + microfluidics**：不做通用高速 solver，做 real physical system + analog model + inverse solver 的闭环平台。

Benchmark 梯度（不要一开始上 NS DNS）：

| 代 | benchmark | 验证什么 | 主 substrate |
|---|---|---|---|
| 1 | Poisson/Laplace、diffusion、reaction-diffusion | physical operator 本身在求解 | 电阻/gm mesh + RC + CMOS NL |
| 2 | stiff ODE、DAE/SPICE、SDE/Langevin | multirate、constraint、noise-as-computation | CMOS + T10 + T8 |
| 3 | MD、Navier–Stokes、wave/Maxwell | heterogeneous 组合 | CMOS + MEMS/photonic mesh |

哪些问题值得换 substrate：

| 问题 | 数字难点 | 物理 solver 潜力 | 最适 substrate |
|---|---|---|---|
| Poisson/Laplace | 大 sparse linear solve | ★★★★★ | 电阻/gm mesh |
| diffusion/heat | 大网格、长时间积分 | ★★★★★ | RC / MEMS / fluidic mesh |
| reaction-diffusion | 非线性 + 空间耦合 | ★★★★★ | CMOS + fluidic/chemical |
| stiff ODE | 极端时间尺度 | ★★★★ | CMOS + multirate |
| DAE/SPICE | 每步 Newton + 稀疏解 | ★★★★★ | CMOS constraint network |
| wave | 二阶 PDE | ★★★★ | MEMS lattice |
| Maxwell | 3D mesh + CFL | ★★★ | photonic structure |
| incompressible NS | 全局压力解 | ★★★★ | CMOS + physical Poisson mesh |
| MD long-range | N-body | ★★★★ | CMOS local + field solve |
| SDE / optimization | 大规模 ensemble | ★★★★★ | noisy analog |

**哪些不会被救**（与 §9 一致）：

- state count 不减少：1024³ 仍是十亿 states；analog 不降低自由度。
- 精度–速度守恒仍成立，只是每域用自己的噪声机制（MEMS 的 thermo-mechanical `S_x=4kT·c`，光子的 shot noise）。
- 高-Re DNS、6D Boltzmann、quantum many-body、长期 chaotic trajectory 不在承诺内；chaos 只承诺统计量（Lyapunov spectrum / attractor statistics / correlation / spectrum），与 §1.5 Level 2 一致。

**不变的是判定机器**：`κ_hw` 变 per-domain，`Γ` 按 transducer 耦合矩阵算，routing 从 RC 换成 loss/thermal/calibration 或 fluidic 公差。§1.2 的 slack 向量仍然适用，只是每轴多一个 domain 下标；§3.2/§6.15 的 per-edge closure 也仍然适用，只是 `H_ij(s)` 换成该域的传递函数。

---

## 4. 编译器：pass 结构

    equation
      → P0 离散化/结构化        (M,K,f) 或 (L,N) 分裂
      → P0.5 随机语义            ito/stratonovich、修正漂移、G 可微性、Q-Wiener trace-class
      → P1 谱与正则性分析        λ, κ_t, Γ, 谱系数衰减率 p
      → P1.5 噪声可实现性        PSD/带宽/τ_c/ρ、D 的动态范围
      → P2 表示选择              basis T / splitting / 局部表示
      → P3 尺度对齐              A_t, 频窗检查
      → P3.5 随机尺度对齐        G 的 range scaling、D 与状态摆幅
      → P4 扰动预算              H2 per source, C 按敏感度分配
      → P5 布局与路由            config c, Y_c, Δ, small-gain
      → P5.5 噪声路由与相关      寄生对 PSD 的影响、ρ 校准
      → P6 可行性证书            slack + 随机证书（带宽/τ_c/D 范围/读出）
      → P7 配置生成 + 校准计划    含噪声增益、seed、T9 统计配置

### 4.0 前端：从方程到 descriptor

P0 之前先做这套规范化，顺序不能反：

    ODE/PDE/SDE
      → jet / first-order reduction        z=(x,x′,…)；隐式判 det ∂G/∂x^(r)
      → nondimensionalization              x_i=X_i x̂_i, t=T₀τ；Re/Pe 等无量纲数
      → state augmentation                 sin/cos/exp/tanh/Bessel；记录不变流形
      → differential-algebraic reduction    消冗余状态；找不到有限增广就拒绝
      → spectral / spatial projection      û=P_N u；算 closure defect
      → sparse polynomial + special-function IR
      → state/time scaling                 V_i=S_i x_i, J_V=T₀ S J_x S⁻¹
      → analog primitive graph

对应关系：P0 做 jet/DAE 与无量纲化；P2 做增广/消元和投影；P3 做 S/T₀；P4 做 NL/MVM 预算。
P2 的投影与剥离已落地：`lowering.closure`（P_N 投影 + η_closure 证书，
`bench/run_closure_split.py` 34/34，`results/closure_split.json`）与
`lowering.split`（CSP/QSSA 快慢剥离，谱间隙 auto/τ_fast，物理坐标 Schur +
slaving 残差实测）。P2 的换基判定与接口合约也已落地：`lowering.basis`
（候选集 {identity/schur/gramian/pod/eigenvector} 按失配总预算
Γ·amplification 选基，`bench/run_basis_interface.py` 31/31，
`results/basis_interface.json`）与 `lowering.interface`（单通量源合约 +
§6.5 fitting/capturing selector）。

**scaling 是预条件，不是单位换算。** V=Sx 不改变特征值/κ，但改变 Γ、瞬态增长和硬件扰动范数；S 要同时满足 |V_i|≤V_max 和 Γ(T₀SAS⁻¹) 尽量小。逐项 O(1) 是目标，不是教条——把 J_ij 强行压到 O(1) 可能把物理刚度藏进 C_i 的失配里。

**DAE 检查**：det ∂G/∂x′=0 时先做 index reduction；index>1、pencil 不正则、或解对输入导数敏感时，P6 直接拒绝，不用 Lyapunov/H∞ 硬算。

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
   合同（review 第三支柱，待 `twin/multirate.py` 实现）：
   - 线性化分裂 Eẋ−A_diag x = A_off x_prev + Bu 的频域迭代算子 G_WR(jω)=(jωE−A_diag)⁻¹A_off；收敛条件是 sup_ω σ_max(G_WR(jω)) = ρ_sup < 1，不是只看 DC 值。
   - 硬件波形重放每次叠加 DAC 量化 + 重构滤波误差 ε_replay，k 次迭代后的误差界是 ‖e^(k)‖ ≤ ρ_sup^k·‖e^(0)‖ + (1−ρ_sup^k)/(1−ρ_sup)·‖(jωE−A_diag)⁻¹‖∞·ε_replay。第二项给出**有效迭代次数上限 k_max**：超过后重放底噪主导，继续迭代不再降误差。k_max 是编译器必须输出的数，不是仿真里试出来的。
   - 验收（`bench/run_multirate.py`，33/33）：正规分裂实测每迭代衰减率 = ρ_sup ±1.6%；非正规分裂 σ_max 证书保守（×5 gap，实测率 = sup|λ(G_WR)|——σ_max 充分不必要），IVP 因果窗口比界更快（Volterra 结构）；replay 平台实测 1.33e-3 vs 界 2.06e-3（×0.64，界有效），k_max 公式 8 vs 观测拐点 8；强耦合负控制实测发散。ZOH 三项独立预算已验收：Nyquist 处 |H|=2/π=0.6366（PureDelay 错报 1.0）、相位仍 =−π/2（=T_s/2 group delay）、高带宽负载下 ZOH 的 gain 判据实测 NO（gain_err=0.363≈1−2/π）而 PureDelay 错误地 PASS；droop<ε(0.01) 的退化阈值 ωT_s*=0.4906（≈√(24ε)，droop≈(ωT_s)²/24），退化显式记录 `degraded=True`；混叠 flag 0.5/0.9 Nyquist 两档（warning/violation）、重构滤波是独立行（sinc-only 镜像衰减 20log₁₀|sinc(1.05π/2)|=−4.37 dB 单独报告，不折进 droop）。k_max 是编译器必须输出的数，不是仿真里试出来的。注意平台落在界之下（0.64×）的原因：`wr_error_bound` 的 (1−ρ) 求和是确定性最坏情形的范数叠加，而独立零均值 replay 噪声按方差以 1/√(1−ρ²) 叠加——界作为最坏情形是正确的，√ 解释了这个 gap。
2. **S&H 同步**：每次交换付 kT/C + 孔径抖动，误差进 B_p；必须验证 ‖G_loop·Δ_sync‖∞ < 1。
   ZOH 的等效传递 H_ZOH(s)=(1−e^{−sTs})/(sTs)=e^{−sTs/2}·sinc(sTs/2)：`twin/route.py` 的 `ZOH` 块携带真实幅度——相位=PureDelay(T_s/2)，但幅度带 sinc droop（Nyquist 处 2/π≈0.637，−3.92 dB；PureDelay 会错报 0 dB）。`zoh_edge(T_s, omega_max=)` 只在 droop<ε（ε=0.01，ωT_s≲√(24ε)=0.49）时显式记录地退化成 `PureDelay(T_s/2)`；droop 进 route gain 轴，**混叠**（信号带宽逼近 Nyquist 的折叠）与**重构滤波**是两项独立预算，不许折进 droop。

否则不严格。写进 compiler 合法性检查，不接受"差不多"。

### 4.4 随机 pass 的合法性条件

P0.5/P1.5/P6 不是"建议"，是拒绝编译的条件：

1. G 必须在状态域上 C¹（Itô 修正需要 ∂G/∂x）；不满足就只允许加性噪声。
2. τ_c/τ_min ≤ 0.01，否则白噪声近似不成立；要么换物理噪声源，要么把 τ_c 写进模型（增广状态）。
3. B_noise ≥ 100·B_req；replay 模还要 f_s ≥ 2·B_noise 和足够位数。
4. D 的可用区间：D_min（高于本征底噪）到 D_max（状态不饱和）必须包含目标 D。
5. SPDE 的 Q 必须 trace-class；2D/3D 空间白噪声直接拒绝，要求相关长度。
6. 读出：N·p·2B_req 超过 T9/ADC 预算时，必须降 N、降带宽或加片上统计。

任一条不满足，P6 写"不可行"，不允许"差不多"。

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

在 ISA 里它就是一个显式原语 `BASIS_TRANSFORM(T, T⁻¹)`，硬件是 T3/T7 的 MVM。稠密 T 是 O(n²) 权值/时间；Fourier/DCT/wavelet 用结构化因子（O(n log n) 或稀疏），不要假装 O(n²) 的 T 是免费的。

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

**三类跳变要分开**：理想模型跳变（R_OFF/R_ON = 10¹²/1）→ 有限斜率 G(V_c) = G_off + (G_on−G_off)·S((V_c−V_T)/ΔV)；真实 MOS 开关 → 用 compact-model 的有限带宽，不真的去开关 routing transmission gate；拓扑事件（ideal diode/comparator/reset）→ T6 的 comparator/event/reset。不要用无限快的理想开关逼连续网络产生无限斜率。

### 5.6 多项式基与特殊函数：硬件 ISA 的选择

**不要用 monomial 1,x,x²,… 做硬件多项式基。** |x|>1 时 x²⁰ 爆动态范围；|x|≤1 时 monomial 的 Vandermonde 条件数仍然很差。用 Chebyshev：

    T_{n+1}(x) = 2x·T_n(x) − T_{n−1}(x),    |T_n(x)| ≤ 1 on [−1,1]

代价是 2 个状态 + 1 个乘法器 + 1 个减法器；系数 c_n 用 T3 加权求和。Legendre 同理。这正好落在 T1/T2/T3 上。

**基的选择表：**

| PDE/解类型 | 基 | 硬件映射 |
|---|---|---|
| periodic smooth | Fourier | T3/T4 对角增益 |
| finite interval smooth | Chebyshev/Legendre | T1 递推 + T2/T3 |
| irregular geometry | FEM | T5/T2 局部 |
| localized structures | wavelet | T5/T7 多分辨率 |
| shocks/discontinuity | FV/DG | T5 flux |
| radial/spherical | Bessel/spherical harmonics | T3 权值 + T1 |

**PDE 的 closure defect 要进合同：**

    r_closure = (I−P_N) F(u_N),    η_closure = ‖r_closure‖ / ‖F(u_N)‖

分母用 ‖F(u_N)‖ 或一个固定的参考尺度，避免 ‖F‖→0 时零除。Fourier 的二次非线性会产生 |k|≤2N 的卷积；2/3 规则只消 aliasing，不消 closure。η_closure 超 ε 时，P6 必须降 N、换基或标为不可行。这是 §5.2 “spectral ↔ physical 往返”的定量版，也是 §6.1 的 W_d 在表示层的对应物。

**原生 NL vs 状态增广**：§1.5 的字典给的是“可消”，不是“应该消”。增广状态要付积分器噪声、失调、漂移和不变流形维护；只有当增广 + 维护比原生块更便宜、条件数更好时才用它。

### 5.7 WENO 不是 stencil，是三个算子

经典 WENO 对 `u_t+f(u)_x=0` 构造候选重构

    f̂_{i+1/2} = Σ_k ω_k f̂^{(k)}_{i+1/2},   ω_k = α_k/Σ_j α_j,   α_k = d_k/(ε+β_k)^p

`β_k` 是 local smoothness indicator。所以它实际做的是

    u → {β_k} → local representation selection → f̂

这不是一个固定 stencil，而是 `z=T(u)u`：`T` 依赖当前状态。光滑区 `T→T_high-order`，间断面 `T→one-sided/local`。等价地，它定义一个状态依赖的有效相互作用长度

    ℓ_eff = ℓ_eff(x,t,u,∇u,…),   smooth: ↑,   shock: ↓

对硬件，重要的不是复现 WENO5 的算术，而是把这层拆成三个可独立实现的算子：

    S:   u → σ(x)          singularity / smoothness detector
    T_σ: u → z             state-dependent representation / scale transform
    F_σ: z → F_{i+1/2}     conservative flux operator

    ∂_t u = −∇·F_{S(u)}[ T_{S(u)} u ]

**σ 是连续量，不是 if/else。** 数字算法的硬切换在模拟域会引入新的 discontinuous dynamics；用

    σ(β) = 1/(1+exp(−k(β−β_0))),   F = (1−σ)F_smooth + σF_shock

保持连续。`σ` 是 representation coordinate，不是 mode flag；`β_0`、`k`、迟滞窗口都是配置，不是物理常数。

**分工：analog 算候选通量，digital 算/选权重。** 三路候选 `F_0,F_1,F_2` 在 analog 并行存在；数字块根据采样到的 `β_k` 算 `w_k`（或只做 `β` 区间选择）；analog 用加权电流求和 `I_F=Σ w_k I_k`。这就是 §3.2 的 "analog data plane + digital control plane" 用在一个具体算子上。但 `w_k` 的更新率必须受限于 supervisor 的判决延迟，否则 `σ` 在阈值附近抖动，重新引入 §6.2 的开关注入噪声；`σ` 需要和 §6.6 一样带迟滞和速率限制。

**最便宜的 sensor 不是 WENO 权重，是归一化二阶差分：**

    S_i = |u_{i+1} − 2u_i + u_{i−1}| / (|u_{i+1}−u_i| + |u_i−u_{i−1}| + ε)

difference → abs → normalize → threshold，输出 `s_i∈[0,1]`。它只需要 T2 的差分/绝对值/归一化，不需要除法器阵列。WENO 的 `β_k` 只在需要更精确的权重时才算；sensor 决定“要不要算”。

`ε` 必须是**相对场幅的 floor**（~1% range），不能是 1e-12 的绝对数：光滑极值处一阶差分为零，`0/0` 会让 σ 饱和成 1（§6.16 实测 sine 的 σ 从 1.0 降到 0.1265）。σ 的上升/下降速率还必须不对称：上升快（抓住激波），下降慢（hold time）。对称速率会形成 `σ↓ → 中心格式 → 振荡 → S↑ → σ↑ → MUSCL → S↓` 的 representation 反馈极限环；实测同一 cell 的穿越次数从 8 降到 2。

**守恒不能因为表示切换而丢。** 无论 `F_smooth` 是谱通量还是局部 FV 通量，混合必须发生在**同一个 face flux** 上：

    C_i·V̇_i = F_{i−1/2} − F_{i+1/2}

相邻 cell 用同一个 `F_{i+1/2}`，内部通量 telescoping，`Σ_i C_i V̇_i = 0` 由拓扑保证（§5.4、§6.4）。禁止先混合状态再各自算通量；那会把守恒误差变成 `O(σ)`，且不可观测。

**与 multiresolution 的关系**：`u=u_coarse+u_detail`，`|d^{(l)}_i|>ε_l` 的区域激活更细一级 state；WENO 的 `β_k` 与小波 detail coefficient 都在检测 local regularity。区别是 wavelet detail 是线性的、可逆的，WENO 权重是非线性的、只用于通量。所以架构上应是

    multiresolution transform → regularity detector → adaptive refinement
    WENO/FV                  → singularity-handling operator（只在 σ 高的区域）

而不是“整个 domain 统一 WENO”。这也是 §3.7 的 substrate 分配在表示层的版本：`σ≈0` 用 Fourier/Chebyshev/photonic transform，`σ≈0.5` 用 local polynomial CMOS，`σ≈1` 用 conservative FV/WENO current network。

### 5.8 TransformContract：forward / inverse / adjoint 是同一个 primitive

§5.1 只写了 `T` 与 `T⁻¹`。PDE 的 adjoint、反问题、preconditioner 还要求第三个对象 `T^*`。三者一般不同：

    biorthogonal:  T⁻¹ ≠ T^*,   orthonormal:  T⁻¹ = T^*

所以硬件合同必须分开写：

    forward:  z = T x
    inverse:  x = T⁻¹ z        （perfect reconstruction）
    adjoint:  λ_z = T^{-*} λ_x （gradient / sensitivity）

把三者当同一个东西，会在 biorthogonal wavelet 或非正交 lifting 上得到错误的梯度；这类错误不会让 forward 解看起来错，只会让优化/反问题慢慢偏。

**等距性要用物理 metric 说。** 裸电压的 Euclidean norm 不是能量；真正的 metric 是存储能量

    ‖x‖_M² = x^* M x,   M = diag(C) 或 mass matrix

所以条件不是 `T^*T=I`，而是

    T^* W T = M

若 transform domain 用 Euclidean metric（`W=I`），则希望 `T^*T=M`；或先 whitening `y=M^{1/2}x` 再做 unitary。§0(B) 的 `σ²=αkT/C` 已经说明 C 是物理量，这里只是把它提升成内积。

**Parseval 对噪声是设计原则。** 若 `T` unitary 且噪声各向同性 `Σ_n=σ²I`，则

    tr(T Σ_n T^*) = tr Σ_n

不产生 `κ(T)` 式的额外放大。反过来，非 unitary 的 `T` 会把噪声按 `σ_max(T)/σ_min(T)` 重新分配；这不是“数学不优雅”，是 SNR 预算里必须出现的一项。

**算子对角化：目标不是 `Λ`，是 `D+B`。** 广义特征问题

    A φ_k = λ_k E φ_k,   Φ^* E Φ = I,   x = Φ z
    ⇒  ż = Λ z + Φ^* B u

对 `E=diag(C)`、`A` 为导体网络的 diffusion/RC 情形，这就是把耦合网络变成一组近乎独立的模态积分器；单模态时 `λ_k=−1/(R_kC_k)`，耦合时 `λ_k` 是 `−G C⁻¹` 的模态值。但 exact diagonalization 只在 `A,E` 构成合适 self-adjoint pair 时可用；非正规时 `κ(Φ)≫1`，硬用特征基会把模拟误差放大 `‖Φ‖‖Φ⁻¹‖`。所以 compiler 的目标放宽成

    T A T⁻¹ = D + B,   D diagonal,   B sparse / small norm

`B` 是允许的局部耦合，`D` 是本地增益。对非正规问题，优先 unitary Schur、orthogonal block basis、POD/SVD、wavelet sparse representation，而不是病态特征基。判据仍是 §0(C) 的 resolvent：`Γ` 比 `κ(Φ)` 更能说明实际放大。

**forward 与 adjoint 共享变换。** 若 `z=Tx`，`δJ=⟨λ_x,δx⟩`，`δx=T⁻¹δz`，则

    λ_z = T^{-*} λ_x

unitary 时 `λ_z=Tλ_x`；forward 与 adjoint 用同一个变换，只是方向相反。若 `A_T=TAT⁻¹` 被对角化，则 `A_T^*=TA^*T⁻¹`；normal operator 的 forward/adjoint 是同一组 mode、共轭增益：

    forward mode k:  gain = λ_k
    adjoint mode k:  gain = λ_k^*

self-adjoint 的 real PDE 里 `λ_k∈ℝ`，两者完全相同。这是“一块硬件同时跑 forward 与 adjoint”的数学依据，不是技巧。

**时间方向要单独处理。** 空间 Fourier 在同一时刻对整个场做变换，没问题；时间 Fourier/因果滤波器的 `L2` adjoint 是 time-reversed `h^*(−t)`，需要未来数据。连续时间 adjoint 写成

    −E^T λ̇ = J_F(x)^T λ + q
    τ = T − t  ⇒  E^T dλ/dτ = J_F(x(T−τ))^T λ + q

所以流程是 forward run → checkpoint → reverse replay → transpose routing/connectivity → forward-in-τ。**`TRANSPOSEABLE` 的 edge 应该在架构第一版就有**：

    EDGE i j gain=g TRANSPOSEABLE=true
    forward:  I_j += g V_i
    adjoint:  I_i += g V_j

`A ↔ A^T` 只是重新配置 switch matrix，不需要复制一套模拟硬件。这与 §5.1 的 `BASIS_TRANSFORM` 是同一类原语：一个可转置的稀疏线性算子，而不是一个固定功能的模块。

### 5.9 小波：lifting、undecimated 与 scale-dependent g_m/C

对 PDE solver，小波的价值不是“像 CWT 那样做时频检测”，而是同时提供三件事：局部性、可逆性、可预测的 adjoint。所以优先 DWT/lifting，而不是几十路 bandpass CWT filter bank。

**lifting 的硬件形态。** 把样本分成 `x_e,x_o`：

    d = x_o − P x_e
    s = x_e + U d

    [s;d] = [[I,U],[0,I]] · [[I,0],[−P,I]] · [x_e;x_o]

所有连接都是 local，系数少，可用电流求和，可 hierarchical reuse。adjoint 按 `(AB)^*=B^*A^*` 反向执行 transpose：

    adjoint lifting = 反向顺序 + 转置/共轭的 predict/update

所以硬件只需要 switchable edge direction + programmable sign/gain，不需要两套 filter bank。这与 §5.8 的 `TRANSPOSEABLE` edge 是同一个原语。

**不要真的 decimate。** 连续时间 field state 做 `↓2` 会引入 sample clock 和 aliasing；用 undecimated/stationary wavelet representation，保留所有 location，只改变 scale：

    N_coeff > N

代价是 redundancy，收益是 translation invariance、shock 移动时 coefficient 不跳格、adjoint 直接。第一代机器宁可多花 state，也不要引入 hidden S/H。若必须 decimate，decimation 必须进 §3.2 的 route contract，而不是当作免费的索引操作。

**scale-dependent time normalization。** 对 `ψ_{j,k}(x)=2^{j/2}ψ(2^j x−k)`，

    ‖∂_x^m ψ_{j,k}‖ ~ 2^{mj}

Laplacian `m=2` 给 `2^{2j}`：高尺度 detail 天然是快 dynamics，也是 stiffness 的来源。若所有 state 用同一个 C，pole 阶梯就是 `4^L`。两种都可行的实现方式：

    mode C:   C_j ∝ 4^{−j}, g_m 固定   → g_m 范围小，噪声 ∝ 4^{j}
    mode g_m: g_m,j ∝ 4^{j}, C 固定    → 噪声不变，g_m 范围 4^{L}

两者都不改变 `κ_t`（§0(A) 仍然成立），改变的是把 pole 跨度放在 g_m 还是 C/噪声上；`κ_t` 本身仍由 `λ_j` 决定。判据：校准后每级 pole 的 `g_m/C` 残差、相对增益误差和噪声（`σ²=αkT/C_j`）都要进 slack；不能只调 C 不管噪声，也不能只调 g_m 不管范围。

**adjoint-driven analog AMR。** 小波系数本身就是 singularity sensor：smooth 区 `|c_{j,k}|` 随 j 快速下降，shock 附近在多个 j 上持续存在。再加上 adjoint 系数 `λ_{j,k}`，定义

    R_{j,k} = |c_{j,k}| · |λ_{j,k}|

它直接告诉硬件：这个 location、这个 scale 值不值得保持高精度。于是可以开启 finer state、提高 bias/current、给更大电容、分配低噪声 tile、reroute 到短路径。这是 §6.7 资源分配律的 adjoint 加权版本：不是按 `1–5%` 的面积猜，而是按 `R_{j,k}` 排序。

**Gram matrix calibration。** 真实模拟变换 `T̃=T+ΔT`，`T̃^*T̃≠I`。不要逐个 OTA 调 gain；直接测

    G_T = T̃^* T̃

若 `G_T≈I+ΔG`，数字端算 `K≈G_T^{−1/2}`，**右乘** `T_cal=T̃ K`，使 `T_cal^*T_cal≈I`。左乘要用**左 Gram** `T̃T̃^*` 与 `T_cal=K T̃`；两者一般不可交换。实测用右 Gram 配左乘 `K T̃` 得到 isometry 0.032（0.2% 扰动；2% 时 0.342），右乘后 3.46e-15（§6.16）——这是硬件标定最容易犯的一个矩阵边错误。这是 hardware whitening/orthogonalization，也是 §3.6 “校准数学 contract，而不是晶体管”的直接实例。biorthogonal 情形测 `Ψ̃^*Ψ`，目标是 `I`；`G_T` 的条件数决定 `G_T^{−1/2}` 是否会放大噪声，必须一起报。

**TransformContract（compiler 的输入，不是事后报告）：**

    TRANSFORM wavelet_lifting_L3
      metric            = energy_M        # M=diag(C) 或 mass matrix
      forward           = W
      inverse           = W_inv
      adjoint           = W_star
      isometry_error    < 0.5%            # ‖W*W − M‖ / ‖M‖
      PR_error          < 0.2%            # ‖W_inv W − I‖
      group_delay_skew  < 20 ps
      noise_gain        < 1.03
      transposeable     = true
      scale_levels      = 3

物理综合不要求 `W_HW=W_ideal` 逐项相同，而要求上面这些**数学合同**成立，再加上 `Δτ_g、S_n、DR`。这和我们对待 route 的方式一致：不要求每条 wire 的 R、C 等于 nominal，而要求 `H_ij(s)` 在带宽内满足 gain/phase/noise/absolute-delay。

**实现优先级**：local lifting wavelet fabric + transposeable gm/current edges + scale-dependent `g_m/C` + Gram-matrix calibration。它同时满足局部互连、multiresolution、shock handling、adjoint 复用，而且不会重新引入一个巨大的 global analog crossbar（§9 不做项）。

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

对湍流/混沌，五元组里的 x_s(t) 和 smooth 区 L2 只能作短期验证；长期契约换成 E(k)、⟨u_i u_j⟩、相关函数、涡量统计。逐点轨迹契约不合理，噪声/失配的容忍度按统计量重算；但硬件噪声 PSD 必须低于要解析的物理 fluctuation，或明确作为 subgrid stochastic model 的一部分。

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

两处值得记住：κ 估计器误差 2–5%，可以直接当 QoI 用；**半离散迎风的数值扩散是 ½·a·h，与 CFL 无关**（(1−CFL) 因子只属于显式欧拉的时间离散），所以两个 CFL 给出同一个 κ_num——这也顺带验证了时间积分器已收敛。红线换算：κ_num < 1e-6 m²/s ⇒ 一阶迎风需要 h < 2e-5 m（a=0.1 m/s；半离散 ν=½·a·h 与 CFL 无关，(1−CFL) 因子只属于显式欧拉），这就是海洋模型必须用高阶传输的原因。

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
| IHP SG13G2 npn13G2 | 20 µA | 37.4 /V | 0.014 /V | 3.61 MΩ | 0.9990 | 4.7 nV/√Hz |

> 噪声列不是 ngspice `.noise` 的实测值，而是由提取到的 g_m 按 4kTγ/g_m 折算（MOS γ=2/3，BJT γ=1/2，T=300 K）。extract.py 只做 op/dc；要改成实测，需加 `.noise` 分析并重跑。IHP MOS 的 λ 与 58.6% 镜像误差说明短沟道器件不适合做接口镜像，与 HBT 的 2.8% 形成对照。

接回 twin/coupling.py 的 InterfaceModel(mode="gm") 后，重跑 §6.8(d) 同一算例：

| 接口 | drift | max\|ΔF\| |
|---|---|---|
| 理想镜像 | ≤1e-15（舍入） | 0 |
| IHP HBT，仅 λ 项（共享参考，λ=0.0138@20µA） | −3.29e-6 | 2.56e-5 |
| IHP HBT，两条独立镜像（0.1% 系统误差） | −2.2e-4 | 8.6e-4 |
| gf180，仅 λ 项（共享参考） | −8.1e-6 | 6.3e-5 |
| gf180，两条独立镜像（含 3.3% 系统误差） | 7.1e-3 | 2.8e-2 |
| sky130，仅 λ 项（共享参考） | −1.9e-5 | 1.5e-4 |
| sky130，两条独立镜像（含 2.2% 系统误差） | 4.8e-3 | 1.9e-2 |
| IHP MOS，仅 λ 项（共享参考） | −2.4e-4 | 1.9e-3 |
| IHP MOS，两条独立镜像（含 58.6% 系统误差） | 0.179 | 0.523 |

（HBT λ 于 2026-11 修正为 0.0138：此前 vsweep 在 I_D=20 µA 测 g_ds 却除以 10 µA 的镜像电流，λ 被高估约 2×；现 λ=gds/I_D 在 vsweep 实际偏置点自洽取值，见 `extract.py analyse` 的 `i_op_vsweep` 字段。indep 行的 eps_sys=ratio−1 在 10 µA 测量，不受影响。）

三条结论：

1. **共享参考节点把系统误差整个删掉。** sky130 从 4.8e-3 降到 1.9e-5（250×），HBT 从 2.2e-4 降到 3.3e-6（67×），gf180 从 7.1e-3 降到 8.1e-6（880×），IHP MOS 从 0.179 降到 2.4e-4（740×）。这是"单通量源"在真实器件下的确切价值——不是"零泄漏"，而是**把可预测的系统误差换成不可预测的随机项与 λ 项**。indep 行的 eps_sys 取实测 ratio−1（含 λ·ΔV 项），所以这些倍数是上界。
2. **残余泄漏 = λ·ΔV·F，且被四个工艺的实测比例确认。** 共享参考下按 λ 升序：HBT 3.29e-6 (λ=0.0138) → gf180 8.10e-6 (0.0341) → sky130 1.93e-5 (0.0810) → IHP MOS 2.43e-4 (0.970)；相邻 drift 比值 2.46 / 2.38 / 12.6 对 λ 比值 2.47 / 2.38 / 11.97。这不是巧合，是把"漏电流 ∝ (1/r_o)·节点电压差"这条机制钉死了。
3. **理想镜像不可达。** 最好的实现（HBT + 共享参考）仍比理想差约 10¹⁰ 倍。10⁻¹⁶ 是双精度舍入极限，10⁻⁵ 是器件极限。设计目标应定在后者，不能拿前者做验收。

噪声：由 g_m 折算的输入热噪声 sky130 7.9 nV/√Hz、gf180 9.0、IHP MOS 5.6、HBT 4.7（非 .noise 实测，见上表注）。§6.2 的乐观重构注入 45 nV/√Hz（30 pF 节点）相对它们是 5.7× / 5.0× / 8.0× / 9.6×；1 pF 节点情形是 1.4 µV/√Hz，即 ~180× / 156× / 250× / 300×。两种口径都支持"重构噪声主导"，但引用时必须带节点电容。

**HBT vs MOS（回答"IHP SiGe 是否值得重点建模"）**：同一电流下 HBT 的 λ 小 2.5×、r_o 大 3×、g_m 大 2×、镜像系统误差小 20×（0.1% vs sky130 的 2.2%）。把四个工艺放在一起，接口层排序是 **HBT ≫ gf180 > sky130 ≫ 短沟道 MOS**：HBT 的 λ 比 IHP MOS 小 70×（0.0138 vs 0.970），镜像系统误差小 ~590×。这解释了为什么接口层应优先落在 BiCMOS/RF 工艺，而数字 supervisor 留在低成本 CMOS。代价是偏置电路（V_BE 温度系数 −2 mV/K）必须进校准回路。

**原先两个失败的工艺已解决**（都是 netlist 引入方式，不是 PDK 模型问题）：
- gf180mcu：`sm141064.ngspice` 的 noia/noib 表达式引用 `fnoicor`，而 `design.ngspice` 才给它默认值。extract 直接 include 模型文件，需补 `.param fnoicor=0`；
- ihp-sg13g2 MOS：必须 include `cornerMOSlv.lib`（它定义 `sg13_lv_nmos` 子电路并 include parm 文件），只 include parm 文件会报 "could not find a valid modelname"；PSP103 走 OSDI，子进程需设 `PDK_ROOT`/`PDK`。
两处修好后四个 PDK 均提取成功；sky130 仍不得设 PDK_ROOT（见 AGENTS.md）。

### 6.10 Phase 0 判定器：五个验收算例 + DAE 噪声注入（可复跑）

    python3 -m bench.run_phase0        # ~2 min；38/38 checks passed

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
4. **判定落在正确的轴**：diffusion N=64 无违规；N=1024 在 κ_hw=1e6 下 1.049×（5% 越界，§2.4 的"贴边"）；biharmonic → stiffness；Burgers → representation；Robertson → stiffness，且 robustness_A 同量级。38/38 全过（原 23 条 + 8 条 DAE/P0-G 验收 + 3 条能量度量 Γ + 1 条 freqresp-integral 独立路径 + 3 条相似变换回归）。
5. **奇异 E 的 H2 判定先做代数块消去（`qz-dae-eliminated`）**：index-1 DAE（E=diag(1,0)、A 非对称）在代数约束 z₂=−A₂₂⁻¹(A₂₁z₁+B₂u) 消去后 σ_y²=0.5，与直接计算一致；naive 判据（B 在代数子空间有分量 ⇒ ∞）会错判。两个负对照成立：直接 feedthrough 与 index-2 nilpotent 块仍判 ∞。QZ 左投影必须用 `Q1ᵀ`（ordqz 的 `QᵀAZ` 只有 Q 在左侧才三角化）；五系统 σ_y 在修正前后逐位一致（它们都是 E=I 或对称 pencil，Q=Z 的旧代码碰巧等价）。
6. **边缘模态的 driven≠observable（H2 是输入→输出性质）**：旧判据"边缘（零/虚轴）块被驱动 ⇒ σ_y=∞"对 E=I、A=diag(−1,0)、B=[1;1]、C=[1,0] 错报 ∞——第二个状态是噪声驱动的积分器，但 C 不观测它，H(s)=1/(s+1)、σ_y²=0.5。修正（`_drop_unobservable_marginal`，`bench/run_lyapunov.py` 26/26）：对每个边缘特征值做 PBH 检验（rank[λE−A; C]<n ⇒ 不可观），收集 PBH 零空间方向（不可观子空间，pencil 不变），正交剔除后得最小实现，在 ω∈[1e-4,1e4] 网格上运行时核对剔除前后 H(s) 逐点一致（对消是恒等式；失败——如 Jordan 链部分可观——回落到诚实 ∞），再递归求解。剔除是精确的：稳定极点残差由双正交关系逐项相等，不可观边缘模态经耦合进入稳定行的通道（数值上会翻转结果的路径，例如 A=[[−1,1],[0,0]]、B=[1;1]、C=[1,−1] 的 H≡0 vs 稳定块直算给 1/(s+1)）随剔除消失。诚实负对照不变：驱动且可观（C=[0,1] 控制组、虚轴振荡器）⇒ ∞；index-1 馈通与 index-2 ⇒ ∞；σ_y 经 (1/π)∫|H(jω)|²dω 数值积分独立对拍。

**鲁棒轴探针**（1% 失配/接口预算，只加在 Burgers/Robertson 上）：Burgers 的
robustness_A = 36.8×1% = 0.37、robustness_io = 6.13×1% = 0.061，两轴都有限——鲁棒轴不是它的失败轴。
Robertson 的 robustness_A = 6.19e3×1% = 61.9（与 κ/κ_hw=62.2 同量级，慢模态是同一根杠杆）、
robustness_io = 22.3×1% = 0.22。其余三个算例没有失配模型，这两轴记 n/a（未知），不是 0。

三个口径决策，已经写进 `analysis/`：

- **κ 只去掉数值零模态**（|λ| < 1e-12·max|λ|），物理慢模态保留。Robertson 的 κ 随观测窗变化：t=1e-3 时 ~5e3、t=1e2 时 4.7e5、t=1e4 时 6.2e7、t=1e6 时 5.1e9；10¹¹ 是速率常数比，不是 Jacobian 谱（§8 已改）。
- **鲁棒轴是两个轴**：Γ_proj‖ΔA‖（失配，离不稳定多远）与 ‖H‖∞‖Δu‖（接口，small-gain）。全 Γ 含零模态极点，只进诊断不进 slack；1% 预算只是投影验收的探针，没有失配模型时记 NaN。
- **表示阈值 = tol_L2²**（默认 1% ⇒ 1e-4）。**Burgers 的表示轴必须用物理状态算**：精确解 p=1.44、E_tail=2.6e-4（representation=2.57）；WENO 捕捉解 p=7.28、E_tail=4.4e-5（representation=0.44）。捕捉格式用 ν_num 换掉了谱尾，所以它不能用来证明"表示轴可行"。

### 6.11 SDE 正问题：三个可证伪实验

| 实验 | 解析对照 | 验什么 | 通过线 |
|---|---|---|---|
| OU: dx=−λx dt+√(2D)dW | Var=D/λ, ACF=e^{−λt} | 加性噪声、带宽、标定 | Var ±3 dB；ACF 时间常数 ±5% |
| GBM: dx=μx dt+σx dW | Itô E=x₀e^{μt}；Stratonovich E=x₀e^{(μ+σ²/2)t} | Itô/Stratonovich 语义 | 两条预测分开 >3σ，实测落在声明的那条 ±10% |
| 双势阱: dx=−V′(x)dt+√(2T)dW | p∝e^{−V/T} | 采样、动态范围、长时间漂移 | 1D 直方图 KL < 0.05 |
| CRN vs 独立噪声（§6.12 路径一验收） | 共享实现梯度噪声底 ∝1/δ² 抵消 | replay 合同 | Var(IND)/Var(CRN) ≥ 10×；λ̂<10% |

四个都先用数字孪生跑（`bench/run_sde.py`，27/27，seed 20261118），再接 T8/T9 硬件。OU 先测"噪声源本身"，GBM 先测"乘性语义"，双势阱测"分布"，CRN/IND 方差比测"replay 合同"。实测：OU Var 0.5%、ACF λ̂ 2.8%；带宽实验 6.69% vs 预测 6.4%（窗口 4.5–8.3%）；GBM Itô/Strat 均值分离 51.9σ；双势阱 KL 5e-4；CRN/IND 梯度方差比 2.3e5、λ̂ 4.96%（moment 路径 0.14%）。

**噪声带宽的单独验收**：把 B_noise 从 100·B_req 降到 10·B_req，OU 的稳态方差应下降约 6.4%；如果测不到这个下降，说明模型或标定有错。这个实验把"噪声源带宽"从规格表变成了可证伪的数。

口径（`analysis/sde.py ou_variance_deficit`）：6.4% 对应 §1.3 合同里的 **flat/brickwall 平坦带宽**约定（Var(k)=(D/λ)·(2/π)arctan k，k=10→6.345%、k=100→0.637%）。若噪声源是一阶低通的 Lorentzian（`shape="lorentzian"`，如 OU 滤波源），同一 k=10 的方差亏损是 9.09%（Var=(D/λ)·k/(k+1)）。合同必须写明 shape，否则差 2.7 个百分点。

**广义 Wong–Zakai 修正（第三评审支柱 5，已实测）**：乘性噪声 G(x)·ξ 经延迟 τ_d 的硬件乘法器/反馈极点后，Stratonovich→Itô 漂移修正不是固定的 ½·G∂G/∂x 而是 θ(χ)·G∂G/∂x，χ=τ_d/τ_c、`wong_zakai_factor`=1/(2(1+χ))。χ=0→½（经典 Stratonovich），χ→∞→0（白噪声+延迟环路退化为 Itô）。`bench/run_sde.py` case F 用一阶极点延迟模型（不是纯延迟）在 χ∈{0,0.25,1,4,16} 上实测均值增长率，与 θ(χ) 全程偏差 ≤14.9%——插值公式在诚实证伪下存活。约定：白且瞬时的双重极限是经典歧义，按 §1.4 `hardware_limit` 字段显式声明，默认 ½。

**硬件伴随的失配下界定理（同支柱，已实测）**：`TransposeableEdge` 的 A↔Aᵀ 在 2.7e-16 假设理想开关；晶体管级反向传输 Ã_adj=Aᵀ+Δ_asym。梯度相对误差界 ‖∇̂J−∇J‖/‖∇J‖ ≤ Γ_proj(A)·‖Δ_asym‖·T_obs，case G 实测 err/bound≈0.71（非空、不紧），log-log 斜率 1.000。用途：当 Γ_proj·‖Δ_asym‖·T_obs 超过容差时禁止免校准转置复用，必须走 §5.9 Gram 白化——界给出了定量判决点。

### 6.12 反问题：数字优化器 + 模拟 ensemble + 片上统计

正问题是"给定 θ 跑轨迹"；反问题是"给定 y(t) 求 θ、x₀，甚至 f/G"。不建议全模拟：

    digital host                     analog fabric
    ------------                     -------------
    θ_k, 优化器, 先验, 校准   →      N 条轨迹 x^{(n)}(t; θ_k)
    likelihood / 重采样        ←      T9: 均值/协方差/直方图/似然
    checkpoint / seed 管理     ↔      replay noise

三条路径都记录、都在数字孪生里实现，但适用域不同：

| 路径 | 需要什么 | 适合 | 代价 |
|---|---|---|---|
| CRN 有限差分 | 同一 noise realization 重放 | 低维参数、黑箱 | 2p 个 ensemble；差分方差 ~ δ²Var(∂L) |
| EnKF / SMC | 粒子统计量 | 低维参数、非线性 | N 个粒子；T9 读出 |
| 模拟 adjoint | ∂f/∂x、∂f/∂θ、x(t) checkpoint | PDE 高维系数、ODE/加性噪声 | 反向配置 + 存储/重放 |

Langevin 优化 dθ=−∇L dt+√(2T)dW 单独记录：它只在 ∇L 可在硬件内计算、且漂移是梯度流时成立；否则稳态不是 e^{−L/T}。乘性噪声的 pathwise adjoint 有额外项，且反向 SDE 可能不稳定；第一版不作为主路径。

**读出预算是反问题的第一约束**：N 条轨迹 × p 个输出 × 2B_req 的原始数据率，N=100、p=10、B=10 MHz 就是 20 Gsample/s。T9 把每个 ensemble 压成 O(p²) 个统计量，数字端才接得住。没有 T9，"模拟 ensemble"只是把 ADC 换成更大的瓶颈。

**可辨识性要单独检查**：单条轨迹的 Fisher 信息随 T 增长；参数维数高时 CRN 的 2p 成本会爆。P6 先算敏感度矩阵，再决定 N 和 T，不能先跑再说。

**稀疏编码是同一套动力学的应用**：`min_a ½‖y−AΦa‖²+λ‖a‖₁` 的连续时间 LCA 电路就是 `transform → threshold → sparse state`，落到 T2/T3/T9 的组合，不需要新 tile。它和 Langevin 一样，是“把优化写成 ODE”的实例，不是通用 posterior 机器。

### 6.13 前端可划归性：五个算例的实际分类（§1.5 的验收）

    python3 -m bench.run_reduce        # 41/41 checks passed（+ manifold-γ 随机验收、native_primitives 成本、poly_frac 同分母回归）

| 系统 | 增广 | states | kind | degree | monomials | dense | sparsity | 结论 |
|---|---|---|---|---|---|---|---|---|
| Lorenz | — | 3 | polynomial | 2 | 7 | 10 | 0.233 | 直接落 T2/T3 |
| Duffing | sin(6t/5) → (s,c) | 4 | polynomial | 3 | 7 | 35 | 0.05 | s²+c² 漂移 9.9e-15 |
| HH | 6×exp | 10 | rational | 6 | 65 | 8008 | 8.1e-4 | 3 个分母，需 divider |
| Bessel J0 | J0(t) → (j,j′) | 4 | rational | 2 | 6 | 15 | 0.10 | t≥1 可增广；t=0 拒绝 |
| Gamma | — | 1 | rejected | — | — | — | — | Hölder，无有限增广 |

四条结论：

1. **Lorenz/Duffing 确实落到同一 sparse polynomial IR**：两者都是 7 个 monomial，dense 上界分别是 10/35。这是 §2.7 的 `N_mono=C(n+p,p)` 的直接对照。
2. **标准 HH 不是 polynomial。** α_m/α_n 的 1/(1−exp) 把它变成 rational；6 个 exp 增广后 10 个状态、3 个分母、65 个 numerator monomial（dense 8008）。要上硬件必须显式有 divider 原语，或者做一次有文档的 polynomial fit；不能因为“HH 很常见”就把它算进 polynomial 类。这是对 §7 Phase 1 早期措辞的修正。
3. **exp 增广的 degree 会继承 V̇ 的复杂度。** HH 的 exp 状态方程是 ė=e·(−V̇/10)，而 V̇ 本身是 degree 5 的多项式，于是 exp 方程 degree 6。增广不是免费的：它把 V 的复杂度复制到每个 exp 状态上；但 65 个 monomial 仍远小于 dense 8008。
4. **不变流形不是自稳定的，但可以被稳定化（已实测）。** Duffing 的理想 RK4 漂移 9.9e-15；把 ṡ 乘 (1+1e-3) 后漂移升到 1.0e-3。模拟硬件中法向特征值为 0，失调电流使偏离随 t 线性/布朗增长；`reduce_system(..., manifold_gamma=γ)` 可选加 Baumgarte 项 −γ∇g·g（sin/cos 为 −γs(s²+c²−1)，每对增广状态多一组三次乘法反馈），把法向特征值移到 −2γ|∇g|²<0，稳态方差饱和而非无界增长。`manifold_variance` 实测拟合 E[g²]=1.08e-8/γ+1.12e-6/γ²（残差 8.8e-4），分项标度 1/γ（噪声）与 1/γ²（失调）都与两项界一致；γ 可行区间 [0.1, 5e5]（κ_hw=1e6 下）非空——对本参数不触发"必须用原生原语"判决，但区间为空时的判决规则已写死。
5. **增广 vs 原生非线性的成本可以直接比（已实测）。** `reduce_with_primitives` 把 exp/tanh 留作 T2 tile 原语而不是增广状态：HH 从 10 状态/65 monomial/degree 6 降到 4 状态/29 monomial/degree 5 + 6 次原生 NL 求值。degree 5 来自 HH 自身的门控幂（n⁴、m³h），不是增广引入；增广额外付的是 6 个积分器状态 + 继承 V̇ 全部复杂度。盈亏平衡是编译器的显式选项，不是默认行为。

Gamma 与 Bessel 的边界也按 §1.5 执行：Gamma 直接拒绝（native-only）；Bessel J0 增广成 1/t 的有理系统，t≥1 时与 `scipy.special.jv` 的相对误差 5e-15，t=0 时拒绝——这正是“x 远离 0”的工程含义。

### 6.14 互连延迟不是 latency，是 E 的改变（analog skew 的一阶模型）

若 x_j 到 i 的路径延迟为 τ_ij：

    ẋ_i(t) = Σ_j A_ij x_j(t − τ_ij)

对 ωτ ≪ 1：

    [I + (A∘T)] ẋ = A x ,    E = I + A∘T

所以 delay 的一阶效应是把 ODE 偷偷变成带 mass matrix 的 descriptor，而不是“结果晚了几 ps”。这个 E 可以直接进 §1 的 descriptor 分析器。

`python3 -m bench.run_skew`（15/15）的四组数：

| 算例 | 结果 |
|---|---|
| 均匀 2×2，τ=0.01 | 广义极点与闭式 λ/(1+τλ) 的误差 2.31e-16；一阶 −τλ² 的误差 3.14e-2；κ 3.205→3.275；Γ 1.057→1.057；cond(E)=1.022 |
| 标量 DDE，a=−1 | τ=1e-3/1e-2/1e-1 的相对误差 5.01e-7/5.12e-5/6.45e-3；精确 τ_crit=π/2=1.571，一阶给 1.000，ratio=0.6366=2/π |
| 扩散链 n=8，τ=5e-3 | κ 32.163→32.780；Γ 8.291→8.291；cond(E)=1.019；一阶 τ_crit=0.2578；0.9τ_crit 稳定、1.1τ_crit 不稳定 |
| 相位预算 | 100 MHz：1°→27.78 ps；10°→277.78 ps |

三条结论：

1. **均匀延迟有闭式**：E=I+τA 与 A 可交换，广义极点就是 λ/(1+τλ)。所以 pole shift 的符号/量级不用仿真就能算：实稳定模态变得更稳定，复模态的阻尼会被改变。
2. **一阶模型只在 ωτ≪1 时可信**：标量 DDE 的精确稳定边界是 |a|τ=π/2，而 E=I+τA 给 |a|τ=1，乐观 2/π≈0.637 倍。§2.8 的 ω_max Δτ_route≪1 不能省；一阶 mass matrix 不能替代相位/稳定边界检查。
3. **相位预算直接给 A_t 设上限**：1° @100 MHz 只有 27.8 ps。芯片几何、buffer 延迟、wire RC 一旦超过这个量级，A_t 就不能再往上加；此时 wire 已经不是 interconnect，而是被求解动力学的一部分。

### 6.15 Route 是 transfer function：per-edge closure 与 sensitivity-driven budget

`python3 -m bench.run_route`（73/73）把 §3.2 的 per-edge 条件做成可测的：

    H_ij(s) = H_wire(s) · ∏_k H_buf,k(s) · e^{−sτ_ij}

七组结果：

| 算例 | 结果 |
|---|---|
| RC wire 400 Ω / 400 fF | DC group delay 80.00 ps（distributed/lumped 相同）；distributed t50=60.65 ps=0.3791 RC（FD 实测，与 0.3795 差 0.1%）；lumped 55.45 ps；f_3dB 2.42 vs 1.99 GHz |
| buffered fanout | g=0.999、f_p=2 GHz、σ=10 µV：1 级 79.6 ps/L2、增益 0.225%、相位 2.86°、噪声 10 µV → pass；4 级 318.3 ps/L3、增益 0.896%、相位 11.45° → fail |
| all-pass vs RC | τ=200 ps、500 MHz：all-pass 增益误差 3.3e-16、相位 1.12°、J_H 5.6e-5、J_τ 68.1 ps²；RC 15.3%、3.86°、J_H 6.0e-3（108×）、J_τ 769.7 ps²（11.3×）；RC f_3dB 0.80 GHz |
| deskew | 20/45/60 ps 三条 edge：skew 39.8 ps→3.3e-7 ps，group-delay spread 267.7→0.029 ps²；共同延迟 59.8 ps 在 100 MHz 仍是 2.15°，1° reserve 失败、5° 通过 |
| feedback | ω_c=100 MHz、φ_reserve=5° → τ_max=138.9 ps；130 ps 通过、150 ps 失败；两条 150 ps 的 skew=0 但都失败 |
| crosstalk | C_m=20 fF、C=200 fF：HF 地板 C_m/(C+C_m)=0.0909（−20.8 dB）；2 fF→−40.1 dB；1 V 摆幅下 20 fF 给 90.9 mV，1 mV 预算要求 C_m<0.11 fF |
| sensitivity | 慢/快模态 τ_tol 比 200（=a_f/a_s）；Q=10 的 50 MHz 共振把每条预算再压 2Q=20.05×；worst ω=50.0 MHz；排序 0.69/6.93/13.85/138.5 ps |

后五组（review 追加，case K–M）：

| 算例 | 结果 |
|---|---|
| loop closure | Loop A（阻尼振荡器，全边均匀延迟）：σ_min(jωI−J∘H) 在 τ=0→300 ps 从 3.14e7 单调降到 1.56e6，精确 DDE 边界 τ_ex=315.7 ps（`scalar_dde_root`，jω 轴残差 7e-17）；一阶 E=I+A∘T 给 315.2 ps（比值 0.998）。deskew 负对照：skew 150 ps→0 但 loop margin 仍按 250 ps 共同延迟算（6.53e6 vs 100 ps 级的 2.15e7）。‖S‖∞=‖(I+L)⁻¹‖∞ 从 10.0→202.1 随 τ 增长。Loop B（近标量）：τ_fo/τ_ex=**0.6467 ≈ 2/π=0.6366**——一阶 mass matrix 的乐观性在 loop 级复现（差 1.6%） |
| PhysicalScaling | t_phys=alpha·t_model：λ=2π·8 MHz 模型在 B_hw=10 MHz 下裸比通过，alpha=0.1 s/model-unit ⇒ 80 MHz 物理 ⇒ 失败（判定翻转）；λ_hw=λ/α 与 x=V/S_x 双向换算成立 |
| phase delay @ωc | PureDelay 的 phase delay 恒等于 DC latency；in-band 4×AllPass(50 ps)@10 GHz 下两者分叉：latency 205 ps vs phase delay 132.8 ps（naive `np.angle` wrap 会给 32.8 ps——bug）；marching unwrap 与 ∫τ_g dω 差 1.4e-5。feedback_critical 判据改用 phase-delay@ωc；deskew 后的 case-D 判定不变（两值差 2.5e-4，latency 仍偏保守） |

八条结论：

1. **同一个 DC 斜率不是同一个模型。** distributed 与 lumped π 的 DC group delay 都是 RC/2=80 ps，但 t50 差 8.7%、f_3dB 低 18%。用单极点替代 wire 会在 delay 和 bandwidth 上同时给错方向；`twin/route.py` 默认分布式 1/cosh√(sRC)，0.3795 这个系数是可测的（FD 0.3791）。
2. **fanout 深度的第一个约束是相位，不是噪声。** 2 GHz 极点、3° 预算、100 MHz：1 级 2.86° 刚好过，4 级 11.45° 失败，而增益只掉 0.9%、噪声只有 20 µV。§3.2 的“buffer tree 深度受 N_bσ_b² 限制”要加一条同样硬的相位限制。
3. **deskew 修的是 matched，不是 instantaneous。** 三条 edge 对齐到 3.3e-7 ps 后，共同延迟 59.8 ps 在 100 MHz 仍是 2.15°；1° reserve（27.8 ps）失败。零 skew 不恢复原 ODE（§2.8），feedback edge 必须单独做绝对延迟检查。
4. **crosstalk 有地板，而且不能靠 buffer 修。** HF 极限是 C_m/(C_2+C_m)=9.09%，与 R、buffer 无关；1 mV/1 V 的预算要求 C_m<0.11 fF，20 dB 屏蔽不够。它进的是相关噪声项，不是可以 ensemble 平均掉的独立噪声（§9 风险 25）。
5. **延迟预算可以从 J 直接算出来。** τ_tol=ε_φ/(ω|J_ij|‖(jωI−J)⁻¹‖)；高频极限是 ε/|J_ij|，共振处再乘 resolvent 峰值（Q=10 时约 2Q）。所以 router 的输入是 (J, ε_φ)，不需要手写 critical path：

    equation → J → S_τ,ij → τ_tol → latency class → placement/routing

`route_closure` 把这四条 closure（gain / phase / noise / feedback absolute）与 η、J_H、J_τ、latency class 一起返回，正好对应 §2.8 的 analog STA。注意最后一条与前面三条不同：前三条可以靠 pad/deskew 修，feedback absolute delay 只能靠 locality 或重新分解动力系统。

6. **per-edge 全过 ≠ loop 稳定，但一阶 descriptor 在 loop 级仍偏乐观 2/π。** `loop_margins` 在 σ_min(jωI−J∘H(jω)) 与 S=(I+L)⁻¹ 两个观测量上同时检查：Loop A 中 deskew 把 skew 清零但共同延迟不变，margin 仍按最坏边算；Loop B 的 τ_fo/τ_ex=0.6467 复现了 §6.14 的 2/π 乐观——`E=I+A∘T` 是一阶 Padé，不是 DDE 的精确判据。
7. **判定必须用相位延迟，不是 DC group delay。** AllPass pad 之后 `edge.latency`（DC 群延迟和）与 −∠H(jωc)/ωc 分叉；前者是保守侧（pd ≤ latency），但 naive `np.angle` 会在 −π 处 wrap 出偏小的假值（32.8 ps vs 真实 132.8 ps），marching unwrap 是必需实现细节。
8. **模型时间/物理时间/电压是 IR 字段，不是约定。** `PhysicalScaling(S_x, alpha, v_swing)` 让 λ→λ_hw 的换算成为编译器的显式检查；raw model bandwidth 与 B_hw 直接比会在 alpha≠1 时给出翻转的错误判定。

已知简化（RouteEdge docstring）：`H` 是开路传递函数乘积（不含级间负载，ABCD cascade 未实现）；`noise_var` 用 DC 增益²累加（无 ENBW 压缩，偏保守）。两者留给 PEX/SPICE 环节，不在本模型层。

**与已有 FPAA P&R 的关系。** 这条 sensitivity-driven routing 不是从零开始。Baskaya–Reddy–Lim–Anderson（TVLSI 2006）已经用 switch 数 + path balance 做 clustering/placement；Baskaya–Anderson–Lim（TCAS-II 2009）进一步把每个 net 对三种 wire type 的 SPICE 性能偏差做成 3×n net-sensitivity matrix（2n+1 次仿真），并以 `α·num_sw + β·Σ(nc_j+num_col_j)` 为 SA 的 cost，报告 fc error −50%、gpass −90%、ripple −91%，且只最小化 switch 数反而会恶化性能。Hasler–Twigg（ISCAS 2007）用 OTA-based routing 提高 fabric 速度；Wang–Vrudhula（TODAES 2002）给出 behavior → function decomposition → macrocell → P&R 的流程；George 等（TVLSI 2016）把 interdigitated A/D block 与 MSP430 放在同一 routing grid 上。它们的 sensitivity 仍然是 scalar 性能指标（fc、gain、ripple、rolloff）对 wire type 的偏差；本节把它换成 edge 的 `H_ij(s)` / `τ_g` / `S_n` / Jacobian，并把 feedback absolute delay 作为不可 pad 的第四条。无法满足 instantaneous contract 时，按 Lelarasmee 等的 waveform relaxation（TCAD 1982）把系统 partition 成 multirate islands，而不是继续加 buffer。文献与阅读状态见 `ref/README.md`。

### 6.16 Adaptive representation 的 50 项验收：WENO 三算子与 TransformContract

`python3 -m bench.run_transform`（56/56）把 §5.7–§5.9 的三条规则变成可测的：

    S: u → σ,   T_σ: u → z,   F_σ: z → F_{i+1/2}
    forward / inverse / adjoint 分开验证；混合只在同一个 face flux 上

六组结果：

| 算例 | 结果 |
|---|---|
| WENO sensor | 理想 step σ=1.0000；光滑 sine max σ=0.1265（`rel_eps=1%`） |
| Burgers adaptive | N=256、640 步：守恒漂移 2.22e-16；激波位置与 WENO5 都是 x=0.7480；L2=4.59e-3；每 cell 最大 σ 穿越 2 次（对称 tracker 是 8 次）；逐步熵不等式 d/dt∫u²/2≤0 恒成立 |
| Riemann 精确参考（P0-F） | 扩展域 u_L=1,u_R=0、激波速度 1/2、熵产生解析值 −(u_L−u_R)³/12=−1/12；N=128/256/512：WENO5 L1 3.47e-3→8.68e-4（rate 1.000，激波主导不是 5 阶）、adaptive L1 ≈ WENO5 的 1.34×、熵产生率在 ±1% 内、激波位置误差 0 cell、ν_num 随 h 减半 |
| σ tracker 速率界（第三评审） | 解析界 r_attack≫|s_shock|/Δx=128/s、r_release≤π²ν_num·Δσ_hyst/Δx²=46/s（实测 ν_num=8.89e-4、滞环 0.08）；16 点扫掠：release 界强约束（越界穿越 2→4→6 次），attack 界在当前滞环下不 bind（50 与 800 同穿越数——该界在本配置中经验上空，诚实记录）；400/10 落在可行盒内 |
| lifting wavelet | Haar：isometry 3.51e-16、PR 0、adjoint dot 1.73e-17；CDF53：PR 2.17e-16、`W_inv≠W*` 0.684；stationary 平移误差 0、逆变换 8.88e-16 |
| transposeable edge | `A↔Aᵀ` 恒等式 2.72e-16；二次型梯度 vs FD 1.48e-10；离散 adjoint ODE 梯度 vs FD 1.77e-10；checkpoint 2424 B（3 状态 × 100 步） |
| scale poles | L=6：pole spread 1024（两种 mode 相同）；mode C：gm spread 1、noise spread 1024；mode gm：gm 1024、noise 1；两种 model-time 误差均 0；不归一化误差 1.0 |
| Gram calibration | Haar isometry 2.27e-2→3.46e-15；CDF53 0.651→3.70e-15；PR 3.2e-16；noise gain 1.023；cond(G_T) 1.094 / 18.649；naive left 0.032，correct left 3.42e-15 |

七条结论：

1. **`ε` 必须是相对场幅的 floor，不是 1e-12 的绝对数。** 光滑极值处一阶差分为零，绝对 ε 会让 `0/0` 饱和成 σ=1；取 `ε=1%·range` 后，step 仍是 1.0，sine 降到 0.1265。这是硬件偏置（ADC/DAC 满量程的小分数），不是数值细节。
2. **对称的 σ tracker 会形成 representation 反馈极限环。** 实测同一 cell 的 σ 穿越 8 次：σ 降 → 中心格式 → 振荡 → S 升 → σ 升 → MUSCL → S 降。把 attack/release 改成 400/10 后降到 2 次。所以 §5.7 的“迟滞 + 速率限制”必须写成不对称 hold time；这是 §9 开放问题“adaptive representation 闭环稳定性”的第一个实测实例。**第三评审的解析界已验证一半**：`tracker_rate_bounds` 给出 r_release≤46/s（实测 ν_num 与滞环代入；评审建议的 ≲15/s 对应滞环 0.008 与 ν=3e-3 的参数选择，定性 bracket 成立但具体数依赖未声明参数），release 扫掠确认越界确实抬穿越数；r_attack 下界在本配置中不 bind（滞环已压制其防的机制），记录为经验上空而非假阳性。
3. **守恒确实是拓扑，但"ΣV²守恒"不是验收。** 混合发生在 face flux 上，质量漂移 2.22e-16；激波位置与 WENO5 完全一致（同一网格）；L2 4.59e-3。一旦激波形成，无黏 Burgers 的 u²/2 必须耗散——正确的验收是逐步熵不等式 + 熵产生率对解析值 −(u_L−u_R)³/12（Riemann 参考实测 ±1%）。Riemann 算例同时给出诚实的收敛阶：L1 rate ≈1.0（激波主导），不是 WENO5 名义的 5 阶。
4. **PR ≠ adjoint ≠ inverse。** CDF53 的 PR 是 2.17e-16，但 `W_inv−W*` 是 0.684；一个变换可以完美重构却给出错误梯度。`TransformContract` 必须同时测三者，不能把 synthesis 当 adjoint。
5. **Gram 标定乘错边不会正交化。** 右 Gram `G_T=T̃*T̃` 必须配右乘 `T_cal=T̃ G_T^{-1/2}`；用 `K T̃` 实测 isometry 0.032（2% 扰动时 0.342），右乘后 3.46e-15。左乘要用左 Gram `T̃T̃*`。两者一般不可交换。
6. **per-scale C/gm 是 trade-off，不是 κ 降低。** L=6 的 pole spread 在两种 mode 下都是 1024；mode C 把跨度放在噪声（1024），mode gm 放在 g_m（1024）。两种 model time 都精确；`κ_t` 不变，变的只是跨度落在哪个物理量上。
7. **undecimated wavelet 的 adjoint 是局部的。** stationary Haar 的平移误差 0、逆变换 8.9e-16；3 级只多 4 个数组，不需要 ↓2 的隐藏 S/H。lifting 的 predict/update 与 `TRANSPOSEABLE` edge 是同一个原语。

### 6.17 编译器证书链：P3→P6 物理时间衔接与 ΔA 量纲一致性（`results/compiler.json`、`pipeline.json`、`constraint.json`）

P0–P6 全链在三锚上跑通（`python3 -m bench.run_pipeline`，34/34，记录 `results/pipeline.json`）。**P3→P6 的衔接有一个必须钉死的方向与一个量纲陷阱**，两者都已修复并写入验收：

| 项目 | 错误写法 | 修复后 | 实测 |
|---|---|---|---|
| 时间缩放方向 | `α·A`（eig 放大 α 倍，poisson scaled bw=2e18，反向） | `α⁻¹·S·A·S⁻¹`（`t_phys=α·t_model` ⇒ `λ_phys=λ/α`；α=1.58e6 ⇒ 硬件比模型慢） | scaled.bandwidth()=B_hw·margin ≤ B_hw |
| 状态缩放 | `A·S⁻¹`（单边右乘，非预条件） | 双边相似 `S·A·S⁻¹`（V=Sx，x=S⁻¹V，左乘 S） | Γ_after=1.60 < Γ_before=2.00 —— 预条件真正生效 |
| 输入缩放 B_v | `B_v = S·B`（缺 α⁻¹——**2026-11 修复**；外部 DC 增益错 S/α 因子：repro dx/dt=−x+u 单位阶跃终值 0.0938 而非 S=0.471） | `B_v = α⁻¹·S·B`（**输入缩放契约**：u 携带物理强迫单位 [x̂/model-time-unit]，u_p(t_p)=u_m(t_p/α) ⇒ 同一个 d/dt_m=α·d/dt_p 代换打在 RHS 的 A、B 两项上；漏掉 B 的 α⁻¹ 内部 eig/A/B/C round-trip 查不出——scaled 系统自身自洽，只有**外部**传递函数 `C_v(jωE_v−A_v)⁻¹B_v` 对拍 `C(jωαE−A)⁻¹B` 才暴露） | `bench/run_scaling.py` 21/21，`results/scaling.json`：repro V_ss=S·x_ss 精确（5.6e-17）、四频点 TF 逐点 ≤4.4e-15、多输入逐通道保持、eig 不变性仍在（≤9.3e-10）；p7 b_ports/netlist `effective_B` 消费的 gain 已含 α⁻¹（p7.py 的 scaled-缺失回退路径同步修复）。替代口径（契约已命名）：端口已是物理速率单位（V/s per V_in，如直接注进物理求和结点的噪声电流）**不**带 α⁻¹、不走 scaled.B；slack 的 B_p 扰动端口与主输入同口径（物理单位扰动源经 scaling.scaled 评估同吃 α⁻¹）。契约显式记录在 `ScalingResult.input_contract`（进 as_dict） |
| ΔA 量纲 | scaled 系统直接吃 model-time 绝对 `delta_A_norm` ⇒ "缩时间白嫖失配裕度"（naive 得 4e-10 伪裕度） | `delta_A_norm` 按**相对失配**解释：`δ_rel=‖ΔA‖/‖A‖ ⇒ ‖ΔA_v‖=δ_rel·‖A_v‖`，`evaluate_sys(scaling=)` 内部自动缩放（`budget.py`） | shear：raw robA=0.0512，scaled=0.0020（真实预条件增益，非伪值）；naive-absolute 会是 3.98e-10 |

关键语义：`sc.scaled` 是 slack/Γ/σ_y/‖H‖∞ 的唯一一致物理时间入口（`evaluate_sys(sys, spec, scaling=sc)`）；不给 scaling 就退化为 model-time 原始系统。poisson 的 `delta_A_norm=None` 是**正确声明**——KCL 松弛网格的失配是泄漏/漂移型扰动（slack.py：判 drift budget 不用 Γ_proj），`robustness_A` 轴对它"不适用/未评估"，其误差进 `floor_bound`（`results/constraint.json`：floor_meas=1.38e-4 ≤ bound 3.63e-3）。

`results/compiler.json`（22/22）覆盖证书聚合的独立验收：结构化拒绝优先于 slack 违规、NaN 轴不记"feasible"、manifold/loop/multirate/sde_cert/constraint 各证据的 verdict 聚合。`results/constraint.json`（44/44）是 T10 KCL 松弛孪生：`τ_relax=C/λ_min` 解析命中（Δ0.00%）、大 C 在 `τ_state` 视界残差 11.31≈‖I‖（椭圆变抛物）vs 在自身 `τ_relax` 收敛到 0.87、`mismatch_floor` 失配地板 ∝1/λ_min。`results/constraint_timing.json`（36/36）钉死 ε_c 语义：外部时标必须显式声明（None ⇒ raise）、纯稳态走 deadline+residual 事实（verdict STEADY/PASS/FAIL，eps_c=None）、耦合 FAIL 进证书（evaluate_sys(constraint=) ⇒ rejected）、纯稳态事实不标 infeasible、畸形证据 ⇒ rejected。

### 6.18 Stage A 收口：回退率口径与"加速器不是求解器"定位（`results/stagea.json`、`ledger.json`、`substrate.json`）

`python3 -m bench.run_stagea`（26/26，记录 `results/stagea.json`）关闭退出条件 4/5/7；`bench/run_substrate`（40/40，`results/substrate.json`）关闭 A7。

**退出条件 5（数字回退率）**：11 个 workload（phase0 五系统 + 3 anchors + shear/VdP/viscoelastic），规则 = certificate infeasible/rejected ⇒ 回退。**原始 6/11 = 54.5% ⇒ 按 §9 风险 4 的口径，这台机器是"连续时间混合加速器/协处理器"，不是通用全模拟求解器**；剔除故意不可行探针（biharmonic、两个 Burgers 激波探针、VdP 平衡点探针）后 2/7 = 28.6%。两个数字都如实进正文——语料偏向失败模式探针，不挑好看的。真实 workload 的回退原因集中在 stiffness（diffusion N=1024、Robertson）与 representation（Burgers），正是 §8 表预测的绑定轴。

**退出条件 7（假设敏感度账本，`results/ledger.json`）**：七项假设每项给敏感度数字（现场算自既有机制或引用 results/*.json，不新增拟合）：κ_hw 0.0486（−4.86% 翻 diffusion N=1024）、alpha_B_hw 0.8（margin=0.8 ⇒ B_hw 缩 20% 翻判）、noise_psd 0.0063、mismatch 5.22/unit ε_dev（floor 线性 + 1/λ_min 放大）、parasitics 2.86°/buffer level、calibration_residual 17.1/unit（isometry 0.342@2% vs 标定后 3.46e-15）、excess_noise_gamma 0.5（σ²=αkT/C 的精确对数敏感度）。每项带 `falsification_observation`——账本不是免责清单，每条都写明"哪个观测会推翻它"。

**A7 异构介质（`results/substrate.json`，40/40）**：MEMS `ω0=√(k/m)`/`Q=mω0/c` 精确 + Duffing 骨架线 vs RK4 <1%；gm/C 锁 ω_ref 后残差≈10ppm 测量底（非 1000ppm 温漂）；photonic MVM shot ∝1/√N、漂移地板 N-无关。**诚实边界：transducer 数字是 TARGET-标注（待 §8B 实测替换），MEMS transducer NaN+"no model fabricated"——A7 验收过的是物理/映射，不是器件实测。**

**Stage A 签字口径**：所有判定与敏感度都引用 `results/*.json` 的实测数；§10 skeleton 是文件地图，`results/*.json` 是机器可读判据。进入 Stage B0 前 §8 表剩余行（airfoil/bore/front 等）是**待补**——它们不改变已验收的三类锚点结论。

---

### 6.19 B2 pre-silicon 原语实测：.noise γ_eff + corner/MC 分离 + P(workload)（`results/b2_extract.json`、`b2.json`）

`python3 -m twin.spice.extract_b2`（~10–25 min，负载相关；24 次 ngspice：5 corner + 2 MC + 9 PVT + 4 joint corner×temp + 3 parasitic + 1 transient）+ `python3 -m bench.run_b2`（26/26 checks）。对象是 B0 tile spec 的行为级原语，**不是** §6.9 的二极管镜像（extract.py）：HBT 差分对（npn13G2，20 µA/边、10k 负载——接口 gm 级）、HBT 电流镜像（20 µA）、LV MOS gm-cell×2（sg13_lv_nmos 10u/0.13u @ Vgs=0.3837 = 10 µA 二极管点、饱和区；gm/Id≈25/V，用孤立 diode sweep 交叉验证——固定 Vgs=0.40 会掉进三极管区，gm/Id 只剩 8.7/V，是第一个被实测抓住的偏置陷阱）、传输门（2u nmos ∥ 4u pmos，Ron=452 Ω）。

**(1) γ_eff 实测（关 §9 风险 3 的一半）**：HBT（VBIC level 9，native model）的 `.noise` 可测：输入参考白噪声 **6.4 nV/√Hz**（1k–1 GHz 积分 var_in=4.07e-8 V²，100 M–1 G 波段全平——带内无可见 1/f 拐点）。分母 g_m=I_C/V_T=773 µS 来自实测 I_C（gm/IC=1/V_T 是精确物理，不是拟合）。相对理想白噪声 4kT/g_m（γ=1 口径）的 white factor = **1.90×**；相对 §6.9 采用的推导假设 γ=1/2 的 γ_eff = **3.79×** ⇒ SNR 乐观 **2.8 dB**。风险 3 的"5–10× ⇒ 7–10 dB"场景在 HBT 接口上**未触发**（比 feared 的小），但 1.90 也不是 1.0——账本 `excess_noise_gamma` 的 α≈1 应按 HBT 实测 1.9 修正。**MOS/传输门的 .noise 不可测**：PSP103 OSDI 器件噪声在 ngspice-44.2 未实现（孤立单管验证 onoise/inoise 谱恒为 0、op `[gm]` 参数恒 ~0 常数）；§6.9 的 MOS 噪声列仍是推导口径、未升级——如实记录，不造数。**替代工具评估与登记决定**：Xyce 不实现 PSP103 OSDI（其 ADMS 库无该模型），BSIM4 是另一套模型卡——两者都验证不了**本 PDK 的 PSP103 噪声**，只会给出另一颗器件的教科书 γ，因此不做"替代验证"；MOS/TG `gamma_eff` 正式登记为 **Stage C1 硅片首测项**（C1 的 noise PSD 行），在此之前保持 4kTγ/gm 推导口径 + `derived-not-measured` 标签，不进 P(workload) 的实测轴。

**(2) corner ≠ MC（三路分布分开跑、分开报）**：
- **corner（全局工艺，mos_tt/ss/ff + hbt_typ/wcs/bcs，器件同向平移）**：固定 Vgs 的 gm_fd 115(ss)/151(tt)/94(ff) µS——亚阈值下 gm 对 corner **非单调**（ff 提升电流同时降低 gm/Id）；单调的是 Id：4.85 < 13.9 < 21.1 µA；实测 AC 极点（Rl≈ro 负载 + 1 pF，−3 dB）：**2.82 / 5.01 / 10.0 MHz**（3.6× 张成）。HBT corner 族移动镜像比 ±0.13%（wcs −0.21% / bcs +0.13%）。镜像比对 MOS corner 完全不敏感（0.0 σ_MC shift——MOS corner 不动 HBT，口径正确）。
- **MC-局部失配（`mos_tt_mismatch` + 逐实例 `mm_ok=1`，48 样本）**：HBT 差分对输入失调 **σ=11.33 mV**（SEM≈1.17 mV）、镜像比 **σ=4.62%**（≈ is 3.8%·√2，机制对账）、gm **σ=1.87 µS（1.25%）**、Ron **σ=7.64 Ω（1.69%）**、I_C σ=4.30%。gm 的 corner shift 是失配 σ 的 **16–25×**——全局工艺平移 ≠ 逐器件失配，两条预算不能合并。
- **MC-全局工艺（`moslv/hbt_stat` libs，24 样本）**：`_stat` 库的 gauss() 是**全局 .param**——所有器件同一抽签：失调 σ≈1e-8 µV（≈0，两器件同漂，这是它与失配的判别性差异）、gm σ=26.9 µS。三路分布的可分性就是它们的定义：corner=系统性平移、stat-MC=全局随机、mismatch-MC=逐器件随机。
- **工具链事实（复现于本机 ngspice-44.2）**：逐实例 agauss 重抽必须用 PDK 自带配方 `setseed`+`mc_source`+实例 `mm_ok=1`（`libs.tech/gnucap/tests/ngspice/moslv/tb_moslv_mc_mm_tt.sp`），复现其 reference `.out` 逐位一致；**纯 `reset` 循环不重抽 per-instance agauss**（实测配对 ΔVgs/ΔVbe≡0）；全局 `.param mm_ok=1` 会破坏电路（镜像比塌到 1e-7）；mos_tt_mismatch 子电路在 mm_ok=0 时二极管点收敛到 0 V（退化），只在 mm_ok=1 下可用。

**(3) CalibratedTwin 换入实测参数 + P(workload passes)**：`twin/calib.py` 的 CalibratedTwin 以实测值替换 nominal（gm/C=1.509e8 rad/s = gm_fd 151 µS/1 pF、镜像比 0.999565、差分增益 −3.654、Ron 451.7 Ω、失调 σ=11.33 mV；全部 source="foreground"，residual=100 ppm 测量底合同）。**P(workload passes)**（积分器极点 |pole/target−1|≤tol，48 个 mismatch-MC 样本的真实分数）：±10% ⇒ **1.00**、±5% ⇒ **1.00**、±2% ⇒ **0.917**。±2% 容差下 8.3% 样本出局——corner 数字（单点最坏值）给不出的就是这个分布口径。

B2 仍是 pre-silicon 行为级：无 PEX/EM/substrate/thermal；corner×MC×cal-loop 组合已由 §6.21 关闭（3 anchors × 9 实测点 × 48 失配 × 4 校准模式），post-layout slack 重跑仍是 B2 后续条目。

Fail-closed B2 extension（`extract_b2.py`、`results/b2_extract.json` 的 `status`/`blocks`）：任何 required block 运行/解析失败即 status≠"complete"，run_b2 报为真失败；PSP103 OSDI 器件噪声保持 `known_limitation`（required=False），绝不为静默 null。**PVT**（−40/27/125 °C 经 ngspice `.temp` × 电源 1.35/1.5/1.65 V）：固定 Vgs=0.3837 下 MOS gm 实测 65.4/150.9/40.0 µS（1.5 V；125 °C 弱反型塌缩），±10% 电源 ±8.5%（pole-cell 偏流由电阻定）；积分器极点 3.16/5.01/19.95 MHz；HBT gm = I_C/V_T(T)，**V_T(T)=k_B(T+273.15)/q**，-40/27/125 °C ⇒ **0.995/0.772/0.582 mS**（1.708× = V_T 比）——早先"HBT gm 全网格 772–773 µS 不变"是固定 300 K VT 的后处理错误（已修；run_b2 用 gm(-40)/gm(125)==VT(125)/VT(-40) 与 1.65–1.77 区间双重钉住）。**偏置头room**：TT/125 °C 与 FF/125 °C 的固定 Vgs + 65 kΩ 负载把 MOS 拉到 Vd=82/26 mV（三极管），联合评估把它们标为 bias-invalid（P=0 + 原因），不包装成"模型预测通过"。**失配 MC 加深 24→48**（setseed/mc_source 配方不变）：vos σ=11.33±1.17 mV、镜像 σ=4.62%±0.48%、gm σ=1.87±0.19 µS、Ron σ=7.64±0.79 Ω（JSON 内含 σ/√(2(N−1)) 误差棒）。**Transient**（10 MHz、dt=200 ps；100 MHz PSP tran 过慢——已记录）：HBT 差分对 10 mV 阶跃 → ΔVout −73.1 mV、1% 整定 0.8 ns、THD 0.272%（h3 主导、偶次抑制）；MOS gm-cell 1 mV 阶跃 → Δ −9.82 mV、整定 146.8 ns ≈ 5 MHz 极点的 4.6τ、摆率 0.303 V/µs、THD 0.227%（h2 主导）。**Pre-PEX 寄生**：尚无版图，PEX 做不了——改为在积分器输出加 C 重测极点（0/10/100 fF → 4.942/4.893/4.495 MHz；实测对数斜率 **−9.496e11 /F** vs 两点预测 **−9.531e11 /F**，差 +0.37%），明确标注"pre-PEX 敏感度估算，非 PEX 结果"；早先 measured/fF 与 predicted/F 的 **1e15 单位错配**已统一到 /F，run_b2 检查同量纲、同号、<2%。

### 6.20 B2 行为级 macromodel：把 SPICE 实测非理想性装进原语模型（`twin/macromodel.py`、`bench/run_b2_behavioral.py`、`results/b2_behavioral.json`）

§6.19 测到了数字，本节把它们变成**可复算的模型**并重算 P(workload)。每条 macromodel 都必须复现自己的实测点，否则验收即 FAIL（32/32，`bench/run_b2_behavioral.py`）。

**(a) 积分器极点是 (gm+g_out)/C，不是 gm/C（headline）。** b2 AC 记录（gm-cell B，Rl=65k，C=1pF）的实测低频增益 A0（9.84/7.30/6.48 @tt/ss/ff）与工作点器件 gm（gm_mos2_param）经 GBW/A0 恒等式给出节点总电导 g_out_total=gm/A0，扣除 65k 负载后得器件自身输出电导 **15.93 µS @TT（ro=62.8k，λ_fit=1.147/V @Id=13.9µA）**。macromodel 预测极点 gm2/(A0·C) 对实测 pole_hz 的偏差为 **−0.54% / +4.94% / +2.26%**（tt/ss/ff，1-pole 拟合残差 ≤2.1e-4 证实模型类）。器件漏电导使 MOS 自系数比 gm/C 口径**系统性 +10.6%**；HBT 接口（λ=0.0138/V，ro=3.63 MΩ）漏电/gmC 仅 **0.036%**。λ_fit 随 corner 张成 **0.66–2.32 /V**——§6.9 的 0.970/V 是二极管偏置下的点值，不是常数。

**(b) P(workload passes) 四口径（48 片实测 MC，n=48）。** (i) 线性参数口径（run_b2.py，gm 轴，σ1.25%）：**1.00 / 1.00 / 0.917**（±10/5/2%，逐点复算一致）；(ii) 非理想联合判据（有效极点 + DC 增益，TT 实测参考）：**0.96 / 0.71 / 0.35**——线性口径漏掉了两条轴：镜像比 σ4.6% 直接进增益、Id σ4.3% 经 λ 进漏电；(iii) 前台增益标定后增益轴收敛到 100 ppm 底但**极点不动**：1.00/1.00/0.98（极点剩余散布 λ·σ_Id/gm=0.45%）；(iv) 若仍以线性 gm/C 为设计目标，漏电把 ±5%/±2% 窗口**清空**（0.52/0.00/0.00）——**target_gm_over_c 必须定义在 (gm+g_out)/C 上，或由标定环 trim gm code 吸收 g_out**。级联 DC 增益从线性口径的 0.9996 降到实测 **0.818（−18.2%）**，同为漏电的系统性效应。轴归因：镜像 4.6% > Id 4.3% > gm 1.2%，系统项 +10.6% 支配均值。

**(c) 噪声与非线性。** HBT 白噪密度由 var_inoise_white/bw 复原为 **6.376 nV/√Hz**，白噪因子 (e_meas/e_{γ=1})² = 1.8961 vs 记录 1.8969（<0.1%）；平谱带内白噪因子经二阶系统传递函数后仍是**纯方差标度**（输出方差比 1.896947），SNR 代价 10·log10 = **2.78 dB**，与 §6.19 的 2.8 dB 一致。MOS 路径保持 4kT·γ/gm 推导口径（7.41 nV/√Hz @γ=0.5，标注 derived-not-measured——PSP103 OSDI 器件噪声在 ngspice-44.2 仍未实现）。差分对 tanh 律对实测扫描斜率的复现差 **+4.6%**（VBIC Early/高注入残余，作为有效域记录）；最差芯片 26.4 mV 失调在工作点吃掉**局部 gm ×0.78（−22%）**——前台 vos 标定是必需项不是装饰。

**(d) 网表承载。** `lowering/netlist.py` 新增 `BehavioralNetlist.nonideal`（空 dict = 理想默认，effective_A/round-trip 不受影响）与 `attach_nonidealities`（键校验、拒绝重复挂载）；`twin/macromodel.b2_nonideal_map` 从 b2_extract.json 为每个积分器生成 ro/leak/A0/v_swing/v_sat/噪声密度+来源/vos/TG 实例参数。v_swing（0.5 V 头room估算）、TG c_par（30 fF）、q_inj_fc（0.5）是**声明的模型参数，不是实测**；TG 串联 RC 极点 352 MHz = 级极点的 71×（模型预测）。

### 6.21 B2 联合闭环：corner × MC × calibration-in-the-loop（`bench/run_b2_joint.py`、`results/b2_joint.json`）

**问题。** §6.19/§6.20 的 P(workload) 只在 TT/27 °C 的 48 组局部失配 MC 上算过，corner 与 MC 是两张分开的表。剩余闭环是把 worst-case corner（SS/FF × −40/125 °C）、48 组局部失配、标定环叠在一起，在 `diffusion-1d`/`poisson-2d`/`burgers-1d` 三个 anchor 上各给一张 P 表。

**做法**（复用现有入口，不新建求解器）：
1. `bench.problems.build_problem` 的三个 bundle → `lowering.compose.run_chain`（boundary→closure→split→scaling→basis→cert→interface→contracts→p7），同一 masked descriptor 端到端。poisson 的 descriptor 先做 **E=I 规范化**（E=c_node·I 使 split 的 det 判据下溢；E x'=Ax+Bu 与 x'=E⁻¹Ax+E⁻¹Bu 是同一个 ODE，时间单位不变），并沿用 run_pipeline 的 `delta_A_norm=None, gamma_M=inf`（robustness 轴由漂移预算判；`run_chain` 新增 `gamma_M` 直通，避免 256 态上 ~15 min 的 resolvent 扫频）。
2. 编译后的物理时间 descriptor `A_v=α⁻¹SAS⁻¹`、`B_v` 是扰动载体。逐点 self 系数比：
   - 积分器 anchor（diffusion/burgers）：对角用 AC 实测 `k_pole=pole_hz(点)/pole_hz(tt,27)`（漏电 g_out/C 的实测比）；输入/耦合边用 `k_gm=gm2(点)/gm2(tt)`（**真小信号 gm**，不是 5 mV 差分 gm——125 °C 下两者差 ~8×，差分 gm 会给出错误符号的角点趋势）。
   - poisson：`A_v=-G/c` 是电导矩阵，对角与耦合同一 gm 原语 ⇒ 全矩阵用 `k_gm`；QoI 在 `T=α·1 ns=237τ_phys` 取稳态，KCL 的 DC 解由电导**比**决定，均匀角点缩放被 `G⁻¹I` 抵消，这是它与积分器 anchor 的判别性差异。
   - 耦合边再乘镜像比 `k_mirror`；读出 `C` 视为已标定（不扰动）。
3. 局部失配：48 组实测 (gm, Id, mirror) 的**相对**因子，按 `(state+k) mod 48` 循环分配到 n 个状态（bootstrap；不是逐状态独立抽样，声明的）。mismatch 只在 TT/27 测过，叠加到角点是声明的组合假设。
4. QoI：编译后 descriptor 的零初值阶跃响应 x(T)（diffusion T=α·1；burgers T=α·0.3；poisson 的 1 ns 是**模型时间**求解 deadline，P3 后 T=α·1 ns=1.543e-4 s=237τ_phys，取稳态比值口径），相对 L2 误差 vs 名义。步响应用增广矩阵 expm（奇异 A 不求逆）；固定数值平移 `shift=max(0, max Re λ_nom)`，`x(T)=e^{shift T}expm((A−shift I)T)b`，指数溢出或非有限即判失败，不造有限数。**坐标错配修复**：早先把 1 ns 当物理时间，T=0.0015τ_phys，响应落在预松弛区（x≈bT），电导比从未进入 QoI，poisson 的角点误差退化成输入增益误差；改为模型时间后 `run_b2_joint` 的 J11（比值口径跨角点平坦）才真正成立。
5. 校准模式（残差下限统一 100 ppm 测量底）：
   - `uncal`：角点 × mismatch 直接进系数。
   - `no_track`：在测试条件（同一颗芯片、27 °C）前台 trim 掉工艺角与静态失配；运行时只有温度偏移，无背景跟踪 ⇒ 残差 = 同工艺 27 °C 参考下的温度偏移。
   - `bg_track`：`no_track` + replica 背景跟踪，刷新粒度**声明** 1 °C ⇒ 残差 = 温度偏移 × min(1, 1 °C/|T−27 °C|) + 地板。
   - `full`：工作点连续 trim 的上界 ⇒ 100 ppm 地板（bracket，不是实测量）。
6. 偏置头room 门：固定 Vgs=0.3837 + 65 kΩ 在 TT/125 与 FF/125 把 MOS 拉到 `V_d=1.5−I_d·65kΩ ≈ 82/26 mV`（三极管）⇒ `bias_ok=False`，该点所有模式 P=0 并在 JSON 里带原因。

**结果**（P at 1%/2%/5%，n=48；`k_diag`/V_d/逐点误差见 `b2_joint.json`）：

**diffusion-1d（速率定义：极点/漏电）**

| point | k_diag | V_d/mV | bias | uncal 1/2/5 | no_track 1/2/5 | bg_track 1/2/5 | full 1/2/5 | err_uncal med |
|---|---|---|---|---|---|---|---|---|
| tt_27 | 1.000 | 596 | ok | 0.54/0.92/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.95% |
| ss_27 | 0.562 | 1185 | ok | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 56.23% |
| ff_27 | 1.995 | 128 | ok | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 34.25% |
| tt_-40 | 0.631 | 1108 | ok | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 41.87% |
| tt_125 | 3.981 | 82 | **FAIL** | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 12.00% |
| ss_m40 | 0.501 | 1445 | ok | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/1.00/1.00 | 1.00/1.00/1.00 | 89.81% |
| ss_p125 | 1.122 | 464 | ok | 0.00/0.10/0.94 | 0.00/0.00/0.00 | 0.00/1.00/1.00 | 1.00/1.00/1.00 | 3.59% |
| ff_m40 | 1.122 | 517 | ok | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 15.88% |
| ff_p125 | 22.387 | 26 | **FAIL** | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 16.25% |

**poisson-2d（比值定义：G⁻¹I 稳态）**

| point | k_diag | V_d/mV | bias | uncal 1/2/5 | no_track 1/2/5 | bg_track 1/2/5 | full 1/2/5 | err_uncal med |
|---|---|---|---|---|---|---|---|---|
| tt_27 | 1.000 | 596 | ok | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.52% |
| ss_27 | 0.440 | 1185 | ok | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.52% |
| ff_27 | 1.351 | 128 | ok | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.52% |
| tt_-40 | 0.585 | 1108 | ok | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.52% |
| tt_125 | 1.129 | 82 | **FAIL** | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.52% |
| ss_m40 | 0.103 | 1445 | ok | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.52% |
| ss_p125 | 0.970 | 464 | ok | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.52% |
| ff_m40 | 1.166 | 517 | ok | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.52% |
| ff_p125 | 0.852 | 26 | **FAIL** | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.52% |

**burgers-1d（增长指数：pre-shock）**

| point | k_diag | V_d/mV | bias | uncal 1/2/5 | no_track 1/2/5 | bg_track 1/2/5 | full 1/2/5 | err_uncal med |
|---|---|---|---|---|---|---|---|---|
| tt_27 | 1.000 | 596 | ok | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 14.65% |
| ss_27 | 0.562 | 1185 | ok | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 81.56% |
| ff_27 | 1.995 | 128 | ok | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 84.62% |
| tt_-40 | 0.631 | 1108 | ok | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/1.00/1.00 | 1.00/1.00/1.00 | 60.02% |
| tt_125 | 3.981 | 82 | **FAIL** | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 95.59% |
| ss_m40 | 0.501 | 1445 | ok | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 98.00% |
| ss_p125 | 1.122 | 464 | ok | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/1.00 | 1.00/1.00/1.00 | 67.37% |
| ff_m40 | 1.122 | 517 | ok | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 155.54% |
| ff_p125 | 22.387 | 26 | **FAIL** | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 99.33% |

† `bias=FAIL`：固定 `Vgs=0.3837` + 65 kΩ 在 TT/125 与 FF/125 把 MOS 拉到 `V_d=82/26 mV`（三极管）；P=0 是偏置门，不是模型判决。

**读法**（每格 P at 1%/2%/5%，n=48；`k_diag` 是逐点 self 系数比，`V_d` 是偏置门）：

1. **uncal（不做任何标定）**：角点直接打穿 1% 窗。diffusion 在 27 °C 只有 TT 过半（0.54/0.92/1.00），SS/FF 全 0（中位误差 56.2%/34.3%）；全部温度点（TT PVT + joint SS/FF × −40/125 °C）全 0，ss/−40 中位 89.8%。burgers 连 TT/27 都是 0（中位 14.7%），增长指数把失配与角点一起放大（ff/−40 中位 156%）。poisson 是唯一 uncal 就 1% 全过的锚点（48/48，中位 0.52%）：QoI 取在 237τ_phys 的稳态，均匀角点/温度缩放被 `G⁻¹I` 的比值抵消，只剩逐节点电流源失配。**这是"比值定义 vs 速率定义"的判别性差异，不是 poisson 更宽松。**
2. **no_track（测试温度前台 trim、无背景跟踪）**：27 °C 所有工艺角回到 1.00（trim 把该芯片的工艺角与静态失配吸收）；diffusion/burgers 的温度点全部 0（−40/125 °C 的 gm/漏电漂移无人跟踪）。poisson 在所有点保持 1.00——比值口径下温度对稳态无影响。
3. **bg_track（1 °C 刷新 replica）**：把温度残差压到 `frac×1/67`。diffusion 的 ff/−40 与 tt/−40 全恢复；ss/−40 与 ss/125 在 1% 为 0（残差 ≈0.77×1/67≈1.15%），2% 起通过。burgers 的同一残差被增长指数放大：ff/−40 与 ss/−40 到 5% 仍为 0，ss/125 到 5% 才通过。poisson 全 1.00（均匀残差再次被比值抵消）。
4. **full（连续 trim 上界）**：所有 bias-valid 点 1%/2%/5% 全 1.00。QoI 误差：diffusion 9.995e-5、burgers 9.12e-5、poisson 9.76e-16（均匀 100 ppm 在比值口径下精确抵消）。**本 lane 没有看到增长指数放大 100 ppm 地板**；早先"~4%"的估计来自单轴扰动，不是均匀系数缩放（`validity_limits` 已按实测改写）。
5. **标定环买到什么**：`uncal → no_track` 买"工艺角 + 静态失配"（27 °C 全部恢复）；`no_track → bg_track` 买"温度漂移"（−40/125 °C 的失败基本消失，残余 = 1 °C 刷新粒度 × 温度行程）；`bg_track → full` 买"刷新残差 + 工作点漂移"的上界。三列之差就是 §6.20 标定环的预算，`full` 列是 bracket，不是实测量。
6. **偏置门优先于模型**：tt_125/ff_p125 的 P=0 不是"模型预测失败"，而是固定偏置网络失效；要先换偏置（或降 Vgs）再谈这一点的角点通过率。

**全仓验收（2026-11-20）**：`python3 -m bench.run_all` 顺序跑完 25 个套件，无失败、无跳过，1324.6 s；计入 check 计数器的 868/868 项全部通过（`run_twin_experiments` 是无 check 计数器的实验驱动）。记录在 `results/run_all.json`。

---

## 7. 分阶段落地（每阶段都有可证伪的验收）

### Phase 0：数字孪生 + 可行性判定器（无硬件）—— 判定器已验收
- descriptor + 广义 Lyapunov（QZ）+ Hamiltonian 二分：已实现于 `analysis/`，见 §6.10
- 5 个验收算例 + 四张图：已跑通（`bench/run_phase0.py`）；§8 的 20 个 benchmark 仍待补齐
- **验收**：diffusion(N=64/1024)、biharmonic(N=128)、Burgers、Robertson 的判定与解析对照 30/30 通过（±3 dB / ±0.5 decade，+7 条奇异-DAE H2 验收：index-1 消去 σ_y²=0.5、feedthrough/index-2 判 ∞）

### Phase 1：单 island（4 积分器 + 1 小 MVM + 比较器）
- **前端验收（`bench/run_reduce.py`，25/25 通过）**：Lorenz/Duffing 落到同一 sparse polynomial IR（3/4 状态、degree 2/3、各 7 个 monomial）；sin/cos 增广的 s²+c² 漂移 9.9e-15，把 ṡ 乘 (1+1e-3) 后升到 1.0e-3；**标准 HH 落到 rational IR（6 个 exp 增广 + 3 个分母），不是 polynomial**；Bessel J0 在 t≥1 可增广（相对误差 5e-15），t=0 拒绝；Gamma 拒绝。见 §6.13。
- **验收**：实测 σ² = α kT/C 与 σ²τ = α kT/g_m 各在 ±3 dB 内；实测 Q 容差规则（Q = 10/50/200 三档，验证 pole shift ∝ Q·Δ）

### Phase 2：spectral tile（8 模态扩散）
- **验收**：τ_k ∝ 1/k² 实现误差 < 5%；预测 κ 与实际一致；与数字参考解 L2 误差符合 σ_y 预测；实测 §6.7 的 E_smooth(N) 曲线（决定 N 该买多少）
- **约束 fabric 验收**：电阻/gm mesh 解 2D Poisson，残差 <1e-3；实测 τ_relax ∝ 1/λ_min(A)；KCL 节点 |ΣI| <1e-6

### Phase 3：flux tile（1D Burgers 含激波）—— 按 §6.1 五元组验收
- 守恒：Σ V_i（质量）漂移 < 1e-4；**熵：d/dt ∫u²/2 ≤ 0 恒成立，且形成 shock 后熵产生率落在解析值 −(u_L−u_R)³/12 的 ±10%**（P0-F 修正：无黏 Burgers 的 ΣV_i² 在 shock 后必须耗散，强制"守恒"是错的判据）
- **W_d → ν_num 反演**：实测剖面拟合 tanh，与预测值一致（±20%）
- x_s(t) 轨迹误差（wave drag 的代理量）
- **接口守恒对照实验**：单通量源+镜像 vs 两边各算一遍，泄漏量应差 2 个数量级以上
- limiter 前向不变性（检查边界向量场；可能失败，失败要给出失效边界）

### Phase 4：分层路由 + 换基 + 事件
- **验收**：η_route 与 ‖G_loop·Δ‖∞ 的预测-实测误差；一次真实 basis switching（smooth↔shock）的切换瞬态与噪声代价；实测重构注入噪声是否落在 §6.2 的 45 nV/√Hz 量级

### Phase 4.5：adaptive representation 与 TransformContract（§5.7–§5.9）—— 已验收 35/35

`twin/transform.py` + `bench/run_transform.py`；结果见 §6.16。核心实现：`smoothness_indicator`（相对场幅的 ε floor）、`SigmaTracker`（attack/release 不对称）、`blend_face_flux`（同一 face）、`LiftingWavelet`（Haar 与 CDF53）、`gram_calibrate`（右/左 Gram）、`adjoint_ode_gradient`（离散 RK4 map 的转置）、`scale_poles`（mode C/gm）。

- **4.5A WENO 三算子**：Burgers 移动激波，`S→σ→F_σ` 与 WENO5 参考对比。
  验收：守恒量漂移 <1e-4（混合发生在 face flux 上，不是先混状态）；`σ` 在阈值附近无极限环（迟滞/速率限制生效）；shock 离开后 `σ→0` 且谱表示恢复；QoI（`W_d`/`x_s`）与 WENO5 差 <10%。
- **4.5B TransformContract**：3 级 lifting wavelet，测 `W`、`W_inv`、`W_star`。
  验收：`isometry_error<0.5%`、`PR_error<0.2%`（§5.9 的合同字段）；adjoint 内积测试 `⟨Wx,y⟩=⟨x,W^*y⟩` 误差 <1e-3；`noise_gain<1.03`；`W_inv` 与 `W_star` 分开验证（biorthogonal 情形不得混用）。
- **4.5C adjoint 复用**：线性系统 forward/backward，transposeable edge 只重配置方向。
  验收：adjoint 梯度与有限差分差 <5%；`A↔A^T` 后模拟残差与数字转置一致；时间反演需要 checkpoint 的存储量可报告。
- **4.5D scale-dependent `g_m/C`**：扩散方程在 wavelet 基下按 scale 归一化 pole；同时验证 mode C（调 `C_j`）与 mode g_m（调 `g_m,j`）。
  验收：校准后每级 pole 残差 <5%；相对增益误差 <1%；两种 mode 的 `g_m` 范围 / 噪声范围 trade-off 与预测一致；model time 恢复误差 <1%。
- **4.5E Gram calibration**：注入已知扰动，测 `G_T=T̃^*T̃`，数字端 `G_T^{−1/2}`。
  验收：校准后 `‖T_cal^*T_cal−I‖<0.5%`；`G_T` 的条件数与噪声放大一起报告；校准前后 adjoint 梯度误差下降一个量级以上。

### Phase 5：explicit singular state（shock fitting）
- 1D nozzle 移动激波 + 跨声速翼型截面：x_s(t) 与 RH 解析解对比
- 拓扑事件：生成 / 合并 / 消失各触发一次，验证 reset map 与事件后重初始化（这里最容易崩）
- 同 QoI 对比 Phase 3 的 capturing：验证"M_int 小、T_int 长时，fitting 用更少状态达到同等 W_d"

### Phase 6：随机动力学与反问题（SDE 扩展）—— 正问题已验收 27/27

`analysis/sde.py`（二阶矩 Sylvester + 均方稳定 + Itô↔Stratonovich 修正 + `noise_realizability`）、`twin/noise.py`（OU 滤波色噪声 + replay + ρ_ij）、`bench/run_sde.py`（OU/GBM/双势阱/CRN-moment 参数恢复），见 §6.11。

- **6A 加性 SDE**：OU 的 Var/ACF、噪声带宽、T8 标定；验收见 §6.11。
- **6B 乘性语义**：GBM 的 Itô/Stratonovich 分离；修正漂移的符号与幅度；验收 ±10%。
- **6C 反问题**：CRN 有限差分 + EnKF/SMC + adjoint，三条都在数字孪生里同时实现；硬件验收先做低维参数恢复，验收：θ 恢复误差 <10%，且 CRN 的方差比独立噪声低 10×。
- **6D Langevin/双势阱**：直方图 vs e^{−V/T}；KL < 0.05；T 的标定。
- **6E 研究项**：模拟 adjoint 的乘性噪声版本、neural SDE、低维 Fokker–Planck 验证；不作为 Phase 6 的验收承诺。
- **总则**：每个实验先数字孪生（`bench/run_sde.py`），再硬件；硬件阶段必须报告 τ_c、B_noise、D 标定残差三个数。

### Phase 7：pre-silicon 与第一颗 characterization chip

验证金字塔（每层只回答一个问题）：

    PDK/device model
      → transistor SPICE（primitive 的 gain/pole/noise/offset）
      → primitive characterization database
      → layout + PEX/EM/substrate/thermal（cluster）
      → calibrated behavioral model（当前芯片的数字孪生）
      → full workload（不是全芯片 transistor SPICE）

五条验收：

- **corner ≠ Monte Carlo**：global variation 与 local mismatch 分开跑，再跑 combined；每个 MC sample 是一个 hardware model，最后看 P(workload passes)，不是只看 3σ gain。
- **噪声不能靠 corner**：thermal/1f/shot/supply/substrate/digital coupling 分开表征；多节点要 S_ij(f)，不只 S_ii。
- **transient noise**：comparator/switch/SDE 的事件抖动 σ_t 比输入 PSD 更有意义；`.noise` 只覆盖线性化 PSD。
- **calibration-in-the-loop**：SPICE 给出 g_m/C，pilot 测量，DAC 更新，再测；报告 post-calibration 分布，不是 raw。
- **第一颗芯片是 Characterization Vehicle，不是 processor**：primitive bank + 不同长度/扇出/模式的 route + C bank + 器件 bank + replica/温度/噪声/loopback + 同一 cell 的 32 份不同位置。目标是把“PDK model → measured primitive → calibration residual → system error”闭环跑通，而不是跑 CFD。

**已实现（B1，`twin/calib.py` + `lowering/p7.py`，验收 `bench/run_calib.py` 62/62，`results/calib.json`）**：前台标定残差 = **测量噪声底**（合同：不低于底；FSM IDLE→STIMULATE→MEASURE→CORRECT→VERIFY）；VERIFY 用 **held-out** 验证点集（`run_foreground(..., verify_stimulus=, verify_reading=)`）——correction 作用于独立读数、残差须 ≈ 噪声底；无验证集时回退训练点集并如实记录 `independent=False`（训练残差，降级口径）。过拟合负控制：2 点拟合 2 参数（dof=点数）把训练残差吃到底、看似合格，held-out 残差 ≈29×底被 VERIFY 抓住。**终态语义（P1-FSM）**：IDLE 只是"循环结束"，不是"合格"——结果必查 `result.accepted`/`terminal`（ACCEPTED=VERIFY 通过；FAILED=重试耗尽/最后一次 VERIFY 失败或没有新测量可用，accepted=False，`CalibratedTwin.apply_foreground` 拒绝折入——这是 P7 只消费 CalibratedTwin 所以那道门就是 certificate-cal 一致性的关口）；重试**不是旧数据重拟合**：失败后拟合集/验证集清空，重试消耗 `run_foreground(..., retry_sets=[...])` 里的新测量（重新激励+重新测量，history 的 `read_digest` 可核对），无新测量 ⇒ 立即 FAILED，不悄悄重拟合。后台标定残差 = `rate·interval/(1+loop_gain) + 底`（与 A7 `gm_c_calibrate` 同语义；间隔加倍 ⇒ 漂移残差加倍——诚实的调度权衡）；温漂孪生更新 `drift_update(p, ΔT, tempco)` 把当前这颗芯片的实测参数交给 compiler（不是 TT_25C）。P7 把 (Discretized, Certificate, ScalingResult, CalibratedTwin) 编成 per-element 配置 + 校准计划；噪声 σ_scale 不自造裕度；`residual_budget = floor/noise_axis` 是**启发式**（跨轴 dJ 敏感度分配是 B2 refinement，plan notes 里注明）；噪声轴 ≤1 才承诺预算，>1 置 None（此时前台计划与 update_interval 也不排程，不编造）；证书判 infeasible 的量 **flagged，不悄悄计划**。P0-3 系数图已编码进 config（不再只是一张速度 spec）：每积分器**各自**的有效自系数 `a_ii` = 缩放后 `A_v[i,i]`（`target_gm_over_c = |a_ii|`——diag(-1,-100) 出两个不同系数，不再收敛到全局 max|λ|/alpha；tile 速度 spec 保留在 `source.tile_speed_spec`）、每条非对角 A_v 非零元一条 `coeff_edges`（物理增益 + 引用 `Config.route_contracts` 契约序号）、`b_ports`/`c_taps`（B/C 图逐元展开）、C_bank/gm 码位（`C_BANK_CODES` 档位显式选中）、逐元素 cal bindings；`lowering/netlist.py` `to_behavioral()` 把 config 转成结构化行为级网表（integrator 带极点/edge 带增益/噪声 scale），其 `effective_A()` 重构出的特征值 ≈ scaled 描述符特征值（round-trip，`bench/run_p7_netlist.py` 26/26，`results/p7_netlist.json`；diag 系统与 4×4 稠密系统误差 ≤1.3e-15）。仍是 plan + 行为级模型，不是 SPICE 网表；per-edge route-contract 已含（`lowering/route_contract.py`，验收 `bench/run_route_contract.py` 53/53，`results/route_contract.json`）：`Config.route_contracts` 逐边携带四判据契约——需求（omega_max=max|λ|/alpha 物理带宽；feedback 边 omega_c + tau_max_abs=phase_budget(ωc, φ_reserve)，有限非 inf）+ 代表性 candidate 分布式 RC 线的 `route_closure` verdict（强制长线 ⇒ verdict False，pad_can_fix 在 feedback 边诚实为 False——deskew 负控制的编译层重演：matched 延迟 ≠ instantaneous）+ `edge_sensitivity` 的 1/|A_ij| 延迟预算排序（**物理单位合同**：S_tau/tau_tol/hf_tau_tol 一律在物理时间轴评估——J_p = E_p⁻¹·A_p（ScalingResult.scaled 的 E_v/A_v；无 scaling 时 alpha=1）、ω = omega_max 物理、weight/hf_tau_tol 读物理系数 A_v[i,j] = α⁻¹·SAS⁻¹[i,j]；alpha≠1 时 S_tau 缩 1/α、tau_tol/hf_tau_tol 缩 α，bench 逐项核对解析缩放比；奇异 E_p 不朴素求逆——J_p 取有限（微分）子空间上的 pinv 提升，代数行的边敏感度如实为 0/tau_tol=inf，"无有限动力学"，pinv 失败 ⇒ "sensitivity not computable"，不造假数；`bench/run_route_contract.py` 53/53）；纯对角/无描述符 ⇒ 空契约表（真"无布线"），NaN/inf 边 flagged 不静默。契约是**需求+代表性可行性**，不是布线结果（那是 P5/B2）；聚合 "route-edges" ElementConfig 保留为向后兼容摘要。

**投入顺序**：behavioral → primitive SPICE → PEX/EM/substrate/thermal → calibration loop → 小规模 test chip → 完整 VLSI。分立器件 PCB 原型只验证 compiler/time scaling/校准流程，不验证片上 mismatch、kT/C、switch 寄生和 substrate coupling。

### Phase 7.5 / B0：架构冻结（process + tile counts + 冻结规格书）

B0 把 §3 的 10-tile 表收敛成一颗可流片的 characterization vehicle。冻结的不是"最高性能"，而是**让 Stage C 每个待测量都有专用结构**（§7 Phase 7 / B3）。三条输入：§6.9 的 λ 表、§2.1 的尺度律、§1.2 的 slack 轴。**所有未进 `results/*.json` 的数字标 TARGET——B0 的功率/面积是一阶估算，不是实测。**

**(a) 工艺冻结 = IHP SG13G2 SiGe BiCMOS（130 nm）。** 依据是实测 λ 表（`results/device_extract.json`），不是偏好：

| 器件 | λ = gds/ID | r_o | 用途判据 |
|---|---|---|---|
| SG13G2 npn13G2 **HBT** | 0.0138 /V | 3.61 MΩ | 接口镜像/缓冲/line driver（λ 比 IHP MOS 小 70×，镜像系统误差 ~590× 小） |
| SG13G2 sg13_lv_nmos | 0.970 /V | 65 kΩ | **不适合接口镜像**（§6.9：58.6% 镜像误差）；只做低阻节点 SUM/开关/digital |
| MIM（MOS 电容阵） | — | — | 存 state（`σ²=αkT/C`，C 是资源） |
| CMOS digital | — | — | supervisor / 校准 FSM / 配置 |

分工与 §3.1 一致且被 λ 表钉死：**HBT 做高速电流接口与缓冲（它快、ro 高、g_m/I_C≈38.7/V），MOS/MIM 做状态积分，CMOS 做 supervisor。** 代价记账：V_BE 温度系数 −2 mV/K ⇒ 接口偏置必须进 background 校准环（`twin/calib.py` 的 replica-locked drift 语义）；HBT 不存 state（§3.1）。同一 die 上有 HBT+CMOS 是选 SG13G2 而非纯 RF CMOS 的决定性理由（§6.9：接口层排序 HBT≫gf180>sky130≫短沟道 MOS）。

**(b) Tile 数冻结（第一颗 characterization vehicle，不是全 fabric）。** 每类至少一个实例 + §7 Phase 7 的"同一 cell 的 32 份不同位置"用于 mismatch 地图：

| Tile | 数量 | 冻结规格（(E,A,B,C) 抽象 + 关键数） | 绑定 slack 轴 |
|---|---|---|---|
| T1 State | 4–8 积分器 | C·V̇=I；C_bank = 1 pF（`C_BANK_F`，`lowering/p7.py`）梯度 1×/4×/16×；`target_gm_over_c = max|λ|/α`（P7 实测口径） | noise `σ²=αkT/C`、leakage |
| T2 Algebra | 1 簇 | 电流域 SUM/MUL/tanh（低阻节点）；HBT/MOS | mismatch Γ·ΔA |
| T3 Operator | 1 个 ≤64 MVM | 可编程权值；失配 ∝1/√(WL)；**cap 256，本 chip 用 64**（§3 T3） | robustness_A |
| T4 Stiff | 1 组极点阵 | ż_k=−λ_k z_k；τ 覆盖 1e-8…1e-3（§3 T4） | stiffness κ_hw=1e6（账本敏感 4.86%） |
| T5 Flux | 1 条 1D 链（~64 cell） | cell C + NL flux + limiter；单通量源镜像（HBT） | conservation、representation |
| T6 Event | 1 比较器+锁存+S&H | 迟滞防 Zeno；抖动 σ_t 进统计 | event jitter |
| T7 Route | 可编程长度/扇出阵 | `H_ij(s)=H_wire·∏H_buf·e^{−sτ}`；wire 分布式 1/cosh√(sRC)；buffer 单极+噪声；AllPass pad；**per-edge contract 由 `route_contract.py` 提供四判据**（B1 落地） | routing η_route、phase、feedback ωc |
| T8 Noise | 1 bank（m 独立源） | 物理热噪声 + replay PRNG+DAC；PSD/τ_c/ρ 可编程；`B_noise≥100·B_req`（§3 T8） | noise realizability（账本 0.63%/unit） |
| T9 Statistics | 1 累加器组 | 片上均值/协方差/直方图；只读 O(p²) | readout budget |
| T10 Constraint | 1 稀疏 gm mesh（~16×16） | `C_y ẏ=−Ay+b` 松弛解 Poisson/KCL；`ε_c=τ_constraint/τ_external`（外部时标显式声明，§2.8 #3）；λ_min 判收敛；纯稳态 ⇒ deadline+residual 事实 | relaxation ε_c、mismatch floor ∝1/λ_min |
| Supervisor | 1 | CMOS FSM + DAC + ADC + replica/reference；校准 FSM 见 `twin/calib.py` | 全轴校准残差（账本 17/unit） |

**(c) 冻结规格书抽象。** 每个 tile 在 compiler 里暴露 `(E,A,B,C)` descriptor 片段 + `‖H‖₂`/`‖H‖∞`（由 `analysis/` 在其 (E,A,B,C) 上算）+ 功耗 + 面积。功率/面积为 TARGET（§2.1：1000 节点 @10µV@10MHz ⇒ ~0.1 W、~3 mm² C——本 chip 只有几十节点，功率/面积**不是瓶颈**，瓶颈是 κ_hw、route 寄生、校准漂移）。**B0 退出 = 这张表 + 每 tile 的 (E,A,B,C)/‖H‖₂/‖H‖∞/power/area 五行规格冻结**；具体数值待 §7 Phase 7 的 primitive SPICE/PEX 填入，B0 只冻结结构与口径。

**(d) B0 的诚实边界。** (1) 第一颗是 characterization vehicle，不是 processor——§7 Phase 7 已冻结。(2) 功率/面积是 §2.1 一阶估算（TARGET），不是实测；§9 风险 3 已半关：HBT 接口的 γ_eff 已 `.noise` 实测（white factor 1.90× vs γ=1 口径，3.79× vs 推导 γ=1/2，SNR 乐观 2.8 dB，§6.19）——"5–10×"场景在 HBT 上未触发；MOS 一侧因 PSP103 OSDI 器件噪声在 ngspice-44.2 未实现仍是推导口径，实测要等 SPICE 器件噪声支持或硅测（C1）。(3) 工艺包络不只看 f_T：κ_hw=1e6 是 nominal，账本敏感度 4.86% 意味着实测窗口若只到 1e4，diffusion N=1024 这类 stiffness-bound workload 翻判——**κ_hw 必须在 C1 实测，不能只信 PDK**。(4) per-edge route contract 由 B1 `lowering/route_contract.py` 在四判据（gain/phase/noise/feedback-ωc）上发，T7 的数量只是容量冻结。(5) B2 已把 primitive 级 corner/MC/噪声/P(workload) 数字从 TARGET 升级为实测（§6.19，`results/b2.json`）：T1 极点 5.01 MHz@TT（corner 张成 2.82–10.0 MHz）、接口失调 σ=11.33 mV（48 样本）、镜像比 σ=4.62%——功率/面积仍是 TARGET，但失配/噪声/极点的预算轴不再靠 3σ 假设。提取是 fail-closed（b2_extract.json 的 `status`/`blocks`，required block 失败 run_b2 即非零退出），覆盖 PVT 与 transient 激励，并以加 C 的 pre-PEX 敏感度探针代替 PEX（尚无版图）；primitives 的偏置源是理想源，故偏置网络自身的电源敏感度仍是 B3 测量项。

### Phase 8：heterogeneous substrates（研究阶段，按 §3.7 的优先级）

不做"三种 substrate 一起上"。每次只加一个域，并且先有 `substrate.py` 的 constitutive law + transducer cost，再有数字孪生。

- **8A CMOS + MEMS**：resonator bank（2–4 个模态）+ CMOS programmable coupling/readout。
  验收：实测 `ω0` 与 `√(k/m)` 一致（±10%）；Q 与 `mω0/c` 一致；Duffing `α` 的符号与幅度可辨识；用 `ω_ref` 校准 `g_m/C` 的残差 <1%；2×2 coupled resonator 的模态频率与 `Kφ=ω²Mφ` 对比 <5%。
- **8B CMOS + photonics**：先单波长 link 的 `S_n`/linearity/thermal drift，再 WDM 的 `y=Wx`。
  验收：PD 电流对光功率的线性度 <1%；ring 热漂移在闭环校准后 <1 pm；4×4 MVM 误差 <5%；`η_route` 与 §6.15 的电子 route 同一量级。
- **8C CMOS + microfluidics**：不做 solver，做 1D reaction-diffusion cell + 电子模型闭环。
  验收：`θ` 的在线恢复误差 <10%（与 Phase 6C 同口径）；`e=c−ĉ` 收敛到测量噪声底；蒸发/温度慢漂移被 background calibration 跟踪。
- **总则**：硬件阶段必须报告每域的 `κ_hw`、transducer 的 gain/噪声/带宽/能量四个数，以及跨域边界的 sample/锁相误差。任何一个没进 §1.2 的 slack，就不算验收。

---

### Stage A 收尾：A4/A8/A9 closeout（`bench/run_stagea.py` + `bench/ledger.py`）

Stage-A 退出条件 1–3 由 §6.10/§6.13 与 `bench/problems.py` 三锚覆盖；本小节记录 4/5/7 的关闭证据（复跑 `python3 -m bench.run_stagea`，26/26，结果 `results/stagea.json`）：

- **退出条件 4（负控制按预测原因失败）——三个新判定**：
  - 非正规 shear（A=[[-1,10],[0,-2]]，§8 "非正规 shear + block Schur" 行）：Γ=5.123（与 `results/skew.json` 的 gamma_ideal 对账一致），谱本身良态（对角 Γ=1）——Γ≫1 是基/非正规性，不是谱。换基对照实测：Gramian 对角 scaling（P3 预条件）κ 不变（eig 缩 α 不变性 0.00e+00）；**特征向量基把 Γ 降到对角值 1.000，但 κ(Φ)=cond(V)=20.05，失配总预算 Γ·κ(Φ)=20.05 比原基 5.12 更差**——"禁止病态特征基、用块正交基"是定量结论不是偏好；正交 Schur 基（本例 A 已上三角，基=恒等）保留诚实 Γ。六轴证书下 shear 在已评估轴可行（robustness_A=Γ·ΔA=0.051<1）：Γ≫1 的瞬态增长风险在六轴之外，A8 判定签名照实记录，不判回退。
  - Van der Pol（μ=1，平衡点线性化 λ=0.5±0.866j）：stable=False 是**预期**（同 Burgers anchor 的口径）；前端 reduce 归类 polynomial；RK4 证实极限环 max|x|≈2.009。六轴签名：不稳定极点使 κ_t 返回退化 inf ⇒ stiffness 轴"违规"——但语义是工点不稳定，不是删模态/多速率问题，判定必须是证书级数字回退/事件调度（§8 HH 行同款 hybrid）。环上工作点是时变（Floquet）线性化，不在判定器范围——记录的限制。
  - Lorentzian 记忆粘弹性（§8 带核粘弹性行）：Maxwell/SLS 内变量 Markov 嵌入，n = 2 + m（m 个弛豫模态，m=4 时 n=6，≤64 保持有限维）；力→应变频响与解析柔量 J(s)=1/(Ms²+G_e+Σ g_k s/(τ_k s+1)) 对到 4.5e-16；Γ(m=4)=14.3 有限；模态跨度 3 decades 时 κ 从 100 涨到 1e4（极点跨度是代价，与 §5.9 pole-spread 同型）。幂律核 t^{-1/2} **没有有限 Markov 嵌入**（Hölder 奇异，同 Gamma 的拒绝口径）：只能用 Lorentzian 和逼近（相对 L2 误差 m=4→12：0.181→0.013，"辅助状态数 vs 精度"），这是已记录的限制，不是隐藏失败。
- **退出条件 5（数字回退率已测）**：tally 覆盖 phase0 五系统 + 3 anchors（`results/phase0.json`/`pipeline.json`）+ shear/VdP/viscoelastic 共 11 个 workload，规则 = certificate infeasible/rejected ⇒ 回退（"no violation on evaluated axes" 记模拟，证书只覆盖已评估轴）。**原始 6/11 = 54.5% ⇒ 按 §9 风险 4 的口径，这批语料上机器是"加速器而非求解器"；剔除故意不可行探针（biharmonic、两个 Burgers 激波探针、VdP 平衡点探针）后 2/7 = 28.6%。** 两个数字都报告——语料偏向失败模式探针，不挑好看的那个。真实 workload 的回退原因集中在 stiffness（diffusion N=1024、Robertson）与 representation（Burgers），正是 §8 表预测的绑定轴。
- **退出条件 7（假设账本已建）**：`bench/ledger.py` 七项假设，每项 = {nominal, sensitivity, sensitivity_definition, source, falsification_observation}，敏感度数字全部现场算自既有机制或引用 results/*.json（不新增拟合），输出 `results/ledger.json`：

  | 假设 | nominal | 敏感度 | 来源/证伪观察 |
  |---|---|---|---|
  | kappa_hw | 1e6 | 4.86%（−Δκ_hw 即翻判 diffusion N=1024） | phase0.json；§9 风险 1：实测窗口可能 1e4 |
  | alpha_B_hw | 10 MHz | 贴在 margin=0.8 ⇒ B_hw 缩 20% 翻判 | scaling.py margin + pipeline.json（0.08→0.8） |
  | noise_psd | B_noise/B_req=10 | 0.63%/unit ratio（r=10 处亏损实测 6.69%、预测 6.35%） | sde.json A_bandwidth + bandwidth_drop_factor |
  | mismatch | ε_dev=1% | floor 线性（2%/1%=2.000），0.052 per 1%；1/λ_min 放大 | constraint.mismatch_floor 现场 + constraint.json |
  | parasitics | 3° 预算 | 2.86°/buffer 级（depth4 11.45° ⇒ 1 级余量） | route.json buffers |
  | calibration_residual | 2% | isometry ≈17/unit 残差（0.342@2%，标定后 3.46e-15） | transform.json gram + §6.16 |
  | excess_noise_gamma | α≈1 | d log σ_y/d log α = 1/2（公式精确）；γ_eff 5–10× ⇒ SNR 乐观 7–10 dB；**B2 HBT 实测 white factor 1.90×（vs γ=1）/3.79×（vs 推导 γ=1/2）⇒ SNR 乐观 2.8 dB，MOS 侧仍推导口径（§6.19）** | §1.2(B) + §9 风险 3 + §6.19 |



## 8. Benchmark 集（每个针对一条尺度律或一条 QoI）

| 系统 | 针对 | 预期判定 |
|---|---|---|
| 1D diffusion（spectral） | κ = N² | N ≤ 1024 可行 |
| biharmonic / KS | κ = N⁴ | N = 128 不可行 → 必须降维 |
| Burgers（激波） | Gibbs + flux tile | 需 basis switching |
| 线性对流 | 离散守恒 | flux tile 必须 |
| Robertson | κ ~ 10⁶–10⁹（Jacobian 谱，随 T_obs）/ 10¹¹（速率常数比） | 必须 QSSA/多速率 |
| 非正规 shear | Γ ≫ 1 | 换基或数字回退（实测 Γ=5.123；特征向量基 Γ=1 但 κ(Φ)=20，总预算更差——run_stagea） |
| Van der Pol（stiff 版） | κ + 极限环 | 边界（run_stagea：平衡点不稳定 ⇒ 回退/事件调度；reduce=poly；环幅≈2） |
| HH 神经元 | 事件原语 | 必须 hybrid |
| 带核的粘弹性 | Lorentzian 展开 | 辅助状态数 vs 精度（run_stagea：n=2+m 有限 Markov 嵌入，柔量对到 4.5e-16；幂律核无有限嵌入，Lorentzian 和逼近 m=4→12 误差 0.181→0.013） |
| 分数阶扩散 | s^α 有理逼近 | 同上 |
| 跨声速翼型（Euler, NACA0012, M≈0.8） | x_s、ν_num、wave drag | fitting 优先，否则窄 capturing |
| 高超声速 bow shock（驻定） | post-shock T/p/ρ 的 L2 | sharp shock + 高阶后方 |
| 浅水 tidal bore | bore speed、质量/动量守恒 | flux tile；κ_num 要求松 |
| 温盐锋面 / 密度流 | κ_num、front thickness | 必须保正 + 超前换基 |
| Rayleigh–Taylor / 界面破碎 | M_int、T_int 判据 | capturing，fitting 会崩 |
| OU 过程 | 加性噪声 / 噪声带宽 | Var=D/λ，ACF=e^{−λt} |
| GBM（乘性） | Itô vs Stratonovich | 两条均值曲线必须分开 |
| 双势阱 Langevin | 采样 / 标定 | 直方图 vs e^{−V/T} |
| 1D stochastic Burgers/heat | 噪声离散化 / trace-class | 谱与二阶矩 |
| OU/GBM 参数辨识 | CRN + T9 读出 | θ 恢复误差 <10% |
| MEMS resonator bank（2 模态） | 二阶 ODE / 高 Q | `ω0=√(k/m)`、`Q=mω0/c`；τ_settle~2Q/ω0 |
| MEMS Duffing | 物理非线性 | α 可辨识；不承诺任意 α |
| 1D fluidic diffusion | graph Laplacian 同构 | `P↔V`；τ~R_f·C_f |
| 1D reaction–diffusion | 化学 ODE + 输运 | `ċ=S·v(c)` 提供乘法 |
| photonic 4×4 MVM | 线性变换 | `y=Wx` 误差 <5% |
| Burgers + adaptive σ | representation switching | 守恒漂移 <1e-4；无阈值抖动 |
| lifting wavelet（3 级） | isometry / PR / adjoint | `isometry<0.5%`、`PR<0.2%`、内积测试 <1e-3 |
| biorthogonal wavelet | `W_inv≠W_star` | 用 synthesis 当 adjoint 会产生可测梯度偏差 |
| diffusion + wavelet 预条件 | scale-dependent `g_m/C` | pole 跨度降低，model time 恢复 <1% |
| 非正规 shear + block Schur | `κ(Φ)` vs `Γ` | 禁止病态特征基；用块正交基 |

---

## 9. 明确不做 + 已知风险

**不做**
- >256×256 的全 crossbar（寄生与容差不可控）
- 纯模拟（必须有数字 supervisor 做配置/校准/换基）
- "能表达任意 ODE"的宣称，只做 region 判定
- "两块 analog 各自跑不同时间再直连"
- 不承诺任意 SDE：只做有限维、Markov、Gaussian（白或有色）噪声；跳变/重尾需要 shot-noise tile，暂不做
- 不把 Fokker–Planck 作为硬件模式；它只在 n≤3 做验证
- 不承诺 neural SDE 训练；它是 Phase 6E 的研究项
- 不宣称“解任意 ODE/PDE”；只解 (K,T,Ω,O,ε) 上的投影/截断动力学（§1.5、§5.6）
- 不用稠密 monomial 展开做高维多项式：n=100、p=4 是 4.6e6 项（§2.7）
- 不做没有不变流形维护的状态增广：sin/cos/exp/tanh 只在各自流形上等价
- 不承诺 Gamma 等 differentially transcendental 函数的有限增广
- 不宣称 analog 降低 state count：高-Re DNS、6D Boltzmann、quantum many-body 不在承诺内
- 不承诺长期 chaotic trajectory；只承诺统计量（§1.5 Level 2）
- 不把微流控当通用高速 solver；它只做 physical-model-in-the-loop
- 不把 MEMS 当可任意重编程的 tile；它是 fixed-function bank
- 不做"全光模拟计算"；光子只做 communication + linear transform，不做非线性与存储
- 不把 WENO5 的公式逐项硬件化；只实现 `S → T_σ → F_σ` 三算子
- 不做"大型 analog FFT crossbar"；用 local lifting wavelet fabric + transposeable edges
- 不用病态特征基做非正规算子的对角化；宁可 `D+B` 或块正交

**五条物理设计原则**

1. 不做 arbitrary global crossbar；local interaction + 少量 global field/constraint。
2. 把守恒/reciprocity 做进拓扑（pair force、flux cancellation、KCL），不靠两路独立乘法器恰好匹配。
3. fabric switch 只做准静态配置；被模拟的 switch 是 computational primitive（有限斜率/compact model/event），不是 transmission gate。
4. 不强迫一种物理时间尺度覆盖所有问题；multirate 用 waveform relaxation/S&H，不用两块 analog 直接不同速相连。
5. 表示切换用连续 `σ` 和同一个 face flux，不用 mode flag 硬切；新的 discontinuity 只能来自物理或事件，不能来自编译器。

**风险（按杀伤力排序）**
1. κ_hw 估计过于乐观：实际受低频 1/f 与漏电双重挤压，τ_max 可能只有 100 µs ⇒ κ_hw 降到 10⁴，直接砍掉一半 PDE
2. 校准漂移：温度使 θ(t) 慢变，ΔA 时变 ⇒ H2/H∞ 需周期性重算；校准开销是真实成本
3. α 被低估：真实 OTA 的 γ_eff 与 excess noise 可达 5–10×，SNR 预算系统性偏乐观
4. **降级判定**：若 benchmark 中 >50% 需要数字回退，这台机器就是"模拟加速器"而非"模拟求解器"。可接受的降级，但必须在 Phase 0 就承认，而不是 Phase 4 才发现
5. **模拟无 rewind**：换基若滞后于 Gibbs 出现，误差不可逆（过冲穿过非线性 EOS/饱和元件后信息已丢）；τ_lead 预算必须实测，不能假设
6. tanh limiter 非 TVD：若目标应用要求严格 TVD（如某些 ocean front 诊断），此 tile 方案不成立，需换 limiter 拓扑
7. s ≳ 10% 时架构优势消失（§6.7）：这决定机器只对"奇异区局部化"的问题有意义
8. 噪声带宽不足：T8 的 B_noise 或 replay DAC 的 f_s 不够，OU 方差会低 6% 以上；这是系统性偏差，不是随机误差
9. 1/f 污染：低频 1/f 会改变长时间稳态分布；需要斩波/高通或把 1/f 增广进模型
10. 噪声相关：共享电源/地使 ρ_ij≠0；T3 混合矩阵的失配会把它变成不可控的 D 误差
11. 读出带宽：N 条轨迹的原始波形无法全回传；没有 T9 就没有反问题
12. 可辨识性：低维参数也需要足够 T 和 N；单轨迹的 Fisher 信息有限
13. Itô/Stratonovich 偏差：τ_c 有限时 O(τ_c) 偏差；合同写错会让所有矩系统性偏
14. closure defect 被低估：Fourier 的非线性把能量搬到 |k|>N；η_closure 超 ε 时解会“看起来对、统计量错”
15. 不变流形漂移：sin/cos/exp/tanh 增广离开 s²+c²=1 或 |y|<1 后不再等价；投影/反馈会引入额外噪声
16. DAE index：det ∂G/∂x′=0 且 index>1 时，descriptor 的广义谱/正则性假设失效；必须先 index reduction
17. scaling 把刚度藏进失配：S 能把 J_ij 压到 O(1)，但 C_i 的失配/噪声会变大；κ 不变，Γ 可能变好也可能变坏
18. 低秩假设不成立：B 的秩 R 接近 n 时，O(Rn) 退化回 O(n²)，T3 装不下
19. 特殊函数的 differential-algebraic 复杂度被低估：Bessel 需要 1/x，Gamma 不可有限增广；误判在仿真后期才暴露
20. 约束 fabric 的松弛速度/条件数：λ_min(A) 太小或 A 病态时，T10 的 τ_relax 可能比动态时间尺度还慢
21. routing cost 权重错配：C_route 的权重若不用 dynamical sensitivity，会把 wire length 优化成错误方向
22. current-mode 镜像失配/噪声：低阻节点求和省寄生，但镜像电流的失配和噪声会进入状态
23. 统计契约模糊：对湍流/混沌，O 选错会把"统计正确"误判为"轨迹正确"
24. analog skew 被当成 latency：ω_max Δτ 不满足时 wire 把 ODE 变成 DDE；一阶 E=I+A∘T 乐观 2/π，不能替代相位/稳定边界检查
25. global correlated noise：global reference/电源/地的噪声是低秩相关的，ensemble averaging 消不掉；必须走信号分级 + 差分电流 + 局部 DAC
26. constraint lag：τ_constraint/τ_state,min 不够小，elliptic 约束被硬件松弛成 parabolic；MD 假 retardation、NS pressure 滞后、SPICE KCL 不收敛
27. analog buffer tree 的噪声/失真累积：数字 repeater 可再生，模拟 buffer 不能；N_b σ_b² 限制 fanout 深度。3°/100 MHz、2 GHz 极点时只允许 1 级，相位比噪声更早成为约束（§6.15）
28. HV 器件的寄生进入 RF node：C_off/C_gd 让“关掉的开关”在高频变成负载；必须分 RF/fast 与 HV/large-swing domain
29. 校准把噪声当 offset 校：噪声不可反相消除，只能估计 S_n/Q/ρ；Q_hw>Q_target 时只能改 C/带宽/器件/placement
30. pre-silicon 只跑 corner：corner 不覆盖 mismatch、transient noise、substrate coupling、electrothermal；必须 calibration-in-the-loop + MC + PEX/EM
31. matched skew 被当成 closure：deskew 把 skew 修到 0 不改变绝对 loop delay；feedback edge 必须用 τ<φ_reserve/ω_c 单独判，零 skew 仍然给 ẋ=Ax(t−τ0)（§6.15）
32. crosstalk 地板被当成可修：HF 极限 C_m/(C_2+C_m) 与 R/buffer 无关；1 mV/1 V 要求 C_m<0.11 fF，20 dB 屏蔽不够，且它进的是相关噪声（§6.15）
33. MEMS 的 Q/ω0 温漂与老化：`ω_ref` 本身也漂，ratio accuracy 不能替代绝对表征；`τ_settle~2Q/ω0` 会让高 Q 模态的观测窗变长
34. MEMS 接口非线性被当成 g_m：静电驱动 `F∝V²`、电容读出 `ΔC/C`；当成线性会把 Duffing α 与驱动幅度混在一起
35. MEMS 的 pull-in/stiction/迟滞：驱动接近 pull-in 时状态不再连续；必须留 margin 或进事件模型，不能靠"再校准"
36. 微流控的不可逆慢漂移：蒸发、气泡、堵塞、表面吸附改变 `R_f/C_f` 且不可逆；只能 background calibration + 冗余通道，不能假设流体网络=理想 RC
37. 光子的热漂移与标定：ring `dλ/dT`、laser 波长/功率漂移、PD 暗电流；WDM 通道间串扰是相关的，ensemble averaging 消不掉
38. transducer 成本被低估：每次跨域都付 gain+noise+bandwidth+energy；`V↔x`、`V↔P`、`V↔a` 没有一个是免费的，它必须进 §1.2 的 slack
39. heterogeneous slack 被压成一个数：不同域的 `κ_hw` 不能混用；跨域边界必须显式 sample/锁相，否则"整体可行"是假的
40. `σ` 混合作用在 state 上而不是 face flux 上：守恒误差变成 `O(σ)`，且 `Σ C_i V̇_i=0` 不再由拓扑保证（§5.7、§6.4）
41. weight/σ 更新延迟与抖动：数字权重更新率低于模拟动态时，`σ` 在阈值附近形成极限环，重新引入开关注入噪声；实测对称 tracker 在同一 cell 穿越 8 次，attack/release 400/10 降到 2 次。必须迟滞 + 不对称 hold time（§5.7、§6.16）
42. undecimated wavelet 的冗余被当成免费：`N_coeff>N`，state 数、功耗、读出带宽都按 redundancy 增长；第一代要把它算进 `N_max`
43. biorthogonal 的 inverse 与 adjoint 混用：`W_inv≠W_star`，用 synthesis 当 adjoint 会让 forward 解看起来正确、梯度/反问题缓慢偏（§5.8）
44. 时间反演不可行：事件/reset/开关使轨迹不可逆，checkpoint 也可能放不下；adjoint 只能在可逆段内做，跨事件段要用段间伴随或数字回退
45. Gram 标定本身病态：`G_T` 条件数大时 `G_T^{−1/2}` 放大测量噪声；`isometry_error` 与 `noise_gain` 必须同时报，不能只报前者
46. per-scale `C_j`/`g_m,j` 的副作用：调 pole 的同时改 `σ²=αkT/C`、g_m 范围和相对增益；每级必须单独标定，不能用一个全局 gain 补
47. `D+B` 的 `B` 并不自动小：非正规强耦合时 `B` 可能仍是稠密的，`‖T‖‖T⁻¹‖` 也未必降；必须有 `Γ` 的前后对比，不能只看形式
48. wavelet detail 被当成 shock 概率：湍流/高频光滑结构也会产生大 `|d_{j,k}|`；`σ` 是 regularity coordinate，不是"有激波"的二值判决
49. Gram 标定乘错边：`G_T=T̃*T̃` 配左乘 `K T̃` 不会正交化（0.2% 扰动实测 isometry 0.032、2% 时 0.342，对比右乘 3.46e-15）；右 Gram 必须右乘 `T̃ G_T^{-1/2}`，左乘要用 `T̃T̃*`。公式对、矩阵边错是典型失败（§5.9、§6.16）
50. sensor 的 `ε` 取绝对小量：光滑极值处一阶差分为零，`0/0` 会把 sine 判成 shock（实测 σ=1.0）；`ε` 必须是场幅的 ~1% floor（§5.7、§6.16）

**开放问题（不阻塞当前阶段）**
- **最小 dynamical basis**：是否存在一组 B，使大量科学 ODE/PDE 经 state augmentation 后都落入低阶 polynomial system？
- **closure selection**：给定 (K,T,Ω,O,ε)，如何自动选 Fourier/Chebyshev/FV/DG/wavelet，并给 η_closure 证书？
- **不变流形修正**：如何在不增加噪声/不引入新失配的前提下把 s²+c²=1、|y|≤1 拉回？（Baumgarte −γ∇g·g 已在 lowering/reduce.py 作为可选稳定项实现；稳态方差界 E[g²(∞)]≈αkT/C·τ_hwγ_noise/γ+(I_off/2γCV_FS)² 与 γ 可行区间 [γ_min(offset), γ_max(κ_hw)] 见 bench/run_reduce.py 验收）
- **differential-algebraic complexity 的可计算度量**：minimal augmentation dimension + degree + invariant cost，而不是“这个函数常不常见”
- **统一约束求解器**：把 MD electrostatics、NS pressure projection、SPICE KCL 都写成 A(x)y=b(x)，同一块 T10 要覆盖多大的条件数/规模范围？
- **自适应多分辨率**：quadtree/octree tile 的物理重配置如何在 charge injection 与 τ_lead 约束下做到不破坏守恒？
- **multirate DAE**：waveform relaxation 的收敛率与 S&H 误差如何进 P6 的可行性证书？（§4.3 合同已实现并验收：`twin/multirate.py` + `bench/run_multirate.py` 33/33——ρ_sup=sup_ω σ_max(G_WR) 作为 small-gain 充分证书、k_max 由 replay 底噪决定、σ_max>1 但 |λ|<1 的分裂实测仍收敛即充分不必要；ZOH 是真 ZOH 块 `H=(1−e^{−sTs})/(sTs)`，Nyquist 处幅度 2/π=0.637（PureDelay 错报 1.0），droop 进 route gain 轴，ε=0.01/ωT_s≲0.49 以下才允许显式记录地退化成 PureDelay，混叠/重构滤波是独立预算）
- **异构 operator placement**：给定 (K,T,Ω,O,ε)，如何在 {CMOS, MEMS, fluidic, photonic} 上分配算子，使 slack/energy 最优？每次跨域的 transducer 成本如何进 P6，而不是事后加总？
- **adaptive representation 的闭环稳定性**：`sensor → σ → weights → flux → state` 是一个新的反馈回路，它会不会产生原系统没有的振荡/分岔？如何用 §1.2 的 Γ 和 small-gain 判定，而不是靠调参？（§6.16 已给出部分答案：`tracker_rate_bounds` 的 r_attack≫|s|/Δx、r_release≤π²ν_num·Δσ_hyst/Δx² 把 attack/release 从调参变成可推导的不等式；实测 release 界约束强、attack 界在当前滞环宽度下不 bind）
- **可转置性的边界**：哪些 primitive 天然可转置（gm/current edge、lifting predict/update），哪些不可（比较器、事件、reset）？跨事件段的 adjoint 怎么定义？
- **Gram 标定的可观测性**：`G_T=T̃^*T̃` 的哪些元素能在片测，哪些只能靠假设？测量矩阵的秩/条件数是多少，`G_T^{−1/2}` 的噪声增益如何进随机证书？

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
        ✅ skew.py        # DDE → E=I+A∘T；均匀延迟闭式极点、τ_crit、相位预算
        ✅ sde.py         # Itô/Stratonovich、线性乘性协方差、均方稳定、噪声可实现性、wong_zakai_factor
      lowering/
        ✅ reduce.py      # jet/一阶化、状态增广、rational 化、canonical form
                          # 实测：Lorenz/Duffing polynomial；HH rational；Gamma 拒绝
        ✅ normalize.py    # 坐标中心化 + 无量纲化（x_i=X_i x̂_i+x0_i, t=T₀τ，
                          # Expr 重写；pipeline_normalize 出 Normalization 记录）
        ✅ closure.py      # P_N 投影 + η_closure 证书（实测：η(N) 对照 ‖f_{k>N}‖/ref
                           # 逐点一致；k>N forcing 全额 defect；二次项 defect 落
                           # (N,2N] alias 带 ⇒ change-basis——2/3 规则消 aliasing
                           # 不消 closure；34 项验收 in run_closure_split）
        ✅ split.py        # CSP/QSSA 快慢模态剥离：谱间隙 auto/τ_fast 分类、物理坐标
                           # Schur A_eff=A_ss−A_sf·A_ff⁻¹·A_fs、slaving 残差实测
                           # （gap 1e4 ⇒ ~1e-8 且 ∝ A_fs）；弱间隙/亏损 Jordan/
                           # 不稳定 fast/奇异 fast 块 ⇒ flag 拒绝，不静默求逆
        ✅ basis.py        # 选基：稀疏度 vs κ(Φ) 税——budget=Γ·amp
                          # （worst-case amp=cond(T)；给定 Δ 则 ‖T⁻¹ΔT‖/‖Δ‖），
                          # 不变式 ‖R_x‖≤cond(T)‖R_z‖ ⇒ 纯相似变换预算 ≥ 恒等基；
                          # shear 实测：eigvec Γ=1 但 κ(Φ)=20.05 总预算 20.05>5.12
                          # 被淘汰；Δ=ε·A 时 eigvec 翻身赢（amp=‖Λ‖/‖A‖<1）
        ✅ scaling.py      # 可控/可观 Gramian 对角平衡 → S_x (rms3σ / Moore d_i)
                          # + α=max|λ|/(2πB_hw·margin)；自动构造 PhysicalScaling；
                          # 频窗检查（eig 不变性断言）
        ✅ discretize.py   # P0 半离散化：PDESpec→Discretized (fourier/fd/fv)
        ✅ budget.py       # P6 证书聚合：slack + Γ_M + σ_y² + manifold-γ 窗
                          # + loop ‖S‖∞ + multirate k_max + sde_cert →
                          # Certificate（rejected/infeasible/no-violation/feasible）
        ✅ interface.py    # §6.4 单通量源+镜像合约（dM/dt=(F_A_fv−F_A_spec)−
                          # (F_B_fv−F_B_spec)，单源 ⇒ 拓扑零泄漏，coupling.py
                          # 实跑对账 drift 2.2e-16 vs mismatch 泄漏 8.5e-3）；
                          # §6.5 fitting/capturing selector：光滑 ⇒ capture 免费，
                          # 奇异 ⇒ ν_num=½·α_loc·dx·σ vs remesh 单价×M_int，
                          # ν_num 越阈值判定翻转（31 项验收 in run_basis_interface）
        ✅ p7.py          # P7 配置+校准计划：Certificate+ScalingResult+
                          # CalibratedTwin → per-element {primitive,
                          # target_gm_over_c=|a_ii|（各自有效自系数，
                          # diag(-1,-100) 出两个不同值；tile 速度 spec
                          # = max|λ|/alpha 保留在 source）, a_ii, C_code,
                          # S_x, alpha, noise_setting} + 完整系数图
                          # coeff_edges/b_ports/c_taps（物理增益，coeff_edge
                          # 引用契约序号）+ cal_bindings + plan（foreground/
                          # background/residual_budget=floor/noise轴（启发式，
                          # dJ 分配是 B2），不可行 ⇒ flag）
        ✅ netlist.py     # Config → 结构化行为级网表（integrator 带极点/
                          # edge 带物理增益/噪声 scale；effective_A()/B/C
                          # 重构 → round-trip 特征值对拍 scaled 描述符，
                          # diag/4×4 稠密误差 ≤1.3e-15）（26 项验收
                          # in run_p7_netlist）
        ✅ route_contract.py # per-edge 路由契约（四判据 + feedback ωc）：
                          # 耦合图 = A 非对角非零元；每边 REQUIREMENT
                          # （omega_max=max|λ|/alpha 物理带宽；feedback 边
                          # omega_c + tau_max_abs=phase_budget(ωc,φ_reserve)
                          # 有限非 inf）+ 代表性 candidate 线的
                          # route_closure 四判据 verdict（pad_can_fix 诚实：
                          # feedback 边 False——deskew 负控制的编译层版）+
                          # edge_sensitivity S_τ/τ_tol 排序； defective 边
                          # flag 不静默；契约是需求+代表性可行性，
                          # 不是布线结果（那是 P5/B2）（38 项验收
                          # in run_route_contract）
      twin/
        ✅ spice/extract.py  # 真实 PDK 提取：ngspice 电流镜 → λ, g_m/I_D, r_o, ratio
        ✅ noise.py        # OU 滤波色噪声 + replay + ρ_ij 相关矩阵；色噪声/白噪声极限
        ✅ multirate.py    # WR 谱半径 + replay 底噪 k_max + 真 ZOH（sinc droop，Nyquist 2/π）
        ✅ constraint.py   # T10：A(x)y=b(x) → C_node·V̇+G·V=I_rhs KCL 松弛
                          # τ_relax=C/λ_min(G_sym)、非对称时全谱横坐标守卫、
                          # mirror 失配静差下界 ‖G⁻¹‖·‖δI‖（∝1/λ_min，κ trade-off）、
                          # kcl_relaxation_descriptor → 复用 analysis 栈（44/44）
        calib.py          # ✅ 校准 FSM：前/后台 + 温漂数字孪生更新
                          # （foreground 残差 = 测量噪声底合同；VERIFY 用
                          # held-out 验证点集，无集时如实记 independent=False；
                          # 过拟合负控制：dof=点数 ⇒ 训练残差吃到底、
                          # held-out ≈29×底被抓；background
                          # 残差 = rate·interval/(1+T) + 底，间隔↑残差↑；
                          # CalibrationFSM IDLE→STIMULATE→MEASURE→CORRECT→
                          # VERIFY；CalibratedTwin 是 compiler 的输入）
        ✅ coupling.py     # spectral↔local 接口守恒：单通量源 + 电流镜像
        ✅ qoi.py          # W_d → ν_num；Osborn–Cox → κ_num；x_s(t) 对比
        ✅ route.py        # H(s)=wire+buffer+delay；η/J_H/J_τ、S_τ,ij、latency class、route_closure
                          # 实测：0.3795 RC 可测；fanout 4 级 11.45°；all-pass J_H 108× 优于 RC
        ✅ substrate.py    # §3.7 最小子集：KAPPA_HW 每域时间窗 + 流体→电学
                          # P↔V/Q↔I/R_f↔R/C_f↔C 映射 + Hagen–Poiseuille R_f +
                          # fluidic_mesh_to_conductance + TransducerCost 合同
                          # + MEMS 谐振器（ω0/Q/τ_settle/Duffing 骨架线/
                          # gm_c_calibrate ω_ref 锁定）+ photonic MVM 保真度
                          # （shot 1/√N + 漂移残差地板；无存储断言；A7，40/40）
        ✅ transform.py    # §5.7–§5.9：smoothness sensor/σ（相对 ε floor + attack/release）、
                          # lifting wavelet、W_inv/W_star、Gram 右/左标定、transposeable edge
                          # 实测：守恒 2.2e-16；CDF53 W_inv≠W* 0.684；isometry 2.3e-2→3.5e-15
      bench/
        ✅ run_twin_experiments.py  # §6.8 五组对照（3 min）
        ✅ run_phase0.py            # Phase 0 五算例验收（~2 min）
        ✅ run_reduce.py            # Phase 1 前端：Lorenz/Duffing/HH/Bessel/Gamma（25/25）
        ✅ run_skew.py              # Phase 1.5 互连延迟：E=I+A∘T、τ_crit、相位预算（15/15）
        ✅ run_route.py             # Phase 1.6 route closure：wire/buffer/all-pass/deskew/crosstalk/敏感度（38/38）
        ✅ run_transform.py         # Phase 4.5：WENO 三算子、lifting adjoint、Gram 标定（35/35）
        ✅ run_substrate.py         # A7 异构介质验收：MEMS/光子（40/40）；
                                    # fluidic diffusion 算例仍是 Phase 8 TODO
        ✅ run_sde.py               # OU/GBM/双势阱 + CRN/moment 参数恢复（27/27）
        ✅ run_multirate.py         # §4.3 WR/S&H 合法性验收（33/33）
        ✅ run_compiler.py          # P6 证书聚合层验收（22/22）
        ✅ run_constraint.py        # T10 KCL 松弛孪生验收（44/44）
        ✅ run_constraint_timing.py # T10 ε_c 语义：外部时标显式声明 + 纯稳态 deadline 事实 + 证书可见性（36/36）
        ✅ run_pipeline.py          # 3 锚定基准 e2e（符号→证书→孪生核对）（34/34）
        ✅ run_stagea.py            # A4/A8/A9 收尾：shear 换基判定（Γ 5.12 /
                                    # scaled / eigvec=1+κ(Φ)=20 税）/ VdP
                                    # （不稳定工点 ⇒ 回退，reduce=poly）/
                                    # Lorentzian 粘弹性（n=2+m 有限嵌入）+
                                    # 数字回退率 tally（6/11=54.5%，剔探针
                                    # 2/7=28.6%）+ 假设账本（26/26）
        ✅ run_closure_split.py     # B1 closure+split：η_closure 证书（η(N) 解析
                                    # 对照 / k>N forcing 全额 defect / alias 带
                                    # change-basis）+ QSSA 快慢剥离（Schur 精确
                                    # 复现慢特征值、slaving ∝ gap⁻¹、负对照
                                    # 弱间隙/亏损/不稳定/奇异全 flag）（34/34）
        ✅ run_calib.py             # B1 校准验收：前台标定修回噪声底 /
                                    # VERIFY held-out 独立验证 + 过拟合负控制
                                    # （训练残差=底、held-out ≈29×底被抓）/
                                    # 后台残差 = rate·interval/(1+T) + 底
                                    # （间隔↑残差↑，A7 同语义）/
                                    # P7 diffusion-1d 配置+校准计划
                                    # （逐元素 target=|a_ii| spread 4096、
                                    # S_x、残差预算 = floor/noise轴（启发式）、
                                    # 不可行 ⇒ flagged）（50/50）
        ✅ run_p7_netlist.py        # B1 P7 系数图 + 行为级网表 round-trip：
                                    # diag(-1,-100) ⇒ 两个不同有效系数
                                    # （比值 100）/ diag ⇒ 0 条 off-diag 边、
                                    # 耦合系统 ⇒ 边增益=物理 A_v + 契约消费 /
                                    # 4×4 稠密 12 边逐元对拍 + B/C 重构 /
                                    # diffusion 64 模态 spread=4096 /
                                    # round-trip eig 误差 ≤1.3e-15（26/26）
        ✅ run_basis_interface.py   # B1 选基+接口：shear 最坏情况 eigvec 被淘汰
                                    # （Γ=1×κ(Φ)=20.05=20.05 > 恒等基 5.12，
                                    # 对账 stagea/skew）、不变式预算≥Γ_id、
                                    # Δ=ε·A 时 eigvec 翻身（chooser 不退化）、
                                    # 近正规 cond(V)=1.43 小优势赢；接口：
                                    # 光滑 capture 免费 / 单源合约 dM/dt=0
                                    # （coupling.py 实跑 drift 2.2e-16 vs
                                    # mismatch 泄漏 8.5e-3）/ ν_num 翻转（31/31）
        ✅ run_route_contract.py    # B1 per-edge 路由契约：耦合图=A 非对角
                                    # 非零 → 每边契约（feedback 边 ωc+τ_max=
                                    # phase_budget 有限；非 feedback 绝对判据
                                    # 平凡 True）；candidate 短线四判据全过；
                                    # 1/|A_ij| 敏感度排序；强制长线 ⇒ verdict
                                    # False + pad_can_fix 诚实（feedback False/
                                    # 非 feedback True）；负对照：纯对角/burgers
                                    # ⇒ 空表、NaN/inf 边 flagged（38/38）
        ✅ lowering/compose.py      # B1 组合层 run_chain：链驱动 + 逐 stage
                                    # 坐标声明（model/physical-time/basis-T/
                                    # field）；F1 typed boundary（System IR ⇒
                                    # TypedBoundaryError）、F2 projected 载体
                                    # 上重推 scaling（alpha 比 = 谱比 0.25，双
                                    # 口径入台账）、F3 interface 输入域 = "field"
                                    # （无场 ⇒ 显式 not applicable）、F5 basis
                                    # 赢家复合进 scaled lane（T⁻¹AT 尘埃清理
                                    # 1e-12·max|A|；eig 不变性 9.3e-16、外部
                                    # TF 不变性 6.9e-18、alpha 比 1.0；p7 a_tol
                                    # 边准入 mask 同时发给契约与系数图 ⇒
                                    # 边集由构造相等并断言）（compose 层）
        ✅ run_compiler_e2e.py      # B1 closeout：经 run_chain 走整条链，
                                    # F1–F5 关闭性断言（findings status=
                                    # "closed"；F4 保持正向一致断言：
                                    # omega_max=2π·scaled.bandwidth）；
                                    # diffusion full+projected 两 lane、
                                    # synthetic eigenvector+identity 两 lane
                                    # （48/48；旧 12 条"契约边"实测为
                                    # eigenvector 共轭的 1e-14 尘埃，清理后
                                    # 对角化 ⇒ 0 边；identity lane 6 条边 ==
                                    # coeff 边集，(0,1) 边缩放权重 6.84e-9 <
                                    # a_tol 5.03e-6 被双边一致剔除）
        ✅ ledger.py                # 七项假设敏感度账本 → results/ledger.json
                                    # （κ_hw 4.86% 翻判 / B_hw margin 0.8 /
                                    # 噪声 0.63%/unit / 失配线性 / 2.86°/级 /
                                    # isometry 17·resid / γ_eff log-sens 1/2）
        diffusion.py biharmonic.py burgers.py robertson.py shear.py hh.py
        airfoil.py bore.py front.py   # 20 个 benchmark 待补；3 anchors in bench/problems.py；当前模型在 run_phase0.py 内
      results/                    # report.txt / twin_experiments.json /
                                  # device_extract.json / phase0.json /
                                  # phase0_report.txt / reduce.json /
                                  # reduce_report.txt / skew.json /
                                  # skew_report.txt / route.json /
                                  # route_report.txt / png / 提取日志 /
                                  # transform.json / transform_report.txt（Phase 4.5）/
                                  # sde.json / sde_report.txt（Phase 6）/
                                  # substrate.json / substrate_report.txt（A7 异构介质）/
                                  # stagea.json / stagea_report.txt（A4/A8/A9 收尾）/
                                  # closure_split.json / closure_split_report.txt
                                  # （B1 closure+split）/
                                  # calib.json / calib_report.txt（B1 校准+P7）/
                                  # route_contract.json / route_contract_report.txt
                                  # （B1 per-edge 路由契约）/
                                  # p7_netlist.json / p7_netlist_report.txt
                                  # （B1 P7 系数图+网表 round-trip）/
                                  # ledger.json（七项假设账本）

关键 API（✅ 已实现；SDE 三行是 Phase 6 的目标接口）：

    sys  = Descriptor(E, A, B, C, D)
    ly   = solve_generalized_lyapunov(sys, Q)  # ly.P, ly.sigma_y, ly.rel_residual
    hi   = hinf_norm(sys)                      # hi["value"], hi["method"]
    res  = peak_with_projection(sys)           # res["Gamma"], res["Gamma_proj"]
    sl   = evaluate(sys, HardwareSpec(...),
                    gamma_proj=res["Gamma_proj"], hinf=hi["value"])
                                               # sl.values, sl.violated, sl.status

    red  = reduce_system(System(states, rhs))  # red.kind: polynomial/rational/rejected
                                               # red.canonical, red.augmentations

    sk   = skew_report(A, delay_mass_matrix(A, tau))
                                               # sk["kappa_skew"], sk["tau_crit_first_order"]
    tau1 = phase_budget(2*np.pi*100e6, 1.0)    # 27.78 ps

    e    = RouteEdge((Wire(2000, 0.2, 0.2),))  # distributed 1/cosh√(sRC)
    rc   = route_closure(e, RouteSpec(2*np.pi*1e8, feedback_critical=True,
                                      omega_c=2*np.pi*1e8))
                                               # rc["pass_gain/phase/noise/absolute"]
    S, tol = edge_sensitivity(J, 2*np.pi*1e8)  # S_τ,ij；tol=ε_φ/S
    dk   = deskew(edges, 2*np.pi*2e8)          # dk["skew_before/after"], dk["edges"]

    sde  = linear_sde_covariance(A, [G0, G1], Q)  # sde.P, sde.mean_square_stable
    fS   = ito_to_stratonovich(f, G, x)           # 修正漂移
    nr   = noise_realizability(spec)              # B_noise/B_req, tau_c, D 范围

    tf   = LiftingWavelet(64, "haar", levels=3)   # W, W_inv, W_star（CDF53 用 "cdf53"）
    rep  = gram_calibrate(W_hw)                   # rep["isometry_after"/"pr_after"/"adjoint_after"]
    sig  = SigmaTracker(beta0=0.15, k=16, hysteresis=0.08,
                        max_rate=400, release_rate=10).update(S, dt)
    Fl   = blend_face_flux(F_smooth, F_shock, sig)  # 同一 face；实测 drift 2.2e-16

---

## 一句话总结

不是"模拟电路能不能表示方程"，而是：**给定 (E,A)、工艺窗口 κ_hw 和扰动端口 B_p，是否存在 θ = (T, A_t, σ, c, w) 使 ‖H‖₂ ≤ σ_req、‖GΔ‖∞ < 1、κ_t ≤ κ_hw。** 三个数，一个判定，一张 slack 向量；越界的哪一轴，决定 compiler 拉哪根杠杆。这就是“经过 basis T、时间缩放 A_t 和硬件映射之后，动力学条件数落在硬件可实现域内”的精确版本。

T10 的代数约束也按同一逻辑判定：τ_relax ~ C_y/λ_min(A) 是约束层的条件数；A 病态时，约束解和动态解一起失效。

对 SDE，再加一组**不可交易的随机证书**：噪声带宽、τ_c、D 的范围与读出。它们回答的不是"余量还剩多少"，而是"这个随机过程能不能被忠实地产生"。

而“可划归”本身也是一个合同：存在 E~∈Closure(B)，Cost(E~)≤N_max，且在 (K,T,Ω,O) 上 d_O<ε——不是“多项式能逼近”，也不是“这个函数常见”。

而奇异区不是"降阶"，是**把 QoI 从点值阶数换成 (W_d/ν_num、守恒漂移、x_s(t))**：95–99% 的流场保持高阶，1–5% 的自由度交给一个狭窄、严格守恒、拓扑上不可能泄漏的 flux 层。WENO 式的降阶因此从"精度损失"变成**资源局部化机制**。

异构 substrate 不改变这套判定，只改变取值域：state 是 generalized coordinate `q` 而不是电压，`κ_hw` 按域给，`Γ` 按 transducer 耦合矩阵算，每次跨域的成本（gain/噪声/带宽/能量）必须进 slack。MEMS 提供二阶动力学与时间基准，微流控提供同一类 graph Laplacian，光子提供不随 L² 增长的 global dense 连接——它们各自做自己"天然会做"的算子，而不是把 CMOS 变成万能介质。

WENO 也不是一个 stencil：把它拆成 `S → T_σ → F_σ`，`σ` 连续、混合发生在同一个 face flux 上，才和模拟 fabric 兼容。表示层的合同从 `T` 与 `T⁻¹` 扩成 `T、T⁻¹、T^*`：等距性用 energy metric 说，forward/adjoint 共享同一块可转置硬件，Gram 矩阵标定的是数学合同而不是晶体管。
