# gf180mcu 第一批硬件设计计划（批次 G1：MOS-only characterization vehicle）

## 0. 对现行 plan 的检查结论

按 `DESIGN.md §7` 的验证金字塔，Stage A（理论验证）已收口——2026-11-27 全量回归 28 套件 / 987 项 / 1033.6 s 无失败；B1 编译器 near-complete；B2 的 pre-silicon 闭环（primitive 提取 → macromodel → joint closure）已在 **IHP SG13G2** 上完成并给出实测 P 表。计划的下一步本来就是 **B3：第一颗 characterization vehicle**。本计划把 B3 的第一个流片载体落到 **gf180mcu**，并在 `DESIGN.md §7.5 / B0` 记录一次**显式的工艺修订**（不是推翻冻结：SG13G2 保留为 HBT 接口版 vehicle #2，gf180mcu 是 MOS-only vehicle #1）。

**为什么 gf180mcu 在计划里站得住（全部来自仓库既有实测，不是偏好）：**
- 接口层排序 `HBT ≫ gf180 > sky130 ≫ 短沟道 MOS`（§6.9 λ 表）：gf180 `nfet_03v3` @10 µA 实测 λ=0.034/V、r_o=2.84 MΩ，是**纯 CMOS 里最好的接口器件**（比 IHP MOS 好 28×，比 HBT 差 2.5×）；
- 单通量源方案在 gf180 上实测把镜像误差从 7.1e-3 压到 8.1e-6（**880×**，四工艺中最大）——T5 的核心机制在无 HBT 时仍然成立；
- gf180 用 BSIM4，ngspice 原生支持 `.noise`：**MOS γ_eff 可以在 pre-silicon 实测**（IHP MOS 的 PSP103 OSDI 噪声在 ngspice-44.2 未实现、只能推导口径——gf180 反而补上这个洞）；
- open PDK + 本机工具链齐全：ngspice / xschem / magic 8.3.685 / klayout / yosys 均已安装；PDK 在 `~/.ciel/ciel/gf180mcu/versions/792ef670.../gf180mcuD`（libs.ref/libs.tech），标准单元与 SRAM 宏在 `/grid/gf180mcu`；
- 已知坑已有记录：直接 include `sm141064.ngspice` 必须补 `.param fnoicor=0`（`twin/spice/extract.py` 已处理）。

## 1. 需要先冻结的决策（B0-gf180 修订，写进 DESIGN.md §7.5）

| 决策 | 内容 | 依据 |
|---|---|---|
| 工艺 | **GF180MCU D（gf180mcuD）B1L**（3.3V 主 + 6V 模拟档；PDK `792ef670`） | D 是唯一 m4m5 MIM 物理合法的变体（A 的 mimcap 落 m2m3）；A/D 的 ngspice kit 逐字节相同，primitive bank 零差异继承（§6.25 变体锁定段）；5 层金属给长主干 M5 40 Ω/□ |
| 模拟器件 | nfet/pfet_03v3（逻辑接口）+ nfet/pfet_05v0（积分器 gm、镜像）| 5V 饱和摆幅 vs gm/ID 折衷，两档都提取后定 |
| 存 state | MIM 电容（C_bank 梯度 1×/4×/16×，`C_BANK_F=1pF` 对齐 `lowering/p7.py`） | §3.1「C 是资源」 |
| 接口镜像 | MOS 单通量源（shared reference），1:1/1:4/1:16 阵列 | §6.9：880× 机制在 gf180 实测成立 |
| supervisor | gf180 数字标准单元（`/grid/gf180mcu` osu/as_sc 7T 库）+ yosys 综合 | CMOS 做 FSM 的 B0 分工不变 |
| 流片路径 | GF180MCU open shuttle（TinyTapeout/ChipFoundry 类），Caravel harness 可选 | open PDK 生效前提 |

T1–T10 的 tile 子集沿用 B0 冻结表，第一颗只上 **T1（4–8 积分器）+ T5（缩短为 16-cell 链）+ T7（route tester）+ T8（noise bank 2 源）+ Supervisor**；T2/T3/T4/T9/T10 留 vehicle #2。功率/面积仍是 TARGET。

## 2. 门与交付物（每道门有可证伪数字，TARGET 直至实测）

### G1 — gf180 primitive 提取数据库（软件先行，~1 周）
新建 `twin/spice/extract_gf180.py`（对齐 `extract_b2.py` 的 fail-closed 框架）+ `bench/run_gf180.py` 验收：
- 器件工作点提取：nfet_03v3 / pfet_03v3 / nfet_05v0 / pfet_05v0 的 gm、λ=gds/ID、ro @ 指定偏置（复用 `PDKS` 的 vsweep 口径）；
- **`.noise` 实测 γ_eff**（BSIM4 原生支持）——MOS 一侧从"推导口径"升级为实测，直接回填 §9 风险 3；
- corner ≠ MC 三分布：global（tt/ss/ff × 3.3V/5V）与 Pelgrom mismatch MC(48) 分开，再加 PVT(3T×3V) 与联合 ss/ff×−40/125 °C；
- 镜像阵列 1:1/1:4/1:16：独立 vs 共享参考两种接法的 ratio（复现 §6.9 的 880× 机制，gf180 实测数字）；
- MIM cap bank 密度/失配；TG 开关 Ron/寄生；transient step/THD；
- 交付 `results/gf180_extract.json`（fail-closed status/blocks）+ `DESIGN.md` 新 §6.25。
验收线：γ_eff 实测值与 4kTγ/g_m 推导的偏差记录在案；MC offset/镜像比/gm σ 与 IHP MOS 表并列成表。

### G2 — macromodel + joint closure（pre-silicon P 表，~1 周）
- `twin/macromodel.py` 加 gf180 器件映射（ro/leak/A0/v_swing/噪声密度/vos/TG，来源标 MEASURED(gf180_extract)）；
- `bench/run_gf180_joint.py`：3 anchors（diffusion/poisson/burgers）× 48 mismatch × 4 cal 模式走 `compose.run_chain`，产 **P 表（1%/2%/5%）**——pre-silicon 口径，与 §6.21 的 IHP 表并排；
- 校准计划 floor 用 G1 实测底（不是沿用 IHP 数字），`p7` deliverability 用 gf180 floor 重算。
验收线：uncal/no_track/bg_track/full 四档结论与 IHP 表的结构一致（温度点、bias-invalid 标记语义不降级）。

### G3 — 原理图与行为级对账（xschem + ngspice，~2 周）
- xschem 画 batch-1 全部模拟块：gm-cell + C_bank 积分器（T1）、16-cell 单通量源链（T5 缩版）、route tester（长度/扇出矩阵 + all-pass pad，T7）、噪声源 bank（T8）、SAR ADC + 电流 DAC 测试通道；
- 每块先跑 transistor SPICE 再过 macromodel，两者极点/增益/噪声 PSD 对到 ±20%（口径记录）；route tester 的实测 t50/相位与 `route_contract.py` 四判据对账（wire RC 用 **D 版**五层 stack：M3 10.094 aF/µm²+30.021 aF/µm、90 Ω/□，M5 40 Ω/□；线电容只有 magic tech 有，ngspice kit 无此模型，故 T7 网表用 10 段 π 型梯显式建出 80.4 fF）；
- 交付 `hw/gf180/batch1/`（xschem 原理图 + ngspice testbench + 对账记录进 results）。
验收线：每块有一张"SPICE vs macromodel vs §6.xx 预测"三方表；对不上的项按"按实测记录"原则写回 DESIGN，不许修齐。

### G4 — supervisor 数字流（yosys + gf180 std cells，~1.5 周，可与 G3 并行）
- `twin/calib.py` 的 FSM（IDLE→STIMULATE→MEASURE→CORRECT→VERIFY，held-out 语义）综合到 gf180 7T 库；
- OpenLane 未安装：走 yosys + magic 布局布线的简化流，或先装 openlane2（二选一在开工时定）；
- 验收线：后仿（post-PEX 时序）通过 + FSM 终态语义（ACCEPTED/FAILED/retry 消耗新测量）在 RTL 仿真里复现 62/62 合同的对应子集。

### G5 — 版图 + PEX（magic/klayout，~3 周）
- floorplan：primitive bank + 32 份同 cell 错位副本（mismatch 地图，B3 铁律）+ route 阵 + 数字区隔离（guard ring / 深N阱按 gf180 规则）；
- DRC/LVS clean（magic gf180 tech file），post-PEX 网表回灌：G1 的探针在 **提取后网表** 上重跑一遍（寄生敏感度探针从"加 C 模拟"变成真实 PEX，与 §6.19 pre-PEX 探针口径衔接）；
- 验收线：PEX 前后极点/镜像比漂移量化入表；任何 DRC 修复不改变电路语义（LVS 全等）。

### G6 — test plan 映射 Stage C（~1 周）
- pin map、激励/回放（CRN replay 对齐 `twin/noise.py` 的 start_capture/replay）、loopback 与 replica 结构逐一对应 C0–C4 待测量；
- 每 tile 的 (E,A,B,C)/‖H‖₂/‖H‖∞/power/area 五行规格从 G1–G5 数据填实，TARGET→PRE-SILICON-MEASURED；
- 验收线：`bench/ledger.py` 七项假设的 gf180 版本（κ_hw nominal、噪声 PSD、mismatch、parasitics、cal residual）各带敏感度数字。

## 3. 风险与诚实边界

1. **无 HBT**：接口精度上限比 SG13G2 方案差 ~2.5×（λ 0.034 vs 0.0138）；characterization vehicle 的 ε 目标按 §6.9 的 8.1e-6 泄漏口径评估，若某 workload 的 ε_J 分配不可交付，`deliverability=False` 如实上报，不放宽 floor。
2. **180nm 速度**：gm/C 速度 spec `max|λ|/α` 与 route τ 预算须用 gf180 器件/金属参数重推；deadline 冲突的 workload 按 §6.24 的约束求解口径报告"无可行点"，不静默放宽。
3. **1/f 噪声**：gf180 noia/noib（fnoicor=0）的 flicker 比 130nm BiCMOS 差，直接影响长窗口校正的 ζ 底——G1 必须给出 1/f 拐点频率，background-cal 的 `drift_update` 用它重算。
4. **κ_hw=1e6 是 nominal**（账本敏感 4.86%）：gf180 上须重推并在 C1 实测；若实测只到 1e4，stiffness-bound workload 翻判，按既有规则诚实降级。
5. **数字流工具**：OpenLane 未装，yosys 已有；若 openlane2 路线受阻，G4 降级为手工布局 + magic，工期 +1 周。
6. **全部 pre-silicon 数字都是 TARGET/PRE-SILICON 口径**：硅上真值以 Stage C 为准，这批计划的产出是"让 C 阶段每个待测量都有专用结构 + 失败可回溯的档案"。

## 4. 顺序与依赖

```
B0-gf180 冻结修订（§7.5 一段） ──► G1 提取 ──► G2 macromodel/joint ──► G3 原理图对账 ──► G5 版图/PEX
                                        └────────────► G4 数字 supervisor（并行）──► G6 test plan
```
总量约 7–8 周到 G6 完成；G1+G2 结束即有一个中间检查点（P 表 + γ_eff 实测），若 MOS-only 接口在 anchors 上不可交付，到那时改向（回 SG13G2 或改接口方案）的成本仍然最低。

## 5. 第一步（批准后立即执行）

1. `DESIGN.md §7.5` 追加 B0-gf180 修订小节（上面 §1 的表）；
2. 写 `twin/spice/extract_gf180.py`：先复跑 `PDKS["gf180mcu"]` 的 diode 镜像提取确认环境不回归，再扩到 5V 器件 + `.noise` + MC；
3. `bench/run_gf180.py` 验收骨架 + `results/gf180_extract.json` 落盘。
