# Computo 投稿框架 v2：架构依赖性 + 可复现流水线

> 本文件是写作框架（working document），不是论文正文。英文部分是可直接进 `.qmd` 的草稿文字，中文部分是写作指令与证据台账。
> 目标期刊：[Computo](https://computo-journal.org/site/about.html)（法国统计学会，Diamond OA，CC-BY，ISSN 2824-7795）。投稿形态为 Quarto `.qmd` + `.bib` + 可复现环境配置 + 原始数据/代码，**数值可复现是接收的必要条件**。

---

## 0. 一页速览

**Title**
Which Observables Track Training Phase Transitions? A Systematic Comparison Across ConvNets, Transformers, and MLPs

**Keywords**
Training Dynamics; Phase Transitions; Order Parameters; Reproducibility; Cross-Architecture Comparison

**TL;DR**
Weight entropy tracks a sharp, learning-rate-driven transition in ConvNets but not in Transformers or MLPs: order parameters for training transitions are architecture-conditional.

**Abstract 草稿（198 词，LAR 不出现，只保留 ηc≈0.035 一个数字）**

> Are order parameters for training phase transitions architecture-independent? We address this question with a controlled comparison that holds the measurement fixed and varies the architecture: one definition of the thermodynamic observables (internal energy U, weight-histogram entropy S, free energy F = U − TS), one scanning protocol, and three architectures — a ConvNet on CIFAR-10, and a Transformer and an MLP on modular arithmetic — each scanned over 9–13 learning rates with three seeds and complete checkpoint records.
>
> In the ConvNet we find a sharp entropy instability at η ≈ 0.035, supported by three independent signatures — entropy collapse, a peak in seed-to-seed susceptibility, and bistability at the transition — and robust to a twentyfold change in weight decay (5e-4 to 1e-2); a weight decay of 1.0 destroys the trainable phase and serves as a negative control. The same order parameter fails elsewhere: in the Transformer it is inverted (m peaks at η = 1e-3 and then decreases; Spearman ρ = −0.71), and in the MLP it is flat (m ≈ 0.003) while accuracy remains 97–100% across the entire grid. Behavioural phase transitions are therefore architecture-independent, whereas the entropy order parameter is not.
>
> A second-order free-energy expansion of SGD reproduces the qualitative direction but misses the transition location by a factor of three to four, which we trace to the isotropic-Hessian approximation under AdamW. We release the full pipeline: protocols, per-step metrics, analysis scripts and figure-generation code.

---

## 1. 投稿形态与合规清单

Computo 的评审会实际运行工作流，因此"能否在 CI 里跑完"是形式审查的一部分。我们的训练是小时级（200K 步 × 27–54 runs），不能整包进 CI。

| 项 | 决定 | 说明 |
|---|---|---|
| 正文载体 | `main.qmd` | 图表由代码块生成或读取归档 CSV，二者都保留 |
| 引用 | `references.bib` | 每条必须在 Crossref / arXiv 上核对（见 §5 核验列） |
| 环境 | `environment.yml`（conda）+ `requirements-lock.txt` | 固定 torch / numpy / matplotlib / pandas 版本 |
| 数据 | Zenodo DOI（CSV + 图 + config.json） | 检查点 258 文件 21MB 的 zip 可一并放 Zenodo；`.pt` 不入 git |
| CI 复现 | `demo/` 5K 步 × 1 seed × 3 η | 目标 < 20 分钟，仅验证流水线可跑通，不代表论文结论 |
| 全量复现 | `reproduce_all.sh` + `TRAINING_CONFIGS.md` | 记录硬件与实测耗时（本地 MPS / 远端 2×3090） |
| 许可 | CC-BY 4.0 | 代码 MIT，文本 CC-BY |

**必须在 §2.4 写明的一点**：CI 跑的是 `demo/` 配置，论文所有数字来自 `full/` 配置，两者用 `config.json` 的 `spec` 与 `used_steps` 字段区分——这是诚实性要求，也是审稿人最容易抓的点。

---

## 2. 核心叙事与 claim–evidence 台账

### 2.1 一句话逻辑链

> 序参量的选择在训练动力学里是开放问题 → 我们用统一口径在三架构上扫描 → ConvNet 有尖锐熵失稳（三重签名、跨 20× wd 稳定） → Transformer 倒置、MLP 平坦，但行为相变都真实存在 → 因此"行为相变普适、熵序参量不普适"；二阶自由能理论方向对、定量错 → 结论落地为一条方法论主张：**序参量必须架构感知**，并给出可复用的检验协议。

### 2.2 台账（每条 claim 的证据强度必须与措辞匹配）

| # | Claim | 强度 | 证据 | 若被推翻的观察 |
|---|---|---|---|---|
| C1 | ConvNet 在 η≈0.035 有尖锐熵失稳 | 强 | 13 η × 3 seeds；三重签名；[phase5_theory_scan/](../phase5_theory_scan) | η 网格加密后 m(η) 变平滑 |
| C1b | 该失稳与"训练失败"解耦 | 强（已核对） | m 从 0.002 跳到 0.565（η=0.001→0.05）期间，val_acc 全程 0.700→0.733→0.720（仍在平台）；准确率首次离开平台在 η=0.1（m 已 0.80）；[phase3_free_energy/metrics_per_epoch.csv](../phase3_free_energy/metrics_per_epoch.csv) | 若 m 跳变与 acc 崩塌在同一步发生，C1 需降级为 "entropy tracks failure" |
| C2 | 该失稳不是 wd=5e-4 的微调产物 | 强 | 中点 0.040 vs 0.038（5e-4 vs 1e-2）；[phase8_wd/](../phase8_wd) | 20× wd 内中点移动 > 因子 2 |
| C3 | 该失稳不可由二阶 SGD 自由能理论定量解释 | 强（负结果） | P1 过零 0.09–0.13；P2 差 ~10⁴；[phase5_theory_scan](../phase5_theory_scan) | 各向异性 H̃ 修正后 P2 回到 O(1) |
| C4 | Transformer 上熵序参量倒置 | 强 | m 峰 0.54 @ η=1e-3，ρ=−0.71；[phase7_transformer_fe](../phase7_transformer_fe) | 扩到 20+ η 后出现单侧转变 |
| C5 | MLP 上熵序参量全 η 平坦 | 强 | m 0.002–0.005，acc 97–100%；[phase7_mlp_fe](../phase7_mlp_fe) | 换 wd/宽度后出现转变 |
| C6 | 行为相变（grokking 窗口）在三架构都存在 | 强 | 行为相图；[phase7_critical](../phase7_critical) | — |
| C7 | 注意力温度冷却先于 grokking | 中 | 32→0.94 / 1600 步，领先 5200–8200 步，单架构 3 seeds | 换任务后不再领先 |
| C8 | 熵产可作为无验证集的停止准则 | 中 | 触发 epoch 60/61/71 vs 最优 13/14/19；acc 74.18 vs 75.20 | 在更多任务上差距消失 |
| C9 | ηc 与 β 的定量值 | 弱（探索性） | ηc=0.0345 (0.0295–0.0495)，β=0.243 (0.075–0.394) | — |
| C10 | 有限尺寸标度可识别 | 弱（不成立） | 最优参数顶在网格边界 | 加密 η 后塌缩稳定 |

**写作强制规则**：C9/C10 只能出现在 §3.4 与 §8.4，措辞必须是 "exploratory / not identified"，不得进入摘要与结论句。C4/C5 的因果表述只能是 "the entropy order parameter does not track"，**不能**写成 "Transformers have no phase transition"（行为相变确实存在）。

### 2.3 与旧版的三处叙事改动

| 维度 | 旧版 | 本版 |
|---|---|---|
| 核心贡献 | 热力学签名 | 系统数值检验 + 架构依赖性 + 可复现流水线 |
| 负面结果 | 谨慎措辞、藏附录 | 提升为核心发现（C3/C4/C5 平权） |
| 贡献主语 | 我们发现的物理 | 我们提供的**比较方法与协议**（Computo 的 neutral comparison 定位） |

---

## 3. 分节写作框架

字数按 Computo 版式（约 550 词/页）估算。

### §1 Introduction（1.5 页，约 800 词，最后写）

- 段 1：训练动力学中的相变现象（grokking、double descent、edge of stability）——**注意**：这里只陈述"存在突然的结构重排"，不预设热力学语言。
- 段 2：问题陈述（Computo 式）：*which observable, under which conditions, reliably indicates a training phase transition?* 明确指出序参量选择缺少中立比较。
- 段 3：方法概述（四句）：统一观测量定义、统一扫描协议、三架构、全量公开。
- 段 4：主要发现（四句，对应 C1/C2 + C4/C5 + C3）。
- 段 5：六条贡献（与 §9 逐条一致，**编号必须相同**）。
- 段 6：组织。

引用：Power 2022、Nakkiran 2020、Cohen 2021（现象）；Bahri 2020（统计力学视角）；Pineau 2021（可复现性）。

### §2 Experimental Framework and Reproducibility（2 页，约 1100 词，先写，定调）

**2.1 热力学量定义**（每个量一行公式 + 一句物理解释）
- U：交叉熵损失在固定评估集上的均值。
- S：权重值直方图熵，100 个固定 bin，`S_norm = S / ln 100 ∈ [0,1]`（**必须写归一化**，否则 m 不可比）。
- F = U − T S，其中 T 由 η 扮演（本版统一称 "noise scale set by η"，**不使用 T 指学习率**）。
- T_attn：Transformer 的 softmax 温度参数，由权重直接读出（32→0.94）。**术语冲突预警**：T_attn 与"η 扮演温度"必须显式区分，全文只用两个符号：`η`（学习率/噪声尺度）与 `T_attn`（模型温度）。建议在 2.1 加一句 "We use η, not T, for the learning-rate scale to avoid collision with T_attn"。
- σ = dS/dt + G − ηQ（细节入 §7，此处只定义）。

**2.2 序参量**：`m = 1 − S_final`，值域 [0,1]；三条签名判据（塌缩幅度、跨种子磁化率峰、转变点双稳态）。

**2.3 扫描协议**（表格化，一表给全）：
- η 网格：ConvNet 13 点 `[1e-4 … 0.15]`；模加 9 点 `[1e-4 … 0.1]`。
- seeds=3；步数：ConvNet 200K、Transformer 25K、MLP 100K；记录间隔 1000（基线实验 100）。
- 固定 split_seed=42、30% 训练子集；AdamW。
- **bin edges 由 seed-0 初始化确定并在全部 run 复用**——这是跨 η、跨 wd 可比的前提，必须成段解释，否则审稿人会问"熵的定义随 run 变化"。

**2.4 可复现性基础设施**
- 每个 run 保存 `metrics_*.csv`（逐 1000 步）、`config.json`（`spec` vs `used_steps`）、最终检查点。
- 熵产触发规则的稳健化：`w=21` 滑动平滑 + 连续 5 epoch 判据；朴素单点翻转保留在 `naive_signflip_epoch` 列（**保留naive列是诚实性的关键**，避免"事后调参"指控）。
- 目录结构 + 环境 + 随机种子控制（torch/numpy/python 三处 seed，MPS 无 float64 的显式转换顺序）。
- CI demo 与 full 的区分（见 §1）。

**2.5 中立比较原则**（本版新增的实质章节，Computo 会重点看）
同一序参量定义、同一记录口径、同一分析脚本、跨架构不加任何特判；所有架构的拟合用同一 `fit_once` + 同一 bootstrap 流程；失败/发散 run 按同一规则排除（未跑满步数或 S 非有限）。

### §3 ConvNet: A Sharp Entropy Instability（2 页，约 1100 词）

- **3.1 熵塌缩 + 与准确率解耦**：13 η 点，低 η 平台 m<0.002，高 η 平台 m≈0.84，转变区仅 2–3 个 η 点宽。**必须同图同轴给出 val_acc**：η=0.001/0.01/0.03/0.05/0.1 时 m=0.002/0.017/0.108/0.565/0.803，而 val_acc=0.700/0.722/0.733/0.720/0.612——熵在 η≈0.05 已跳掉 56%，准确率仍处在平台最高点；准确率首次离开平台在 η=0.1，此时 m 已 0.80。这一段是回应"m 只是训崩了"的决定性证据，写作上要放在 3.1 而不是留到讨论。
- **3.2 磁化率峰**：`Var_seed(S)` 峰在 η=0.035，与拟合 ηc 重合（这是独立于拟合的签名，重点强调）。
- **3.3 双稳态**：L=8、η=0.03 时 seed 间 std=0.106；单 run 时间轨迹是否也双稳？（若没有，措辞降为 "seed-level bistability"）。
- **3.4 临界指数**：ηc=0.0345（0.0295–0.0495），β=0.243（0.075–0.394），5000 次自助。**必须写**：bootstrap 分布多峰、紧邻 ηc 只有 2 个点、故 β 为探索性；给出"要定论需在 0.030–0.040 以 0.002 密铺 × 5 seeds"。
- **3.5 wd 鲁棒性**：5e-4→1e-2 中点 0.040 vs 0.038；wd=1.0 为死网络对照（η=1e-3 时 m 已 0.20、acc 0.58；η≥0.03 全 chance、S≈0.15），说明高 m 可来自"权重被钉在零"而非"超阈值谱集中"——**这一段同时是 C2 的正面证据和"m 高≠相变"的负面控制**，写作上要并列呈现，不能只讲前者。

图 1、图 2。

### §4 Transformer: Attention Temperature（1.5 页，约 800 词）

- **4.1 温度坍缩**：32→0.94，1600 步内完成；grokking 步 5700/7200/7400，领先 5200–8200 步。
- **4.2 熵–温度扫描**：Tc=2.71±0.48；稳态 T 在 Tc 以下约 2.9 倍。**注意**：这是"T 与 S 的关系"，不是"η 与 S"的关系，必须与 §5 的 η 扫描分开陈述，否则读者会混。
- **4.3 热力学解读**："cooling precedes generalization"——只作为描述性解读，不作因果断言（单架构、单任务）。
- **4.4 补充观测**：一句话交代 LAR（谱几何量、与泛化同步但非热力学量），指向附录 B.9。**正文不再给 LAR 数字**。

图 3。

### §5 Cross-Architecture Comparison（2 页，Computo 核心章节，约 1100 词）

- **5.1 Transformer 倒置**：m 峰 0.54 @ η=1e-3 → 0.35 @ 0.1，Spearman −0.71；低 η 处 m 已 0.48，**不存在 m≈0 的有序相**。
- **5.2 MLP 平坦**：m 0.002–0.005 全网格；acc 97–100%；ρ=−0.36。
- **5.3 行为 vs 熵**：双面板图；Transformer 的 grokking 窗口 1e-3–5e-3（~100%），1e-4 约 15%，0.1 约 14% 且种子方差大；MLP 从 3e-4 到 0.1 全部 97–100%。结论句只写：*behavioural transitions are present in all three architectures; S_final is not responsive to them.*
- **5.4 三架构汇总表 + 讨论**：为什么直方图熵在注意力架构上失效？给出三条候选解释（几何 vs 直方图；AdamW 自适应度规改变权重尺度分布；任务/正则化混杂），并明确说明**本轮不能区分**（这是 8.4 局限的前置）。
- **5.5 协议模板**（本版新增，也是方法贡献）：一张 6 步清单，供他人在新架构上复用——(i) 固定 bin edges 于 seed-0；(ii) 至少 9 个对数间距 η；(iii) 3 seeds 起；(iv) 三条签名齐备才算转变；(v) 同期记录行为指标作为对照；(vi) 用同一 fitter 与同一 bootstrap。

图 5。

### §6 Free Energy Theory: Qualitative Success, Quantitative Gap（1.5 页，约 800 词）

- **6.1 推导**（正文只给结论式，完整推导进附录 A）：
  `E[ΔU] = −ηG + (η²/2)·tr[H(ggᵀ + Σ/B)]`，记 `Q = tr[H(ggᵀ+Σ/B)]`；连续极限 `dF/dt ≈ −ηG(1 − η/ηc*)`，`ηc* = G/Q`。
  P1：dF/dt 的零点在 ηc*；P2：ηc* 与序参量拟合的 ηc 应一致。
  引用：Engel & Van den Broeck 2001；Seifert 2012（熵产）；Cohen 2021（η 作为有效噪声尺度）。
- **6.2 对比**：P1 外推 0.09–0.13（早期窗口 0.127 / 0.115），差 3–4 倍；P2 的 G/Q 为 826–3195，即 η²Q 比 ηG 小约 10⁴，**与 0.035 差四个数量级**。
- **6.3 各向异性修正**：AdamW 下有效度规 `H̃ = D^{-1/2} H D^{-1/2}`，D 为二阶矩；`Q_iso` 被高估/低估的原因写清楚；给出可检验预测（修正后 Q 应与 G/ηc 同量级）——**这是"数值研究阐释理论结果"的落点**，也是审稿人最欢迎的"可证伪的下一步"。
- **6.4 定位**：定性方向正确（F 单调下降、存在零点）、定量需修正；明确写 "we report this as a quantitative failure of the isotropic approximation, not of the thermodynamic framing as a whole"（避免把负结果写成全盘否定，也避免护短）。

图 4（a/b/c/d）。

### §7 Entropy Production as a Stopping Criterion（1 页，约 550 词）

- **7.1 定义**：σ = dS/dt + G − ηQ ≥ 0；系统=权重，热浴=小批量数据流；三处噪声源（直方图估计器、批量扩散 ∝ η²、跨种子临界涨落）。
- **7.2 实证**：触发 epoch 60/61/71（三 seed）vs 验证最优 13/14/19；节省 87.5%（熵产）vs 97.2%（验证早停）；准确率 74.18 / 72.24 / 75.20。→ 结论：**次优于验证早停，无验证集时可用**。数字提交前需用 `phase4_entropy_stop/` 重算一次并写入附录 B.5。
- **7.3 量纲修正**：1000 步分辨率下 std(ΔS)=0.001–0.006，估计器地板 √2·std(S)=0.0024–0.075，8 个 η 中 7 个低于地板 → 物理 dS/dt 需聚合到更大时间窗。**这一段是本版相对旧版的实质改进**（旧版有量纲错误，已修正），写作时要给出"同单位比较"的判据，不能只报比值。
- **7.4 定位**：工具性结论，不外推到"熵产是普适停止准则"。

图 6。

### §8 Discussion（1.5 页，约 800 词）

- **8.1 五条热力学事实**（逐条一句 + 强度标签）：(i) ConvNet 存在尖锐熵失稳；(ii) 跨 20× wd 位置不变；(iii) 跨架构不复现；(iv) 二阶理论定性对、定量差 3–4 倍/4 个量级；(v) 临界涨落峰与转变点重合。
- **8.2 核心发现**：架构依赖性不是"理论失败"，而是"序参量选择必须架构感知"。给出机制假设（直方图熵对权重尺度分布敏感，而注意力架构的泛化以低秩几何重组为主）并明确标为假设。
- **8.3 方法贡献**：可复现流水线 + 中立比较协议（含 5.5 模板）。
- **8.4 局限**（逐条，务必与 C 台账的强度标签一致）：协议不完全同构（任务/wd 差异）；β 精度不足；温度只在一个架构；FSS 未识别；单序参量为主（其他候选量见附录 B.6/B.9）。
- **8.5 未来工作**：用本流水线检验其他候选序参量（谱熵、稳定秩、参与比、Jacobian 谱）——**并列一张"候选量清单 + 已测结果"小表**，这是对标题的正面回应。

### §9 Conclusion（0.5 页，约 250 词）

一段。三句结构：问题 → 发现（行为相变普适、熵序参量不普适）→ 方法论主张（架构感知的序参量检验 + 公开协议）。不重复数字。

---

## 4. 图表归置（修正原大纲的一处编号错误）

**修正说明**：原大纲把"图 4（F/U/S 轨迹、F_min vs η、U–S 相图、熵产停止）"放在 §4.5–4.6，但 §4 是 Transformer 章节且只到 4.4，且这四组数据分别属于 ConvNet 自由能与停止准则。本版按"图随章节"原则重排，仍为正文 6 张：

| 图 | 位置 | 面板 | 数据来源 | 重绘入口 |
|---|---|---|---|---|
| 图 1 | §3.1–3.3 | (a) m(η)，右轴叠加 val_acc（关键：解耦） (b) Var_seed(S) 峰 (c) 双稳态 | `phase5_theory_scan/`、`phase3_free_energy/metrics_per_epoch.csv`、`phase7_critical/` | `phase7_critical_analysis.py` |
| 图 2 | §3.5 | 三档 wd 的 m(η) | `phase8_wd/` | `phase8_wd_analysis.py` |
| 图 3 | §4.1–4.2 | (a) T_attn 轨迹 + grokking 标记 (b) S–T 扫描 + Tc | `phase2_attention_temperature/` | 该目录脚本 |
| 图 4 | §6.1–6.2 | (a) F/U/S 轨迹 (b) F_min vs η (c) U–S 相图 (d) P1/P2 对比 | `phase3_free_energy/`、`phase5_theory_scan/` | `phase5_theory_scan.py` |
| 图 5 | §5.1–5.3 | (a) 三架构 m vs η (b) 行为相图 acc vs η | `phase7_critical/` | `phase7_critical_analysis.py` |
| 图 6 | §7.2 | 熵产触发 vs 验证最优（epoch / 节省 / 准确率） | `phase4_entropy_stop/` | 该目录脚本 |

附录图：B.4 自助分布、B.6 基线信号、B.7 FSS 原始曲线、B.8 噪声地板、B.9 LAR。**LAR 不进正文**。

每张图在 `.qmd` 里必须带 `#| label: fig-xxx`，正文用 `@fig-xxx` 引用；图的 caption 只写"看到什么"，结论放正文（Computo 评审反感 caption 里下结论）。

---

## 5. 参考文献（真实条目，提交前仍需 Crossref/arXiv 复核）

核验列：`✓` = 本次会话已通过检索确认标题/出处；`—` = 领域标准条目，需在 `.bib` 生成时复核（不核对不提交）。

**Grokking 与训练动力学**
1. ✓ Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177, 2022.
2. ✓ Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023; arXiv:2301.05217.
3. ✓ Liu, Kitouni, Nolte, Michaud, Tegmark, Williams. *Towards Understanding Grokking: An Effective Theory of Representation Learning.* NeurIPS 2022.
4. ✓ Liu, Michaud, Tegmark. *Omnigrok: Grokking Beyond Algorithmic Data.* ICLR 2023.
5. ✓ Gromov. *Grokking Modular Arithmetic.* arXiv:2301.02679, 2023.
6. — Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever. *Deep Double Descent.* ICLR 2020; arXiv:1912.02292.
7. — Zhang, Bengio, Hardt, Recht, Vinyals. *Understanding Deep Learning Requires Rethinking Generalization.* ICLR 2017; arXiv:1611.03530.
8. — Barak, Edelman, Goel, Kakade, Malach, Mehta. *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit.* NeurIPS 2022.

**统计力学与相变**
9. ✓ Bahri, Kadmon, Pennington, Schoenholz, Sohl-Dickstein, Ganguli. *Statistical Mechanics of Deep Learning.* Annu. Rev. Condens. Matter Phys. 11:501–528, 2020.
10. ✓ Bahri, Dyer, Kaplan, Lee, Sharma. *Explaining Neural Scaling Laws.* PNAS 121(27):e2311878121, 2024.
11. — Engel, Van den Broeck. *Statistical Mechanics of Learning.* Cambridge University Press, 2001.
12. — Zdeborová, Krzakala. *Statistical Physics of Inference: Thresholds and Algorithms.* Advances in Physics 65:453–552, 2016.
13. — Saad, Solla. *On-Line Learning in Soft Committee Machines.* Phys. Rev. E 52:4225, 1995.
14. — Goldenfeld. *Lectures on Phase Transitions and the Renormalization Group.* Addison-Wesley, 1992.

**优化、Hessian 与噪声尺度**
15. — Cohen, Kaur, Li, Kolter, Talwalkar. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR 2021; arXiv:2103.00065.
16. — Loshchilov, Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019; arXiv:1711.05101.
17. — Jastrzębski, Kenton, Arpit, Ballas, Fischer, Bengio, Storkey. *Three Factors Influencing Minima in SGD.* arXiv:1711.04623, 2017.
18. — Chaudhari, Soatto. *Stochastic Gradient Descent Performs Variational Inference, Converges to Limit Cycles for Deep Networks.* ICLR 2018; arXiv:1710.11029.
19. — Keskar, Mudigere, Nocedal, Smelyanskiy, Tang. *On Large-Batch Training for Deep Learning.* ICLR 2017; arXiv:1609.04836.
20. — Yao, Gholami, Keutzer, Mahoney. *PyHessian: Neural Networks Through the Lens of the Hessian.* arXiv:1912.05270, 2020.

**熵估计与非平衡热力学**
21. — Paninski. *Estimation of Entropy and Mutual Information.* Neural Computation 15(6):1191–1253, 2003.
22. — Grassberger. *Estimating Mutual Information.* Phys. Rev. E 69:066138, 2004.
23. — Seifert. *Stochastic Thermodynamics, Fluctuation Theorems and Molecular Machines.* Rep. Prog. Phys. 75:126001, 2012.

**可复现性**
24. — Pineau et al. *Improving Reproducibility in Machine Learning Research.* JMLR 22(1):1–20, 2021.

**引用纪律**
- 不引用任何"权重直方图熵是训练序参量"的先验工作——我们检索未发现直接对应文献，这正是 §1 段 2 的 gap 陈述基础。**不得为了显得有根基而编造近邻文献**。
- §3.4/§6 的物理类比只引 9/11/14 三条（统计力学综述 + 教材 + 相变教材），不引具体"某模型某相变"的个案，避免被要求做严格对应。

---

## 6. 审稿人预判与应答（写作时就把答案埋进正文）

| # | 预判质疑 | 应答策略（写在哪） |
|---|---|---|
| 1 | "m 的跳变只是高 η 训崩了" | **已核对，可正面反驳**：m 从 0.002 跳到 0.565 期间 val_acc 一直是 0.70–0.73 的平台，准确率崩塌发生在 η=0.1、m 已 0.80（§3.1 + 图 1a 右轴）；再加 wd=1.0 的死网络对照（§3.5）反证"m 高≠相变" |
| 2 | "三架构任务/正则化不同，架构依赖性是混杂" | 承认（8.4），但 wd 对照（§3.5）排除了 ConvNet 侧的正则化解释；给出决定性交叉实验（ConvNet on mod-add / Transformer on CIFAR）作为未来工作 |
| 3 | "β 不可信" | 主动写在 3.4，报 CI 与多峰，并给加密网格方案；位置改用无模型中点作为主估计 |
| 4 | "只有一个序参量，标题过大" | §8.5 候选量清单表 + 附录 B.6/B.9；标题用 "Observables"（复数）并用表格兑现 |
| 5 | "熵的定义随 bin 变化" | §2.3 明确 seed-0 固定 edges + 复用规则，并给 bin 数敏感性检查（附录） |
| 6 | "理论检验是否循环论证" | §6.1 强调 P2 用局域量（无需终态拟合），与 ηc 的拟合路径独立 |
| 7 | "CI 跑不完" | §1 的 demo/full 双配置 + Zenodo 归档 |

---

## 7. 术语与口径一致性（提交前逐项 grep）

| 符号 | 唯一含义 | 禁用写法 |
|---|---|---|
| `η` | 学习率 / 噪声尺度 | 不用 T 指学习率 |
| `T_attn` | Transformer softmax 温度 | 不简写为 T |
| `S` | 归一化权重直方图熵（S/ln K） | 不写"entropy"无限定 |
| `m` | 1 − S_final ∈ [0,1] | 不写 "order parameter" 不加架构限定 |
| `ηc` | 熵序参量转变点 | 不与 ηc*（理论预测）混用——**两个符号必须区分**：`η_c` 与 `η_c^*` |
| `β` | 临界指数 | 不与 β（AdamW 的 betas）混用 |
| "phase transition" | 仅用于**有 ≥2 条签名**的场合 | 单条签名只能写 "instability" / "crossover" |
| "architecture-dependent" | 结果陈述 | "universal / architecture-independent" 只用于**行为**相变 |

---

## 8. 待补分析与已知缺口（提交前必须处理）

1. ~~S vs val_acc 同期核对~~ **已完成**：解耦成立，且是强证据（见 C1b、§3.1、图 1a）。**待做的是把它做进图 1(a) 的右轴**——目前是两条数据源拼接（phase3 的 `metrics_per_epoch.csv` + phase5 的 `metrics_theory_scan.csv` 无 acc 列），绘图时须用同一 lr 对齐并标注哪 5 个点来自 phase3、哪 8 个点无 acc 而不画。

   附：`phase5_theory_scan/metrics_theory_scan.csv` **不含任何准确率列**，13 点合并图里只有 5 个 η 有 acc。若要将图 1(a) 的 acc 覆盖全部 13 点，需从 `final_states.csv` 或补一次轻量评估（无需重训，加载检查点评估即可）。**这是提交前必须解决的数据缺口**。
2. **bin 数敏感性**：用 50/100/200 bins 重算 m(η) 曲线（只需重算 CSV，不需重训），证明 ηc 位置不随 bin 数移动。
3. **β 的加密网格**：η∈[0.030,0.040] 步长 0.002 × 5 seeds（约 30 runs）才能把 §3.4 从"探索"升为"定量"。不做就在正文保持弱表述。
4. **§7 数字重算**：`phase4_entropy_stop/` 三 seed 的触发 epoch、节省比例、准确率，以脚本输出为准写附录 B.5。
5. **图 4 与图 2 的 caption 去结论化**；所有图在 `.qmd` 中可重生成。

---

## 9. 写作与提交流程

1. §2 → §3 → §5（先把定调与最硬数据写死）
2. §4 → §6 → §7
3. §8 → §9
4. §1 + Abstract（回填与全文一致性）
5. 附录 A（推导）、B（补充）、C（复现指南，与 §2.4 呼应，不重复）
6. 图重绘 + caption 审阅 + `@fig-` 引用检查
7. 内部"审稿人演练"：按 §6 表格逐条自答，任何答不上来的条目回到 §8 补分析
8. BibTeX 复核（§5 的 `—` 条目全部核对）→ Zenodo DOI → CI demo 跑通 → 提交

**贡献列表（六条，与 §1 段 5、§9 必须逐字一致）**
1. 系统性跨架构数值检验：同一序参量定义、同一协议，ConvNet / Transformer / MLP 上 9–13 个 η × 3 seeds。
2. ConvNet 熵失稳的稳健性证据：三重签名 + 跨 20× wd 不变 + 死网络对照。
3. 架构依赖性的明确发现：行为相变普适，熵序参量不普适。
4. 自由能理论的定量边界：P1/P2 偏差 + 各向异性修正方向。
5. 熵产停止准则的实证评估：次优于验证早停，无验证集时可用。
6. 可复现流水线：完整 metrics/config、分析脚本、复现指南，作为后续研究的实验框架。
