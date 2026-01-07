# STAMP 项目说明：对抗训练开关、损失函数设计与阅读路线

> 适用范围：本仓库 `STAMP/`（支持 `pred_model={gat,mamba}` 与 `recon_model={ae,mamba}` 的组合）。
>
> 本文解释：
> 1) 为什么需要新增 `--use_adv`（关闭对抗训练）参数；
> 2) `use_adv=True/False` 时损失函数分别是什么，且对 GAT+AE 与 Mamba 的差异影响；
> 3) 要深入理解算法与工程实现，应从哪些文件开始阅读（给出推荐路线）。

---

## 1. 为什么要新增 `--use_adv`：原始损失在 mamba 上有什么问题？

### 1.1 背景：训练不是“单纯预测 + 单纯重构”
本项目的训练本质上包含两类目标：

- **基础目标（稳定）**：
  - 预测分支最小化预测误差 `pred_loss`
  - 重构分支最小化真实窗口的重构误差 `ae_loss`

- **耦合/对抗目标（不稳定来源）**：
  - 预测分支不仅要预测准确，还要让“拼接后的生成窗口”能被 AE 重构得更好
  - 重构分支除了要重构真实窗口，还会对“生成窗口”的重构误差使用 **减号**（相当于鼓励生成窗口重构变差），形成典型的 min-max / 对抗结构

当模型容量较强（如 Mamba 重构器），这种对抗结构更容易出现“走极端”的优化行为，从而造成训练曲线在早期变好、后期发散。

### 1.2 你观察到的现象如何对应到代码与公式
你在日志里看到：

- 前 1～2 个 epoch 很快达到最小 val loss
- 后续 epoch 几乎“无效训练”，并且 loss 逐渐变大
- GAT+AE 相对更平稳

这通常不是简单的学习率问题，而是**目标函数本身的权重调度导致训练后期更偏向对抗项**。

在训练代码里，对抗/耦合项的系数随 epoch 增大而增大（例如 `(1 - 1/epoch)` 逐渐接近 1），导致：

- 前期：以稳定项为主（`pred_loss`、`ae_loss`），所以很快下降
- 后期：对抗项权重变大，优化方向发生偏移，AE 分支会更倾向于“破坏某个生成窗口重构”，从而使验证损失（尤其 AE 相关）上升

**Mamba** 作为预测/重构的 backbone 表达能力更强，对抗项更容易被“优化到极端”，因此更容易出现后期损失变差。

### 1.3 为什么要加开关，而不是直接调学习率

- 学习率调小（如 1e-3 → 1e-4）只能减缓发散速度，但无法改变“后期对抗项占主导”的事实。
- 在实际工程上，我们需要一个**稳定基线**：先用稳定目标训练出可靠模型，再考虑是否引入对抗耦合提升效果。

因此新增 `--use_adv` 的意义是：

- **快速切换稳定/对抗训练模式**
- 方便定位问题是否来自对抗项（而非数据、模型或实现细节）
- 为 Mamba 组合提供可复现、可控的训练路径

---

## 2. 打开与关闭对抗训练：损失函数分别是什么？对 GAT+AE 与 Mamba 有何差异？

> 下面的符号约定：
> - `x`: 输入窗口 `[B, T-n_pred, N, C]`
> - `y`: 目标未来片段 `[B, n_pred, N, C]`
> - `pred(x)`: 预测模型输出
> - `AE(z)`: 重构模型输出
> - `generate = concat(x, pred(x))`: 生成窗口（拼接后的完整窗口）

### 2.1 关闭对抗训练（`--use_adv False`）：稳定的“分支独立”目标
训练时每个 batch 只包含两次更新（各更新一次）：

#### (1) 预测分支：
\[
L_{pred} = \mathrm{MSE}(pred(x),\ y)
\]

#### (2) 重构分支（真实窗口重构）：
\[
L_{ae} = \mathrm{MSE}(AE(batch),\ batch)
\]

日志中为了兼容原格式：
- `loss1 = L_pred`
- `loss2 = L_ae`

这一模式的特点是：

- 目标函数单调、稳定
- 对模型训练更友好
- 更适合作为 Mamba 组合的默认训练方式与调参基线

### 2.2 打开对抗训练（`--use_adv True`）：4 次更新（2 次基础 + 2 次耦合/对抗）
该模式下，每个 batch 内会发生 **4 次更新**：

1) pred 基础更新（最小化 `L_pred`）
2) AE 基础更新（最小化 `L_ae`）
3) pred 耦合更新（最小化 `loss1`）
4) AE 对抗更新（最小化 `loss2`，含减号项）

#### (3) 预测分支耦合项：让生成窗口也易于被 AE 重构
定义生成窗口重构误差：
\[
L_{adv} = \mathrm{MSE}(AE(generate),\ generate)
\]

然后 pred 分支的耦合损失（代码里权重随 epoch 变化）：
\[
loss1 = a(e)\,L_{pred} + b(e)\,L_{adv}
\]
其中常见形式（仓库实现中类似）：
- `a(e) = 5/e`
- `b(e) = 1 - 1/e`

直觉：pred 不仅预测准，还要让拼接后的生成序列在 AE 意义下“看起来更正常”。

#### (4) 重构分支对抗项（关键）：包含减号项
同样定义生成窗口重构误差（在 AE 更新时记作 `L_adv2`）：
\[
L_{adv2} = \mathrm{MSE}(AE(generate),\ generate)
\]

AE 的对抗式耦合损失：
\[
loss2 = c(e)\,L_{ae} - d(e)\,L_{adv2}
\]
其中常见形式：
- `c(e) = 3/e`
- `d(e) = 1 - 1/e`

**注意：`- d(e) * L_adv2` 是“减号”。**

这意味着：AE 一方面要重构真实窗口（小 `L_ae`），另一方面又被鼓励让生成窗口重构变差（大 `L_adv2`）。
当 `d(e)` 在后期变大时，训练可能发生目标偏移，导致：

- 验证 AE loss 上升
- 最佳 epoch 出现在很早期
- 后期训练越跑越差

### 2.3 对 GAT+AE 与 Mamba 的差异影响

#### GAT + 传统 AE（`pred_model=gat, recon_model=ae/mamba`）
- GAT/传统 AE 通常参数量与动态表达能力相对“温和”
- 对抗目标即使存在，也不容易把训练推到极端
- 更常见的表现是：收敛慢一些，但曲线更平稳，后期仍有波动但不至于发散

#### Mamba（尤其 `pred_model=mamba, recon_model=mamba`）
- 模型容量更强、优化更“锋利”
- 对抗项（尤其 AE 的减号项）更容易被优化器走到极端解
- 结果就是你观察到的：
  - 前期快速下降（基础项好优化）
  - 后期随着对抗项权重变大而偏离（验证 loss 变差）

因此 `--use_adv` 对 Mamba 的价值更大：
- 用 `False` 提供稳定训练基线
- 用 `True` 才去尝试对抗耦合收益，并进一步引入更细粒度超参（例如对抗系数、warmup、逐步启用等）

---

## 3. 想深入了解算法与执行流程：阅读路线（按推荐顺序）

下面给出一条“从入口到细节”的阅读路线。建议按顺序看，先搞清整体数据流、训练循环，再钻进模型和评估。

### 3.1 第 0 步：快速了解运行方式
- `README.md`
  - 理解项目目标、数据集支持情况、常用命令

### 3.2 第 1 步：入口脚本（训练/测试）
- `run.py`
  - 训练入口：解析参数 → 加载数据 → 构建模型 → 交给 `Trainer`
  - 这里也能看到：`--use_adv`、学习率、MAS、downsample 等超参从哪里传入
- `test.py`
  - 测试入口：加载模型 → 推理得到 score → 搜索阈值/计算 F1
- （可选）`test_explain.py`
  - 在 `test.py` 的基础上增加 explain pipeline（用于异常段解释/模板说明）

### 3.3 第 2 步：训练/测试循环与损失函数实现
- `trainer.py`
  - **核心文件**：
    - `Trainer.train_epoch()`：训练过程、`use_adv` 开关逻辑、loss1/loss2 计算
    - `Trainer.val_epoch()`：验证过程
    - `Tester.testing()`：测试阶段如何计算 score（包含 loss1/loss2/loss3 加权）

阅读建议：
1) 先定位 `Trainer.train_epoch`，看 `use_adv` 分支。
2) 再看 `Tester.testing`，理解最终 anomaly score 是怎么来的。

### 3.4 第 3 步：数据加载与窗口构造
- `lib/dataloader_smd.py`
- `lib/dataloader_msl_smap.py`
- `lib/dataloader_swat.py`
- `lib/dataloader_wadi.py`
- `lib/utils.py`
  - 包含 `MyDataset`、downsample、moving average（MAS）等

阅读要点：
- 原始序列如何切成 `[num_windows, window_size, nnodes, channels]`
- 标签对齐、downsample 与 MAS 对维度的影响

### 3.5 第 4 步：模型结构（GAT 与 Mamba）
- GAT / STAMP 结构：
  - `model/net.py`（若存在）或 `model/layers.py`
  - `model/utils.py`（初始化、打印参数等）
- Mamba 结构：
  - `model/mamba_wrappers.py`（构建预测/重构两分支的统一入口）
  - `model/mamba_forecast.py`
  - `model/mamba_recon.py`

建议路线：
1) 先看 `model/mamba_wrappers.py`，因为它告诉你预测/重构分支如何拼装。
2) 再分别看 forecast / recon 的 forward 输入输出维度。

### 3.6 第 5 步：评估、阈值搜索、指标
- `lib/evaluate.py`
  - `get_final_result` 等阈值搜索与 F1 计算逻辑
- `lib/metrics.py`
  - 基础指标/损失（如 masked MSE）
- `lib/spot.py`
  - POT（Peaks-Over-Threshold）相关（如果你跑 unsup/POT 方案）

### 3.7 第 6 步（工程相关）：日志、输出目录、可复现
- `lib/logger.py`
  - 日志文件命名、tqdm 兼容输出、超参数摘要
- `lib/paths.py`
  - 输出目录 `expe/{log,pdf,pth}` 的统一解析

---

## 4. 实践建议：如何用这些知识定位训练问题

1) 先用稳定配置跑通：
   - `--use_adv False`
   - 观察 `val_pred_loss` 与 `val_ae_loss` 是否都稳

2) 再开启对抗：
   - `--use_adv True`
   - 重点观察 `val_ae_loss` 与 `val_adv_loss` 的关系，如果 AE loss 后期持续上升，很可能就是对抗项权重过大导致

3) 若必须保留对抗训练：建议扩展为可调超参（后续可做）
   - 对抗项系数 `lambda_adv`
   - warmup epoch（前若干 epoch 不启用对抗项）
   - 只做一次耦合更新（省掉 batch 内 4 次更新中的一部分）

---

## 附：关键文件索引（快速跳转）
- 训练入口：`run.py`
- 测试入口：`test.py`
- 对抗训练与损失核心：`trainer.py`
- 数据加载：`lib/dataloader_*.py`
- Mamba 组装：`model/mamba_wrappers.py`
- Mamba 预测：`model/mamba_forecast.py`
- Mamba 重构：`model/mamba_recon.py`
- 评估/阈值：`lib/evaluate.py`
- 日志/输出目录：`lib/logger.py`, `lib/paths.py`

