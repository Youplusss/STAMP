# STAMP + Mamba Hybrid 消融实验计划

> 目标：用**结构拆分 + 训练策略拆分 + 分数融合拆分**三条主线，验证你把 **S-D-Mamba（预测）**与 **MambaAD（重构）**改造成 1D 并接入 **STAMP 框架**后，各组件对最终异常检测 **F1 / Precision / Recall** 的贡献。

本计划配套你当前工程的可控维度：
- **预测分支** `pred_model ∈ {gat, mamba}`
- **重构分支** `recon_model ∈ {ae, mamba}`
- **STAMP 对抗耦合训练** `use_adv_train ∈ {True, False}`
- **预测 Mamba 结构开关**：bi/uni、FFN、norm、residual、MAS-token
- **重构 Mamba 结构开关**：multi-scale、global/local、kernel sizes、bi/uni、global depth、输出激活

> 建议先做“框架级消融”把大方向验证清楚，再做细粒度结构消融。否则很容易把超参/融合权重差异误判成结构贡献。

---

## 1. 实验统一设置（保证可对比）

### 1.1 数据集与划分
- **SMD**：用于主结果与消融主战场（你希望达到 F1>0.95 的核心）
- **MSL**：用于跨域验证（防止过拟合 SMD 配置）

> 如果你论文/报告里还需要，可以补充 SMAP；但消融优先 SMD+MSL。

### 1.2 统一超参数（建议默认值）
- `window_size = 15`
- `n_pred = 3`
- `batch_size = 128`
- `epochs = 40`（配合 early-stop）
- `lr = 1e-3`
- `seed ∈ {1,2,3}`（至少 3 个随机种子取均值±方差）
- `topk = 1`（先固定；后续再做 topk 消融）

### 1.3 统一评估协议
- 评估指标：`best_f1 / precision / recall / auc / mcc`（你现有 `evaluate.py` 已支持）
- 阈值搜索：沿用 `bf_search`（保证所有消融在同一阈值策略下比较）

### 1.4 统一输出与记录
每个实验保存：
- checkpoint：`log_dir/best_model_{data}_{model}.pth`
- json：`log_dir/result.json`（建议用 `test.py --save_result_json`）

你也可以用脚本自动汇总：`scripts/run_ablation_grid.py`（见第 6 节）。

---

## 2. 消融总览（先大后小）

把消融分成 5 组：

1) **G0 框架级**：预测/重构分支替换是否有效？耦合训练是否有效？
2) **G1 预测 Mamba（S-D-Mamba 风格）**：bi-Mamba、FFN、norm、residual、MAS token 等贡献
3) **G2 重构 Mamba（MambaAD 风格 1D 改造）**：multi-scale、global/local、kernel、bi-Mamba 等贡献
4) **G3 训练/损失策略**：对抗耦合是否必须？loss schedule 是否必须？freeze-other 是否影响稳定性？
5) **G4 分数融合**：alpha/beta/gamma 与 topk 对最终 F1 的影响

> 实操顺序：G0 →（若 FullMamba 明显优于 baseline）→ G1/G2 →（稳定后）→ G3 → G4。

---

## 3. G0 框架级消融（必须做，先跑通）

固定：`use_adv_train=True, adv_loss_mode=stamp`，其余保持默认。

| ID | 目的 | pred_model | recon_model | 备注 |
|---|---|---|---|---|
| E0 | 原始基线 | gat | ae | STAMP baseline |
| E1 | 只换预测分支 | mamba | ae | 预测端贡献 |
| E2 | 只换重构分支 | gat | mamba | 重构端贡献 |
| E3 | 双分支都换 | mamba | mamba | FullMamba（主模型） |

再加一组“无耦合训练”对照（对应 STAMP 论文里 w/o adversarial optimization module 的思想）：

| ID | 目的 | pred_model | recon_model | use_adv_train |
|---|---|---|---|---|
| E0a | baseline w/o adv | gat | ae | False |
| E3a | FullMamba w/o adv | mamba | mamba | False |

**你需要在 SMD / MSL 都跑**，并在报告里给出：
- E0 vs E3（总提升）
- E1/E2（分支贡献）
- E0a/E3a（耦合训练贡献）

---

## 4. G1 预测分支 Mamba 消融（参考 S-D-Mamba 的关键结论）

以 **E3 FullMamba** 为 base（或在 G0 中最优者），仅改动预测端。

### 4.1 结构开关（优先）

| ID | 目的 | 关键开关 |
|---|---|---|
| P0 | base | 默认配置 |
| P1 | w/o bi-Mamba | `--mamba_bidirectional False` |
| P2 | w/o FFN | `--mamba_use_ffn False` |
| P3 | w/o norm | `--mamba_use_norm False` |
| P4 | w/o last residual | `--mamba_use_last_residual False` |

### 4.2 MAS 相关（根据你的实现路径）
- MAS 数据是否加载：`--is_mas {True/False}`
- MAS token 是否送入预测 Mamba：`--mamba_use_mas {True/False}`

建议最少做两组：

| ID | 目的 | is_mas | mamba_use_mas |
|---|---|---|---|
| P5 | 不用 MAS token | True | False |
| P6 | 使用 MAS token | True | True |

> 如果你担心“is_mas=False 改变数据/输入分布”，可以把它单独作为补充实验，不放进主表。

### 4.3 深度与 SSM 超参（可选，若你要写更完整）
- 层数：`--mamba_e_layers {1,2,3,4}`
- SSM：`--mamba_d_state {8,16,32}`，`--mamba_d_conv {2,4,8}`

建议做小网格但不要爆炸式组合：
- 固定 d_model，先扫 e_layers
- 再固定 e_layers=最佳，扫 d_state/d_conv

---

## 5. G2 重构分支 Mamba 消融（对应 MambaAD 的核心模块）

以 **E3 FullMamba** 为 base（或 G0 最优），仅改动重构端。

### 5.1 multi-scale（金字塔尺度数）

| ID | 目的 | recon_num_scales |
|---|---|---|
| R0 | base | 3 |
| R1 | w/o multi-scale | 1 |
| R2 | 2-scale | 2 |

### 5.2 global/local 贡献（LSS 的核心）

| ID | 目的 | recon_use_global | recon_use_local |
|---|---|---|---|
| R3 | global only | True | False |
| R4 | local only | False | True |

> 一般预期：global only 对长依赖好、local only 对短形状好；二者融合最好。

### 5.3 local kernel sizes（局部形状建模能力）

| ID | 目的 | recon_local_kernels |
|---|---|---|
| R5 | 单核 | `3` |
| R6 | 双核小 | `3,5` |
| R0 | base | `5,7` |
| R7 | 双核大 | `7,11` |

### 5.4 bi/uni Mamba（重构端）

| ID | 目的 | recon_bidirectional |
|---|---|---|
| R8 | w/o bi-Mamba | False |

### 5.5 global depth（每个 LSS block 的 global Mamba 层数）

| ID | 目的 | recon_global_mamba_layers |
|---|---|---|
| R9 | 浅 | 1 |
| R0 | base | 2 |
| R10 | 深 | 3 |

### 5.6 输出激活（非常建议做，关系到对抗训练稳定性）
如果你数据经过 `MinMaxScaler` 到 [0,1]，重构输出最好也约束到 [0,1]。

| ID | 目的 | recon_output_activation |
|---|---|---|
| R11 | 无约束 | none |
| R12 | Sigmoid 约束 | sigmoid |
| R0 | auto | auto |

---

## 6. G3 训练/损失策略消融（STAMP 耦合机制是否必要）

### 6.1 是否启用对抗耦合训练

| ID | 目的 | use_adv_train |
|---|---|---|
| T0 | base | True |
| T1 | w/o adv train | False |

### 6.2 loss schedule：STAMP 原始 schedule vs 常数权重

| ID | 目的 | adv_loss_mode | lambda_pred / lambda_ae / lambda_adv |
|---|---|---|---|
| T2 | STAMP schedule | stamp | n/a |
| T3 | 常数权重 | constant | 例如 `5/3/1` |

> 说明：
> - `stamp`：复现 STAMP “随 epoch 增强耦合项”的策略
> - `constant`：固定权重，便于分析“是否必须随 epoch 变化”

### 6.3 freeze-other（训练稳定性/速度）

| ID | 目的 | adv_freeze_other |
|---|---|---|
| T4 | freeze | True |
| T5 | no-freeze | False |

---

## 7. G4 分数融合消融（alpha/beta/gamma + topk）

你当前 anomaly score：

`score = alpha * pred_error + beta * recon_error + gamma * adv_error`

建议先做 3 个“极端”确认分支是否有用：

| ID | 目的 | alpha | beta | gamma |
|---|---|---:|---:|---:|
| S1 | pred only | 1 | 0 | 0 |
| S2 | recon only | 0 | 1 | 0 |
| S3 | adv only | 0 | 0 | 1 |

再做一个小网格（不要太大）：
- `alpha ∈ {0.2,0.5,0.8}`
- `beta  ∈ {0.2,0.5,0.8}`
- `gamma ∈ {0.0,0.1,0.2}`
- 约束：`alpha+beta+gamma` 不必等于 1（你实现里是线性组合），但建议控制总量在合理范围

最后做 topk：
- `topk ∈ {1,3,5}`

> 经验：SMD 上常见 topk=1 就不错；MSL/SMAP 有时 topk>1 更稳。

---

## 8. 推荐的执行顺序（省时间 & 易排错）

1. **只跑 SMD**：E0/E1/E2/E3
2. 在 SMD 上确定主模型（通常 E3）后：跑 E0a/E3a（验证耦合训练贡献）
3. 在 SMD 上跑 P1/P2 + R1/R3/R4（最核心结构消融）
4. 在 MSL 上复现关键结论（不要全跑，挑主消融）
5. 最后做 G4 分数融合与 topk

---

## 9. 命令模板（可直接复制）

### 9.1 单次训练

```bash
python run.py \
  --data SMD --dataset AIOps \
  --pred_model mamba --recon_model mamba \
  --log_dir exp/E3_FullMamba \
  --window_size 15 --n_pred 3 \
  --epochs 40 --batch_size 128 --lr 0.001
```

### 9.2 单次测试（输出 json，便于自动汇总）

```bash
python test.py \
  --data SMD --dataset AIOps \
  --pred_model mamba --recon_model mamba \
  --load_checkpoint exp/E3_FullMamba/best_model_SMD_stamp.pth \
  --score_method max \
  --save_result_json exp/E3_FullMamba/result.json
```

### 9.3 快速跑一套（推荐）

```bash
python scripts/run_ablation_grid.py \
  --suite framework \
  --data SMD --dataset AIOps \
  --out_root exp/ablations
```

---

## 10. 开关速查表（你需要关注的新增/常用参数）

### 10.1 训练策略
- `--train_pred True/False`：是否训练预测分支（pred-only / recon-only）
- `--train_recon True/False`：是否训练重构分支
- `--use_adv_train True/False`：是否启用 STAMP 对抗耦合训练
- `--adv_loss_mode stamp/constant`：loss 组合方式
- `--lambda_pred / --lambda_ae / --lambda_adv`：constant 模式下的固定权重
- `--adv_freeze_other True/False`：更新一侧时冻结另一侧

### 10.2 预测 Mamba（pred_model=mamba）
- `--mamba_bidirectional True/False`：bi-Mamba / uni-Mamba
- `--mamba_use_ffn True/False`：是否保留 FFN
- `--mamba_use_norm True/False`
- `--mamba_use_last_residual True/False`
- `--mamba_use_mas True/False`
- `--mamba_e_layers / --mamba_d_state / --mamba_d_conv / --mamba_d_model`

### 10.3 重构 Mamba（recon_model=mamba）
- `--recon_num_scales 1/2/3`
- `--recon_use_global True/False`
- `--recon_use_local True/False`
- `--recon_global_mamba_layers {1,2,3}`
- `--recon_local_kernels '3' / '3,5' / '5,7' ...`
- `--recon_bidirectional True/False`
- `--recon_use_ffn True/False`
- `--recon_output_activation auto/none/sigmoid/tanh`

### 10.4 测试融合
- `--test_alpha / --test_beta / --test_gamma`
- `--topk`

---

## 参考（你可以在报告里引用）
- STAMP：Compatible Unsupervised Anomaly Detection with Multi-Perspective Spatio-Temporal Learning
- MambaAD：Exploring State Space Models for Multi-class Unsupervised Anomaly Detection
- S-D-Mamba：Is Mamba Effective for Time Series Forecasting
