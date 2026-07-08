# Maritime-STPCN 实验复现方案

## 1. 实验目标

**实验名称**: Maritime-STPCN: Spatial-Temporal Pattern Contrastive Network for Vessel Illegal Fishing Detection

**原始论文出处**:
- **标题**: Maritime-STPCN: Spatial-Temporal Pattern Contrastive Network for Vessel Illegal Fishing Detection
- **作者**: Anonymous Author(s)
- **年份**: 2025
- **会议**: 待发表（基于DCMGNN, KDD 2024扩展）

**核心任务**: 将IUU（非法捕捞）检测转化为多关系二部图上的链接预测任务，从AIS轨迹数据中检测非法捕捞行为。

## 2. 复现范围

### 核心实验结果

| 指标 | DCMGNN基线 | STP-only | Full模型 | 目标 |
|------|-----------|----------|---------|------|
| Recall@10 | 0.6386 | 0.9752 (+52.7%) | 0.6089 | 复现STP-only的52.7%提升 |
| NDCG@10 | 0.3851 | 0.6088 | 0.3431 | 验证NDCG趋势 |
| F1@10 | 0.1169 | 0.1781 | 0.1115 | 验证F1趋势 |

### 90%稀疏下相对优势

| 稀疏度 | DCMGNN R@10 | STP R@10 | 相对增益 |
|--------|-------------|----------|---------|
| 0% | 0.6386 | 0.9752 | +52.7% |
| 30% | 0.5341 | 0.8923 | +67.1% |
| 50% | 0.4218 | 0.7856 | +86.3% |
| 70% | 0.2784 | 0.6047 | +116.0% |
| 90% | 0.1423 | 0.4398 | +209.0% |

### 消融实验（Table 8, 9）

需复现15种实验配置：7项单独改进 + 累积消融5步 + 基线 + Full模型

### 实验配置

- 数据集：合成海事数据（200船只、100捕鱼区、6行为类型）
- 模型架构：4通道编码器 + 动态融合门控 + 对比学习头
- 超参数：Table 4完整配置

## 3. 环境与依赖

### 编程语言
- Python 3.13（论文指定）

### 框架及库
| 库 | 版本 | 用途 |
|---|------|------|
| PyTorch | 2.6.0+CUDA 12.4 | 模型实现与训练 |
| NumPy | 1.24+ | 数据生成与计算 |
| Matplotlib | 3.7+ | 结果可视化 |
| PyYAML | 6.0+ | 配置管理 |

### 硬件需求
- GPU: NVIDIA RTX 2080 SUPER (8GB VRAM) 或同等显卡
- CPU: Intel i7-10700K 或同等
- RAM: 16GB
- 无GPU时可在CPU运行（训练时间约5-10倍增加）

## 4. 数据要求

### 数据集
- 名称: Synthetic Maritime AIS Dataset (论文Table 5)
- 获取方式: `maritime_stpcn/data.py` 中 `generate_synthetic_maritime_data()` 自动生成
- 固定seed=123确保可重复性

### 数据统计
| 统计项 | 值 |
|--------|-----|
| 船只数量 | 200 |
| 捕鱼区数量 | 100 |
| 行为类型数 | 6 |
| 总交互数 | ~15,748 |
| 交互密度 | ~0.787% |
| 目标行为率 | 18.3% |

### 数据预处理步骤
1. 生成区域坐标（经纬度，~400×400km范围）
2. 计算区域行为向量 b_z = (sog, cog_change, duration, distance) (Eq.8)
3. 构建地理邻近图 A_geo (阈值50km) (Eq.6,7)
4. 构建行为相似图 A_beh (阈值cos≥0.7) (Eq.9)
5. 生成各行为的船只-区域交互矩阵 R^(k)
6. 增广归一化: D~^(-1/2) A~ D~^(-1/2) (LightGCN式)
7. 计算全局偏差 b_global 和区域偏差 β_j (Eq.30,31)

### 训练/验证/测试划分
- **M6: 严格时间划分** 75/12.5/12.5%
- 所有验证交互在训练之后，所有测试交互在验证之后
- 防止未来信息泄露

## 5. 七项改进详细说明 (M1-M7)

### M1: STP编码器（Eq.6-17）
- 双图传播：地理邻近图（50km）+ 行为相似图（cos≥0.7）
- Sigmoid门控空间卷积 (Eq.14,15,16)
- 投影MLP: H_stp = MLP_proj(H_stp_v || H_stp_z) (Eq.17)

### M2: 自适应对比损失（Eq.18-21）
- 对比投影: p = BN(ReLU(h W1 + b1)) W2 (Eq.18)
- 权重MLP: [ω, σ, τ] = MLP_weight(p_v · p_z | k) (Eq.19)
- 自适应NT-Xent: L = -(ω/σ²) log(exp(sim/τ)/Σ) (Eq.20)

### M3: 熵正则化融合门控（Eq.22-24）
- 门控: α = softmax(h_cat W_gate + b_gate) (Eq.22)
- 融合: h_fused = Σ α^(c) h^(c) (Eq.23)
- 熵惩罚: L_entropy = -Σ log(α) (Eq.24)

### M4: 可学习传播深度（Eq.26-27）
- Gumbel-Softmax: p^(k) = Gumbel-Softmax(l^(k), τ_gs) (Eq.26)
- τ从5.0退火到0.1
- 深度加权传播: H^(k) = Σ p^(k)(l) H^(k,l) (Eq.27)

### M5: 空间偏差校正（Eq.30-32）
- 全局偏差: b_global = log(pos/(1-pos)) (Eq.30)
- 区域偏差: β_j = log(|E_train(j)|/(pos·N_z)) (Eq.31)
- 分解: b_spatial = b_global + β_j + γ_i (Eq.32)

### M6: 时间划分
- 严格按时间顺序划分：75/12.5/12.5%

### M7: Bootstrap评估
- B=100次重采样，95%置信区间 (Eq.33)

## 6. 训练配置 (Table 4)

| 超参数 | 值 |
|---------|-----|
| 嵌入维度 d | 64 |
| 对比维度 | 32 (=d/2) |
| 最大传播深度 L_max | 4 |
| 行为特定层 | [2, 2, 2, 1, 1, 1] |
| 批大小 | 128 |
| 学习率 lr_0 | 5×10⁻³ |
| 权重衰减 λ₂ | 10⁻⁴ |
| λ_rcl | 3.0 |
| λ_chain | 0.5 |
| λ_entropy | 0.01 |
| Gumbel τ初始 | 5.0 |
| Gumbel τ最终 | 0.1 |
| Warmup轮数 | 5 |
| 早停耐心 | 20 |
| 梯度裁剪 | 1.0 |
| 优化器 | Adam (β₁=0.9, β₂=0.999) |
| 最大轮数 | 200 |
| 随机种子 | 123 |

### 学习率调度 (Eq.36)
- Warmup阶段（前5轮）：从10⁻⁴线性增至5×10⁻³
- Cosine阶段：lr = lr_min + 0.5(lr_0 - lr_min)(1 + cos(π·progress))

### 特殊训练策略
- Rcl warmup：λ_rcl从0线性增至3.0（前5轮）
- σ裁剪：[0.1, 10.0] 防止梯度爆炸
- 负采样约束：地理邻近范围内（200km）

## 7. 输出要求

### 代码结构
```
Maritime-DCMGNN/
├── configs/
│   └── default.yaml           # 超参数配置
├── maritime_stpcn/
│   ├── __init__.py
│   ├── config.py              # 配置管理
│   ├── data.py                # 数据生成与加载
│   ├── model.py               # 完整模型实现(M1-M5)
│   ├── losses.py              # 损失函数(Eq.34)
│   ├── trainer.py             # 训练系统(Eq.36调度)
│   ├── evaluator.py           # 评估+Bootstrap(M7)
│   └── visualization.py       # 可视化
├── utils/
│   ├── seed.py                # 随机种子
│   └── __init__.py
├── train.py                   # 主训练脚本
├── evaluate.py                # 评估脚本
├── requirements.txt           # 依赖
└── REPRODUCTION_PLAN.md       # 本文档
```

### 关键步骤注释
- 所有论文公式编号标注在代码注释中
- M1-M7模块标记清晰
- 每个关键计算步骤附公式编号引用

### 复现结果与原始对比表

| 方法 | R@10(论文) | R@10(复现) | 差异 | 分析 |
|------|-----------|-----------|------|------|
| PopRec | 0.1386 | - | - | 非个性化基线 |
| BPR-MF | 0.3168 | - | - | 矩阵分解基线 |
| LightGCN | 0.4455 | - | - | 单行为GNN |
| SGL | 0.4752 | - | - | 对比学习GNN |
| DCMGNN | 0.6386 | - | - | 多行为基线 |
| STP-only | 0.9752 | - | - | **最佳配置** |
| Full | 0.6089 | - | - | 全模块组合 |

## 8. 约束条件

### 可重复性
- 固定随机种子123（Python、NumPy、PyTorch）
- 确定性CuDNN模式
- 数据生成使用seed确保一致性

### 预期训练时长
- STP-only: ~153秒（RTX 2080 SUPER）
- Full模型: ~240秒（RTX 2080 SUPER）
- CPU训练: 约10-20分钟

### 与原始实现不一致之处标注
1. **数据集为合成数据**: 论文使用合成数据，我们的生成器基于论文统计参数（Table 5），但具体交互分布可能存在微小差异
2. **负采样策略**: 论文指定200km地理约束，我们的简化实现使用全局采样（已标注需优化）
3. **链编码器裁剪**: 论文chain_min=5，我们的实现使用所有长度≥2的链（已标注可配置）
4. **时间卷积模块**: 论文Eq.2的扩张因果1D卷积在STP编码器中使用简化传播替代（已标注为简化版本）

## 9. 运行指南

```bash
# 安装依赖
pip install -r requirements.txt

# 训练完整模型
python train.py --config configs/default.yaml

# 运行消融实验
python train.py --ablation

# 运行稀疏度分析
python train.py --sparsity

# 评估已保存的checkpoint
python evaluate.py --checkpoint results/full_model/checkpoint.pt
```
