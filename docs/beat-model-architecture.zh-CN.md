# Terb 当前节拍模型技术说明

> **历史文档：** 本文描述已淘汰的 tiny TCN。当前生产模型请看
> [`current-beat-model.zh-CN.md`](current-beat-model.zh-CN.md)。

> 对应模型：`tiny-causal-tcn-guitarset-v1`
> 对应权重：`assets/beat_tracker.onnx`
> 文档状态：描述 2026-08-19 仓库中的实际实现，不代表后续设计目标。

## 1. 系统到底在做什么

当前神经方案不是“输入一首歌，直接回归一个 BPM”。它把问题拆成两层：

1. 小型因果 TCN 每 20ms 判断一次“这一帧像不像 beat/downbeat”；
2. Rust 解码器观察最近 4–16 秒的 beat activation，再估计 BPM、置信度和相位。

这样做的好处是模型只学习局部与中期的节拍线索，BPM 范围、半拍/倍拍选择、连续性和静音策略仍由容易调试的 Rust 代码控制。

```mermaid
flowchart LR
    A[44.1/48kHz 音频块] --> B[单声道]
    B --> C[状态化线性重采样<br/>22050Hz]
    C --> D[STFT<br/>1024 / hop 441]
    D --> E[128-band Slaney Mel<br/>ln(1 + 1000x)]
    E --> F[最近 256 帧<br/>约 5.12s]
    F --> G[因果 TCN<br/>189378 参数]
    G --> H[beat / downbeat logits]
    H --> I[sigmoid activation]
    I --> J[4–16s Rust 解码器]
    J --> K[BPM / confidence<br/>beat pulse / phase]
```

现在的 TUI 仍使用旧的 `src/bpm.rs`。新模型有独立 API 和 CLI，但由于跨数据集测试结果不够好，尚未替换主程序算法。

## 2. 输入特征

### 2.1 音频与重采样

WAV CLI 会先把多声道样本逐帧求平均得到单声道。实时 API 接受调用方提供的单声道 `f32`。

所有输入统一到：

| 参数 | 值 |
|---|---:|
| 模型采样率 | 22050Hz |
| FFT | 1024 |
| hop | 441 |
| 帧率 | 50fps |
| Mel 数量 | 128 |
| Mel 范围 | 30Hz–11025Hz |

重采样器是状态化线性插值器。它保存前一输入样本、绝对输入位置和下一个输出采样位置，所以无论调用方每次传 128、512 还是 2048 个样本，输出采样序列不应随块边界变化。

线性重采样不是最高质量的音频重采样，但它有三个现实优势：

- 算法简单、确定；
- Python/Rust 容易做到一致；
- 目标只是低带宽节拍特征，不是重建高保真音频。

后续若换成 soxr/rubato，必须重新做训练侧一致性测试，不能只改 Rust。

### 2.2 STFT 与固定 lookahead

每帧使用 periodic Hann window：

[
w[n] = 0.5 - 0.5cos(2pi n/N),quad N=1024
]

为了匹配 Beat This! 发布频谱的 centered STFT，流开始时在缓冲区前放 512 个零。产生某时刻特征需要看到其右侧约 512 个采样，因此固定未来信息约为：

[
512 / 22050 approx 23.22\text{ms}
]

严格说，整个系统不是“零未来帧”，而是模型因果、特征提取有 23.2ms 固定 lookahead。

### 2.3 Mel filterbank 与幅度缩放

实现使用 Slaney Mel 标度和三角滤波器。每个 Mel band 聚合的是 FFT 幅度，不是功率：

[
M_m(t)=sum_k |X_t[k]|F_m[k]
]

送入模型的值是：

[
S_m(t)=lnleft(1+\frac{1000M_m(t)}{sqrt{1024}}\right)
]

当前没有逐歌曲均值方差归一化，也没有在线全局归一化。这是为了直接兼容 Beat This! 发布的频谱数值定义，并避免流开始时统计量不稳定。

Rust 与 Python 在同一个合成 WAV 上的实测误差：

- 最大绝对误差：`1.15e-4`
- 平均绝对误差：`1.34e-5`

主要误差来自 FFT 和浮点实现差异，量级足够小。

## 3. 训练标签

原始标注给出 beat 时间，以及某些数据集中的拍号位置。准备器把时间映射到 50fps 网格。

beat 不是只在某一个 frame 上置 1，而是在标注位置附近生成高斯软标签：

[
y[t] = max_i expleft(-\frac{(t-50b_i)^2}{2sigma^2}\right),quad sigma=1.5
]

其中 (b_i) 是第 (i) 个 beat 的秒数。1.5 帧约为 30ms。软标签允许标注与帧网格之间有少量偏差，也避免单帧正例过稀。

如果标注的 beat position 为 1，该时刻同时生成 downbeat 软标签。没有 downbeat 标注的数据，其 downbeat mask 全为零；训练时不计算这部分 loss，绝不会把“未知”伪造成负例。

## 4. 网络结构

### 4.1 张量形状

训练/ONNX 输入：

[
[batch, 128, time]
]

Rust 部署输入固定为：

[
[1, 128, 256]
]

即每次观察最近约 5.12 秒。开始不足 256 帧时在左侧补零。

输出：

[
[1, 2, 256]
]

- channel 0：beat logit；
- channel 1：downbeat logit。

Rust 当前只把 beat channel 送入 BPM 解码器。downbeat head 已训练并存在于 ONNX，但 `downbeat_pulse` 目前仍返回 `None`，这是已知未接线功能。

### 4.2 CausalConv

每个因果卷积使用 kernel 5。PyTorch 导出时采用普通 Conv padding，然后裁掉右侧多余输出：

```python
Conv1d(..., kernel_size=5, dilation=d, padding=4*d)(x)[..., :T]
```

这样输出位置 (t) 只依赖 (t) 及之前的输入。最初使用显式 `Pad` 时，`tract-onnx` 无法分析导出的节点，因此改成了 ONNX/tract 更稳定的 Conv padding + Slice。

### 4.3 TCN 主体

网络配置：

1. front：`128 -> 64`，kernel 5 的因果卷积 + ReLU；
2. 6 个残差块，dilation 依次为 `1, 2, 4, 8, 16, 32`；
3. 每个残差块：
   - `64 -> 64` causal Conv1d，kernel 5；
   - ReLU；
   - `1x1 Conv1d`；
   - 与 block 输入相加；
4. head：`64 -> 2` 的 1x1 Conv。

概念上：

[
h_0=operatorname{ReLU}(C_{front}(x))
]

[
h_{i+1}=h_i+C_{1\times1,i}left(
operatorname{ReLU}(C_{causal,i}(h_i))
\right)
]

[
z=C_{head}(h_6)
]

front 提供 5 帧感受野；dilated blocks 继续增加：

[
R=1+4+4(1+2+4+8+16+32)=257\text{ frames}
]

约为 5.14 秒。部署窗口是 256 帧，因此基本覆盖整个有效感受野。

模型总计：

- 189,378 参数；
- ONNX 大小 767,780 字节；
- opset 17；
- 只使用 Conv、ReLU、Add、Slice 等简单算子。

它比原计划优先的 5–15MB 更小。当前没有理由为了达到文件大小目标而机械扩宽网络：跨领域失败首先是训练数据覆盖问题，扩大单领域模型很可能只增加 CPU 成本和过拟合。

## 5. Loss 与训练过程

beat 使用带正例权重的 BCE with logits：

[
L_{beat}=operatorname{BCEWithLogits}(z_b,y_b, pos_weight=8)
]

downbeat 使用 `pos_weight=16`，并乘标注 mask：

[
L_{down}=
\frac{sum_t m_toperatorname{BCEWithLogits}(z_d[t],y_d[t])}
{max(1,sum_t m_t)}
]

总 loss：

[
L=L_{beat}+L_{down}
]

若当前 batch 没有 downbeat 标注，则只使用 beat loss。

训练参数：

| 参数 | 值 |
|---|---:|
| seed | 20260818 |
| epochs | 10 |
| batch | 16 |
| clip 长度 | 800 帧（16秒） |
| optimizer | AdamW |
| learning rate | 2e-4 |
| weight decay | 1e-4 |
| gradient clipping | 3 |
| 训练录音 | 153 |
| 验证录音 | 27 |

每个 epoch 中，每条 recording 被随机抽取多次 16 秒片段。保存 validation loss 最低的 checkpoint，之后再导出 ONNX。

## 6. 数据增强

当前增强全部在特征域完成：

- 高斯噪声：标准差随机取 ([0,0.04])；
- 增益：随机乘 ([0.85,1.15])；
- 近似变调/EQ：Mel 轴滚动 -3 到 +3 bands；
- 变速：随机取 ([0.94,1.06])。

变速时，Mel、beat、downbeat 和 downbeat mask 使用同一套新时间坐标插值，因此标签与音频特征同步变化。

要注意 Mel 轴 `roll` 会把一端滚到另一端，它只是便宜的近似增强，并不等价于物理正确的 pitch shift。后续可以改为边缘补零或在波形域做真正变调。

## 7. 数据划分与泄漏控制

当前模型只训练了 Beat This! GuitarSet：

- 153 首 train；
- 27 首 validation；
- 使用官方 `single.split`；
- manifest 中每条记录有稳定的 `recording_id`；
- loader 启动时检查同一 recording ID 是否出现在不同 split；
- 所有在线增强继承原 recording ID。

GTZAN 的 999 首只用于最终 test。准备器强制把 GTZAN 标记为 `test`，不读取为 train/validation，也不对其做训练增强。

这能防止最常见的数据泄漏：同一首歌的原版、变速版或变调版分别落入训练集和测试集。

## 8. Rust 流式推理

### 8.1 为什么不是每 20ms 跑一次模型

特征仍以 50fps 产生，但模型每积累 10 个新 frame 才运行一次：

[
10/50=0.2\text{s}
]

每次重新推理最近 256 帧，然后只取最后 10 个 beat logits。这样不需要在 ONNX 图中维护多层卷积 state，部署简单，同时比每帧重复计算整个 5.12 秒窗口便宜约 10 倍。

代价是模型结果按 200ms 批量到达。最坏算法等待大约：

- STFT lookahead：23.2ms；
- 等待下一次推理：最多 200ms。

因此最坏约 223ms，平均约 123ms。它略微超过“尽量不超过 100–200ms”的理想上限；要进一步降低，应实现显式 TCN state 或把 inference hop 从 10 调到 5，并重新测 CPU。

### 8.2 内存复用

Rust 对以下对象复用内存：

- FFT plan；
- Hann window；
- FFT complex buffer；
- Mel filterbank；
- 重采样队列；
- 256×128 模型输入 buffer；
- feature frame 临时 vector；
- activation/rms 历史队列。

`tract` 的输出 tensor 和当前 sigmoid结果仍会在每次模型调用时产生小规模分配，所以这里是“限制和摊销分配”，不是完全零分配。

模型只在 `RealtimeBeatTracker::new/with_model` 中加载一次。文件不存在或 ONNX 不兼容时返回明确错误。

## 9. Rust BPM 解码

神经网络不直接决定 BPM。它只产生：

[
a_t=sigma(z_{beat,t})
]

Rust 保存最多 800 帧，即 16 秒 activation；至少 200 帧，即约 4 秒后才允许输出。

### 9.1 静音门控

同时保存每个 STFT frame 的 RMS。最近 100 帧中，至少 25 帧必须超过 `0.00012`。否则衰减 confidence 并不输出新随机 BPM。

这只是能量门，不等价于“无节奏检测”。安静但节奏明确、响但完全无节奏的情况仍要依赖 activation 能量与 confidence。

### 9.2 候选搜索

对 activation 减均值并裁掉负值：

[
x_t=max(a_t-\bar a,0)
]

如果能量过低则不输出。随后以 0.25 BPM 步长枚举 60–210 BPM。每个候选对应 lag：

[
ell=50cdot60/BPM
]

候选分数由三部分构成：

[
score=0.62,AC+0.28,Interval+0.10,Continuity
]

- `AC`：支持小数 lag 插值的归一化自相关；
- `Interval`：activation 峰相邻间隔对一拍和两拍间隔的支持；
- `Continuity`：相对当前稳定 BPM 的对数距离先验。

峰候选阈值为 0.18。interval support 对直接间隔权重 1，对两倍间隔权重 0.65。

### 9.3 半拍/倍拍

找到最高分候选后，显式构造：

[
{0.5B, B, 2B}
]

仅保留 60–210 BPM 内的候选，再比较它们各自在搜索表里的得分。这比只靠一个偏向 120 BPM 的 tempo prior 更透明，但当前 GTZAN 仍有 24.5% 半拍错误，说明 activation 和候选打分还不足以稳定区分 metrical level。

### 9.4 连续性与变速跟随

新估计先折叠到离旧 BPM 最近的 octave。平滑速度取决于变化大小和置信度：

- 小变化：follow 0.28；
- 明显变化但高置信度：0.14；
- 明显变化且置信度不足：0.045。

所以无理由跳变会很慢，真实变速在高置信度下仍能跟随。这个策略也有风险：如果第一次锁到半拍，octave-nearest 会强化错误层级。

confidence 综合最佳候选分数与它和 runner-up 的分离度，然后用 0.25 的系数做时间平滑。它是内部启发式置信度，不是经过概率校准的正确率。

### 9.5 beat pulse 与 phase

检测到当前 activation 上升且超过 0.25 时记录 beat；否则 pulse 指数衰减。phase 根据距离最近一次 peak 的帧数除以当前 BPM 周期得到。

当前 peak 判断是低延迟的上升沿判断，不等待未来帧确认局部极大值，因此 phase 响应快，但可能比真正峰值略早。

## 10. 当前实验结果

这里必须区分三个对象，不能把它们的成绩混在一起：

1. **当前部署学生模型**：仓库中的 0.77MB GuitarSet-only 因果 TCN；
2. **官方参考模型**：Beat This! small0，约 8.1MB、210 万参数的非因果模型，只作为 teacher/上限参考；
3. **蒸馏候选**：仍是相同的小型因果 TCN，但训练时额外拟合 teacher 的软 activation，尚未通过生产门槛，未替换部署权重。

### 10.1 当前学生模型

使用改进后的 production decoder，在 GuitarSet validation 上：

| 指标 | 结果 |
|---|---:|
| 严格 BPM 准确率 | 85.2% |
| 半/倍拍容忍准确率 | 88.9% |
| 半拍错误 | 3.7% |
| 完全错误 | 11.1% |
| 4 秒严格准确率 | 66.7% |

同一模型固定参数后在 GTZAN 999 首 test-only 集合上：

| 指标 | 结果 |
|---|---:|
| 严格 BPM 准确率 | 40.64% |
| 半/倍拍容忍准确率 | 64.46% |
| 半拍错误 | 13.71% |
| 倍拍错误 | 10.11% |
| 完全错误 | 35.54% |
| 4 / 8 / 16 / 30 秒严格准确率 | 27.3 / 32.2 / 37.1 / 40.7% |

这说明解码器优化确实减少了 GuitarSet 的 metrical-level 错误，但模型 activation 的跨音乐类型泛化仍然不够。当前权重不能作为生产替代方案接入 TUI。

### 10.2 官方 Beat This! 参考模型

对完全相同的 GTZAN 999 首和相同 production decoder，官方 small0 参考模型得到：

| 指标 | 结果 |
|---|---:|
| beat F1 | 84.86% |
| downbeat F1 | 71.76% |
| 严格 BPM 准确率 | 75.58% |
| 半/倍拍容忍准确率 | 85.19% |
| 半拍错误 | 6.41% |
| 倍拍错误 | 3.20% |
| 完全错误 | 14.81% |
| 4 / 8 / 16 / 30 秒严格准确率 | 72.67 / 73.57 / 74.47 / 75.58% |

这个结果证明逐帧 activation 加 Rust 风格 tempo decoder 的方向可行，也给学生模型提供软标签和质量上限。它不是最终运行模型：官方网络含非因果结构，CPU 成本和部署方式不符合当前实时约束，因此不会原样塞进 terb。

### 10.3 当前验收门槛

| 项目 | 门槛 |
|---|---:|
| GTZAN 严格准确率 | >= 70% |
| 半/倍拍容忍准确率 | >= 85% |
| 半拍错误 | <= 10% |
| 完全错误 | <= 15% |
| 4 秒严格准确率 | >= 55% |
| CPU RTF | <= 0.25 |
| 模型大小 | <= 15MiB |

当前学生模型只通过运行速度和模型大小门槛，准确率门槛没有通过。官方参考模型的准确率达到目标附近，但不满足最终因果实时部署约束。

## 11. 蒸馏是怎么接进训练的

当前训练器支持 distill-alpha。teacher 先为每条训练录音生成 beat/downbeat 概率序列；学生训练时同时看人工标注和 teacher 软标签：

[
L=(1-alpha)L_{GT}+alpha L_{teacher}
]

已完成 alpha=0.7（30 epoch）与 alpha=0.3（15 epoch、逐 epoch 完整录音选模）两组候选。二者最佳 GuitarSet 严格准确率都为 81.48%，低于当前 85.2% 基线，因此均被拒绝且没有进入 GTZAN 全测。数据增强发生时间伸缩时，Mel、人工标签、mask 和 teacher 概率使用相同坐标同步重采样，避免 activation 与节拍位置错位。

蒸馏的目标不是复制 teacher 的非因果结构，而是把其更平滑、更丰富的节拍判断压缩到 18.9 万参数的因果网络中。是否有效只由独立验证集和 test-only 结果决定；候选未过门槛前不会覆盖部署权重。

## 12. 性能

在 x86_64 Linux 普通 CPU 路径、30 秒 48kHz 双声道合成输入、512 样本块上：

| 项目 | 结果 |
|---|---:|
| 模型大小 | 767,780 bytes |
| 参数量 | 189,378 |
| 总 RTF | 0.172 |
| 特征提取总耗时 | 18.4ms |
| 模型推理总耗时 | 3.721s |
| BPM 解码总耗时 | 1.411s |
| 峰值 RSS | 20.7MB |
| 首次估计 | 约 4.01s |

RTF 0.172 表示处理 30 秒音频约用 5.17 秒计算时间，约 5.8 倍实时速度。推理已满足单路实时，但接入主程序前仍需在目标 macOS/Linux CPU 上与动画刷新同时压测。

## 13. 已知限制

- 当前部署权重只由 GuitarSet 训练，数据域过窄；
- GTZAN 跨域严格准确率只有 40.64%；
- Rust 尚未消费 downbeat head；
- 200ms 一次的滑窗推理简单可靠，但有重复计算，最坏响应延迟约 223ms；
- confidence 是启发式分数，尚未做概率校准；
- Zenodo 频谱测试无法直接跑依赖原始 WAV 的旧 src/bpm.rs，所以目前没有公平的旧/新全链路结论；
- TUI 仍保留旧 BPM 算法，这是有意的质量闸门，不是遗漏。

## 14. 当前工程决策

现阶段保留两条路径：

- src/bpm.rs：主程序仍在使用的传统 STFT / spectral-flux / autocorrelation 算法；
- src/features.rs + src/beat.rs：新神经模型的 Rust 特征、ONNX 推理和 activation decoder，由独立 CLI/基准工具验证。

下一步是完成 Candombe/SMC 多领域训练、按完整录音验证选模，然后只在所有生产门槛通过后替换主程序路径。这样模型文件是否进入生产由可重复指标决定，而不是由单个 demo 的观感决定。
