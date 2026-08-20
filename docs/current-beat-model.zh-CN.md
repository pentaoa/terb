# Terb 当前节拍模型：从音频流到 BPM

> 更新时间：2026-08-19
> 本文描述 `assets/beat_tracker.onnx` 中当前实际部署的模型和 Rust runtime。

## 先说结论

我们不让神经网络直接猜一个 BPM。模型每 20ms 输出一次 beat/downbeat activation，Rust 再根据最近 4–16 秒 activation 解码 BPM、置信度和节拍相位。

目前仓库里有两代模型：

| 模型 | 状态 | 参数/大小 | GTZAN 严格准确率 | 主要问题 |
|---|---|---:|---:|---|
| Tiny causal TCN | 已淘汰，保留实验记录 | 189,378 / 0.77MB | 40.64% | 跨音乐类型泛化太差 |
| bounded-causal small Transformer v2 | 当前默认模型，已接入 CLI/TUI | 2,099,960 / 9.33MB | 76.68% | 采用 112 帧窗口、200ms 调度 |

当前默认模型是第二个。Rust 使用纯 CPU RTen，权重嵌入发布二进制；TUI 通过有界非阻塞 worker 使用它，模型推理不占用音频/UI 线程。旧 `src/bpm.rs` 仍保留给基准工具，不再是 TUI 默认实现。

## 1. 整条数据流

```text
44.1/48kHz 音频小块
    │
    ├─ 单声道化
    ├─ 状态化重采样到 22050Hz
    ├─ STFT: FFT=1024, hop=441
    ├─ 128-band Slaney Mel
    ├─ log(1 + 1000 · magnitude / sqrt(1024))
    │
    └─ [batch, time, 128] Mel 序列
             │
             ├─ 受限因果 Transformer
             ├─ beat logits  ── sigmoid ── beat activation
             └─ downbeat logits ─ sigmoid ─ downbeat activation
                                      │
                                      └─ Rust tempo decoder
                                          ├─ activation 自相关
                                          ├─ 峰间距
                                          ├─ 0.5× / 1× / 2× 比较
                                          ├─ 连续性与静音门控
                                          └─ BPM/confidence/phase
```

这种拆法比“整首歌回归 BPM”更适合实时流：activation 可以连续更新，速度变化可以被后端跟踪，也为以后输出 downbeat 和 AutoMix 相位留下接口。

## 2. Rust 特征提取

### 2.1 固定参数

| 参数 | 值 |
|---|---:|
| 模型采样率 | 22050Hz |
| FFT 窗长 | 1024 samples，约 46.44ms |
| hop | 441 samples，20ms |
| 帧率 | 50fps |
| Mel bands | 128 |
| 频率范围 | 30Hz–11025Hz |
| window | periodic Hann |

输入可以是 44.1kHz 或 48kHz。`MelExtractor` 保存跨音频块的插值位置和上一采样，不会因为调用方把音频切成 128、512 或 2048 samples 而改变结果。

### 2.2 STFT 和 Mel 数值定义

第 `t` 帧、第 `m` 个 Mel band：

```text
M[m,t] = Σk |FFT(window · audio)[k]| · filter[m,k]
S[m,t] = ln(1 + 1000 · M[m,t] / √1024)
```

这里聚合的是幅度谱，不是功率谱；Mel 使用 Slaney 标度。没有做逐歌曲全局归一化，因为流式开始时还没有整首歌统计量，而且训练数据就是这套数值定义。

Python/Rust 对同一 WAV 的一致性实测：最大绝对误差 `1.15e-4`，平均绝对误差 `1.34e-5`。

### 2.3 特征自身的未来信息

为了兼容 Beat This! 的 centered STFT，流开头补 512 个零，一帧特征需要约 512 个未来采样：

```text
512 / 22050 = 23.22ms
```

因此整个系统不是数学意义上的零 lookahead。特征端有固定约 23.2ms future context。

## 3. 当前主候选网络

主候选从 Beat This! 官方 `small0` 权重出发，将所有“沿时间轴”的 attention 改造成受限因果 attention，再微调恢复精度。沿频率轴的 attention 保持双向，因为它只在同一时刻混合不同频带，不会偷看未来帧。

### 3.1 输入输出

训练/导出输入：

```text
spectrogram: [batch, time, 128]
```

输出是两个独立的逐帧 logit：

```text
beat:     [batch, time]
downbeat: [batch, time]
```

推理后做 sigmoid 得到 0–1 activation。模型不直接输出 BPM。

### 3.2 网络骨架

小模型的关键超参数：

| 参数 | 值 |
|---|---:|
| Transformer width | 128 |
| 主 Transformer 层数 | 6 |
| attention heads | 4 |
| head dimension | 32 |
| FFN expansion | 4× |
| stem channels | 32 |
| 总参数 | 2,099,960 |
| ONNX | opset 17，9,333,229 bytes |

前端先把二维 Mel 图压缩成每帧一个 128 维 token：

1. `Conv2d(1→32, kernel=(4,3), stride=(4,1))`；
2. 三个 frontend block，每个 block 先做 frequency/time partial attention，再用 `Conv2d(kernel=(2,3), stride=(2,1))` 将频率维减半、通道翻倍；
3. 展平通道和剩余频率维；
4. Linear 投影到 128 维；
5. 六层 RoFormer/Transformer；
6. beat 和 downbeat 两个任务头。

频率维压缩路径大致为 `128 → 32 → 16 → 8 → 4`。时间维不下采样，所以输入输出都是 50fps。

### 3.3 “因果”具体是什么意思

原官方模型可以让时间 attention 看完整首歌，不适合实时。我们的适配器对每个时间 attention 使用如下 mask：

```text
当前 query t 只能访问 key k，满足：t - 83 <= k <= t
```

即每一层最多看当前帧和此前 83 帧。多层堆叠后总体过去感受野约为 751 帧，约 15 秒；这与 Rust 解码器最长 16 秒历史相匹配。

卷积前端仍有四个对称的 time-kernel=3 卷积串联，因此总共保留 4 帧未来信息：

```text
4 frames / 50fps = 80ms
```

加上 centered STFT 的约 23.2ms，网络路径的理论最低 lookahead 约 103ms。实际滚动调度还会加最多一个 inference hop。

因果性不是口头假设，已有两个自动测试：

- 修改未来输入，不应改变早于 4 帧边界的输出；
- 当 attention context 设为 28 时，足够久远的输入不应影响当前输出。

## 4. 实时滚动方式

最终生产调度如下：

| 参数 | 值 |
|---|---:|
| 滚动窗口 | 112 帧，2.24 秒 |
| 每次前进 | 10 帧，200ms |
| 模型未来帧 | 4 帧，80ms |
| 调度最坏算法延迟 | 260ms |

为什么最坏是 260ms：一个目标帧在批次中会带 4–13 帧未来上下文，取决于它落在 10 帧 inference hop 的哪个位置。因此最大约 `13 × 20ms`。这是为满足普通 CPU 上 RTF≤0.25 作出的固定折中；没有增加用户延迟设置。

112 帧/hop10 在 GuitarSet 验证集上得到：

| 指标 | 结果 |
|---|---:|
| strict BPM accuracy | 88.89% |
| 允许 0.5×/2× | 96.30% |
| half-time | 7.41% |
| double-time | 0% |
| 完全错误 | 3.70% |
| 4s / 8s / 16s / 30s | 85.19 / 85.19 / 81.48 / 88.89% |

网络每次只看最近 2.24 秒，长期 4–16 秒证据由廉价的 Rust activation decoder 保存。这个拆分将真实 48kHz 流式 RTF 控制在约 0.226，同时保留通过门槛的跨域准确率。

## 5. 训练方式

### 5.1 teacher 初始化与因果微调

我们没有把约 78MB 的官方模型放进最终程序。训练过程是：

1. 加载 Beat This! `small0` checkpoint；
2. 将所有时间 attention 替换为 causal/local mask；
3. 用真实 beat/downbeat 标签和 teacher activation 共同微调；
4. 导出独立 ONNX；
5. Rust 运行时只加载这个 9.34MB ONNX，不需要 teacher、Python、网络或 GPU。

最终模型先完成 GuitarSet local-84 微调，再进行两轮电子音乐域蒸馏。40 首 GiantSteps 音频只使用官方 teacher 的逐帧 activation；单个 tempo 标注没有被展开为伪造的 beat/downbeat 标签。第二轮配置：

| 参数 | 值 |
|---|---:|
| seed | 20260819 |
| clip | 800 frames / 16s |
| attention context/layer | 84 frames |
| batch | 4 |
| learning rate | 5e-6 |
| distillation alpha | 1.0（teacher-only continuation） |
| GiantSteps 蒸馏录音 | 40，按原始录音划分 |
| 选中 checkpoint | round 2 / epoch 1 |

最终 continuation 的 `distillation alpha=1.0` 表示这一阶段只拟合 teacher activation，避免把只有 BPM 的 GiantSteps 曲目伪装成逐帧标注。导出的学生模型本身仍是独立模型，运行时不需要 teacher。

### 5.2 标签和 loss

beat/downbeat 时间先映射到 50fps。标签不是孤立的单帧 1，而是在标注附近生成窄高斯软目标，以容忍几十毫秒标注偏差。

训练使用类别加权 BCE。官方 small 配置的正例权重为：

```text
beat pos_weight = 19
downbeat pos_weight = 86
```

没有 downbeat 标注的数据通过 mask 跳过 downbeat loss，绝不把“未知 downbeat”当作负样本。

### 5.3 数据与泄漏控制

已准备 GuitarSet、Candombe、SMC；GTZAN 固定为测试集，不参与训练。划分单位是原始 recording，不是随机切片。某一首歌的切片、变速或变调版本只能继承原歌 split，不能跨 train/validation/test。

多域小 TCN 已做过训练，但结果没有超过 GuitarSet 单域基线，因此没有为了“看起来有更多数据”而替换当前候选。详细数据来源和许可证见 `docs/beat-data.zh-CN.md`。

## 6. Rust BPM 解码器

模型只负责“哪里像拍点”。Rust 的 `ActivationDecoder` 才负责 tempo 决策。

### 6.1 历史长度和静音

- 至少 200 帧，即 4 秒 activation 后才允许输出；
- 最多保留 800 帧，即 16 秒；
- 每 10 帧，即 200ms 更新一次 BPM；
- 最近约 2 秒可听帧不足时降低 confidence 并不产生新 BPM。

### 6.2 候选产生

对 activation 去均值并截断负值：

```text
x[t] = max(activation[t] - mean, 0)
```

然后在 60–210 BPM 对应 lag 范围内计算归一化自相关。与此同时从局部峰提取相邻 beat 间距，将峰间距证据与自相关证据合并。

对每个候选显式比较：

```text
bpm / 2, bpm, bpm * 2
```

只保留落在 60–210 的值。这样半拍/倍拍不是事后硬修，而是候选评分的一部分。

### 6.3 稳定与跟随

已有稳定 BPM 时，新候选会先映射到最接近旧值的节拍层级，再按置信度平滑：

- 很接近旧 BPM：快速跟随；
- 差异较大但置信度高：中速跟随；
- 差异大且证据弱：慢速跟随，避免无理由跳变。

输出包含 `bpm`、`confidence`、`beat_pulse`、`phase` 和 `downbeat_pulse`。downbeat 当前是逐帧概率，尚未做完整小节状态机。

## 7. 当前最重要的测试结果

GTZAN 999 首严格留出测试，最终 112 帧/hop10 调度配固定 Rust 等价 BPM decoder：

| 指标 | 结果 |
|---|---:|
| beat F1 | 53.00% |
| strict BPM accuracy | 76.68% |
| 允许 0.5×/2× | 92.49% |
| half-time | 6.31% |
| double-time | 9.51% |
| 完全错误 | 7.51% |
| 4s / 8s / 16s / 30s | 64.46 / 70.47 / 74.17 / 76.68% |

对比旧 tiny TCN：

| 指标 | tiny TCN | 当前 Transformer |
|---|---:|---:|
| strict | 40.64% | 76.68% |
| metrical | 64.46% | 92.49% |
| half-time | 13.71% | 6.31% |
| 完全错误 | 35.54% | 7.51% |

准确率提升明显，并且最终形态同时通过准确率、模型大小和 CPU RTF 门槛。

## 8. Rust 部署现状

- ONNX opset 17，固定输入 `[1,112,128]`，输出 `beat/downbeat [1,112]`；
- 纯 Rust RTen 0.24，无 Python、ONNX Runtime、网络或 GPU；
- 默认模型通过 `include_bytes!` 嵌入二进制，`--model`/`TERB_BPM_MODEL` 可显式覆盖；
- 有界 16-block worker 使用 `try_send`，音频/UI 线程不执行模型推理；
- 48kHz、155.69 秒真实音乐、512-sample 块：RTF 0.226；
- 分阶段耗时：特征 0.108s、推理 34.533s、解码 0.515s；
- 128/512/2048-sample 块得到完全相同最终 BPM/confidence；
- 首次估计 4.09s，稳定锁定 4.49s；
- 本轮实测峰值 RSS 约 132MiB，模型 9,333,229 bytes。

同音频三方封存测试使用第四批、预先按流派和 ID 排序选出的 20 首 GiantSteps Beatport 预览；音频逐个通过官方 MD5，且在模型、112 帧窗口和解码参数完全冻结后才下载与评测：

| 最终 GiantSteps 留出 | 旧 `src/bpm.rs` | 当前模型 | Beat This! small 参考 |
|---|---:|---:|---:|
| strict | 25% | 55% | 70% |
| 允许 0.5×/2× | 55% | 55% | 75% |
| half / double / wrong | 30 / 0 / 20%（另 25% 不稳定） | 0 / 0 / 45% | 5 / 0 / 25% |
| 4 / 8 / 16 / 30 秒 strict | 25 / 25 / 25 / 25% | 60 / 60 / 55 / 55% | 65 / 60 / 65 / 70% |
| 平均锁定 / 抖动 | 4.92s / 0.04 BPM（仅已锁定样本） | 5.30s / 2.67 BPM | 不适用（离线双向） |

这组较难的封存结果说明当前模型的 strict 明显优于旧算法，并消除了旧算法的 half-time 与无稳定锁定问题；但 metrical 只持平，完全错误仍有 45%，strict 与官方非因果参考还有 15 个百分点差距。它不是“所有指标全面超过 teacher”的结果。

## 9. 相关文件

- `training/causal_beat_this.py`：时间 attention 因果化与 local mask；
- `training/train_causal_beat_this.py`：因果模型微调；
- `training/export_causal_beat_this.py`：opset-17 ONNX 导出；
- `training/evaluate_rolling_causal.py`：真实滚动窗口评测；
- `src/features.rs`：Rust 流式重采样/STFT/Mel；
- `src/beat.rs`：当前 RTen runtime、非阻塞 worker 和 activation decoder；
- `docs/beat-data.zh-CN.md`：数据集、许可证、校验和与 split；
- `docs/experiment-report.md`：逐次实验记录。

一句话概括：模型学“拍点概率”，Rust 决定“这些拍点意味着什么 BPM”；最终实现用短窗口低频滚动换取可控 CPU 成本，并将所有重推理隔离到后台线程。
