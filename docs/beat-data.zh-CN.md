# Terb 节拍模型数据卡

> 状态：2026-08-19。本文记录训练/验证/测试数据的来源、许可、固定版本、下载和隔离规则。第三方原始音频、频谱与解压数据均不得提交到 terb 仓库。

## 数据源与许可

| 数据源 | 用途 | 本项目采用的许可证据 | 版本或记录 |
|---|---|---|---|
| Beat This! annotations | beat/downbeat 标注与官方 recording split | 仓库 LICENSE 为 MIT；各原始数据集仍需引用其上游论文和条款 | v1.1，commit 890407d158078527ab396b49fea3c8a83e5734ee |
| Beat This! spectrograms | 统一 22.05kHz/128-Mel 训练特征 | Zenodo 记录元数据为 CC BY 4.0；发布模型前仍保留各上游数据集 attribution | Zenodo 13922116 |
| GuitarSet | 当前 train/validation 领域 | GuitarSet 官方 Zenodo 3371780 元数据为 CC BY 4.0 | DOI 10.5281/zenodo.3371780 |
| GiantSteps Tempo | 40 首 teacher-only 蒸馏、独立开发集与 20 首封存三方评测 | 仓库未声明 Beatport 音频的统一再分发许可；只提交代码/标注，不提交下载音频 | commit d51ab2422e76abacfaa86616a57054bc222ec9fd |
| Beat This! code/weights | 参考模型与离线 teacher | 上游代码仓库 MIT；权重仅作实验 teacher，最终程序不运行 teacher | CPJKU/beat_this |
| GTZAN 频谱 | 固定 test-only | 使用 Zenodo 13922116 发布物；不进入训练或调参切片 | 999 recordings |

Zenodo 13922116 的记录级元数据是 CC BY 4.0，但它汇集了多个既有研究数据集。本项目把该许可视为发布频谱包的许可证据，不据此声称拥有原始音频的再分发权。发布模型前需要保留 Zenodo、Beat This! 论文以及实际采用子数据集论文的 attribution。

## 当前下载清单

| 文件 | 官方大小 | MD5 | 状态 |
|---|---:|---|---|
| gtzan.zip | 306,944,985 | 39a7dfe6a6b0a5279a94d770506db879 | 已完成并校验；test-only |
| guitarset.zip | 1,356,129,024 | 2bd210bf3e994065641410f2c0bb00fe | 已完成并校验 |
| candombe.zip | 2,072,957,923 | 6c30e2114f358e543a7decf955b28c0c | 断点续传中；完成前不得解压或训练 |
| smc.zip | 2,096,168,666 | 32c2640f854ba29fb86be9ac6b84532f | 尚未下载 |
| ballroom.zip | 4,780,542,478 | 8c2bc5363dd505d9122cbc65af0a58a1 | 后续消融才下载 |

已完成文件的本地 MD5 与 Zenodo API checksum 一致。

## 下载方法

通过 Zenodo API 的 content 端点下载，支持断点续传：

~~~sh
curl -fL --retry 20 -C -   -o data/downloads/candombe.zip   https://zenodo.org/api/records/13922116/files/candombe.zip/content
~~~

网络较慢时可以显式使用本机 mihomo mixed port，但端口必须从 mihomo API 的 configs 查询，不硬编码到仓库脚本：

~~~sh
curl -x http://127.0.0.1:PORT -fL --retry 20 -C -   -o data/downloads/candombe.zip   https://zenodo.org/api/records/13922116/files/candombe.zip/content
~~~

下载完成后先验证字节数和 MD5，再解压。失败或部分 ZIP 不允许送入准备器。

## 划分与泄漏规则

- GTZAN 永久 test-only；任何 GTZAN 原曲、切片、变速、变调或 teacher activation 都不得出现在训练和验证。它只作为候选生产门槛，不参与梯度更新。
- GuitarSet、Candombe、SMC 等采用 Beat This! annotations 的 recording-level 官方 split。
- 同一个 recording_id 只能属于一个 split。
- 所有切片和在线增强继承原 recording_id。
- 时间伸缩必须同步变换 beat、downbeat、mask 和 teacher activation。
- 没有 downbeat 标注的数据以 mask=0 排除 downbeat loss，不伪造负标签。
- GiantSteps tempo 标签不展开成伪 beat；其 40 首训练素材只使用 Beat This! teacher 的逐帧 activation，并与最终封存 ID 完全不重叠。
- 模型/解码器超参数只能在 train/validation 上选择；GTZAN 只用于冻结候选的最终一次报告。

training/train.py 在启动时会拒绝同一 recording_id 跨 split。training/prepare.py 强制把 GTZAN 写成 test。

## 仓库边界

允许提交：

- 下载/准备脚本；
- 来源、许可、校验值和固定版本；
- recording-level manifest 的结构说明；
- 不含第三方特征的指标 JSON；
- 通过许可审核后的最终小模型。

禁止提交：

- 原始音乐音频；
- Beatport preview；
- Zenodo 第三方频谱压缩包及解压数组；
- teacher 生成的逐曲 activation；
- 能还原第三方录音内容的大型中间缓存。

本项目的 data/、runs/ 和训练虚拟环境由 .gitignore 排除。
