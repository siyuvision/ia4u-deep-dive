# KongNet vs ClassPose — 病理核检测对比实操

Windows 上对两个核检测/分类模型的整片推理 (WSI) 与 QuPath 回显对比。

| 模型 | 论文 | 参数量 | 预训练 |
|------|------|--------|--------|
| [KongNet](https://arxiv.org/abs/2510.23559v2) | Jiaqi Lv et al., Medical Image Analysis 2026 | 176M | CoNIC (6 类) |
| [ClassPose](https://github.com/sohmandal/classpose) | Mandal et al. | — | CoNIC (6 类) |

**主要结论**（详见 [docs/RUN-LOG.md](docs/RUN-LOG.md)）：

- 同一张 TCGA-AA-3534 结肠 WSI（29952×60160 px），KongNet 检出 160,354 个核，ClassPose 检出 152,253 个
- 速度：KongNet ~7.1 min vs ClassPose ~15.6 min（RTX 4090 Laptop, GPU boost）
- 6 类分布高度一致；无逐细胞真值故不计算准确率

## 前置要求

- **Windows 10/11** 64-bit
- **NVIDIA GPU** + 兼容 CUDA 驱动（已测试 RTX 4090 Laptop）
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — 快速 Python 包管理器
- **Git**（含 submodule 支持）
- **PowerShell** 5.1+

## 一键安装

```powershell
# 克隆（含子模块）
git clone --recurse-submodules https://github.com/siyuvision/ia4u-deep-dive.git
cd ia4u-deep-dive/projects/kongnet-vs-classpose

# 一键安装：submodule → 权重下载 → 双环境创建 → CUDA 验证
powershell -ExecutionPolicy Bypass -File install/setup.ps1
```

安装脚本会自动完成：
1. 初始化 git submodule（KongNet + ClassPose 作者源码）
2. 从 HuggingFace 下载 KongNet CoNIC 权重（~176 MB, SHA256 校验）
3. 创建两个 Python 虚拟环境并安装所有依赖
4. 验证 CUDA 可用性

## 快速运行

准备一张 `.svs` 或其他 OpenSlide 支持的整片图像。

### KongNet 整片推理

```powershell
.venv\Scripts\python.exe scripts\run_kongnet_wsi_windows.py `
    --input_dir D:\data `
    --single_wsi slide.svs `
    --mask_dir results\masks `
    --output_dir results\kongnet `
    --cache_dir results\.cache `
    --local_weights weights\KongNet_CoNIC_1.pth `
    --no_tta --batch_size 8
```

### ClassPose 整片推理

```powershell
.venv-classpose\Scripts\python.exe scripts\run_classpose_wsi.py `
    --wsi D:\data\slide.svs `
    --weights .venv-classpose\Lib\site-packages\classpose\model_weights `
    --output results\classpose
```

### QuPath 回显

推理完成后，输出目录会自动生成 `.qupath.geojson` 文件。在 QuPath 中：

1. 打开原始 WSI
2. File → Object data → Import objects from file → 选择 `.qupath.geojson`
3. Class list 右键 → Populate from existing objects → All classes

## 目录结构

```
kongnet-vs-classpose/
├── install/                  # 安装脚本
│   ├── setup.ps1             ← 一键入口
│   ├── download-weights.ps1  ← HuggingFace 权重下载
│   ├── install-environments.ps1  ← 双环境创建
│   ├── requirements-kongnet.txt
│   └── WEIGHTS-LICENSE.txt
├── scripts/                  # 推理与工具脚本
│   ├── run_kongnet_wsi_windows.py
│   ├── run_classpose_wsi.py
│   ├── run_patch.py
│   ├── export_qupath.py
│   ├── evaluate_points.py
│   ├── extract_conic_sample.py
│   └── prepare_wsi.py
├── tests/                    # 验证脚本
├── vendor/                   # Git submodules（作者原始仓库）
│   ├── KongNet_Inference_Main/
│   └── classpose/
├── docs/
│   └── RUN-LOG.md            # 完整实验记录
└── README.md
```

## 许可

- 本项目适配脚本：MIT License（见根目录 [LICENSE](../../LICENSE)）
- KongNet 权重：CC BY-NC-SA 4.0（见 [install/WEIGHTS-LICENSE.txt](install/WEIGHTS-LICENSE.txt)）
- KongNet 作者代码：BSD-3-Clause（见 vendor/KongNet_Inference_Main/LICENSE）
- ClassPose 代码：见 vendor/classpose/LICENSE

## 致谢

- [KongNet](https://github.com/Jiaqi-Lv/KongNet_Inference_Main) — Jiaqi Lv et al., TIA Centre, University of Warwick
- [ClassPose](https://github.com/sohmandal/classpose) — Sohyun Mandal et al.
- [CoNIC Challenge](https://conic-challenge.grand-challenge.org/) — 训练数据来源
- [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) — 整片推理框架
