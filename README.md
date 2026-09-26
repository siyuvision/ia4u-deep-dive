# ia4u-deep-dive

**思语视觉深度解析系列** — 长视频内容配套代码与一键安装。

本仓库收录「思语视觉」深度解析视频的可复现实操代码，每个子项目对应一期或一组视频，提供一键安装脚本、运行入口和实验记录。

## 项目列表

| 项目 | 说明 | 状态 |
|------|------|------|
| [kongnet-vs-classpose](projects/kongnet-vs-classpose/) | KongNet 与 ClassPose 病理核检测对比：Windows 整片推理 + QuPath 回显 | ✅ 可用 |
| [nnfoundation-tumseg-ct](projects/nnfoundation-tumseg-ct/) | nnFoundation 小鼠肿瘤 CT：两例微调、四例开发性复测、远端清理及完整证据 | ✅ 代码与结果 |

## 快速开始

```powershell
git clone --recurse-submodules https://github.com/siyuvision/ia4u-deep-dive.git
cd ia4u-deep-dive/projects/kongnet-vs-classpose
powershell -ExecutionPolicy Bypass -File install/setup.ps1
```

详细步骤见各项目 README。

## 环境要求

- Windows 10/11 64-bit
- NVIDIA GPU + CUDA 驱动（已测试 RTX 4090 Laptop）
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python 包管理
- Git（含 submodule 支持）
- PowerShell 5.1+

## 许可

本仓库的适配脚本采用 [MIT License](LICENSE)。  
第三方模型代码和权重各有独立许可，详见各项目 `install/WEIGHTS-LICENSE.txt` 和 vendor 子模块内的 LICENSE 文件。
