# 来源与许可（核验于 2026-09-27）

- **nnFoundation**：Harsy et al., *nnFoundation: 3D Foundation Models for Radiology*, [arXiv:2609.26924](https://arxiv.org/abs/2609.26924)，2026-09-22 提交的预印本。预训练规模 210 万份 CT/MRI/PET 三维影像。
- **nnFoundation-CNN 权重**：German Cancer Research Center (DKFZ) and contributors，[官方模型页](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) 标注 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。原始文件 `checkpoint_final.pth`，SHA256 `ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd`。适配代码的 MIT 不覆盖权重或修改权重；再分发应遵守上游署名、修改声明与同许可等条件。本仓库只提供下载/校验入口。
- **nnU-Net**：[MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)，Apache-2.0；固定提交 `202f6baa0adc2ef5f7b615df19cc4da970412cc0`。[上游微调文档](https://github.com/MIC-DKFZ/nnUNet/blob/202f6baa0adc2ef5f7b615df19cc4da970412cc0/documentation/finetuning_from_nnssl_checkpoints.md)。安装时从上游克隆并保留原许可。
- **TumSeg 数据**：Jensen et al., *3D whole body preclinical micro-CT database of subcutaneous tumors in mice with annotations from 3 annotators*, Scientific Data 11, 1021 (2024)，[论文](https://doi.org/10.1038/s41597-024-03814-y)。[数据 DOI](https://doi.org/10.17894/UCPH.F7BCF864-BE18-4A16-95AD-6F22DEDB4265)，[官方档案](https://erda.ku.dk/archives/ba4fcd9bfa0fb581d593297dd43d1fd1/published-archive.html)。档案写作 **CC-BY**，未明确版本；DOI 元数据 rightsList 为空，因此本项目不补写版本号。
- **本仓库图像**：思语视觉依据上述 TumSeg CT/标注及本实验预测重新生成，新增预测轮廓、指标与排版。数据署名：Jensen et al. / University of Copenhagen / 数据 DOI 同上 / CC-BY。原始数据无重采样，只在实验副本统一空间头信息；图示有窗口、投影及叠加。未复制论文插图。图中的参考与预测分别标注，不表示临床诊断。
- **未使用的第三方实现**：KjaerLab/TumSeg 的模型代码与权重没有被复制或调用。其仓库 README 的 CC BY-NC 4.0 不能代替这里所用数据的许可。
- **本项目代码**：思语视觉，2026，遵循仓库根目录 [MIT LICENSE](../../LICENSE)。其余依赖遵循各自许可，版本见 `repo/requirements-lock.txt`。

原始实验 JSON 中的本机绝对路径在 `evidence/` 发布版被替换为相对路径或缓存占位符。数字与划分不变；原文件及发布版 SHA256 见 `evidence/provenance.json`。原协议内部指向的原始 SHA 不等同于路径已脱敏后的文件 SHA，复现运行会重新冻结自己的协议。
