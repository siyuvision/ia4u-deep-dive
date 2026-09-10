# 实验与验证记录

## 当前采用结果与内容口径 · 2026-09-09

- 本期转入内容提炼，以 KongNet 为主，ClassPose 为上一期方案的对照；教学顺序见 ../handoff/20260908-kongnet-vs-classpose.md。
- 当前选用 KongNet Run 5：results/wsi/kongnet-v2/case01_conic.db 及同目录 case01_conic.qupath.geojson；ClassPose Run 6：results/wsi/classpose-v3/case01_classpose.json 及同目录 case01_classpose.qupath.geojson。文件已核对存在；不把早期 classpose/ 当作最新结果。
- 已核对 ClassPose v3：target_mpp=0.5、resize_factor=0.465；152,253 个对象中，六类合计 152,184，另有 69 个标签 0，保留为 unclassified_0。早期 Run 3 的 92.9% epithelial 已被修复后的结果替代。
- 用户已在 QuPath 检查两模型并反馈“结果表现差不多”。这是用户定性观察，未提供固定 ROI、标注或准确率；不能写为模型等效或准确率相同。
- 当前代码可见 MPP 缩放修复；本轮只整理文档并核对已有输出，未再次运行模型。
- 当前环境核对：KongNet Python 3.11 环境；ClassPose Python 3.13 环境中 cellpose METADATA 为 4.0.8，安装脚本也固定 4.0.8。旧文档的 4.2.1.1 不作为后续安装依据；每次历史运行未保存完整依赖快照，不能仅据当前环境追认。
- 本期代码版本尚未冻结。sources/source-archive-sha256.json 仅记录上游 ZIP；不能替代本地适配代码的版本。

### 速度记录的准确含义

- KongNet Run 5 的 423.05 s 来自日志 Detection completed / Total processing time，对应 base_inference_interface.py 中 inference_instance.start 的计时；包括该函数的推理、后处理、DB 保存及内部清理等待，不包含前置模型加载、后续外层清理和 GeoJSON 导出。日志进度条约 450.78 s 也不是全部 CLI 生命周期。
- ClassPose Run 6 的 938.55 s 为 wall_seconds_total：包含读图准备、模型加载与推理，但在 JSON 保存及 GeoJSON 导出前停止。
- 两个记录值的比值约 2.22；可在本机体验中分别展示并标注计时范围，不能称为严格统一口径的端到端加速比。记录中“均增强模式”不等于实际频率、功率、温度全程一致。
- 如需要论文级公平速度结论，应另做统一计时实验；按当前决定不阻塞 KongNet 介绍和结果展示。不要继续使用旧版 6.5×、Run 4 的约 3× 或无条件“端到端 2.2×”。

### QuPath 显示与历史记录

- 两模型分别输出独立 GeoJSON。导入同一条目会叠加；Annotation list 为空不代表 detection 未导入，Class list 可通过 Populate from existing objects → All classes 补入。
- QuPath 0.7.0 对单点 detection 在部分缩放比例下的显示问题仍保留为注意事项；此前只完成无界面导入测试，不能据此保证所有缩放下可见。用户后来已完成查看，但未记录具体修复方式。
- 下列首次导出和 Run 1–6 为历史证据；使用路径以本节为入口，不把每个历史段落都理解为当前待办。

## QuPath 首次回显导出 · 2026-09-09（历史，当前路径见页首）

- 范围：按用户决定跳过进一步对比条件核实与模型效果检查，仅转换已有结果，不重新推理。此导入检查不代表分割质量或速度比较通过验证。
- 转换脚本：repo/scripts/export_qupath.py；命令在项目根目录运行：`repo/.venv/Scripts/python.exe repo/scripts/export_qupath.py <下列原始结果路径>`。
- KongNet WSI：`results/wsi/kongnet/case01_conic.db` → `results/wsi/kongnet/case01_conic.qupath.geojson`，160,354 个点。
- ClassPose WSI：`results/wsi/classpose/case01_classpose.json` → `results/wsi/classpose/case01_classpose.qupath.geojson`，151,518 个点（包括 22 个原始类别 0，显示为灰色 unclassified_0，未删除或改归其他类别）。
- Patch：`results/patch/kongnet/predictions.json` 与 `results/patch/classpose/predictions.json` 各生成相邻的 `predictions.qupath.geojson`，分别为 141、115 个检测点；不是旧版 results/patch-1001/。
- 坐标：原图全分辨率像素，左上角为原点。KongNet DB 已在上游保存时按 0.5/0.2325 转为 level-0；ClassPose WSI JSON 已含 level-0 坐标，导出不再缩放。WSI 原图为 `data/derived/wsi-input/case01.svs`；patch 原图为 `data/derived/conic-1001/image.png`。
- 对象：QuPath detection / Point，带模型前缀分类、统一类别颜色；KongNet 已有置信度原样写入 Confidence 测量。仅显示保存的点，不补造细胞轮廓。
- 验证：本机 QuPath 0.7.0 通过 `repo/tests/verify_qupath_import.groovy` 在实际 SVS 上读取并添加两组全部对象，检查对象数、Point 类型、非空分类与坐标处于图像范围，两个命令退出码均为 0；未保存或修改用户 QuPath 项目。此为无界面导入验证，尚未人工查看界面。
- Patch 同样通过 QuPath 0.7.0 在 256×256 原图上的实际导入检查，分别为 141 和 115 个对象；四个 GeoJSON 均通过检查。
- 导入操作：QuPath 打开上述原图，选择 File → Object data → Import objects from file，再选相应 `.qupath.geojson`。建议分别查看两模型；重复导入同一文件会叠加重复对象。
- 后续输出：三个现有推理入口已调用共用转换器，保存原生结果后自动附带 GeoJSON。KongNet 只转换本次新写入或改变的 DB，并在上游退出清理抛错时仍尝试导出；本次未重新执行 GPU 推理。
- 格式依据：https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html

## Run 1 — Patch-level smoke test (同图对比)

- 日期：2026-09-09
- 代码版本：repo/scripts/run_patch.py, evaluate_points.py
- 系统／硬件：Windows 10, RTX 4090 Laptop GPU, CUDA 12.4
- 依赖版本：
  - KongNet 环境：Python 3.11.16, torch 2.5.1+cu124, tiatoolbox 1.6.0
  - ClassPose 环境：Python 3.13.15, torch 2.14.0+cu126, cellpose 历史文档曾写 4.2.1.1（无逐次快照可确认；当前安装 4.0.8）, classpose
- 输入数据：data/derived/conic-1001/image.png (256×256, CoNIC patch 1001)
- 参考标注：data/derived/conic-1001/inst_map.png + class_map.png (125 instances, 24 touch border)
- 参考类别分布 (CoNIC 1..6)：[0 neutrophil, 92 epithelial, 5 lymphocyte, 2 plasma, 0 eosinophil, 26 connective]
- 匹配半径：12 像素

### 命令

```
# KongNet
repo/.venv/Scripts/python.exe repo/scripts/run_patch.py --model kongnet --image data/derived/conic-1001/image.png --weights C:\Users\ASUS\.cache\siyu-vision-models\kongnet\KongNet_CoNIC_1.pth --output results/patch/kongnet

# ClassPose
repo/.venv-classpose/Scripts/python.exe repo/scripts/run_patch.py --model classpose --image data/derived/conic-1001/image.png --weights C:\Users\ASUS\.cache\siyu-vision-models\classpose\conic.pt --output results/patch/classpose

# 评估
repo/.venv/Scripts/python.exe repo/scripts/evaluate_points.py --instances data/derived/conic-1001/inst_map.png --image data/derived/conic-1001/image.png --classes data/derived/conic-1001/class_map.png --predictions results/patch/kongnet/predictions.json results/patch/classpose/predictions.json --output results/patch/evaluation.json --radius 12
```

### 结果

| 指标 | KongNet | ClassPose |
|------|---------|-----------|
| 检出数 | 141 | 115 |
| True Positives | 116 | 109 |
| False Positives | 25 | 6 |
| False Negatives | 9 | 16 |
| Precision | 0.823 | 0.948 |
| Recall | 0.928 | 0.872 |
| F1 | 0.872 | 0.908 |
| 匹配分类准确率 | 94.8% | 100.0% |
| 推理耗时 (warm patch) | 0.150 s | 0.810 s |

类别计数对比 (CoNIC 1..6)：

| 类别 | 参考 | KongNet | ClassPose | KongNet 误差 | ClassPose 误差 |
|------|------|---------|-----------|-------------|---------------|
| 1 neutrophil | 0 | 0 | 0 | 0 | 0 |
| 2 epithelial | 92 | 96 | 84 | 4 | 8 |
| 3 lymphocyte | 5 | 6 | 5 | 1 | 0 |
| 4 plasma | 2 | 1 | 2 | 1 | 0 |
| 5 eosinophil | 0 | 0 | 0 | 0 | 0 |
| 6 connective | 26 | 38 | 24 | 12 | 2 |

- 输出路径：results/patch/evaluation.json
- 已知限制：此 patch 来自 CoNIC 训练集公开发布，两个模型均可能在该数据上训练过，结果仅用于演示和流程验证，不能代表模型在独立测试集上的泛化性能。

---

## Run 2 — KongNet 整片推理 (WSI)

- 日期：2026-09-09
- 代码版本：repo/scripts/run_kongnet_wsi_windows.py (Windows 适配器)
- 解释器路径：repo/.venv/Scripts/python.exe
- 系统／硬件：Windows 10, RTX 4090 Laptop GPU, CUDA 12.4
- 输入：data/derived/wsi-input/case01.svs (29952×60160 px, 20×, aperio, 0.2325 mpp)
- Mask：data/derived/masks/case01.png
- 权重：C:\Users\ASUS\.cache\siyu-vision-models\kongnet\KongNet_CoNIC_1.pth (SHA256 已验证)
- 参数：--no_tta --num_workers 2 --batch_size 8

### 命令

```
$env:PYTHONUTF8 = "1"
repo/.venv/Scripts/python.exe repo/scripts/run_kongnet_wsi_windows.py --input_dir data/derived/wsi-input --single_wsi case01.svs --mask_dir data/derived/masks --output_dir results/wsi/kongnet --cache_dir C:\Users\ASUS\Desktop\ia4u-content\.cache\wsi-kongnet --local_weights C:\Users\ASUS\.cache\siyu-vision-models\kongnet\KongNet_CoNIC_1.pth --no_tta --num_workers 2 --batch_size 8
```

### 结果

| 指标 | 数值 |
|------|------|
| Patch 推理（约 2140 patches = 268 batches×batch_size 8；日志进度条是 batch 数） | ~80 s |
| 后处理 + 检测 | 45.78 s |
| 检测记录 | 177,218 |
| NMS stage 1 | 54 s → 160,360 点 |
| NMS stage 2 | 日志进度条约 51 s → 160,354 点；清理等待等额外耗时未单独归属 |
| NMS 合计 | 267.72 s |
| 后处理合计 | 313.54 s |
| 保存结果 | 42.50 s |
| 总 wall time | 515.76 s (~8.6 min) |

- 输出：results/wsi/kongnet/case01_conic.db (32.88 MB, SQLite annotation store)
- 退出码：1（推理和保存成功，仅缓存清理 temp_store_1.db 因 Windows 文件锁失败，结果不受影响）
- 适配器说明：Windows 环境下 OpenSlide 句柄无法 pickle，改用 in-process 读取 + ThreadPool 后处理。后续代码把 SQLite 临时文件清理改为非致命 warning；本次运行仍以退出码 1 结束。非作者 Linux 基准环境。
- 日志：experiments/kongnet-wsi-run2.log

---

## Run 3 — ClassPose 整片推理 (WSI)

- 日期：2026-09-09
- 代码版本：repo/scripts/run_classpose_wsi.py (in-process tile 适配器)
- 解释器路径：repo/.venv-classpose/Scripts/python.exe
- 系统／硬件：Windows 10, RTX 4090 Laptop GPU, torch CUDA runtime 12.6 (torch 2.14.0+cu126)
- 输入：data/derived/wsi-input/case01.svs (29952×60160 px)
- Mask：data/derived/masks/case01.png
- 权重：C:\Users\ASUS\.cache\siyu-vision-models\classpose\conic.pt (SHA256 已验证)
- 参数：tile 1024 px，目标原写约 0.5 mpp，但实际输入为 0.2325 mpp（后续已发现错误）；tissue mask > 4%, fp32, no TTA

### 命令

```
$env:PYTHONUTF8 = "1"
repo/.venv-classpose/Scripts/python.exe repo/scripts/run_classpose_wsi.py --wsi data/derived/wsi-input/case01.svs --mask data/derived/masks/case01.png --weights C:\Users\ASUS\.cache\siyu-vision-models\classpose\conic.pt --output results/wsi/classpose
```

### 结果

| 指标 | 数值 |
|------|------|
| 处理 tiles | 655 |
| 模型加载 | 5.33 s |
| 推理 | 3,356.9 s (~56 min) |
| 总 wall time | 3,362.2 s (~56 min) |
| 检出细胞 | 151,518 |

类别分布 (CoNIC 1..6, 修正后)：

| 类别 | 数量 |
|------|------|
| 1 neutrophil | 146 |
| 2 epithelial | 140,714 |
| 3 lymphocyte | 2,327 |
| 4 plasma | 279 |
| 5 eosinophil | 207 |
| 6 connective | 7,823 |

- 输出：results/wsi/classpose/case01_classpose.json (22.03 MB)
- 退出码：0
- 适配器说明：上游 predict_wsi.py 在 Windows 下 ~63% stall，改用 in-process tiling 逐 tile 调用相同 ClassposeModel.eval。非作者 Linux 管线。
- 修正说明：ClassPose CoNIC 权重原生输出 CoNIC 1..6 标签，class_id_conic 应为 identity（不加 +1）。WSI 结果已用脚本修正。
- 日志：experiments/classpose-wsi-run1.log

---

---

## Run 4 — ClassPose 整片推理 v2（修正 MPP 缩放）

- 日期：2026-09-09
- 代码版本：repo/scripts/run_classpose_wsi.py（已修复 resize 步骤）
- 解释器路径：repo/.venv-classpose/Scripts/python.exe
- 系统／硬件：Windows 10, RTX 4090 Laptop GPU, torch CUDA runtime 12.6 (torch 2.14.0+cu126)
- 输入：data/derived/wsi-input/case01.svs (29952×60160 px)
- Mask：data/derived/masks/case01.png
- 权重：C:\Users\ASUS\.cache\siyu-vision-models\classpose\conic.pt (SHA256 已验证)
- 参数：tile 1024 px, level 0 读取 → resize_factor 0.465 → 实际送入模型 ~476×476 px, fp32, no TTA

### 修复内容

Run 3 发现的 MPP 不匹配 bug 已修正（详见下方"Bug 记录"）：
- 读取 tile 后增加 `cv2.resize(tile, fx=0.465, fy=0.465)` 步骤
- 坐标回算增加 `pred_to_level = 1.0 / resize_factor` 缩放
- 输入实际 MPP 从 0.2325（错误）→ 0.5（正确）

### 命令

```
$env:PYTHONUTF8 = "1"
repo/.venv-classpose/Scripts/python.exe repo/scripts/run_classpose_wsi.py --wsi data/derived/wsi-input/case01.svs --mask data/derived/masks/case01.png --weights C:\Users\ASUS\.cache\siyu-vision-models\classpose\conic.pt --output results/wsi/classpose-v2
```

### 结果

| 指标 | 数值 |
|------|------|
| 处理 tiles | 655 |
| 模型加载 | 8.97 s |
| 推理 | 1,543.7 s (~25.7 min) |
| 总 wall time | 1,552.7 s (~25.9 min) |
| 检出细胞 | 152,253 |

类别分布 (CoNIC 1..6)：

| 类别 | v2（修正后） | v1（Run 3，有 bug） | 变化 |
|------|-------------|-------------------|------|
| 1 neutrophil | 346 (0.2%) | 146 (0.1%) | +137% |
| 2 epithelial | 98,534 (64.7%) | 140,714 (92.9%) | **-30%** |
| 3 lymphocyte | 16,240 (10.7%) | 2,327 (1.5%) | **+598%** |
| 4 plasma | 4,896 (3.2%) | 279 (0.2%) | **+1655%** |
| 5 eosinophil | 1,671 (1.1%) | 207 (0.1%) | +707% |
| 6 connective | 30,497 (20.0%) | 7,823 (5.2%) | **+290%** |

- 输出：results/wsi/classpose-v2/case01_classpose.json (23.3 MB) + case01_classpose.qupath.geojson (36.6 MB)
- 退出码：0
- 速度变化：1,553s vs 3,362s（Run 3），快 2.17×（因为缩放后 tile 面积仅为原来 21.6%，模型运算量大幅降低）
- 日志：experiments/classpose-wsi-run2.log

---

---

## Run 5 — KongNet 整片推理 v2（GPU boost 重跑）

- 日期：2026-09-09
- 代码版本：repo/scripts/run_kongnet_wsi_windows.py（同 Run 2）
- 系统／硬件：Windows 10, RTX 4090 Laptop GPU P0 boost (1275 MHz, 53°C), CUDA 12.4
- 输入：data/derived/wsi-input/case01.svs（同 Run 2）
- 参数：--no_tta --num_workers 2 --batch_size 8

### 结果

| 指标 | Run 5 | Run 2 |
|------|-------|-------|
| Patch 推理 | 93.5 s | ~80 s |
| 后处理 + NMS | 289.2 s (stage1 50s + stage2 47s) | 313.5 s |
| 保存 | 12.1 s | 42.5 s |
| 日志处理耗时（非完整 wall time） | **423.05 s (7.1 min)** | 旧记录 515.8 s（计时出处未完整保留） |
| 检出细胞 | 160,354 | 160,354（一致） |

- 输出：results/wsi/kongnet-v2/case01_conic.db + case01_conic.qupath.geojson
- 退出码：0（temp_store 清理 WARNING 不影响结果）
- 与 Run 2 类别计数一致；记录耗时更短，但未做功率模式与缓存的独立控制实验，不将变化确定归因于 GPU boost 或缓存预热
- 日志：experiments/kongnet-wsi-run5.log

---

---

## Run 6 — ClassPose 整片推理 v3（GPU boost 重跑）

- 日期：2026-09-09
- 代码版本：repo/scripts/run_classpose_wsi.py（同 Run 4，含 MPP 修复）
- 系统／硬件：Windows 10, RTX 4090 Laptop GPU P0 boost, torch CUDA runtime 12.6 (torch 2.14.0+cu126)
- 输入：data/derived/wsi-input/case01.svs（同 Run 4）
- 参数：tile 1024 px, resize_factor 0.465, fp32, no TTA

### 结果

| 指标 | Run 6（boost） | Run 4（非 boost） |
|------|---------------|------------------|
| 模型加载 | 4.68 s | 8.97 s |
| 推理 | **933.9 s (15.6 min)** | 1,543.7 s (25.7 min) |
| 总 wall time | **938.5 s (15.6 min)** | 1,552.7 s (25.9 min) |
| 检出细胞 | 152,253 | 152,253（一致） |

类别分布与 Run 4 完全一致（346 / 98,534 / 16,240 / 4,896 / 1,671 / 30,497）。

- 输出：results/wsi/classpose-v3/case01_classpose.json + case01_classpose.qupath.geojson
- 退出码：0
- Boost 模式下推理快 39.5%
- 日志：experiments/classpose-wsi-run3.log

---

## WSI 速度记录汇总（当前选用 Run 5 + Run 6，计时口径见页首）

| | KongNet (Run 5) | ClassPose v3 (Run 6) |
|---|---------|-----------|
| 各脚本记录耗时（口径不同） | **423.05 s (7.1 min)** | **938.55 s (15.6 min)** |
| 细胞数 | 160,354 | 152,253 |
| Patches/tiles | ~2140 patches（268 batch × 8） | 655 tiles |
| 管线类型 | TIAToolbox WSI pipeline + NMS | In-process tile 适配器 |
| Windows 适配 | ThreadPool 替代 multiprocessing | 直接调用 model.eval |
| GPU 状态 | P0 boost | P0 boost |

注意：两者管线结构不同（KongNet 有 overlap+NMS，ClassPose 为 non-overlapping tiles），速度差异部分来源于管线设计。本次同一 GPU、同一 WSI 下的两个记录值相差约 **2.2 倍**，但计时范围不同，仅描述记录值，不能称为严格端到端速度比。

## WSI 类别分布对比

| 类别 | KongNet | ClassPose v2 | 差异 |
|------|---------|-------------|------|
| 1 neutrophil | 840 (0.5%) | 346 (0.2%) | -0.3pp |
| 2 epithelial | 97,993 (61.1%) | 98,534 (64.7%) | +3.6pp |
| 3 lymphocyte | 11,894 (7.4%) | 16,240 (10.7%) | +3.3pp |
| 4 plasma | 5,364 (3.3%) | 4,896 (3.2%) | -0.1pp |
| 5 eosinophil | 1,801 (1.1%) | 1,671 (1.1%) | 0.0pp |
| 6 connective | 42,462 (26.5%) | 30,497 (20.0%) | -6.5pp |
| 六类合计 | 160,354 | 152,184 | |
| 原始类别 0（未归入六类） | 0 | 69 | |
| **全部对象** | **160,354** | **152,253** | |

修正后两模型的主要类别排序相同，但仍存在数量差异：epithelial 主导（61–65%），connective 次之（20–27%），lymphocyte 第三（7–11%）。最大差异在 connective（6.5pp），其余均在 4pp 以内。

## 已知限制

- 单张 WSI、单 GPU、Windows 适配器；虽有修复与重跑，但不是多次同条件重复的统计基准，也不代表 Linux 官方性能。
- Patch 评估用公开训练集数据，两模型均可能见过该数据。
- KongNet WSI 输出为 SQLite (tiatoolbox annotation store)，ClassPose 输出为 JSON；类别分布需分别从各格式提取后对比。
- WSI 级别无逐细胞标注，无法计算 WSI 分类准确率（见下方说明）。

---

## ⚠ Bug 记录：Run 3 ClassPose WSI 输入分辨率不匹配（已修复）

- 发现日期：2026-09-09
- 影响：Run 3 全部结果（151,518 cells）的分类分布不可信
- 根本原因：`run_classpose_wsi.py` 适配器在选择金字塔层后，**缺少到目标 MPP 的缩放步骤**

### 问题详情

上游 `predict_wsi.py` 分两步对齐输入分辨率：
1. 选择最接近目标 downsample 的金字塔层（`get_best_level_for_downsample`）
2. 计算残余缩放因子（`resize_factor = level_downsample / target_downsample`），逐 tile 用 `cv2.resize` 缩放后再送入模型

我们的 Windows 适配器只做了第一步，漏掉了第二步。对于 case01.svs（base 0.2325 µm/px）：
- 目标 downsample = 0.5 / 0.2325 ≈ 2.15×
- 最近金字塔层 = level 0（downsample = 1.0）；level 1 = 4.0× 更远
- 残余缩放 = 1.0 / 2.15 ≈ **0.465**（需要缩小 53.5%）
- 适配器未做此缩放 → 输入实际 MPP 为 0.2325（而非 0.5），细胞线性尺寸偏大 **2.15×**

### 为什么 patch 级别没出问题

`run_patch.py` 的输入是 CoNIC 数据集的 256×256 patch，本身就是 0.5 µm/px，不需要 MPP 转换。此次 WSI 的 MPP 适配问题不直接作用于该 patch 路径；patch 指标仍仅为一个公开训练样例的流程验证，100% 是匹配成功实例上的分类准确率，不代表独立测试表现。

### 修复

`run_classpose_wsi.py` 增加与上游等价的 resize 步骤：
```python
resize_factor = downsample / target_downsample  # ≈ 0.465
# tile 读取后、送入模型前
if abs(resize_factor - 1.0) > 0.01:
    tile = cv2.resize(tile, None, fx=resize_factor, fy=resize_factor,
                      interpolation=cv2.INTER_LINEAR)
```
坐标回算也已修正（prediction 坐标 → level 坐标需除以 resize_factor）。

### 教训

1. 简化移植时，不能假设"选最近金字塔层"就够了——大多数 WSI 的金字塔层跨度很大（1×→4×→16×），中间分辨率必须靠 resize 补齐
2. `model.eval(..., resample=True)` 是 Cellpose 内部的流场重采样，与输入图像的物理分辨率无关
3. 当前 WSI 与本次 CoNIC patch 是不同输入；没有完整训练来源核对证据时，不声称已排除所有训练重叠或已证明比较公平性。

---

## ⚠ Bug 记录：Run 2–4 GPU 性能模式不一致

- 发现日期：2026-09-09
- 影响：Run 2（KongNet）与 Run 3–4（ClassPose）可能在不同的笔记本 GPU 性能模式下运行，速度对比不可靠
- 根本原因：笔记本 GPU（RTX 4090 Laptop）有多种功率档位（省电 / 标准 / 增强），不同 Run 之间未控制一致
- 修复：Run 5–6 在同一性能模式（增强模式）下背靠背执行，中间不切换设置
