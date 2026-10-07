# 多模态大模型肾小管框选评测

- 项目名称与用途：把同一张 H&E 肾脏图和同一条提示词通过 OpenRouter 发给多个多模态模型，解析它们返回的边界框，对手工 ground truth 评分并生成叠加图。
- 可得到的结果：每次调用的原始响应（含 provider、token、费用、OpenRouter 记录的首 token 延迟）、解析后的框、叠加图、`scores.csv`，以及按模型汇总的 `summary.md`（准确度表 + 速度与价格表）。
- 支持的系统与环境：Windows PowerShell；Python 3.11（uv 管理）；只依赖 requests 和 Pillow，评测部分不需要 GPU。
- 自有代码许可证：MIT；图像来源与使用范围单独见公开项目data/README.md。
- GitHub 项目地址：https://github.com/siyuvision/ia4u-deep-dive/tree/main/projects/mllm-renal-tubule-bbox 。发布状态见研究项目releases/github/RELEASE.md。

## 安装
在项目根目录 `00-research/projects/mllm-renal-tubule-bbox/` 执行：

```powershell
uv venv repo/.venv --python 3.11
uv pip install --python repo/.venv/Scripts/python.exe -r repo/requirements.lock.txt
```

## 放置 OpenRouter API key
把 `repo/.env.example` 复制为 `repo/.env`，写入一行：

```
OPENROUTER_API_KEY=sk-or-...
```

`repo/.env` 被根目录 .gitignore 排除，不会进入 Git。也可以设置同名的环境变量，环境变量优先于 .env。key 不会写入任何结果文件。

## 运行

```powershell
$py = "repo/.venv/Scripts/python.exe"

# 1. 不需要 key：检查模型 ID、价格，生成快照
& $py repo/scripts/check_models.py

# 2. 不需要 key：看计划和最坏费用上界
& $py repo/scripts/run_models.py --dry-run

# 3. 冒烟测试：一个便宜的模型调用一次
& $py repo/scripts/run_models.py --models gpt-6-luna --repeats 1 --run-id smoke
& $py repo/scripts/evaluate_runs.py --run-id smoke

# 4. 正式运行：8 个模型各 3 次，同一个 run-id 重跑会续跑，已成功的调用不会重复付费
#    （重跑也会给已有结果补查首 token 延迟；--no-generation-stats 可跳过）
& $py repo/scripts/run_models.py --run-id 20261007-main
& $py repo/scripts/evaluate_runs.py --run-id 20261007-main
```

模型清单和开关在 `repo/config/models.json`，提示词在 `repo/prompts/bbox_v1.txt`。改动提示词应另存为新版本文件（如 bbox_v2.txt）并用 `--prompt` 指定，不覆盖 v1。

## 输入与输出
- 输入：`data/input/kidney-he-case01-1024x791.jpg`（发给模型的干净图）、`data/annotations/gt-manual.json`（手工 GT）。
- 输出在 `results/<run-id>/`：`raw/`（每次调用一个 JSON）、`parsed/`（换算到 1164×900 的预测框）、`overlays/`（绿 = GT，蓝 = IoU 0.5 下匹配的预测，红 = 未匹配的预测）、`scores.csv`、`summary.md`。

## 指标说明
- 匹配：一对一，IoU 最高的优先，IoU 须不小于阈值。
- 精确率 = 匹配数 ÷ 预测数；召回率 = 匹配数 ÷ GT 数；F1 = 2·匹配数 ÷ (2·匹配数 + 误检 + 漏检)。
- 严格口径：所有未匹配的预测都算误检。边界容忍口径：未匹配且距图像边界 ≤ 10 px 的预测不计误检（`--border-margin` 可改）。
- 「Best IoU」：对每个 GT 框取它与任意预测的最大 IoU，再取平均，与阈值无关。
- 速度：
  - 「Latency」是产出答案的那一次请求的耗时（中位数 [最小–最大]），不含失败重试和退避等待；`scores.csv` 另有含重试的 `wall_s`。调用最多 4 个并发，任务按重复轮次排队，所以同时在飞的是不同模型。
  - 「Out tok/s」= 输出 token（含推理 token）÷ Latency，是端到端速度，不是纯解码速度。推理模型的大部分时间花在思考上，所以这一列主要反映「想了多少」而不只是「生成得多快」。
  - 「TTFT」是 OpenRouter 自己 `GET /api/v1/generation` 记录的首 token 时间（`scores.csv` 里还有 `gen_s`，即它记录的总生成时长）。对推理模型它可能包含首 token 之前的思考时间。这些数据在所有调用结束后统一补查；查不到就留空，不影响其他指标。重跑同一 run-id 也会给旧结果补上。
  - 延迟受当时服务商负载影响，只有这一次运行、一张图、每个模型 3 次调用的样本，不能当作模型的长期速度。
- 价格：
  - 「$/call」和「Total $」是 OpenRouter 在 `usage.cost` 里实际计费的金额，包括被 token 上限截断或无法解析的调用；「List $/M」是快照里的官方单价（输入／输出，美元每百万 token）。
  - 「$ per match」= 平均每次调用的花费 ÷ 平均每次调用在 IoU 0.5 下匹配上的 GT 框数；一个都没匹配上时为空。
  - 汇总表末尾列出「没有被别的模型全面压过」的模型：没有任何其他模型同时做到 F1 不低于它、且更便宜（或更快）。这只是在这 8 个模型里的相对比较。
- 调用失败（网络、HTTP 错误）不计入均值，并在 Flags 列标出；模型返回了但无法解析的答案按 0 个预测计分。
- Flags 中的 `convention?` 表示换一种坐标格式会明显提高分数，说明该模型可能没有按提示词要求的格式作答。

## 测试

```powershell
repo/.venv/Scripts/python.exe -m pytest repo/tests -q
```

测试不联网。`test_manual_ground_truth_is_self_consistent` 在 GT 文件不存在时会跳过。

## 已验证版本与限制
- 已验证：见 `experiments/RUN-LOG.md` 的「2026-10-07 工具链验证」。
- 已用真实 key 完成一次 8 模型 × 3 次的运行，以及一次冒烟调用；请求格式、计费字段、`/generation` 的单位和各模型的输出格式都在实际响应上核对过。
- 未验证：提示词语言、图像分辨率、固定推理强度对结果的影响；只有一张图和一个标注者。
- 目录：`src/mllm_bbox/`（几何与评分、解析、OpenRouter 客户端、评测、叠加图）、`scripts/`、`tests/`、`config/`、`prompts/`。
