# 已完成实验的证据快照

本目录可直接审阅，不要求先训练。运行脚本只向 `data/`、`experiments/`、`results/` 写新产物，不读取本目录来替代实际实验。

- `two-shot-protocol.json`：首轮预先冻结的 2 训 / 5 测个体、采集组、训练预算及完整数据路径记录。
- `two-shot-metrics.json`：五例对 STAPLE 及 A/B/C 的原始指标、宏平均及空集定义。主要结论用 STAPLE。
- `diagnosis.json`：同一份原始预测的全体积、目标切片、远端 FP 和标注者一致性检查。
- `four-shot-protocol.json`：后续四/三开发划分、相同训练预算和规则来源。此时已检查过七只动物。
- `spatial-prior.json`：只从四例训练数据拟合的位置/体积参数及训练标注留出覆盖检查。
- `four-shot-summary.json`：相同三例的旧两例原始、新四例原始、新四例清理后指标；完整 TP/FP/FN、距离统计、预测哈希及模型哈希。
- `component-retention-check.json`：清理后多目标连通块和参考重合的检查。
- `two-shot-training.json` / `four-shot-training.json`：CUDA 环境、用时、峰值 allocated 显存和固定最终模型信息。
- `provenance.json`：原实验文件路径与 SHA256，以及替换本机路径后的发布文件 SHA256。内部引用的原文件摘要不等于已脱敏副本的摘要。

首轮 5 例宏平均与后续同 3 例宏平均属于不同集合，不应直接相减。发布结果来自实际完成的本地训练；跨 GPU / 依赖 / cuDNN 执行不保证逐位一致。

发布验证（2026-09-27）：公开源码 Python 编译通过；指标 3 项测试、清理 3 项测试通过；原始权重 SHA256 通过；原环境对公开依赖文件 `uv pip install --dry-run` 显示无需更改，`uv pip check` 87 个包兼容；PowerShell 安装脚本语法解析通过。未声称在一台全新机器重新训练完成。
