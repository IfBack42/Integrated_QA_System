# RAG离线评测模块

本目录按“程序、输入数据、运行结果”分层，避免评测脚本和大量JSON/CSV报告混放。

```text
rag_assessment/
├── scripts/
│   ├── collect_system_results.py   # 运行真实RAG并采集答案、上下文和耗时
│   ├── rag_as.py                   # 使用RAGAS计算四项指标
│   └── generate_final_report.py    # 汇总路由、拒答、延迟和RAGAS结果
├── data/
│   ├── system_test_cases.json      # 人工构建的黄金测试集
│   └── ragas_runtime_data.json     # 从真实运行结果筛选出的RAGAS输入
└── results/
    ├── collected_system_results.json
    ├── retrained_ragas_evaluation_result.csv
    ├── final_evaluation_summary.json
    └── final_evaluation_report.md
```

## 数据流

```text
data/system_test_cases.json
→ scripts/collect_system_results.py
→ results/collected_system_results.json
→ data/ragas_runtime_data.json
→ scripts/rag_as.py
→ results/retrained_ragas_evaluation_result.csv
→ scripts/generate_final_report.py
→ results/final_evaluation_summary.json
→ results/final_evaluation_report.md
```

三个脚本均保留独立的 `__main__` 入口。默认只运行少量样本进行环境验证；执行全量评测前，应在对应脚本入口中明确调整 `sample_limit`，并固定模型、配置和测试集版本。

## 文件边界

- `data/` 保存评测输入，不保存程序。
- `results/` 保存运行产物和分析报告，不作为程序输入源码。
- `scripts/` 只负责采集、评分和汇总，不保存大模型权重或课程原始资料。
- 含课程正文的原始数据和逐条运行结果默认由 `.gitignore` 排除；公开仓库只保留不泄露原始资料的汇总报告。
