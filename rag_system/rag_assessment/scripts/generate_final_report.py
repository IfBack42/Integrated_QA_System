"""汇总系统运行记录与RAGAS结果，生成最终离线评估报告。

数据来源：
    collected_system_results.json    --- 24条黄金问题的真实系统运行记录
    retrained_ragas_evaluation_result.csv --- 新意图模型下19条专业RAG样本的RAGAS评分
    baseline_old_model_summary_n7.json --- 旧模型基线（n=7），仅用于版本对比
"""

import csv
import json
import math
import statistics
from pathlib import Path


ASSESSMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ASSESSMENT_DIR / "results"
SYSTEM_RESULTS_FILE = RESULTS_DIR / "collected_system_results.json"
RAGAS_RESULTS_FILE = RESULTS_DIR / "retrained_ragas_evaluation_result.csv"
BASELINE_SUMMARY_FILE = RESULTS_DIR / "baseline_old_model_summary_n7.json"
SUMMARY_FILE = RESULTS_DIR / "final_evaluation_summary.json"
REPORT_FILE = RESULTS_DIR / "final_evaluation_report.md"

RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def _safe_mean(rows, field):
    values = []
    for row in rows:
        value = float(row[field])
        if not math.isnan(value):
            values.append(value)
    return statistics.mean(values) if values else None


def _percentile(values, ratio):
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def build_category_routing(records):
    """按问题类别统计可回答样本的专业RAG进入情况，用于定位路由短板。"""
    stats = {}
    for record in records:
        if not record["should_answer"]:
            continue
        bucket = stats.setdefault(
            record["category"], {"answerable": 0, "professional_route": 0}
        )
        bucket["answerable"] += 1
        if record["actual_intent"] == "专业咨询":
            bucket["professional_route"] += 1
    for bucket in stats.values():
        bucket["rate"] = (
            bucket["professional_route"] / bucket["answerable"]
            if bucket["answerable"]
            else None
        )
    return stats


def pick_low_ragas_rows(ragas_rows, limit=3):
    """挑出四项指标平均值最低的样本，作为答案质量排查入口。"""
    scored = []
    for row in ragas_rows:
        values = []
        for metric in RAGAS_METRICS:
            try:
                value = float(row[metric])
            except (TypeError, ValueError):
                continue
            if not math.isnan(value):
                values.append(value)
        if not values:
            continue
        scored.append((statistics.mean(values), row))
    scored.sort(key=lambda item: item[0])
    return [
        {
            "question": row.get("user_input", ""),
            "mean_score": score,
            **{metric: float(row[metric]) for metric in RAGAS_METRICS},
        }
        for score, row in scored[:limit]
    ]


def build_summary(records, ragas_rows):
    """计算系统成功率、路由、拒答、延迟和RAGAS指标。"""
    total = len(records)
    answerable = [item for item in records if item["should_answer"]]
    unanswerable = [item for item in records if not item["should_answer"]]
    successful = [item for item in records if item["error"] is None]
    professional = [
        item for item in records if item["actual_intent"] == "专业咨询"
    ]
    answerable_professional = [
        item
        for item in answerable
        if item["actual_intent"] == "专业咨询"
    ]
    misrouted = [
        item
        for item in records
        if item["expected_route"] == "professional_rag"
        and item["actual_intent"] != "专业咨询"
    ]
    latencies = [item["elapsed_seconds"] for item in successful]
    confidences = [
        float(item["intent_confidence"])
        for item in records
        if item.get("intent_confidence") is not None
    ]

    return {
        "test_set": {
            "total": total,
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "ragas_answerable_samples": len(ragas_rows),
        },
        "system": {
            "execution_success_count": len(successful),
            "execution_success_rate": len(successful) / total,
            "answer_refusal_behavior_count": sum(
                bool(item["behavior_matches"]) for item in records
            ),
            "answer_refusal_behavior_rate": sum(
                bool(item["behavior_matches"]) for item in records
            )
            / total,
            "correct_refusal_count": sum(
                bool(item["behavior_matches"]) for item in unanswerable
            ),
            "correct_refusal_rate": sum(
                bool(item["behavior_matches"]) for item in unanswerable
            )
            / len(unanswerable),
        },
        "routing": {
            "professional_route_count": len(professional),
            "professional_route_rate_all": len(professional) / total,
            "answerable_professional_route_count": len(answerable_professional),
            "answerable_professional_route_rate": (
                len(answerable_professional) / len(answerable)
            ),
            "misrouted_case_ids": [item["case_id"] for item in misrouted],
            "misrouted_queries": [item["query"] for item in misrouted],
            "misrouted_confidences": [
                float(item["intent_confidence"])
                for item in misrouted
                if item.get("intent_confidence") is not None
            ],
            "min_intent_confidence": min(confidences) if confidences else None,
            "high_confidence_error_count": sum(
                1
                for item in misrouted
                if float(item.get("intent_confidence") or 0) >= 0.95
            ),
        },
        "latency_seconds": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p90": _percentile(latencies, 0.90),
            "p95": _percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "ragas": {
            "sample_count": len(ragas_rows),
            "faithfulness": _safe_mean(ragas_rows, "faithfulness"),
            "answer_relevancy": _safe_mean(ragas_rows, "answer_relevancy"),
            "context_precision": _safe_mean(ragas_rows, "context_precision"),
            "context_recall": _safe_mean(ragas_rows, "context_recall"),
            "median_answer_relevancy": statistics.median(
                [float(row["answer_relevancy"]) for row in ragas_rows]
            ),
            "median_faithfulness": statistics.median(
                [float(row["faithfulness"]) for row in ragas_rows]
            ),
            "low_samples": pick_low_ragas_rows(ragas_rows),
        },
        "category_routing": build_category_routing(records),
    }


def _render_category_table(category_routing):
    lines = ["| 问题类别 | 可回答样本 | 进入专业RAG |", "|---|---:|---:|"]
    for category, bucket in category_routing.items():
        lines.append(
            f"| {category} | {bucket['answerable']} | "
            f"{bucket['professional_route']}/{bucket['answerable']} |"
        )
    return "\n".join(lines)


def _render_ragas_comparison(ragas, baseline):
    """与旧模型基线对比；无基线时只输出单列。"""
    if not baseline:
        lines = ["| 指标 | 结果 |", "|---|---:|"]
        lines.extend(
            f"| {METRIC_LABELS[metric]} | {ragas[metric]:.4f} |"
            for metric in RAGAS_METRICS
        )
        return "\n".join(lines)

    old = baseline["ragas"]
    lines = [
        f"| 指标 | 旧模型（n={old['sample_count']}） | 新模型（n={ragas['sample_count']}） | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for metric in RAGAS_METRICS:
        delta = ragas[metric] - old[metric]
        delta_text = f"{delta:+.4f}"
        if abs(delta) < 5e-5:
            delta_text = "0.0000"
        lines.append(
            f"| {METRIC_LABELS[metric]} | {old[metric]:.4f} | "
            f"{ragas[metric]:.4f} | {delta_text} |"
        )
    return "\n".join(lines)


def _render_low_samples(low_samples):
    if not low_samples:
        return "无可用样本。"
    lines = [
        "| 问题 | 平均分 | 忠实度 | 答案相关性 |",
        "|---|---:|---:|---:|",
    ]
    for sample in low_samples:
        lines.append(
            f"| {sample['question']} | {sample['mean_score']:.4f} | "
            f"{sample['faithfulness']:.4f} | {sample['answer_relevancy']:.4f} |"
        )
    return "\n".join(lines)


def render_report(summary, baseline=None):
    """把结构化结果渲染为便于复盘和简历取材的Markdown。"""
    test_set = summary["test_set"]
    system = summary["system"]
    routing = summary["routing"]
    latency = summary["latency_seconds"]
    ragas = summary["ragas"]
    misrouted = "、".join(routing["misrouted_case_ids"]) or "无"
    misrouted_queries = "；".join(routing["misrouted_queries"]) or "无"
    misrouted_confidence = (
        f"{routing['misrouted_confidences'][0]:.4f}"
        if routing["misrouted_confidences"]
        else "无"
    )

    old_rate_text = ""
    if baseline:
        old_routing = baseline["routing"]
        old_rate_text = (
            f"（旧模型为{old_routing['answerable_professional_route_count']}/"
            f"{baseline['test_set']['answerable']}，"
            f"{old_routing['answerable_professional_route_rate']:.2%}）"
        )

    return f"""# 智能问答系统最终离线评估报告

## 评估口径

- 黄金测试集共{test_set['total']}条：可回答{test_set['answerable']}条，资料外问题{test_set['unanswerable']}条。
- 所有问题均调用真实系统入口和意图分类；被判为专业咨询的样本继续进入检索策略、Milvus混合检索、重排和大模型生成链路。
- 本轮使用重训后的BERT意图模型（`bert_query_classifier`）重新采集，RAGAS只统计“应当回答、实际进入专业检索且返回上下文”的{test_set['ragas_answerable_samples']}条样本。
- 该数据集规模仍较小，结果用于当前项目基线，不代表大规模线上效果。
- 旧模型（n={baseline['ragas']['sample_count']}）的成绩保留在`baseline_old_model_report_n7.md`，仅作版本对比。

## 系统层结果

| 指标 | 结果 |
|---|---:|
| 运行成功率 | {system['execution_success_count']}/{test_set['total']}（{system['execution_success_rate']:.2%}） |
| 基础应答/拒答行为符合预期 | {system['answer_refusal_behavior_count']}/{test_set['total']}（{system['answer_refusal_behavior_rate']:.2%}） |
| 资料外问题正确拒答 | {system['correct_refusal_count']}/{test_set['unanswerable']}（{system['correct_refusal_rate']:.2%}） |
| 可回答问题进入专业RAG | {routing['answerable_professional_route_count']}/{test_set['answerable']}（{routing['answerable_professional_route_rate']:.2%}）{old_rate_text} |

## 分类别路由情况

{_render_category_table(summary['category_routing'])}

## RAGAS结果

{_render_ragas_comparison(ragas, baseline)}

说明：Answer Relevancy中位数为{ragas['median_answer_relevancy']:.4f}，均值偏低是被2条退化为“资料不足”的样本拖下来的；Faithfulness中位数为{ragas['median_faithfulness']:.4f}。

## RAGAS得分最低的样本

{_render_low_samples(ragas['low_samples'])}

## 查询延迟

| 指标 | 秒 |
|---|---:|
| 平均值 | {latency['mean']:.3f} |
| 中位数 | {latency['median']:.3f} |
| P90 | {latency['p90']:.3f} |
| P95 | {latency['p95']:.3f} |
| 最小/最大 | {latency['min']:.3f} / {latency['max']:.3f} |

说明：延迟不包含系统启动和模型加载时间，来自本地单用户串行测试。

## 主要结论

1. 24条样本全部执行成功、行为全部符合预期，4条资料外问题全部正确拒答，没有编造价格、年龄限制、课时和就业承诺。注意“行为符合”只说明该答的答了、该拒的拒了，不代表答案都对。
2. 意图路由在本轮开发集上达到{routing['answerable_professional_route_count']}/{test_set['answerable']}（{routing['answerable_professional_route_rate']:.2%}），RAGAS有效样本从7条扩大到{ragas['sample_count']}条，指标覆盖面明显好于上一版。
3. 在{ragas['sample_count']}条真实专业检索样本上，忠实度{ragas['faithfulness']:.4f}、答案相关性{ragas['answer_relevancy']:.4f}、上下文精确率{ragas['context_precision']:.4f}、上下文召回率{ragas['context_recall']:.4f}。
4. Context Precision/Recall接近1.0，主要原因是知识库规模小、检索只返回2段高度同源的资料，指标接近饱和，不能外推为线上检索质量。
5. 剩余误路由样本：{misrouted}（{misrouted_queries}）。它与保留集另外两条失败样本（“二元与三元语法在历史窗口上有什么差异？”、“聊天机器人中Streamlit和LangChain各自承担什么职责？”）属于同一类：教材术语的中文别称、以及“各自承担什么职责”式问法覆盖不足。下一批训练数据补这两类表达即可，不必再调分类阈值。

## 已知问题与边界

- 全部{test_set['total']}条样本的意图置信度都不低于{routing['min_intent_confidence']:.4f}，误路由的{misrouted}同样给出了{misrouted_confidence}。模型整体过度自信，所以“低置信度再兜底”这条常见策略在这批数据上不会触发，不能作为修复方案。
- 检索配置`candidate_m=2`，每次只把2段资料交给大模型。Context Precision/Recall接近1.0是因为知识库小、资料与问题同源且候选很少，不能当作线上检索能力的证据；真实场景需要先调大候选数再重新评估。
- 最典型的真实质量问题是`course_006`（“NLP基础阶段会不会讲Transformer和BERT，除此之外还有哪些模型？”）：知识库里确实有原句“第六部分：NLP自然语言处理基础……诸如RNN、LSTM、GRU、Transformer、BERT等技术”，对应父块也在Milvus中；但用原问题检索时，Top-K提到8、直接检索和子查询都排不进候选，模型只能看到该部分的表格片段，于是答成“BERT无法确认”，Answer Relevancy与Context Recall记0。原因是九个大章节开头都用“可解决的现实问题/可掌握的核心竞争力”这类重复措辞，介绍句本身区分度低。属于召回排序问题，不是分块丢标题，也不是生成模型的问题。
- 候选数实验：Top-K/Top-M从3/2提到6/4后，`llm_008`（“PPL是越大越好还是越小越好”）从“资料不足”变成“越小越好”，耗时没有上升；`course_006`即便提到8/6仍答不出BERT。实验记录见`candidate_m_experiment.json`。所以调大候选数值得做，但它救不了`course_006`这类召回排序问题。
- RAGAS忠实度对答案形式敏感：`course_003`（0.5882）是长枚举答案被判为无依据，`robust_002`（0.6000）是短答案逐句引用仍被判部分不成立，两次人工核对后都能在检索资料里找到原文。低分样本必须人工抽查后才能定性，不能直接当作幻觉证据。
- Milvus中保存的来源路径仍指向已不存在的`rag_system_learning_version`目录，前端引用会指向无效路径，重新入库时需要一并更新元数据。

## 简历可用表述

> 构建覆盖课程问答、LLM基础概念、口语改写与越界问题的24条黄金测试集，打通真实业务链路采集与RAGAS自动评估；重构意图分类训练数据后，可回答问题的专业RAG路由由35%提升至95%，并在有效专业检索子集（n={ragas['sample_count']}）上取得Faithfulness {ragas['faithfulness']:.3f}、Answer Relevancy {ragas['answer_relevancy']:.3f}、Context Precision/Recall {ragas['context_precision']:.3f}/{ragas['context_recall']:.3f}，资料外问题{system['correct_refusal_count']}/{test_set['unanswerable']}正确拒答。

简历中必须保留“有效专业检索子集（n={ragas['sample_count']}）”或等价口径，不能把该RAGAS成绩描述为全部{test_set['total']}条样本的平均分。
"""


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = json.loads(SYSTEM_RESULTS_FILE.read_text(encoding="utf-8"))
    with RAGAS_RESULTS_FILE.open(encoding="utf-8-sig", newline="") as file:
        ragas_rows = list(csv.DictReader(file))
    baseline = None
    if BASELINE_SUMMARY_FILE.exists():
        baseline = json.loads(BASELINE_SUMMARY_FILE.read_text(encoding="utf-8"))

    summary = build_summary(records, ragas_rows)
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_FILE.write_text(render_report(summary, baseline), encoding="utf-8")

    print(f"结构化结果: {SUMMARY_FILE}")
    print(f"最终报告: {REPORT_FILE}")


if __name__ == "__main__":
    main()
