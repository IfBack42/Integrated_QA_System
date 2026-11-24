"""对比正式与候选意图分类模型在黄金问题和通用回归集上的表现。"""

import gc
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from rag_system.core.intent_classifier.bert_intent_model import BertIntentModel


RAG_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MODEL_PATH = RAG_SYSTEM_ROOT / "models" / "bert_query_classifier"
CANDIDATE_MODEL_PATH = (
    RAG_SYSTEM_ROOT / "models" / "bert_query_classifier_candidate_v2"
)
GOLD_CASES_FILE = (
    RAG_SYSTEM_ROOT / "rag_assessment" / "data" / "system_test_cases.json"
)
OUTPUT_FILE = (
    RAG_SYSTEM_ROOT / "rag_assessment" / "results" / "intent_model_comparison.json"
)

GENERAL_REGRESSION_QUERIES = [
    "北京明天的天气怎么样？",
    "帮我计算125乘以48。",
    "用Python写一个冒泡排序函数。",
    "HTTP 404状态码是什么意思？",
    "TCP为什么需要三次握手？",
    "怎么用Git撤销最近一次提交？",
    "Docker容器和虚拟机有什么区别？",
    "Java里的垃圾回收是怎么工作的？",
    "CSS盒模型包括哪些部分？",
    "解释一下操作系统的死锁。",
    "什么是数据库事务的可重复读？",
    "二叉树的前序遍历顺序是什么？",
    "写一个正则表达式校验邮箱地址。",
    "一元二次方程的求根公式是什么？",
    "条件概率应该怎么计算？",
    "请推荐一个周末旅行地点。",
    "英语现在完成时怎么使用？",
    "怎样设置Excel冻结首行？",
    "成年人每天应该喝多少水？",
    "世界上面积最大的海洋是什么？",
]

# 在候选模型训练完成后才加入，不能再根据这组结果继续改训练数据。
HOLDOUT_KNOWLEDGE_BASE_QUERIES = [
    "语言模型为何通常会给通顺句子更高的概率？",
    "一元语法模型是不是完全不使用上下文？",
    "二元与三元语法在历史窗口上有什么差异？",
    "统计语言模型里的OOV问题是怎样产生的？",
    "神经语言模型相对传统统计方法改善了什么？",
    "已经完成预训练的模型为什么还需要微调？",
    "Scaling Law里模型参数、数据和计算量是什么关系？",
    "模型规模扩大后突然出现新能力属于什么现象？",
    "不更新权重，仅根据提示中的示例学习是什么能力？",
    "机器翻译结果一般怎样使用BLEU评估？",
    "评价自动摘要时ROUGE主要关注什么？",
    "两个模型的困惑度分别为20和100，通常哪个更好？",
    "人工智能学习路线如何从Python过渡到机器学习？",
    "机器学习模块是否包含集成学习和聚类？",
    "NLP项目中的模型压缩会涉及哪些方法？",
    "医疗问答项目怎样结合知识图谱？",
    "物流信息咨询项目会怎样使用RAG？",
    "聊天机器人中Streamlit和LangChain各自承担什么职责？",
    "深度学习阶段使用什么框架并学习哪些网络？",
    "最后的大模型阶段有没有Function Call和Agent项目？",
]

HOLDOUT_GENERAL_QUERIES = [
    "唐朝建立于哪一年？",
    "今天美元兑人民币的汇率是多少？",
    "番茄炒蛋应该怎么做？",
    "请帮我写一封请假邮件。",
    "三角形内角和为什么是180度？",
    "Python里怎样删除字典中的键？",
    "JavaScript的Promise怎么捕获异常？",
    "Windows系统如何查看磁盘空间？",
    "怎样把Word文档转换成PDF？",
    "冰箱冷藏室结冰应该怎么办？",
    "篮球比赛一次进攻有多少秒？",
    "日语五十音应该如何记忆？",
    "红绿灯为什么使用这三种颜色？",
    "银行定期存款利息怎么计算？",
    "民事诉讼和刑事诉讼有什么区别？",
    "给猫咪换粮需要注意什么？",
    "如何用Excel计算一列数据的平均值？",
    "电脑蓝屏后应该先检查什么？",
    "太阳系里最大的行星是哪一个？",
    "周末去爬山应该准备哪些装备？",
]


def load_evaluation_cases():
    """黄金集均属于当前知识库入口；通用回归集应绕过专业RAG。"""
    payload = json.loads(GOLD_CASES_FILE.read_text(encoding="utf-8"))
    cases = [
        {
            "case_id": item["case_id"],
            "query": item["query"],
            "expected_label": "专业咨询",
            "group": "knowledge_base",
        }
        for item in payload["cases"]
    ]
    cases.extend(
        {
            "case_id": f"general_{index:03d}",
            "query": query,
            "expected_label": "通用知识",
            "group": "general_regression",
        }
        for index, query in enumerate(GENERAL_REGRESSION_QUERIES, start=1)
    )
    cases.extend(
        {
            "case_id": f"holdout_kb_{index:03d}",
            "query": query,
            "expected_label": "专业咨询",
            "group": "holdout_knowledge_base",
        }
        for index, query in enumerate(
            HOLDOUT_KNOWLEDGE_BASE_QUERIES, start=1
        )
    )
    cases.extend(
        {
            "case_id": f"holdout_general_{index:03d}",
            "query": query,
            "expected_label": "通用知识",
            "group": "holdout_general",
        }
        for index, query in enumerate(HOLDOUT_GENERAL_QUERIES, start=1)
    )
    return cases


def evaluate_model(model_path, cases, device="cuda"):
    """逐条预测并返回分组准确率、专业召回率和错误明细。"""
    model = BertIntentModel(model_path=model_path, device=device)
    predictions = []
    for case in cases:
        label, confidence = model.predict(case["query"])
        predictions.append(
            {
                **case,
                "predicted_label": label,
                "confidence": confidence,
                "correct": label == case["expected_label"],
            }
        )

    metrics = {}
    for group in {item["group"] for item in predictions}:
        group_rows = [item for item in predictions if item["group"] == group]
        metrics[group] = {
            "correct": sum(item["correct"] for item in group_rows),
            "total": len(group_rows),
            "accuracy": sum(item["correct"] for item in group_rows)
            / len(group_rows),
        }
    metrics["overall"] = {
        "correct": sum(item["correct"] for item in predictions),
        "total": len(predictions),
        "accuracy": sum(item["correct"] for item in predictions)
        / len(predictions),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": metrics, "predictions": predictions}


def main():
    cases = load_evaluation_cases()
    result = {
        "formal": evaluate_model(FORMAL_MODEL_PATH, cases),
        "candidate": evaluate_model(CANDIDATE_MODEL_PATH, cases),
    }
    OUTPUT_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for model_name, model_result in result.items():
        print(model_name, model_result["metrics"])
        errors = [
            item
            for item in model_result["predictions"]
            if not item["correct"]
        ]
        print("错误样本:", [item["case_id"] for item in errors])
    print(f"完整对比结果: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
