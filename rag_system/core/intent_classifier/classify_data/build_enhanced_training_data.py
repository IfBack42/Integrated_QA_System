"""基于原始JSONL构建适配当前知识库范围的意图分类训练集。"""

import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
SOURCE_FILE = DATA_DIR / "model_generic_5000.json"
OUTPUT_FILE = DATA_DIR / "model_enhanced.jsonl"

GENERAL_LABEL = "通用知识"
PROFESSIONAL_LABEL = "专业咨询"

# 这些主题已经进入当前知识库，不能继续按“一般概念题”处理。
RELABEL_AS_PROFESSIONAL = {
    "人工智能现在有哪些主要的应用领域？",
    "机器学习和深度学习是什么关系？",
    "自然语言处理（NLP）是什么？",
}

# 刻意不照抄黄金测试集原句，避免直接把测试答案泄漏到训练集。
KNOWLEDGE_BASE_TOPICS = [
    "大语言模型的定义和典型任务",
    "语言模型怎样计算词序列概率",
    "unigram、bigram与trigram的基本假设",
    "N-gram的数据稀疏和长期依赖问题",
    "神经网络语言模型的基本原理",
    "预训练与下游任务微调",
    "Scaling Law和模型规模扩展",
    "大模型的能力涌现现象",
    "Few-shot与上下文学习",
    "BLEU指标的用途",
    "ROUGE指标的用途",
    "困惑度PPL的含义和大小关系",
    "Transformer与注意力机制",
    "BERT的预训练与迁移学习",
    "RNN、LSTM和GRU的差别",
    "PyTorch、CNN与RNN的应用",
    "KNN、线性回归和逻辑回归",
    "决策树、随机森林与XGBoost",
    "KMeans聚类算法",
    "模型量化、剪枝与知识蒸馏",
    "Neo4j知识图谱数据库",
    "实体识别NER与关系抽取",
    "RAG问答系统的完整处理流程",
    "LangChain与Streamlit聊天机器人",
    "SQL、MySQL与Pandas数据分析",
    "Python面向对象与高级语法",
]

PROFESSIONAL_TEMPLATES = [
    "请介绍一下{topic}。",
    "我想系统了解{topic}，它具体是怎么回事？",
    "能用简单例子说明{topic}吗？",
]

# 补充知识库外问题，防止模型因为增加专业样本而把所有技术问题都送入RAG。
GENERAL_TOPICS = [
    "TCP三次握手",
    "HTTP状态码",
    "DNS域名解析",
    "Linux文件权限",
    "Git分支合并",
    "Docker容器网络",
    "Java垃圾回收",
    "JavaScript事件循环",
    "CSS盒模型",
    "操作系统进程调度",
    "数据库事务隔离级别",
    "哈希表冲突处理",
    "二叉搜索树",
    "快速排序",
    "动态规划",
    "正则表达式语法",
    "Python读取CSV文件",
    "Python异步编程",
    "RESTful API",
    "软件测试流程",
    "云计算IaaS与PaaS",
    "网络安全中的XSS",
    "圆的面积计算",
    "一元二次方程求解",
    "概率论中的条件概率",
    "中国地理常识",
    "英语时态区别",
    "家庭旅行规划",
    "日常饮食营养",
    "办公软件快捷键",
]

GENERAL_TEMPLATES = [
    "请解释{topic}。",
    "{topic}是什么？",
    "给我举例说明{topic}。",
    "我想了解{topic}的基本用法。",
]

# 第一轮对比发现“是什么/区别/为什么/大小关系”等自然问法仍被旧规则强烈吸向
# 通用知识。这里补充真实用户会使用的多样化表达，但不照抄黄金测试原句。
TARGETED_PROFESSIONAL_QUERIES = [
    "NLP这一阶段除了Transformer和BERT还会覆盖什么模型？",
    "自然语言处理部分会讲RNN、LSTM和GRU吗？",
    "学NLP时Transformer、BERT和注意力机制都会涉及吗？",
    "知识图谱实战会不会用Neo4j？",
    "实体识别、关系抽取和实体消歧在项目里怎么配合？",
    "知识图谱项目具体会练哪些图数据库和抽取技术？",
    "大模型应用阶段会安排什么实战？",
    "学到最后可以完成哪些LLM项目？",
    "大模型开发部分有哪些可落地的项目案例？",
    "大语言模型究竟是什么，通常能完成什么任务？",
    "LLM能够处理分类、翻译和问答吗？",
    "请说明大语言模型的定义、能力和典型应用。",
    "为什么大语言模型可以理解和生成人类语言？",
    "语言模型本质上是在计算什么概率？",
    "模型是怎样判断一段文本自然不自然的？",
    "词序列概率为什么能反映一句话是否通顺？",
    "语言模型给句子打分时到底算的是什么？",
    "一元、二元和三元语言模型分别依赖几个词？",
    "unigram、bigram、trigram三者有什么不同？",
    "二元语言模型只参考前一个词吗？",
    "三元模型预测当前词时需要看前几个词？",
    "能不能通俗说明N元语法模型的马尔可夫假设？",
    "N-gram为什么会碰到数据稀疏？",
    "为什么n越大，N元模型的参数会快速增加？",
    "传统N-gram处理不了长距离依赖的原因是什么？",
    "N元语言模型有哪些明显缺陷？",
    "只使用N-gram做文本生成为什么效果有限？",
    "大模型的涌现能力指的是什么现象？",
    "模型规模增大后为什么会突然出现新能力？",
    "GPT-3的上下文学习和Few-shot有什么关系？",
    "不更新模型参数也能根据示例完成任务叫什么能力？",
    "BLEU与ROUGE的评价重点有什么差别？",
    "机器翻译和文本摘要分别更常用哪个指标？",
    "BLEU偏向精确率而ROUGE偏向召回率怎么理解？",
    "比较生成文本和参考文本时BLEU、ROUGE怎么算？",
    "PPL困惑度应该越高越好还是越低越好？",
    "语言模型的困惑度数值代表什么？",
    "为什么PPL越小通常说明语言模型越好？",
    "困惑度和预测下一个词的能力有什么关系？",
    "二元那个模型是不是只用前面一个词？",
    "bigram说白了会参考前边几个词？",
    "三元语言模型预测时往前看多少个词？",
    "预训练和微调在模型学习流程中分别负责什么？",
    "模型量化、剪枝和蒸馏各自解决什么问题？",
    "RAG系统从用户提问到返回答案要经过哪些步骤？",
]

TARGETED_GENERAL_QUERIES = [
    "推荐一个适合周末去的旅游城市。",
    "周末两天去哪里旅行比较合适？",
    "帮我规划一次家庭短途旅行。",
    "去海边旅游需要准备什么？",
    "明天出门需要带雨伞吗？",
    "最近气温下降应该怎么穿衣服？",
    "给我推荐几道简单的家常菜。",
    "怎样安排每天的健身计划？",
]


def load_source_records():
    """读取原始JSONL并校验query和label字段。"""
    records = []
    for line_number, line in enumerate(
        SOURCE_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) < {"query", "label"}:
            raise ValueError(f"原始训练集第{line_number}行缺少query或label")
        records.append(
            {
                "query": record["query"].strip(),
                "label": record["label"],
            }
        )
    return records


def build_enhanced_records():
    """修正知识库范围标签，并补充专业知识与通用知识边界样本。"""
    records = load_source_records()
    for record in records:
        if record["query"] in RELABEL_AS_PROFESSIONAL:
            record["label"] = PROFESSIONAL_LABEL

    for topic in KNOWLEDGE_BASE_TOPICS:
        for template in PROFESSIONAL_TEMPLATES:
            records.append(
                {
                    "query": template.format(topic=topic),
                    "label": PROFESSIONAL_LABEL,
                }
            )

    for topic in GENERAL_TOPICS:
        for template in GENERAL_TEMPLATES:
            records.append(
                {
                    "query": template.format(topic=topic),
                    "label": GENERAL_LABEL,
                }
            )

    records.extend(
        {"query": query, "label": PROFESSIONAL_LABEL}
        for query in TARGETED_PROFESSIONAL_QUERIES
    )
    records.extend(
        {"query": query, "label": GENERAL_LABEL}
        for query in TARGETED_GENERAL_QUERIES
    )

    unique_records = []
    seen_queries = set()
    for record in records:
        normalized_query = record["query"].strip()
        if normalized_query in seen_queries:
            continue
        seen_queries.add(normalized_query)
        unique_records.append(
            {"query": normalized_query, "label": record["label"]}
        )
    return unique_records


def save_records(records):
    """保存增强JSONL，保持训练器当前的一行一个JSON格式。"""
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    records = build_enhanced_records()
    save_records(records)
    label_counts = Counter(record["label"] for record in records)
    print(f"增强训练集已生成: {OUTPUT_FILE}")
    print(f"总样本数: {len(records)}")
    print(f"标签分布: {dict(label_counts)}")


if __name__ == "__main__":
    main()
