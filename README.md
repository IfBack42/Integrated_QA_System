# 多层检索智能问答系统

一个面向课程知识库的本地智能问答工程：用 MySQL 保存权威 FAQ，以 Redis Search 承担 FAQ 全文检索与临时答案缓存；FAQ 未可靠命中时，再进入 BERT 意图识别、Milvus 混合检索、CrossEncoder 重排和 LLM 生成链路。

项目源于课程基线，但重点不只是“把组件跑起来”，而是沿真实数据流重构权威数据源、缓存边界、检索职责、异常语义和离线评估。当前版本已经跑通 `FAQ → RAG → 临时缓存` 的完整命令行链路，是可复现的本地工程原型，不宣称为已上线的生产系统。

## 核心架构

```text
启动同步
MySQL 权威 FAQ
→ FAQSnapshot（文档 + SHA-256 内容版本）
→ Redis Hash 正式副本
→ Redis Search 中文全文索引

在线查询
用户问题
→ Redis Search BM25 FAQ 检索
├─ 可靠命中：直接返回权威答案
└─ 未命中
   → Redis 临时 RAG 答案缓存
   ├─ 命中：返回已生成答案
   └─ 未命中
      → BERT 意图识别
      ├─ 通用知识：LLM 直接回答
      └─ 专业咨询
         → LLM 选择检索策略
         → BGE-M3 稠密/神经稀疏编码
         → Milvus 混合召回 + WeightedRanker
         → parent_id 恢复并去重父块
         → CrossEncoder 重排
         → LLM 基于上下文生成答案
         → 写入带 TTL 的临时缓存
```

### 组件职责

| 组件 | 主要职责 |
|---|---|
| MySQL | 保存经过审核的权威 FAQ |
| Redis Hash / Search | 保存可重建的正式 FAQ 副本并执行 BM25 检索 |
| Redis 临时缓存 | 缓存完全相同且不带会话上下文的昂贵 RAG 请求 |
| BERT | 判断通用知识与专业咨询，决定是否进入知识库检索 |
| BGE-M3 | 同时生成稠密向量与神经稀疏向量 |
| Milvus | 执行两路向量召回与加权融合 |
| CrossEncoder | 对恢复后的父块进行更细粒度重排 |
| LLM | 选择检索策略，并依据检索上下文生成最终答案 |

## 已完成能力

- MySQL FAQ 内容快照与 SHA-256 版本计算。
- Redis 正式 FAQ Hash 副本、Search 索引创建、版本一致时跳过刷新。
- 中文分词、停用词过滤、OR 查询构造和 Redis Search BM25 可靠命中判断。
- `INVALID`、业务未命中与基础设施异常的明确区分。
- FAQ 未命中后自动进入 RAG，并缓存可复用的生成答案。
- TXT、Markdown、PDF、DOCX、PPTX 与图片 OCR 加载接口。
- 父子分块、稳定块 ID、BGE-M3 稠密/神经稀疏编码。
- Milvus 混合召回、父块恢复去重和 CrossEncoder 重排。
- 直接检索、HyDE、子查询和 Step-back 检索策略。
- BERT 意图分类训练、独立预测与新旧模型评估脚本。
- 真实系统采集、RAGAS 评估和汇总报告生成流程。

## 离线评估

当前报告基于 24 条小规模黄金测试集，其中 20 条可回答、4 条资料外问题。评估调用了真实业务入口；RAGAS 只统计实际进入专业检索且获得上下文的 19 条样本。

| 指标 | 结果 |
|---|---:|
| 系统执行成功 | 24 / 24 |
| 资料外问题正确拒答 | 4 / 4 |
| 可回答问题进入专业 RAG | 19 / 20（95%） |
| Faithfulness | 0.9051 |
| Answer Relevancy | 0.7948 |
| Context Precision | 1.0000 |
| Context Recall | 0.9474 |
| 本地单用户平均查询延迟 | 5.988 秒 |
| 本地单用户 P95 查询延迟 | 10.322 秒 |

BERT 意图分类在独立保留集上的知识库问题召回率由 10% 提升至 90%，整体准确率由 55% 提升至 95%，通用问题准确率保持 100%。

详细口径和低分样本分析见：

- [`rag_system/rag_assessment/results/final_evaluation_report.md`](rag_system/rag_assessment/results/final_evaluation_report.md)
- [`rag_system/rag_assessment/results/intent_retraining_report.md`](rag_system/rag_assessment/results/intent_retraining_report.md)

这些结果是当前本地小样本基线，不能直接外推为线上大规模效果。

## 项目结构

```text
Integrated_QA_System/
├── main.py                         # 整套系统命令行入口
├── qa_system.py                    # FAQ、缓存与 RAG 总编排
├── domain.py                       # 项目级统一返回对象
├── config.py / config.ini          # 配置读取与普通参数
├── faq_system/
│   ├── mysql_client/               # 权威 FAQ 读取与快照生成
│   ├── redis_client/               # 正式副本、Search 与临时缓存
│   ├── retrieval/                  # Redis Search Retriever
│   └── faq_service.py              # FAQ 业务编排
├── rag_system/
│   ├── core/
│   │   ├── intent_classifier/      # BERT 推理与训练组件
│   │   ├── retrieval_strategies/   # 检索策略及其提示词
│   │   ├── vector_retrieval/       # 文档处理、模型、Milvus 与检索服务
│   │   ├── llm/                    # LLM Client 与提示词加载
│   │   ├── retrieval_service.py    # 策略选择与检索编排
│   │   └── rag_service.py          # RAG 总流程
│   ├── tools/                      # 入库、训练、查询与模型评估入口
│   └── rag_assessment/
│       ├── scripts/                # 结果采集、RAGAS评分与报告生成
│       ├── data/                   # 黄金测试集与RAGAS运行输入
│       └── results/                # 逐条结果、指标汇总与分析报告
├── docs/engineering-decisions.md   # 主要工程取舍
└── sample_data/                    # 不含隐私的最小示例数据
```

## 本地运行

### 1. 环境要求

- Python 3.10
- MySQL 8.x
- Redis Stack（必须包含 Redis Search，普通 Redis 不够）
- Milvus 2.5.x
- 可访问 OpenAI 兼容 Chat Completions API 的模型服务
- 推荐 NVIDIA GPU；CPU 可运行，但模型加载和查询更慢

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要与本项目验证环境完全一致，请根据本机 CUDA 版本单独安装匹配的 PyTorch，不建议盲目复制其他机器的 CUDA Wheel。

### 2. 配置秘密

复制 `.env.example` 为 `.env`，只在本地填写：

```dotenv
MYSQL_PASSWORD=your_mysql_password
REDIS_PASSWORD=your_redis_password
LLM_API_KEY=your_api_key
```

主机、端口、集合名、检索数量和阈值等普通配置在 `config.ini`。`.env` 已被 Git 忽略。

### 3. 准备本地模型和数据

大模型权重、课程原始资料和训练集不提交 GitHub。默认目录约定如下：

```text
rag_system/models/
├── bert-base-chinese/
├── bert_query_classifier/
├── bge-m3/
└── bge-reranker-large/

rag_system/data/ai_data/
└── 课程资料...
```

可以使用自己的兼容模型路径实例化组件；如果使用默认入口，请按上述目录准备模型。仓库中的 `sample_data/` 仅用于理解数据格式，不等同于完整课程数据集。

### 4. 初始化数据

1. 在 MySQL 中创建并导入权威 FAQ，可参考 `sample_data/faq_seed.sql`。
2. 启动 Redis Stack 和 Milvus。
3. 将课程资料放入 `rag_system/data/ai_data/`。
4. 执行文档入库：

```bash
python rag_system/tools/ingest_documents.py
```

如需训练意图模型，先准备训练数据，再运行：

```bash
python rag_system/tools/train_classifier.py
```

### 5. 启动问答系统

```bash
python main.py
```

也可以分别验证组件：

```bash
python rag_system/tools/query_documents.py
python rag_system/tools/query_classifier.py
python rag_system/main.py
```

## 评测复现入口

```text
rag_system/rag_assessment/scripts/collect_system_results.py
→ 通过真实系统采集问题、答案、上下文和耗时

rag_system/rag_assessment/scripts/rag_as.py
→ 对采集结果执行 RAGAS 评估

rag_system/rag_assessment/scripts/generate_final_report.py
→ 汇总路由、延迟、拒答和 RAGAS 指标
```

原始测试问题、课程上下文及逐条输出可能包含课程资料，因此未随公开仓库上传；仓库保留评测代码、汇总 JSON 和 Markdown 报告。

## 主要工程取舍

- MySQL 是正式 FAQ 的权威源；Redis 正式数据随时可以由 MySQL 重建。
- FAQ 相似检索统一交给 Redis Search，而不是用原始 query 做无效的精确 FAQ 匹配。
- 临时 RAG 缓存仍使用精确请求指纹，因为它的目标是避免完全相同的昂贵调用，不承担语义检索。
- 子块用于召回，父块用于生成；当前在 Milvus 子块记录中冗余父块正文，以简化本地原型查询链路。
- 正常未命中返回业务结果；连接故障、索引损坏和返回结构异常直接抛出，避免上层把系统故障误判为“没有答案”。
- 下层组件处理各自第三方 API 格式，上层 Service 只保留清楚的数据流与业务判断。

更多说明见：

- [`docs/engineering-decisions.md`](docs/engineering-decisions.md)：当前代码已经体现的工程取舍。
- [`docs/项目收尾与后续备忘.md`](docs/项目收尾与后续备忘.md)：封版状态、未实现边界和未来备忘。

## 当前限制

- 黄金测试集和知识库规模较小，RAGAS 指标仅作为当前版本基线。
- 仍有少量教材术语问法被 BERT 高置信度误路由。
- 相似章节中的重复标题会干扰个别长文档召回，不能只靠增加候选数解决。
- Milvus 中旧批次文档的来源路径需要重新入库后更新。
- 当前是本地命令行原型，尚未实现 Web/API、多用户会话、鉴权、并发压测和线上监控。

## 安全说明

仓库不提交 `.env`、数据库密码、API Key、模型权重、课程原始资料、训练中间产物和包含检索正文的原始评测输出。公开的示例数据均为最小演示用途。
