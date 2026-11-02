# AI课程多层检索智能问答系统

一个基于课程RAG基线持续审查和重构的个人学习项目。项目围绕AI课程资料构建两条已经独立跑通的检索链路：

1. MySQL权威FAQ → Redis Hash/Search同步 → BM25检索。
2. 文档父子分块 → BGE-M3稠密/神经稀疏向量 → Milvus混合召回 → 父块恢复 → CrossEncoder重排。

当前项目是本地脚本式工程原型，重点是数据流、模块职责、异常边界和真实组件验证，不是已经上线的生产系统。

## 为什么做这个项目

课程基线能够演示基本功能，但在继续阅读和运行时，我发现了以下工程问题：

- FAQ语料每次启动都在Python内存中重新建立BM25统计。
- 原始问题字符串作为精确缓存键，轻微改写就无法命中。
- MySQL、Redis与检索索引之间缺少清楚的权威源和重建关系。
- 正常未命中、低分候选与基础设施故障容易被混成同一种结果。
- 文档加载、切块、Embedding、Milvus字段和重排序职责耦合。

因此，我沿着实际查询和入库数据流逐步拆分职责，并保留课程基线与个人重构边界。

## 当前架构

### FAQ链路

```text
启动
MySQL权威FAQ
→ FAQSnapshot（文档 + SHA-256内容版本）
→ Redis Hash正式副本与Search索引

查询
用户问题
→ 中文分词、停用词过滤与查询转义
→ Redis Search OR召回与BM25排序
→ Retriever判断最低可靠分数
→ FAQ_HIT / MISS / INVALID
```

### 向量检索链路

```text
文档
→ 多格式加载
→ 父块/子块切分与稳定ID
→ BGE-M3生成稠密向量和神经稀疏向量
→ Milvus混合召回与WeightedRanker融合
→ 根据parent_id恢复、去重父块
→ CrossEncoder重排序
→ Top-M上下文
```

## 已完成

- 敏感信息、普通配置和稳定路径的分层管理。
- MySQL FAQ内容快照与SHA-256版本计算。
- Redis Hash正式副本、Search索引创建、全量重建和版本跳过。
- Redis Search中文查询构造、BM25候选召回及可靠命中判断。
- `FAQ_HIT`、`MISS`、`INVALID` 与基础设施异常的明确区分。
- 多格式文档加载以及父子分块。
- BGE-M3稠密/神经稀疏向量生成。
- Milvus混合召回、父块恢复去重和CrossEncoder重排。
- Embedding、Milvus、Reranker与业务Service的职责拆分和依赖注入。
- BERT意图分类的离线训练与在线推理解耦。

本地真实环境中曾验证：467条FAQ同步；3篇文档切分为30个父块、130个子块并写入Milvus。这些数字用于证明链路实际运行，不代表性能基准。

## 尚未完成

- FAQ未命中后自动进入Milvus并调用LLM的总编排。
- RAG生成答案缓存和人工审核闭环。
- 固定评测集、完整自动化测试与性能基准。
- Web/API、多用户、鉴权、限流和线上部署。

因此，本项目不宣称“完整端到端问答平台”或“生产级系统”。

## 目录结构

```text
.
├─ faq_system/                         # FAQ领域对象、MySQL/Redis客户端与检索Service
├─ rag_system_learning_version/
│  ├─ core/
│  │  ├─ vector_retrieval/             # Embedding、Milvus、Reranker和VectorService
│  │  └─ intent_classifier/            # BERT训练与在线分类
│  ├─ edu_document_loaders/             # 多格式文档加载
│  └─ edu_text_spliter/                 # 中文与Markdown切分器
├─ logger/                              # 统一日志入口（运行日志不提交）
├─ sample_data/                         # 自制、脱敏的最小示例数据
├─ docs/engineering-decisions.md        # 关键工程取舍
├─ config.py / config.ini / .env.example
├─ qa_system.py                         # 当前FAQ系统级编排
└─ main.py                              # FAQ命令行入口
```

## 环境要求

- Python 3.10+
- MySQL
- 包含Search能力的Redis（推荐Redis Stack）
- Milvus 2.5.x
- 可选CUDA GPU；没有GPU时模型可使用CPU，但速度会明显下降

模型权重、课程原始资料、真实FAQ数据和运行日志均未提交。请自行准备以下模型目录：

```text
rag_system_learning_version/models/
├─ bge-m3/
├─ bge-reranker-large/
├─ bert-base-chinese/
└─ bert_query_classifier/    # 训练后生成
```

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. 配置本地秘密

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
MYSQL_PASSWORD=your_mysql_password
REDIS_PASSWORD=your_redis_password
DASHSCOPE_API_KEY=your_api_key
```

`.env` 已被Git忽略。请勿提交真实密钥。

### 3. 初始化示例FAQ

在MySQL中执行：

```text
sample_data/faq_seed.sql
```

再根据本地环境调整 `config.ini` 中的MySQL、Redis和Milvus地址。

### 4. 运行FAQ命令行入口

```powershell
python main.py
```

### 5. 文档入库与查询

示例文档位于 `sample_data/ai_course_notes.md`。

```powershell
python -m rag_system_learning_version.core.vector_retrieval.ingest_documents
python -m rag_system_learning_version.core.vector_retrieval.query_documents
```

### 6. 意图分类训练

仓库仅提供极小的格式示例，不能代表真实训练集质量：

```powershell
python -m rag_system_learning_version.core.intent_classifier.train_classifier
```

## 关键工程取舍

- MySQL是正式FAQ唯一权威源，Redis Search是可以重建的查询副本。
- 正常无候选返回MISS；数据库、索引或数据合同错误继续抛出异常。
- 小数据阶段在Milvus子块记录中冗余父块正文，暂不额外引入DocStore。
- 入库时不加载Reranker；只有查询流程真正需要时才加载。
- 不为了展示技术数量强行增加组件，优先保持数据流和职责清楚。

更多说明见 [工程决策记录](docs/engineering-decisions.md)。

## 来源与公开边界

本仓库源于课程RAG基线，公开内容为个人学习过程中审查、重构和验证后的求职展示版本。课程原始数据、文档、模型权重、内部日志及未整理的教学基线均未包含。

当前仓库未附开放源代码许可证；如需复制、再发布或用于其他用途，请先联系作者并同时遵守相关第三方组件的许可证。

