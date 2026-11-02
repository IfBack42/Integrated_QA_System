# 工程决策记录

本文只记录当前代码已经体现的主要取舍，不把规划功能写成现有能力。

## 1. 配置与敏感信息分层

- `.env`：数据库密码、API Key 等本地秘密，不提交仓库。
- `config.ini`：主机、端口、索引名、分块大小等普通参数。
- `config.py`：统一读取、类型转换和必填项检查。

业务模块只依赖 `Config`，不分别解析环境变量和INI文件。

## 2. MySQL是FAQ权威源，Redis Search是可重建副本

```text
MySQL权威FAQ
→ 生成内容版本快照
→ 同步Redis Hash文档
→ Redis Search维护全文索引并执行BM25查询
```

Redis丢失后可以从MySQL重建。正常无候选属于业务MISS；连接或索引故障属于系统异常，两者不混用。

## 3. 用显式领域对象传递结果

`FAQDocument`、`FAQSnapshot`、`RetrievalCandidate`、`RetrievalMatch` 和 `FAQResult` 分别表达不同阶段的数据。`FAQStatus`区分命中、未命中和无效输入，避免使用含义不清的布尔元组。

## 4. 父子分块与Milvus存储取舍

文档先切父块，再切更小的子块。子块负责检索定位，命中后根据 `parent_id` 恢复并去重父块。

当前数据规模较小，因此在Milvus子块记录中冗余父块正文，用少量存储空间换取简单的查询链路；尚未额外引入DocStore。

## 5. 模型、存储与业务编排解耦

```text
EmbeddingModel：字符串 → 稠密/稀疏向量
MilvusStorage：向量与字段 → 混合召回结果
RerankerModel：query与候选正文 → 相关性分数
VectorService：对齐文档、恢复父块、重排并截取结果
```

工厂负责创建和装配真实组件。入库流程不加载Reranker，查询流程才加载，避免无效资源消耗。

## 6. 当前边界

FAQ检索链路和Milvus向量检索链路已经分别跑通，但 `FAQ MISS → Milvus → LLM` 的系统级总编排尚未完成。固定评测集、自动化测试、Web/API和线上部署也仍是后续工作。

