CREATE DATABASE IF NOT EXISTS subjects_kg
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE subjects_kg;

CREATE TABLE IF NOT EXISTS jpkb (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(20),
    question VARCHAR(1000),
    answer VARCHAR(1000)
);

INSERT INTO jpkb (subject_name, question, answer) VALUES
('AI基础', '什么是RAG？', 'RAG先检索外部知识，再结合检索结果生成回答。'),
('AI基础', '为什么需要重排序？', '重排序模型会对召回候选重新评分，帮助筛选更相关的上下文。'),
('AI基础', '父子分块有什么作用？', '子块用于精确检索，命中后恢复父块以保留较完整的上下文。');

