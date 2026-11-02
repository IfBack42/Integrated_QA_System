# 该脚本用于: MySQL数据库的基础操作.

# 导入MySQL操作和数据处理依赖库.
import hashlib
import json
import re
import sys
from pathlib import Path

import pymysql              # 用于构建MySQL数据库连接
import pandas as pd         # 用于数据处理

faq_system_root = Path(__file__).resolve().parents[2] # 向上找两级目录加入到解释器路径
if str(faq_system_root) not in sys.path:
    sys.path.insert(0, str(faq_system_root))

from config import config
from faq_system.domain import FAQDocument, FAQSnapshot
from logger import get_logger

logger = get_logger(__name__)


# 定义MySQL客户端类 -> 封装数据库连接, 表操作, 数据增查, 连接关闭等功能.
class MySQLClient:
    # 初始化方法 -> 建立数据库连接, 创建游标, 处理连接异常.
    def __init__(self, settings=None):
        self.settings = settings or config
        self.faq_table_name = self.settings.MYSQL_FAQ_TABLE_NAME
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.faq_table_name):
            raise ValueError(
                "MySQL FAQ表名只能包含英文字母、数字和下划线，且不能以数字开头"
            )
        # MySQL参数占位符不能代替表名，因此先白名单校验，再安全拼接标识符。
        self._faq_table_sql = f"`{self.faq_table_name}`"
        try:
            # 建立MySQL数据库连接 -> 从config中读取编码.
            self.connection = pymysql.connect(
                host = self.settings.MYSQL_HOST,             # MySQL服务器主机地址
                user = self.settings.MYSQL_USER,             # MySQL登录用户名
                password = self.settings.MYSQL_PASSWORD,     # MySQL登录密码
                database = self.settings.MYSQL_DATABASE      # 数据库名称
            )
            # 创建游标对象 -> 用于执行SQL语句并返回结果.
            self.cursor = self.connection.cursor()

            # 记录info日志 -> 确认MySQL连接成功.
            logger.info('MySQL 连接成功')
        except pymysql.MySQLError as e:
            # 记录error日志 -> 输出错误信息.
            logger.error(f'MySQL 连接失败: {e}')
            raise       # 抛出异常, 不屏蔽错误, 让调用方感知连接失败.


    # 创建数据表 -> 表名从config读取.
    def create_table(self):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self._faq_table_sql} (
            id INT AUTO_INCREMENT PRIMARY KEY,      # 自增主键
            subject_name VARCHAR(20),               # 学科名称
            question VARCHAR(1000),                 # 问题内容
            answer VARCHAR(1000))                   # 答案内容
        """

        try:
             # 2. 执行SQL语句 -> 创建表.
             self.cursor.execute(create_table_query)
             # 3. 提交事务 -> 确认表创建成功.  MySQL默认事务需要手动提交, 确保表创建成功.
             self.connection.commit() # 建表时多余
             # 4. 输出信息日志 -> 确认表创建成功.
             logger.info('表创建成功')
        except pymysql.MySQLError as e:
            # 5. 输出错误信息.
            self.connection.rollback() # 建表无法回滚
            logger.error(f'创建表失败: {e}')
            raise


    # 从csv文件批量插入数据到权威FAQ表中.
    def insert_data(self, csv_path):
        try:
            # 1. 读取csv文件 -> 获取数据.
            data = pd.read_csv(csv_path)
            # 扩展: 打印下, 查看一下数据.
            # print(data.head())

            # 2. 循环遍历csv每一行数据 -> 执行插入操作.
            for _, row in data.iterrows(): # iterrows:pandas行遍历
                # 2.1 定义插入SQL语句, 使用 %s占位符, 避免SQL注入攻击问题.
                insert_query = f"""
                    insert into {self._faq_table_sql}(subject_name, question, answer)
                    values(%s, %s, %s)
                    """
                # 2.2 执行SQL语句 -> 将csv行数据(学科名称, 问题, 答案)传入占位符.
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))

            # 3. 提交事务: 确保所有插入操作批量生效.
            self.connection.commit()
            # 4. 输出信息日志 -> 确认数据插入成功.
            logger.info('MySQL数据插入成功')

        except Exception as e:
            # 5.捕获异常信息, 回滚事务, 抛出异常信息.
            # 5.1 输出错误信息.
            logger.error(f'数据插入失败: {e}')
            # 5.2 事务回滚: 把数据库连接恢复到事务开始前的状态 -> 类似于: Linux的快照.
            self.connection.rollback()
            # 5.3 抛出异常, 不屏蔽错误, 让调用方感知数据插入失败.
            raise


    # 一次读取全部权威FAQ -> 用于同步Redis Search正式副本.
    def fetch_all_faqs(self):
        """按统一FAQDocument结构返回MySQL中的全部权威问答。"""
        query = f"""
            SELECT id, subject_name, question, answer
            FROM {self._faq_table_sql}
            ORDER BY id
        """
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            documents = [
                FAQDocument(
                    qa_id=str(row[0]),
                    subject_name=str(row[1] or ""),
                    question=str(row[2] or ""),
                    answer=str(row[3] or ""),
                )
                for row in rows
            ]
            logger.debug("获取全部权威FAQ成功，共%d条", len(documents))
            return documents
        except (pymysql.MySQLError, ValueError) as e:
            logger.error(f'获取全部权威FAQ失败: {e}')
            raise


    # 读取权威FAQ快照 -> 文档和内容版本一起返回.
    def fetch_faq_snapshot(self):
        """读取完整FAQ，并根据有序文档内容生成稳定的SHA-256版本。"""
        documents = tuple(self.fetch_all_faqs())
        digest = hashlib.sha256()
        for document in documents:
            normalized_document = json.dumps(
                document.to_mapping(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update(normalized_document.encode("utf-8"))
            digest.update(b"\n")

        snapshot = FAQSnapshot(
            version=digest.hexdigest(),
            documents=documents,
        )
        logger.debug(
            "权威FAQ快照生成成功，版本=%s，文档数=%d",
            snapshot.version,
            snapshot.document_count,
        )
        return snapshot


    # 关闭MySQL连接 -> 释放资源, 避免连接泄漏.
    def close(self):
        # 关闭数据库连接
        try:
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            logger.error(f"关闭连接失败: {e}")


def main():
    """直接运行本文件时，只读验证完整FAQ快照加载功能。"""
    mysql_client = None
    try:
        mysql_client = MySQLClient()
        snapshot = mysql_client.fetch_faq_snapshot()
        print(f"权威FAQ内容版本: {snapshot.version}")
        print(f"读取到权威FAQ数量: {snapshot.document_count}")
        if snapshot.documents:
            print("第一条FAQ:")
            print(snapshot.documents[0])
    finally:
        if mysql_client is not None:
            mysql_client.close()


if __name__ == '__main__':
    main()
