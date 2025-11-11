import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel


# # BM25Ingestor：BM25索引构建与保存工具
# class BM25Ingestor:
#     def __init__(self):
#         pass

#     def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
#         """从文本块列表创建BM25索引"""
#         tokenized_chunks = [chunk.split() for chunk in chunks]
#         return BM25Okapi(tokenized_chunks)
    
#     def process_reports(self, all_reports_dir: Path, output_dir: Path):
#         """
#         批量处理所有报告，生成并保存BM25索引。
#         参数：
#             all_reports_dir (Path): 存放JSON报告的目录
#             output_dir (Path): 保存BM25索引的目录
#         """
#         output_dir.mkdir(parents=True, exist_ok=True)
#         all_report_paths = list(all_reports_dir.glob("*.json"))

#         for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
#             # 加载报告
#             with open(report_path, 'r', encoding='utf-8') as f:
#                 report_data = json.load(f)
                
#             # 提取文本块并创建BM25索引
#             text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
#             bm25_index = self.create_bm25_index(text_chunks)
            
#             # 保存BM25索引，文件名用sha1_name
#             sha1_name = report_data["metainfo"]["sha1_name"]
#             output_file = output_dir / f"{sha1_name}.pkl"
#             with open(output_file, 'wb') as f:
#                 pickle.dump(bm25_index, f)
                
#         print(f"Processed {len(all_report_paths)} reports")


# VectorDBIngestorBGE：使用 BAAI/bge-m3 模型的向量库构建工具（本地模型，无需API）
class VectorDBIngestorBGE:
    def __init__(self, model_name: str = 'BAAI/bge-m3', use_fp16: bool = True):
        """
        初始化 BGE-M3 模型
        参数：
            model_name: 模型名称，默认 'BAAI/bge-m3'
            use_fp16: 是否使用 fp16 加速（轻微性能损失但速度更快）
        """
        print(f"Loading BGE-M3 model: {model_name}")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        print("BGE-M3 model loaded successfully")

    def _get_embeddings(self, text: Union[str, List[str]], batch_size: int = 12, max_length: int = 8192) -> np.ndarray:
        """
        获取文本的嵌入向量
        参数：
            text: 单个文本或文本列表
            batch_size: 批处理大小
            max_length: 最大文本长度
        返回：
            embeddings: numpy 数组形式的嵌入向量
        """
        if isinstance(text, str):
            text = [text]
        
        # 过滤空字符串
        text = [t for t in text if t.strip()]
        if not text:
            raise ValueError("所有待嵌入文本均为空字符串！")
        
        # 使用 BGE-M3 进行编码，获取 dense vectors
        embeddings = self.model.encode(
            text,
            batch_size=batch_size,
            max_length=max_length
        )['dense_vecs']
        
        return embeddings

    def _create_vector_db(self, embeddings: np.ndarray):
        """
        用 faiss 构建向量库，采用内积（余弦距离）
        参数：
            embeddings: numpy 数组形式的嵌入向量
        返回：
            index: faiss 索引
        """
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict, batch_size: int = 12, max_length: int = 8192):
        """
        针对单份报告，提取文本块并生成向量库
        参数：
            report: 报告数据字典
            batch_size: 批处理大小
            max_length: 最大文本长度
        返回：
            index: faiss 索引
        """
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks, batch_size=batch_size, max_length=max_length)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path, batch_size: int = 12, max_length: int = 8192):
        """
        批量处理所有报告，生成并保存 faiss 向量库
        参数：
            all_reports_dir: 存放 JSON 报告的目录
            output_dir: 保存 faiss 向量库的目录
            batch_size: 批处理大小
            max_length: 最大文本长度
        """
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports with BGE-M3"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            
            index = self._process_report(report_data, batch_size=batch_size, max_length=max_length)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports with BGE-M3")