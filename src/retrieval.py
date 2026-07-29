import json
import logging
import re
from typing import List, Tuple, Dict, Union, Optional
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.rerank_config import (
    RERANK_MODEL_DASHSCOPE,
    RERANK_MODEL_LABELS,
)
from src.reranking import rerank_documents
from src.chunk_metadata import result_source_from_chunk
import hashlib
import pandas as pd
import time

_log = logging.getLogger(__name__)

_PAGE_MARKER_RE = re.compile(
    r"(?:#\s*Page\s+(\d+)|<!--\s*Page\s+(\d+)\s*-->)",
    re.IGNORECASE,
)


def resolve_chunk_page(chunk: dict, text: str | None = None) -> Optional[int]:
    """从 chunk 元数据或文本中解析页码；无法解析时返回 None。"""
    for key in ("page", "page_number", "page_num"):
        raw = chunk.get(key)
        if raw is None:
            continue
        try:
            page = int(raw)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page

    content = text if text is not None else chunk.get("text", "")
    if content:
        match = _PAGE_MARKER_RE.search(content)
        if match:
            page = int(match.group(1) or match.group(2))
            if page > 0:
                return page
    return None


def format_page_display(page) -> str:
    """将页码格式化为展示字符串，无效页码显示为 '-'。"""
    if page is None:
        return "-"
    try:
        page_int = int(page)
    except (TypeError, ValueError):
        return "-"
    return str(page_int) if page_int > 0 else "-"


# --- 方案 B：检索质量增强参数 ---
DEFAULT_LEXICAL_WEIGHT = 0.30
DEFAULT_MIN_VECTOR_SCORE = 0.12
DEFAULT_MIN_COMBINED_SCORE = 0.25
DEFAULT_MIN_RERANK_COMBINED = 0.22
DEFAULT_CANDIDATE_MULTIPLIER = 5
DEFAULT_MIN_CANDIDATE_POOL = 15


def extract_query_terms(query: str) -> set[str]:
    """从问题中提取中英文关键词/短语，用于词面匹配加权。"""
    terms: set[str] = set()
    for segment in re.findall(r"[\u4e00-\u9fff]+", query):
        if len(segment) >= 2:
            terms.add(segment)
        for n in (2, 3, 4):
            for i in range(len(segment) - n + 1):
                terms.add(segment[i : i + n])
    terms.update(w.lower() for w in re.findall(r"[a-zA-Z0-9]{2,}", query))
    return terms


def lexical_overlap_score(query: str, text: str) -> float:
    """计算问题与 chunk 的词面重合度（0~1），弥补纯向量对关键词不敏感的问题。"""
    terms = extract_query_terms(query)
    if not terms:
        return 0.0
    text_lower = _strip_page_markers(text).lower()
    hits = sum(1 for term in terms if term in text_lower)
    return min(1.0, hits / max(4, len(terms) * 0.35))


def _strip_page_markers(text: str) -> str:
    text = re.sub(r"<!--\s*Page\s+\d+\s*-->", "", text, flags=re.IGNORECASE)
    text = re.sub(r"#\s*Page\s+\d+", "", text, flags=re.IGNORECASE)
    return text


def enrich_candidate_scores(candidates: list[dict], query: str) -> list[dict]:
    """为候选 chunk 融合向量分与词面分（向量分在候选池内 min-max 归一化）。"""
    if not candidates:
        return candidates

    raw_scores = [float(item.get("distance", 0.0)) for item in candidates]
    min_score, max_score = min(raw_scores), max(raw_scores)
    span = max_score - min_score

    for candidate in candidates:
        raw = float(candidate.get("distance", 0.0))
        if span > 1e-9:
            norm_vector = (raw - min_score) / span
        else:
            norm_vector = 1.0
        lexical_score = lexical_overlap_score(query, candidate.get("text", ""))
        combined = (
            (1.0 - DEFAULT_LEXICAL_WEIGHT) * norm_vector
            + DEFAULT_LEXICAL_WEIGHT * lexical_score
        )
        candidate["vector_score"] = round(raw, 4)
        candidate["vector_score_norm"] = round(norm_vector, 4)
        candidate["lexical_score"] = round(lexical_score, 4)
        candidate["combined_score"] = round(combined, 4)
    return candidates


def apply_relevance_filter(
    candidates: list[dict],
    top_n: int,
    min_vector_score: float = 0.12,
    min_combined_score: float = 0.25,
) -> list[dict]:
    """过滤与问题明显不相关的 chunk；不足时回退到最高分候选。"""
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda item: item.get("combined_score", item.get("distance", 0.0)),
        reverse=True,
    )
    filtered = [
        item
        for item in ranked
        if item.get("vector_score_norm", 0.0) >= min_vector_score
        or item.get("combined_score", 0.0) >= min_combined_score
        or item.get("lexical_score", 0.0) >= 0.28
    ]

    if not filtered:
        fallback = ranked[: max(1, min(top_n, len(ranked)))]
        for item in fallback:
            item["low_relevance"] = True
        _log.warning(
            "所有候选 chunk 低于相关性阈值，回退保留 top-%d（已标记 low_relevance）",
            len(fallback),
        )
        return fallback

    return filtered[:top_n]


class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        # 初始化BM25检索器，指定BM25索引和文档目录
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        # 按公司名检索相关文本块，返回BM25分数最高的top_n个块
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")
        # 加载对应的BM25索引，文件名用 sha1
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # 获取文档内容和BM25索引
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"].get("pages", [])
        
        # 计算BM25分数
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            chunk_page = resolve_chunk_page(chunk)
            parent_page = None
            if pages and chunk_page is not None:
                parent_page = next(
                    (page for page in pages if page.get("page") == chunk_page),
                    None,
                )

            if return_parent_pages and parent_page:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"],
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk_page,
                    "text": chunk["text"],
                }
                retrieval_results.append(result)
        
        return retrieval_results



class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_provider: str = "dashscope"):
        # 初始化向量检索器，加载所有向量库和文档
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.last_vector_db_elapsed = 0.0
        self.last_embedding_elapsed = 0.0
        self.all_dbs = self._load_dbs()
        # 默认使用 dashscope 作为 embedding provider
        self.embedding_provider = embedding_provider.lower()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        # 根据 embedding_provider 初始化对应的 LLM 客户端
        load_dotenv()
        if self.embedding_provider == "openai":
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=None,
                max_retries=2
            )
            return llm
        elif self.embedding_provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return None  # dashscope 不需要 client 对象
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _get_embedding(self, text: str):
        # 根据 embedding_provider 获取文本的向量表示
        if self.embedding_provider == "openai":
            embedding = self.llm.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return embedding.data[0].embedding
        elif self.embedding_provider == "dashscope":
            import dashscope
            rsp = dashscope.TextEmbedding.call(
                model="text-embedding-v1",
                input=[text]
            )
            # 兼容 dashscope 返回格式，不能用 resp.output，需用 resp['output']
            if 'output' in rsp and 'embeddings' in rsp['output']:
                # 多条输入（本处只有一条）
                emb = rsp['output']['embeddings'][0]
                if emb['embedding'] is None or len(emb['embedding']) == 0:
                    raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}")
                return emb['embedding']
            elif 'output' in rsp and 'embedding' in rsp['output']:
                # 兼容单条输入格式
                if rsp['output']['embedding'] is None or len(rsp['output']['embedding']) == 0:
                    raise RuntimeError("DashScope返回的embedding为空")
                return rsp['output']['embedding']
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    @staticmethod
    def set_up_llm():
        # 静态方法，初始化OpenAI LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    def _load_dbs(self):
        # 加载所有向量库和对应文档，建立映射
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        for document_path in all_documents_paths:
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            # 用 metainfo['sha1'] 拼接 faiss 文件名
            sha1 = document.get('metainfo', {}).get('sha1', None)
            if not sha1:
                _log.warning(f"No sha1 found in metainfo for document {document_path.name}")
                continue
            faiss_path = self.vector_db_dir / f"{sha1}.faiss"
            if not faiss_path.exists():
                _log.warning(f"No matching vector DB found for document {document_path.name} (sha1={sha1})")
                continue
            try:
                vector_db = faiss.read_index(str(faiss_path))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
            report = {
                "name": sha1,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        # 计算两个字符串的余弦相似度（通过嵌入）
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    @staticmethod
    def _find_reports_for_company(all_dbs: list, company_name: str) -> list:
        """匹配 company_name 字段或文件名中包含公司名的所有文档。"""
        matched = []
        for report in all_dbs:
            metainfo = report.get("document", {}).get("metainfo", {})
            if metainfo.get("company_name") == company_name:
                matched.append(report)
            elif company_name in metainfo.get("file_name", ""):
                matched.append(report)
        return matched

    @staticmethod
    def _metainfo_display_name(metainfo: dict) -> str:
        file_name = metainfo.get("file_name", "")
        if file_name.lower().endswith(".md"):
            return file_name[:-3]
        return file_name

    def _build_chunk_result(
        self,
        distance: float,
        chunk: dict,
        metainfo: dict,
        pages: list,
        seen_pages: set,
        return_parent_pages: bool,
    ) -> dict | None:
        chunk_page = resolve_chunk_page(chunk)
        parent_page = None
        if pages and chunk_page is not None:
            parent_page = next(
                (page for page in pages if page.get("page") == chunk_page),
                None,
            )

        if return_parent_pages and parent_page:
            if parent_page["page"] in seen_pages:
                return None
            seen_pages.add(parent_page["page"])
            text = parent_page["text"]
            page = parent_page["page"]
        else:
            text = chunk["text"]
            page = chunk_page

        source_fields = result_source_from_chunk(chunk, metainfo)
        return {
            "distance": distance,
            "page": page,
            "text": text,
            "chunk_id": chunk.get("id"),
            **source_fields,
        }

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = None,
        top_n: int = 3,
        return_parent_pages: bool = False,
        candidate_size: int | None = None,
    ) -> List[Tuple[str, float]]:
        # 跨所有相关文档检索：扩大候选池 → 词面+向量融合 → 相关性过滤 → top_n
        matching_reports = self._find_reports_for_company(self.all_dbs, company_name)
        if not matching_reports:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        pool_size = candidate_size or max(
            top_n * DEFAULT_CANDIDATE_MULTIPLIER, DEFAULT_MIN_CANDIDATE_POOL
        )
        per_doc_k = max(5, pool_size // max(len(matching_reports), 1))

        t_embed = time.time()
        embedding = self._get_embedding(query)
        self.last_embedding_elapsed = time.time() - t_embed
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)

        candidates = []
        t_faiss_total = 0.0
        for report in matching_reports:
            document = report["document"]
            metainfo = document.get("metainfo", {})
            vector_db = report["vector_db"]
            chunks = document["content"]["chunks"]
            pages = document["content"].get("pages", [])
            if not chunks:
                continue

            k = min(per_doc_k, len(chunks))
            t_faiss = time.time()
            distances, indices = vector_db.search(x=embedding_array, k=k)
            t_faiss_total += time.time() - t_faiss
            seen_pages: set = set()
            for distance, index in zip(distances[0], indices[0]):
                distance = round(float(distance), 4)
                chunk = chunks[index]
                result = self._build_chunk_result(
                    distance,
                    chunk,
                    metainfo,
                    pages,
                    seen_pages,
                    return_parent_pages,
                )
                if result:
                    candidates.append(result)

        self.last_vector_db_elapsed = t_faiss_total
        enrich_candidate_scores(candidates, query)
        candidates.sort(
            key=lambda item: item.get("combined_score", item.get("distance", 0.0)),
            reverse=True,
        )
        retrieval_results = apply_relevance_filter(candidates, pool_size)[:top_n]
        for i, result in enumerate(retrieval_results, 1):
            result["chunk_index"] = i
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        # 检索公司所有文本块，返回全部内容
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


class HybridRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        rerank_model: str = RERANK_MODEL_DASHSCOPE,
        rerank_api_key: str | None = None,
    ):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.rerank_model = rerank_model or RERANK_MODEL_DASHSCOPE
        self.rerank_api_key = rerank_api_key
        self.last_vector_db_elapsed = 0.0
        self.last_embedding_elapsed = 0.0
        self.last_rerank_elapsed = 0.0
        self.last_rerank_fallback = False
        self.last_rerank_fallback_reason: str | None = None
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        使用混合检索方法进行检索和重排。
        
        参数：
            company_name: 需要检索的公司名称
            query: 检索查询语句
            llm_reranking_sample_size: 首轮向量检索返回的候选数量
            documents_batch_size: 每次送入重排器的文档数（Jina 单次 API 调用，此参数仅保留兼容）
            top_n: 最终返回的重排结果数量
            llm_weight: Jina 重排分数权重（0-1），其余为向量分数权重
            return_parent_pages: 是否返回完整页面（而非分块）
        
        返回：
            经过重排的文档字典列表，包含分数
        """
        t0 = time.time()
        # 首先用向量检索器获取初步结果
        print("[计时] [HybridRetriever] 开始向量检索 ...")
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            candidate_size=max(
                llm_reranking_sample_size * DEFAULT_CANDIDATE_MULTIPLIER,
                DEFAULT_MIN_CANDIDATE_POOL,
            ),
            return_parent_pages=return_parent_pages,
        )
        t1 = time.time()
        print(f"[计时] [HybridRetriever] 向量检索耗时: {t1-t0:.2f} 秒")
        self.last_vector_db_elapsed = getattr(
            self.vector_retriever, "last_vector_db_elapsed", t1 - t0
        )
        self.last_embedding_elapsed = getattr(
            self.vector_retriever, "last_embedding_elapsed", 0.0
        )
        # 使用可配置 Reranker 对结果进行重排（3s 超时熔断）
        model_label = RERANK_MODEL_LABELS.get(self.rerank_model, self.rerank_model)
        print(f"[计时] [HybridRetriever] 开始重排 ({model_label}) ...")
        outcome = rerank_documents(
            query=query,
            documents=vector_results,
            model_choice=self.rerank_model,
            api_key=self.rerank_api_key,
            llm_weight=llm_weight,
        )
        reranked_results = outcome.documents
        self.last_rerank_fallback = outcome.fallback
        self.last_rerank_fallback_reason = outcome.fallback_reason
        t2 = time.time()
        self.last_rerank_elapsed = t2 - t1 if outcome.rerank_applied else 0.0
        if outcome.fallback:
            print(
                f"[计时] [HybridRetriever] 重排熔断降级: {outcome.fallback_reason} "
                f"(耗时 {outcome.elapsed:.2f}s)"
            )
        else:
            print(f"[计时] [HybridRetriever] 重排耗时: {self.last_rerank_elapsed:.2f} 秒")
        print(f"[计时] [HybridRetriever] 总耗时: {t2-t0:.2f} 秒")

        filtered = [
            doc
            for doc in reranked_results
            if doc.get("combined_score", doc.get("distance", 0.0)) >= DEFAULT_MIN_RERANK_COMBINED
            or doc.get("lexical_score", 0.0) >= 0.28
        ]
        if len(filtered) < top_n:
            filtered = reranked_results
        final_results = filtered[:top_n]
        for i, result in enumerate(final_results, 1):
            result["chunk_index"] = i
        return final_results
