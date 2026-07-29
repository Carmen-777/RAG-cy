"""文档重排序：DashScope / Jina 可配置，统一接口 + 超时熔断。"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from http import HTTPStatus
from typing import Callable

import requests
from dotenv import load_dotenv
from openai import OpenAI

import src.prompts as prompts
from src.api_requests import dashscope_chat_call, extract_dashscope_message_content
from src.rerank_config import (
    DASHSCOPE_RERANK_MODEL,
    JINA_RERANK_MODEL,
    RERANK_MODEL_DASHSCOPE,
    RERANK_MODEL_JINA,
    RERANK_MODEL_LABELS,
    RERANK_TIMEOUT_SECONDS,
)

_log = logging.getLogger(__name__)


@dataclass
class RerankOutcome:
    documents: list
    rerank_applied: bool
    fallback: bool
    model_choice: str
    elapsed: float
    fallback_reason: str | None = None


def _call_with_timeout(func: Callable, timeout: float):
    """在独立线程中调用 func，超时则抛出 TimeoutError。"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            raise TimeoutError(f"重排序 API 超时（>{timeout}s）") from exc


def _vector_fallback(documents: list) -> list:
    """降级为原始向量检索顺序。"""
    fallback = [doc.copy() for doc in documents]
    for doc in fallback:
        doc["relevance_score"] = 0.0
        doc["combined_score"] = float(
            doc.get("vector_score_norm", doc.get("vector_score", doc.get("distance", 0.0)))
        )
    fallback.sort(key=lambda x: x.get("combined_score", x.get("distance", 0.0)), reverse=True)
    return fallback


def _merge_rerank_scores(
    documents: list,
    score_by_index: dict[int, float],
    llm_weight: float = 0.7,
) -> list:
    """将 rerank 分数与向量/词法分数融合并排序。"""
    vector_weight = 1 - llm_weight
    results = []
    for idx, doc in enumerate(documents):
        relevance_score = float(score_by_index.get(idx, 0.0))
        doc_with_score = doc.copy()
        doc_with_score["relevance_score"] = round(relevance_score, 4)
        lexical = float(doc.get("lexical_score", 0.0))
        vector_part = float(
            doc.get("vector_score_norm", doc.get("vector_score", doc.get("distance", 0.0)))
        )
        if vector_part > 1.0:
            vector_part = 0.0
        doc_with_score["combined_score"] = round(
            llm_weight * relevance_score + vector_weight * vector_part + 0.1 * lexical,
            4,
        )
        results.append(doc_with_score)
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results


def _dashscope_rerank_api(
    query: str,
    texts: list[str],
    top_n: int,
    api_key: str | None,
) -> dict[int, float]:
    import dashscope
    from dashscope import TextReRank

    load_dotenv()
    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("未配置 DASHSCOPE_API_KEY")

    dashscope.api_key = key

    def _call():
        return TextReRank.call(
            model=DASHSCOPE_RERANK_MODEL,
            query=query,
            documents=texts,
            top_n=top_n,
            api_key=key,
        )

    resp = _call_with_timeout(_call, RERANK_TIMEOUT_SECONDS)
    if resp.status_code != HTTPStatus.OK:
        message = getattr(resp, "message", None) or str(resp)
        raise RuntimeError(f"DashScope Rerank 失败: {message}")

    output = getattr(resp, "output", None)
    results = getattr(output, "results", None) if output else None
    if not results:
        raise RuntimeError(f"DashScope Rerank 返回为空: {resp}")

    return {int(item.index): float(item.relevance_score) for item in results}


def _jina_rerank_api(
    query: str,
    texts: list[str],
    top_n: int,
    api_key: str | None,
) -> dict[int, float]:
    load_dotenv()
    key = api_key or os.getenv("JINA_API_KEY")
    if not key:
        raise ValueError("未配置 JINA_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    data = {
        "model": JINA_RERANK_MODEL,
        "query": query,
        "top_n": top_n,
        "documents": texts,
    }

    def _call():
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers=headers,
            json=data,
            timeout=RERANK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    payload = _call_with_timeout(_call, RERANK_TIMEOUT_SECONDS)
    if "results" not in payload:
        raise RuntimeError(f"Jina rerank API 返回异常: {payload}")

    return {
        int(item["index"]): float(item["relevance_score"])
        for item in payload["results"]
    }


def rerank_documents(
    query: str,
    documents: list,
    model_choice: str = RERANK_MODEL_DASHSCOPE,
    api_key: str | None = None,
    llm_weight: float = 0.7,
    timeout: float = RERANK_TIMEOUT_SECONDS,
) -> RerankOutcome:
    """
    统一重排序入口：按 model_choice 路由到 DashScope 或 Jina。

    超时或报错时自动降级为向量检索顺序，不抛出异常。
    """
    if not documents:
        return RerankOutcome(
            documents=[],
            rerank_applied=False,
            fallback=False,
            model_choice=model_choice,
            elapsed=0.0,
        )

    model_choice = (model_choice or RERANK_MODEL_DASHSCOPE).lower()
    texts = [doc["text"] for doc in documents]
    t0 = time.time()

    try:
        if model_choice == RERANK_MODEL_JINA:
            score_by_index = _jina_rerank_api(query, texts, len(texts), api_key)
        elif model_choice == RERANK_MODEL_DASHSCOPE:
            score_by_index = _dashscope_rerank_api(query, texts, len(texts), api_key)
        else:
            raise ValueError(f"不支持的重排序模型: {model_choice}")

        ranked = _merge_rerank_scores(documents, score_by_index, llm_weight)
        elapsed = time.time() - t0
        if elapsed > timeout:
            _log.warning("重排序耗时 %.2fs 超过阈值 %.2fs，降级为向量顺序", elapsed, timeout)
            return RerankOutcome(
                documents=_vector_fallback(documents),
                rerank_applied=False,
                fallback=True,
                model_choice=model_choice,
                elapsed=elapsed,
                fallback_reason=f"重排序耗时超过 {timeout}s",
            )

        return RerankOutcome(
            documents=ranked,
            rerank_applied=True,
            fallback=False,
            model_choice=model_choice,
            elapsed=elapsed,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        _log.warning(
            "%s 重排序失败（%.2fs），回退向量顺序: %s",
            RERANK_MODEL_LABELS.get(model_choice, model_choice),
            elapsed,
            exc,
        )
        return RerankOutcome(
            documents=_vector_fallback(documents),
            rerank_applied=False,
            fallback=True,
            model_choice=model_choice,
            elapsed=elapsed,
            fallback_reason=str(exc),
        )


# --- 兼容旧接口 -----------------------------------------------------------

class JinaReranker:
    """保留类接口，内部走统一 rerank_documents。"""

    def rerank(self, query, documents, top_n=10, max_retries=0):
        api_key = os.getenv("JINA_API_KEY")
        outcome = rerank_documents(
            query=query,
            documents=[{"text": t} for t in documents],
            model_choice=RERANK_MODEL_JINA,
            api_key=api_key,
        )
        if outcome.fallback:
            raise RuntimeError(outcome.fallback_reason or "Jina rerank failed")
        return {
            "results": [
                {"index": i, "relevance_score": doc.get("relevance_score", 0.0)}
                for i, doc in enumerate(outcome.documents)
            ]
        }

    def rerank_documents(
        self,
        query: str,
        documents: list,
        documents_batch_size: int = 4,
        llm_weight: float = 0.7,
        api_key: str | None = None,
    ) -> list:
        del documents_batch_size
        outcome = rerank_documents(
            query=query,
            documents=documents,
            model_choice=RERANK_MODEL_JINA,
            api_key=api_key,
            llm_weight=llm_weight,
        )
        return outcome.documents


class DashScopeReranker:
    """DashScope qwen3-vl-rerank 封装（兼容类接口）。"""

    def rerank_documents(
        self,
        query: str,
        documents: list,
        documents_batch_size: int = 4,
        llm_weight: float = 0.7,
        api_key: str | None = None,
    ) -> list:
        del documents_batch_size
        outcome = rerank_documents(
            query=query,
            documents=documents,
            model_choice=RERANK_MODEL_DASHSCOPE,
            api_key=api_key,
            llm_weight=llm_weight,
        )
        return outcome.documents


class LLMReranker:
    def __init__(self, provider: str = "dashscope"):
        self.provider = provider.lower()
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks

    def set_up_llm(self):
        load_dotenv()
        if self.provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self.provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return dashscope
        raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_single_block},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_single_block,
            )
            return completion.choices[0].message.parsed.model_dump()
        if self.provider == "dashscope":
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ]
            rsp = dashscope_chat_call(
                model="qwen3.7-plus",
                messages=messages,
                temperature=0,
                result_format="message",
            )
            content = extract_dashscope_message_content(rsp)
            return {"relevance_score": 0.0, "reasoning": content}
        raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join(
            [f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)]
        )
        user_prompt = (
            f'Here is the query: "{query}"\n\n'
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_multiple_blocks,
            )
            return completion.choices[0].message.parsed.model_dump()
        if self.provider == "dashscope":
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ]
            rsp = dashscope_chat_call(
                model="qwen3.7-plus",
                messages=messages,
                temperature=0,
                result_format="message",
            )
            content = extract_dashscope_message_content(rsp)
            return {
                "block_rankings": [
                    {"relevance_score": 0.0, "reasoning": content}
                    for _ in retrieved_documents
                ]
            }
        raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def rerank_documents(
        self,
        query: str,
        documents: list,
        documents_batch_size: int = 4,
        llm_weight: float = 0.7,
    ):
        doc_batches = [documents[i : i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight

        if documents_batch_size == 1:

            def process_single_doc(doc):
                ranking = self.get_rank_for_single_block(query, doc["text"])
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + vector_weight * doc["distance"],
                    4,
                )
                return doc_with_score

            with ThreadPoolExecutor(max_workers=1) as executor:
                all_results = list(executor.map(process_single_doc, documents))
        else:

            def process_batch(batch):
                texts = [doc["text"] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get("block_rankings", [])
                if len(block_rankings) < len(batch):
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append(
                            {"relevance_score": 0.0, "reasoning": "Default ranking due to missing LLM response"}
                        )
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + vector_weight * doc["distance"],
                        4,
                    )
                    results.append(doc_with_score)
                return results

            with ThreadPoolExecutor(max_workers=1) as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)

        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
