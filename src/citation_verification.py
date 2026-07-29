"""引用强制回溯校验：将 LLM 回答映射到真实检索片段，避免张冠李戴。"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.retrieval import resolve_chunk_page
from src.chunk_metadata import result_source_from_chunk, metainfo_source_fields

# (documents_dir, company_name) -> corpus chunks
_CORPUS_CACHE: dict[tuple[str, str], list[dict]] = {}


def _strip_page_markers(text: str) -> str:
    text = re.sub(r"<!--\s*Page\s+\d+\s*-->", "", text, flags=re.IGNORECASE)
    text = re.sub(r"#\s*Page\s+\d+", "", text, flags=re.IGNORECASE)
    return text


def _normalize_text(text: str) -> str:
    text = _strip_page_markers(text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]+|\w+", text))


def text_overlap_score(answer_text: str, chunk_text: str) -> float:
    """估算 answer 片段与 chunk 文本的相似度（0~1），无需 embedding API。"""
    answer_norm = _normalize_text(answer_text)
    chunk_norm = _normalize_text(chunk_text)
    if not answer_norm or not chunk_norm:
        return 0.0

    if len(answer_norm) >= 12 and answer_norm in chunk_norm:
        return 1.0
    if len(chunk_norm) >= 12 and chunk_norm in answer_norm:
        return 0.95

    answer_tokens = _token_set(answer_text)
    if not answer_tokens:
        return 0.0
    chunk_tokens = _token_set(chunk_text)
    token_overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)

    seq_ratio = SequenceMatcher(
        None, answer_norm[:800], chunk_norm[:800]
    ).ratio()
    return 0.65 * token_overlap + 0.35 * seq_ratio


def parse_inline_citations(text: str) -> list[int]:
    """解析 [1]、[2] 形式的角标引用。"""
    if not text:
        return []
    seen: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def extract_sentences(text: str, min_len: int = 8) -> list[str]:
    """按中英文句号切分句子。"""
    if not text:
        return []
    parts = re.split(r"[。！？；\n]+", text)
    return [part.strip() for part in parts if len(part.strip()) >= min_len]


def _metainfo_display_name(metainfo: dict) -> str:
    file_name = metainfo.get("file_name", "")
    if file_name.lower().endswith(".md"):
        return file_name[:-3]
    return file_name


def load_company_corpus_chunks(documents_dir: Path, company_name: str) -> list[dict]:
    """加载某公司在语料库中所有文档的全部 chunk（供反向引用匹配）。"""
    cache_key = (str(documents_dir.resolve()), company_name)
    if cache_key in _CORPUS_CACHE:
        return _CORPUS_CACHE[cache_key]

    chunks: list[dict] = []
    for json_path in sorted(documents_dir.glob("*.json")):
        try:
            document = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        metainfo = document.get("metainfo", {})
        file_name = metainfo.get("file_name", "")
        if metainfo.get("company_name") != company_name and company_name not in file_name:
            continue

        for chunk_idx, chunk in enumerate(document.get("content", {}).get("chunks", [])):
            source = result_source_from_chunk(chunk, metainfo)
            chunks.append(
                {
                    "text": chunk.get("text", ""),
                    "page": resolve_chunk_page(chunk),
                    "chunk_id": chunk.get("id", chunk_idx),
                    "corpus_key": f"{metainfo.get('sha1', json_path.stem)}:{chunk_idx}",
                    **source,
                }
            )

    _CORPUS_CACHE[cache_key] = chunks
    return chunks


def _chunk_corpus_key(chunk: dict) -> str:
    if chunk.get("corpus_key"):
        return str(chunk["corpus_key"])
    sha1 = chunk.get("sha1", "")
    chunk_id = chunk.get("chunk_id", "")
    page = chunk.get("page", "")
    text_prefix = _normalize_text(chunk.get("text", ""))[:80]
    return f"{sha1}|{chunk_id}|{page}|{text_prefix}"


def reverse_find_source_chunks(
    answer_text: str,
    corpus_chunks: list[dict],
    min_sentence_score: float = 0.18,
    min_whole_answer_score: float = 0.12,
    max_chunks: int = 3,
) -> tuple[list[dict], list[dict[str, Any]]]:
    """
    方案 A：用 final_answer 在全语料中反向查找真实来源 chunk。
    返回 (去重排序后的 chunk 列表, 逐句匹配详情)。
    """
    if not corpus_chunks or not answer_text.strip():
        return [], []

    chunk_scores: dict[str, tuple[dict, float]] = {}
    sentence_matches: list[dict[str, Any]] = []

    for sentence in extract_sentences(answer_text):
        best_chunk: dict | None = None
        best_score = 0.0
        for chunk in corpus_chunks:
            score = text_overlap_score(sentence, chunk.get("text", ""))
            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk is not None and best_score >= min_sentence_score:
            key = _chunk_corpus_key(best_chunk)
            prev = chunk_scores.get(key)
            if not prev or best_score > prev[1]:
                chunk_scores[key] = (best_chunk, best_score)
            sentence_matches.append(
                {
                    "sentence": sentence,
                    "score": round(best_score, 4),
                    "verified": True,
                    "corpus_key": key,
                }
            )
        else:
            sentence_matches.append(
                {
                    "sentence": sentence,
                    "score": round(best_score, 4),
                    "verified": False,
                    "source": "model_generated",
                }
            )

    for chunk in corpus_chunks:
        score = text_overlap_score(answer_text, chunk.get("text", ""))
        if score >= min_whole_answer_score:
            key = _chunk_corpus_key(chunk)
            prev = chunk_scores.get(key)
            if not prev or score > prev[1]:
                chunk_scores[key] = (chunk, score)

    if not chunk_scores:
        return [], sentence_matches

    ranked = sorted(chunk_scores.values(), key=lambda item: item[1], reverse=True)[:max_chunks]
    cited: list[dict] = []
    for chunk, score in ranked:
        entry = dict(chunk)
        entry["match_score"] = round(score, 4)
        entry["verification"] = "reverse_lookup"
        entry["distance"] = round(score, 4)
        cited.append(entry)
    return cited, sentence_matches


def match_answer_to_chunks(
    answer_text: str,
    retrieved_chunks: list[dict],
    min_score: float = 0.12,
) -> list[dict[str, Any]]:
    """将答案中的每句话匹配到最相似的检索片段。"""
    matches: list[dict[str, Any]] = []
    for sentence in extract_sentences(answer_text):
        best_index: int | None = None
        best_score = 0.0
        for i, chunk in enumerate(retrieved_chunks, 1):
            score = text_overlap_score(sentence, chunk.get("text", ""))
            if score > best_score:
                best_score = score
                best_index = i

        if best_index is not None and best_score >= min_score:
            matches.append(
                {
                    "sentence": sentence,
                    "chunk_index": best_index,
                    "score": round(best_score, 4),
                    "verified": True,
                }
            )
        else:
            matches.append(
                {
                    "sentence": sentence,
                    "chunk_index": None,
                    "score": round(best_score, 4),
                    "verified": False,
                    "source": "model_generated",
                }
            )
    return matches


def _chunk_display_name(chunk: dict) -> str:
    display = (
        chunk.get("source_display_name")
        or chunk.get("display_name")
        or chunk.get("source_file")
        or chunk.get("file_name")
        or ""
    )
    if display.lower().endswith(".md"):
        return display[:-3]
    return display or "unknown"


def _merge_verified_indices(
    claimed_indices: list[int],
    content_indices: list[int],
    answer_text: str,
    retrieved_chunks: list[dict],
    min_whole_answer_score: float = 0.08,
    max_indices: int = 8,
) -> list[int]:
    """合并 LLM 声明的角标与内容匹配结果，内容匹配优先。"""
    valid_claimed = [
        idx for idx in claimed_indices if 1 <= idx <= len(retrieved_chunks)
    ]

    if valid_claimed and content_indices:
        overlap = [idx for idx in valid_claimed if idx in content_indices]
        merged = overlap if overlap else content_indices
    elif content_indices:
        merged = content_indices
    elif valid_claimed:
        merged = valid_claimed
    else:
        ranked = sorted(
            (
                (i, text_overlap_score(answer_text, chunk.get("text", "")))
                for i, chunk in enumerate(retrieved_chunks, 1)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        merged = [
            idx for idx, score in ranked[:max_indices] if score >= min_whole_answer_score
        ]

    ordered: list[int] = []
    for idx in merged:
        if idx not in ordered:
            ordered.append(idx)
        if len(ordered) >= max_indices:
            break
    return ordered


def _build_citation_metadata(cited_chunks: list[dict]) -> tuple[list[int], list[dict], list[dict], list[int]]:
    verified_indices: list[int] = []
    verified_citations: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    relevant_pages: list[int] = []

    for i, chunk in enumerate(cited_chunks, 1):
        verified_indices.append(i)
        page = chunk.get("page")
        citation = {
            "chunk_index": i,
            "file_name": _chunk_display_name(chunk),
            "page": page,
            "sha1": chunk.get("sha1", ""),
            "verification": chunk.get("verification", "reverse_lookup"),
            "match_score": chunk.get("match_score"),
        }
        verified_citations.append(citation)
        references.append(
            {
                "pdf_sha1": chunk.get("sha1", ""),
                "page_index": page,
                "file_name": citation["file_name"],
                "chunk_index": i,
            }
        )
        if page is not None:
            try:
                page_int = int(page)
            except (TypeError, ValueError):
                continue
            if page_int > 0 and page_int not in relevant_pages:
                relevant_pages.append(page_int)

    return verified_indices, verified_citations, references, relevant_pages


def _inject_citation_markers(final_answer: str, indices: list[int]) -> str:
    text = final_answer.strip()
    if not indices or re.search(r"\[\d+\]", text):
        return text
    markers = "".join(f"[{idx}]" for idx in sorted(set(indices)))
    return f"{text} {markers}".strip()


def verify_citations(
    llm_response: dict,
    retrieved_chunks: list[dict],
    company_name: str | None = None,
    documents_dir: Path | str | None = None,
) -> dict:
    """
    校验并修正 LLM 回答中的引用元数据。

    1. 先在原始检索 Top-K 中做内容匹配
    2. 方案 A：在全语料中反向查找与 final_answer 最匹配的 chunk
    3. 优先采用反向查找结果作为 cited_retrieval_results 与角标来源
    """
    result = dict(llm_response)
    final_answer = str(result.get("final_answer") or "")
    analysis = str(result.get("step_by_step_analysis") or "")

    claimed_indices: list[int] = list(result.get("source_chunk_indices") or [])
    for idx in parse_inline_citations(final_answer) + parse_inline_citations(analysis):
        if idx not in claimed_indices:
            claimed_indices.append(idx)

    content_matches = match_answer_to_chunks(final_answer, retrieved_chunks)
    content_indices = [
        match["chunk_index"]
        for match in content_matches
        if match.get("verified") and match.get("chunk_index")
    ]
    retrieval_indices = _merge_verified_indices(
        claimed_indices, content_indices, final_answer, retrieved_chunks
    )
    best_retrieval_score = 0.0
    if retrieved_chunks:
        best_retrieval_score = max(
            text_overlap_score(final_answer, chunk.get("text", ""))
            for chunk in retrieved_chunks
        )

    result["original_retrieval_results"] = list(retrieved_chunks)
    result["citation_matches"] = content_matches

    reverse_chunks: list[dict] = []
    reverse_matches: list[dict] = []
    if company_name and documents_dir:
        corpus = load_company_corpus_chunks(Path(documents_dir), company_name)
        reverse_chunks, reverse_matches = reverse_find_source_chunks(final_answer, corpus)
        result["reverse_citation_matches"] = reverse_matches

    use_reverse = bool(reverse_chunks)
    retrieval_usable = bool(retrieval_indices) and best_retrieval_score >= 0.15

    if use_reverse:
        cited_chunks = reverse_chunks
        for i, chunk in enumerate(cited_chunks, 1):
            chunk["chunk_index"] = i
        result["cited_retrieval_results"] = cited_chunks
        result["citation_source"] = "reverse_lookup"
        result["answer_grounded"] = True
        verified_indices, verified_citations, references, relevant_pages = _build_citation_metadata(
            cited_chunks
        )
    elif retrieval_usable:
        cited_chunks = [dict(retrieved_chunks[i - 1]) for i in retrieval_indices]
        for i, chunk in enumerate(cited_chunks, 1):
            chunk["chunk_index"] = i
            chunk["verification"] = "retrieval_match"
        result["cited_retrieval_results"] = cited_chunks
        result["citation_source"] = "retrieval_match"
        result["answer_grounded"] = True
        verified_indices, verified_citations, references, relevant_pages = _build_citation_metadata(
            cited_chunks
        )
    else:
        result["cited_retrieval_results"] = []
        result["citation_source"] = "none"
        result["answer_grounded"] = False
        verified_indices = []
        verified_citations = []
        references = []
        relevant_pages = []
        result["unverified_sentences"] = [
            match["sentence"]
            for match in (reverse_matches or content_matches)
            if not match.get("verified")
        ]

    result["source_chunk_indices"] = verified_indices
    result["verified_citations"] = verified_citations
    result["references"] = references
    result["relevant_pages"] = relevant_pages
    result["final_answer"] = _inject_citation_markers(final_answer, verified_indices)

    if result.get("answer_grounded"):
        result["unverified_sentences"] = [
            match["sentence"]
            for match in (reverse_matches or content_matches)
            if not match.get("verified")
        ]

    return result
