"""Chunk 级来源元数据：切块写入、索引查询、检索结果补全。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SOURCE_FIELD_ALIASES = (
    "source_display_name",
    "display_name",
    "source_file",
    "file_name",
    "source_sha1",
    "sha1",
)


def display_name_from_file_name(file_name: str) -> str:
    name = str(file_name or "").strip()
    if name.lower().endswith(".md"):
        return name[:-3]
    if name.lower().endswith(".pdf"):
        return name[:-4]
    return name


def metainfo_source_fields(metainfo: dict) -> dict[str, str]:
    """从文档 metainfo 提取标准来源字段。"""
    file_name = str(metainfo.get("file_name") or "").strip()
    sha1 = str(metainfo.get("sha1") or "").strip()
    display_name = display_name_from_file_name(file_name)
    return {
        "source_file": file_name,
        "source_sha1": sha1,
        "source_display_name": display_name,
        "file_name": file_name,
        "display_name": display_name,
        "sha1": sha1,
    }


def attach_source_to_chunk(chunk: dict, metainfo: dict) -> dict:
    """为单个 chunk 写入来源元数据（切块阶段调用）。"""
    enriched = dict(chunk)
    source = metainfo_source_fields(metainfo)
    enriched["source_file"] = source["source_file"]
    enriched["source_sha1"] = source["source_sha1"]
    enriched["source_display_name"] = source["source_display_name"]
    if enriched.get("id") is not None:
        enriched["source_chunk_id"] = enriched["id"]
    elif enriched.get("source_chunk_id") is None and "source_chunk_id" not in enriched:
        pass
    return enriched


def attach_source_to_chunks(chunks: list[dict], metainfo: dict) -> list[dict]:
    """批量为 chunk 列表写入来源元数据。"""
    result: list[dict] = []
    for idx, chunk in enumerate(chunks):
        item = attach_source_to_chunk(chunk, metainfo)
        if item.get("source_chunk_id") is None:
            item["source_chunk_id"] = chunk.get("id", idx)
        result.append(item)
    return result


def result_source_from_chunk(chunk: dict, metainfo: dict | None = None) -> dict[str, Any]:
    """解析检索结果应携带的来源字段（chunk 优先，metainfo 兜底）。"""
    meta = metainfo_source_fields(metainfo or {})

    file_name = str(chunk.get("source_file") or chunk.get("file_name") or meta["file_name"]).strip()
    sha1 = str(chunk.get("source_sha1") or chunk.get("sha1") or meta["sha1"]).strip()
    display_name = str(
        chunk.get("source_display_name")
        or chunk.get("display_name")
        or display_name_from_file_name(file_name)
        or meta["display_name"]
    ).strip()

    return {
        "file_name": file_name,
        "display_name": display_name,
        "sha1": sha1,
        "source_file": file_name,
        "source_sha1": sha1,
        "source_display_name": display_name,
        "source_chunk_id": chunk.get("source_chunk_id", chunk.get("id")),
    }


def resolve_source_label(
    result: dict,
    index: int,
    metadata_index: ChunkMetadataIndex | None = None,
    pdf_assets: dict[str, dict[str, str]] | None = None,
) -> str:
    """解析展示用来源名称；最后回退「参考文档 N」。"""
    pdf_assets = pdf_assets or {}

    for key in ("source_label", "source_display_name", "display_name", "source_file", "file_name"):
        val = str(result.get(key) or "").strip()
        if val.lower().endswith(".md"):
            val = val[:-3]
        if val and val not in {"unknown", "未知来源"}:
            return val

    sha1 = str(result.get("source_sha1") or result.get("sha1") or "").strip()
    chunk_id = result.get("source_chunk_id", result.get("chunk_id"))

    if metadata_index:
        looked_up = metadata_index.lookup(sha1=sha1, chunk_id=chunk_id)
        if looked_up and looked_up.get("source_display_name"):
            return looked_up["source_display_name"]

    if sha1 and sha1 in pdf_assets:
        name = str(pdf_assets[sha1].get("display_name") or "").strip()
        if name:
            return name

    return f"参考文档 {index}"


class ChunkMetadataIndex:
    """sha1 / chunk_id / 文档名 -> 来源元数据 全局索引。"""

    def __init__(self) -> None:
        self.by_sha1_chunk: dict[tuple[str, int], dict[str, str]] = {}
        self.by_sha1: dict[str, dict[str, str]] = {}
        self.by_stem: dict[str, dict[str, str]] = {}

    @classmethod
    def from_chunked_dir(cls, chunked_dir: Path) -> ChunkMetadataIndex:
        index = cls()
        chunked_path = Path(chunked_dir)
        for json_path in sorted(chunked_path.glob("*.json")):
            try:
                document = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            metainfo = document.get("metainfo", {})
            meta_fields = metainfo_source_fields(metainfo)
            sha1 = meta_fields["source_sha1"]
            if sha1:
                index.by_sha1[sha1] = meta_fields
            index.by_stem[json_path.stem] = meta_fields
            if meta_fields["source_display_name"]:
                index.by_stem[meta_fields["source_display_name"]] = meta_fields

            for chunk_idx, chunk in enumerate(document.get("content", {}).get("chunks", [])):
                fields = result_source_from_chunk(chunk, metainfo)
                chunk_key = fields.get("source_chunk_id", chunk.get("id", chunk_idx))
                try:
                    chunk_key_int = int(chunk_key)
                except (TypeError, ValueError):
                    chunk_key_int = chunk_idx
                if sha1:
                    index.by_sha1_chunk[(sha1, chunk_key_int)] = fields
                    index.by_sha1_chunk[(sha1, chunk_idx)] = fields

        return index

    def lookup(
        self,
        sha1: str = "",
        chunk_id: Any = None,
        stem: str = "",
    ) -> dict[str, str] | None:
        sha1 = str(sha1 or "").strip()
        if sha1 and chunk_id is not None:
            try:
                key = (sha1, int(chunk_id))
                if key in self.by_sha1_chunk:
                    return self.by_sha1_chunk[key]
            except (TypeError, ValueError):
                pass
        if sha1 and sha1 in self.by_sha1:
            return self.by_sha1[sha1]
        stem = str(stem or "").strip()
        if stem and stem in self.by_stem:
            return self.by_stem[stem]
        return None

    def enrich_result(self, result: dict, index: int) -> dict:
        """用索引补全单条检索/引用结果元数据。"""
        enriched = dict(result)
        sha1 = str(enriched.get("source_sha1") or enriched.get("sha1") or "").strip()
        chunk_id = enriched.get("source_chunk_id", enriched.get("chunk_id"))

        looked_up = self.lookup(sha1=sha1, chunk_id=chunk_id)
        if looked_up:
            for key, val in looked_up.items():
                if val and not enriched.get(key):
                    enriched[key] = val

        if not enriched.get("source_display_name") and enriched.get("display_name"):
            enriched["source_display_name"] = enriched["display_name"]
        if not enriched.get("source_file") and enriched.get("file_name"):
            enriched["source_file"] = enriched["file_name"]
        if not enriched.get("source_sha1") and enriched.get("sha1"):
            enriched["source_sha1"] = enriched["sha1"]

        enriched["source_label"] = resolve_source_label(enriched, index, self, None)
        return enriched


def backfill_chunked_reports(chunked_dir: Path) -> int:
    """为已有 chunked JSON 批量写入 chunk 级 source_* 字段（无需重新向量化）。"""
    updated_files = 0
    for json_path in sorted(Path(chunked_dir).glob("*.json")):
        try:
            document = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        metainfo = document.get("metainfo", {})
        chunks = document.get("content", {}).get("chunks", [])
        if not chunks:
            continue

        new_chunks = attach_source_to_chunks(chunks, metainfo)
        if new_chunks == chunks:
            continue

        document.setdefault("content", {})["chunks"] = new_chunks
        json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        updated_files += 1

    return updated_files


if __name__ == "__main__":
    from pyprojroot import here

    root = here() / "data" / "stock_data" / "databases" / "chunked_reports"
    count = backfill_chunked_reports(root)
    print(f"已更新 {count} 个 chunked JSON 文件的 chunk 来源元数据。")
