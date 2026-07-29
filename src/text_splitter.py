import json
import hashlib
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import pandas as pd
import os

from src.page_markers import (
    build_page_line_map,
    page_for_line,
    prepare_markdown_with_pages,
)
from src.chunk_metadata import attach_source_to_chunks

# 文本分块工具类，支持按页分块、表格插入、token统计等
class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """按页分组已序列化表格，便于后续插入到对应页面分块中"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """将报告按页分块，保留markdown表格内容，可选插入序列化表格块。"""
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            # 加载序列化表格，按页分组
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in file_content['content']['pages']:
            # 普通文本分块
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            # 插入序列化表格分块
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
        metainfo = file_content.get("metainfo", {})
        if metainfo:
            file_content["content"]["chunks"] = attach_source_to_chunks(
                file_content["content"]["chunks"], metainfo
            )
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        # 统计字符串的token数，支持自定义编码
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """将单页文本分块，保留原始markdown表格。"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    #对 json 文件分块，输出还是 json
    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """
        批量处理目录下所有报告（json文件），对每个报告进行文本分块，并输出到目标目录。
        如果提供了序列化表格目录，会尝试将表格内容插入到对应页面的分块中。
        主要用于后续向量化和检索的预处理。
        参数：
            all_report_dir: 存放待处理报告json的目录
            output_dir: 分块后输出的目标目录
            serialized_tables_dir: （可选）存放序列化表格的目录
        """
        # 获取所有报告文件路径
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        # 遍历每个报告文件
        for report_path in all_report_paths:
            serialized_tables_path = None
            # 如果提供了表格序列化目录，查找对应表格文件
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"警告：未找到 {report_path.name} 的序列化表格报告")
                
            # 读取报告内容
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            # 分块处理，插入表格分块（如有）
            updated_report = self._split_report(report_data, serialized_tables_path)
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入分块后的报告到目标目录
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        # 输出处理文件数统计
        print(f"已分块处理 {len(all_report_paths)} 个文件")

    def _find_line_range(self, full_text: str, chunk_text: str, cursor: int = 0) -> tuple[List[int], int, int]:
        needle = chunk_text[: min(200, len(chunk_text))]
        pos = full_text.find(needle, cursor)
        if pos == -1:
            pos = full_text.find(needle)
        if pos == -1:
            fallback_line = full_text[:cursor].count("\n") + 1 if cursor > 0 else 1
            return [fallback_line, fallback_line + chunk_text.count("\n")], cursor, cursor
        start_line = full_text[:pos].count("\n") + 1
        end_line = start_line + chunk_text.count("\n")
        next_cursor = pos + len(needle)
        return [start_line, max(start_line, end_line)], next_cursor, pos

    def _build_chunk(
        self,
        text: str,
        headers: Dict[str, str],
        full_text: str,
        cursor: int,
        page_line_map: Optional[List[Optional[int]]] = None,
    ) -> tuple[Dict[str, Any], int]:
        lines, next_cursor, _char_pos = self._find_line_range(full_text, text, cursor)
        chunk: Dict[str, Any] = {
            "lines": lines,
            "length_tokens": self.count_tokens(text),
            "headers": headers,
            "text": text,
        }
        if page_line_map:
            page = page_for_line(page_line_map, lines[0])
            if page is not None:
                chunk["page"] = page
        return chunk, next_cursor

    def _get_merged_sections(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        if chunk.get("merged_sections"):
            return list(chunk["merged_sections"])
        return [{
            "headers": dict(chunk.get("headers", {})),
            "lines": list(chunk.get("lines", [0, 0])),
            "length_tokens": chunk.get("length_tokens", 0),
        }]

    def _merge_two_chunks(
        self,
        left: Dict[str, Any],
        right: Dict[str, Any],
        combined_text: str,
        combined_tokens: int,
    ) -> Dict[str, Any]:
        left_lines = left.get("lines", [0, 0])
        right_lines = right.get("lines", [0, 0])
        line_start = left_lines[0] or right_lines[0]
        line_end = max(left_lines[1], right_lines[1])

        return {
            "text": combined_text,
            "length_tokens": combined_tokens,
            "headers": dict(left.get("headers", {})),
            "lines": [line_start, line_end],
            "merged": True,
            "merged_sections": self._get_merged_sections(left) + self._get_merged_sections(right),
            **({"page": left["page"]} if left.get("page") else {}),
        }

    def _merge_small_chunks(
        self,
        chunks: List[Dict[str, Any]],
        min_tokens: int = 50,
        max_merged_tokens: int = 2000,
    ) -> List[Dict[str, Any]]:
        """将小于 min_tokens 的碎片 chunk 合并到相邻 chunk，且合并后不超过 max_merged_tokens。"""
        if len(chunks) <= 1:
            return chunks

        merged = [dict(c) for c in chunks]
        i = 1
        while i < len(merged):
            small = merged[i]
            if small.get("length_tokens", 0) >= min_tokens:
                i += 1
                continue

            merged_into_neighbor = False

            if i > 0:
                prev = merged[i - 1]
                combined_text = prev["text"] + "\n\n" + small["text"]
                combined_tokens = self.count_tokens(combined_text)
                if combined_tokens <= max_merged_tokens:
                    merged[i - 1] = self._merge_two_chunks(prev, small, combined_text, combined_tokens)
                    del merged[i]
                    merged_into_neighbor = True

            if not merged_into_neighbor and i < len(merged) - 1:
                nxt = merged[i + 1]
                combined_text = small["text"] + "\n\n" + nxt["text"]
                combined_tokens = self.count_tokens(combined_text)
                if combined_tokens <= max_merged_tokens:
                    merged[i + 1] = self._merge_two_chunks(small, nxt, combined_text, combined_tokens)
                    del merged[i]
                    merged_into_neighbor = True

            if merged_into_neighbor:
                continue
            i += 1

        if merged[0].get("length_tokens", 0) < min_tokens and len(merged) > 1:
            small, nxt = merged[0], merged[1]
            combined_text = small["text"] + "\n\n" + nxt["text"]
            combined_tokens = self.count_tokens(combined_text)
            if combined_tokens <= max_merged_tokens:
                merged[1] = self._merge_two_chunks(small, nxt, combined_text, combined_tokens)
                del merged[0]

        return merged

    def split_markdown_file(
        self,
        md_path: Path,
        max_tokens: int = 500,
        chunk_overlap: int = 50,
        pdf_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        优先按 Markdown 标题（# / ## / ###）语义切分；
        单个标题块超过 max_tokens 时，再用 RecursiveCharacterTextSplitter 二次切分。
        """
        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        text = prepare_markdown_with_pages(raw_text, pdf_path=pdf_path)
        page_line_map = build_page_line_map(text)

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )
        header_docs = header_splitter.split_text(text)

        secondary_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=max_tokens,
            chunk_overlap=chunk_overlap,
        )

        chunks: List[Dict[str, Any]] = []
        cursor = 0
        for doc in header_docs:
            content = doc.page_content.strip()
            if not content:
                continue

            headers = dict(doc.metadata)
            if self.count_tokens(content) <= max_tokens:
                chunk, cursor = self._build_chunk(
                    content, headers, text, cursor, page_line_map
                )
                chunks.append(chunk)
                continue

            sub_texts = secondary_splitter.split_text(content)
            for sub_idx, sub_text in enumerate(sub_texts, start=1):
                sub_headers = {
                    **headers,
                    "sub_chunk": sub_idx,
                    "sub_chunks_total": len(sub_texts),
                }
                chunk, cursor = self._build_chunk(
                    sub_text, sub_headers, text, cursor, page_line_map
                )
                chunks.append(chunk)

        chunks = self._merge_small_chunks(chunks)
        return chunks

    def split_markdown_reports(
        self,
        all_md_dir: Path,
        output_dir: Path,
        max_tokens: int = 500,
        chunk_overlap: int = 50,
        subset_csv: Path = None,
        pdf_dir: Optional[Path] = None,
    ):
        """
        批量处理目录下所有 markdown 文件，分块并输出为 json 文件到目标目录。
        :param all_md_dir: 存放 .md 文件的目录
        :param output_dir: 输出 .json 文件的目录
        :param max_tokens: 单个标题块允许的最大 token 数，超出则二次切分
        :param chunk_overlap: 二次切分的重叠 token 数
        :param subset_csv: subset.csv 路径，用于建立 file_name 到 company_name 的映射
        """
        # 建立 file_name（去扩展名）到 company_name 的映射
        file2company = {}
        file2sha1 = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            # 自动识别主键列
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')
        
        all_md_paths = list(all_md_dir.glob("*.md"))
        output_dir.mkdir(parents=True, exist_ok=True)
        total_chunks = 0
        for md_path in all_md_paths:
            pdf_path = None
            if pdf_dir is not None:
                candidate = pdf_dir / f"{md_path.stem}.pdf"
                if candidate.exists():
                    pdf_path = candidate
            chunks = self.split_markdown_file(
                md_path, max_tokens, chunk_overlap, pdf_path=pdf_path
            )
            total_chunks += len(chunks)
            output_json_path = output_dir / (md_path.stem + ".json")
            # 查找 company_name 和 sha1
            file_no_ext = md_path.stem
            company_name = file2company.get(file_no_ext, "")
            sha1 = file2sha1.get(file_no_ext, "")
            if not sha1:
                sha1 = hashlib.sha1(file_no_ext.encode("utf-8")).hexdigest()
            # metainfo 只保留 sha1、company_name、file_name 字段
            metainfo = {"sha1": sha1, "company_name": company_name, "file_name": md_path.name}
            chunks = attach_source_to_chunks(chunks, metainfo)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({"metainfo": metainfo, "content": {"chunks": chunks}}, f, ensure_ascii=False, indent=2)
            print(f"已处理: {md_path.name} -> {output_json_path.name} ({len(chunks)} chunks)")
        print(f"共分割 {len(all_md_paths)} 个 markdown 文件，合计 {total_chunks} 个 chunks")
