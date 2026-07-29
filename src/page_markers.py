"""在 Markdown 中注入与解析 # Page N 页码标记。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

PAGE_MARKER_RE = re.compile(
    r"(?:^#\s*Page\s+(\d+)\s*$|<!--\s*Page\s+(\d+)\s*-->)",
    re.MULTILINE | re.IGNORECASE,
)


def _marker_page(match: re.Match) -> int:
    return int(match.group(1) or match.group(2))


def _page_marker(page: int) -> str:
    return f"<!-- Page {page} -->"


def has_page_markers(text: str) -> bool:
    return PAGE_MARKER_RE.search(text) is not None


def parse_page_range_start(page_ranges: str) -> int:
    return int(page_ranges.split("-")[0].strip())


def parse_page_range_end(page_ranges: str) -> int:
    parts = page_ranges.split("-")
    return int(parts[1].strip()) if len(parts) > 1 else int(parts[0].strip())


def inject_pages_in_range(md_text: str, start_page: int, end_page: int) -> str:
    """在一段 Markdown 内按字符比例插入 # Page N（页码范围为 start_page..end_page）。"""
    if has_page_markers(md_text):
        return md_text

    total_pages = max(1, end_page - start_page + 1)
    if total_pages == 1:
        return f"{_page_marker(start_page)}\n\n{md_text}"

    length = len(md_text)
    if length == 0:
        return f"{_page_marker(start_page)}\n\n"

    parts: list[str] = []
    for idx in range(total_pages):
        page = start_page + idx
        char_start = idx * length // total_pages
        char_end = (idx + 1) * length // total_pages if idx < total_pages - 1 else length
        segment = md_text[char_start:char_end]
        if idx == 0:
            parts.append(f"{_page_marker(page)}\n\n{segment}")
        else:
            parts.append(f"\n\n---\n\n{_page_marker(page)}\n\n{segment}")
    return "".join(parts)


def inject_pages_from_pdf_page_count(md_text: str, total_pages: int) -> str:
    """根据 PDF 总页数，为整篇 Markdown 按比例注入页码标记。"""
    if total_pages <= 0:
        return md_text
    return inject_pages_in_range(md_text, 1, total_pages)


def get_pdf_page_count(pdf_path: Path) -> Optional[int]:
    if not pdf_path.exists():
        return None
    try:
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return None


def inject_pages_estimated(md_text: str, lines_per_page: int = 45) -> str:
    """无 PDF 时按行数估算页数并注入标记。"""
    if has_page_markers(md_text):
        return md_text
    line_count = max(1, md_text.count("\n") + 1)
    estimated_pages = max(1, (line_count + lines_per_page - 1) // lines_per_page)
    return inject_pages_from_pdf_page_count(md_text, estimated_pages)


def prepare_markdown_with_pages(
    md_text: str,
    pdf_path: Optional[Path] = None,
    lines_per_page: int = 45,
) -> str:
    """确保 Markdown 含 # Page N 标记；优先使用 PDF 页数，否则按行数估算。"""
    if has_page_markers(md_text):
        return md_text

    if pdf_path is not None:
        page_count = get_pdf_page_count(pdf_path)
        if page_count:
            return inject_pages_from_pdf_page_count(md_text, page_count)

    return inject_pages_estimated(md_text, lines_per_page=lines_per_page)


def build_page_line_map(full_text: str) -> list[Optional[int]]:
    """为全文每一行标注当前页码（由最近的页码标记继承）。"""
    page_lines: list[Optional[int]] = []
    current_page: Optional[int] = None
    for line in full_text.splitlines():
        match = PAGE_MARKER_RE.search(line.strip())
        if match:
            current_page = _marker_page(match)
        page_lines.append(current_page)
    return page_lines


def page_for_line(page_line_map: list[Optional[int]], line_number: int) -> Optional[int]:
    if line_number <= 0:
        return None
    idx = min(line_number, len(page_line_map)) - 1
    for pos in range(idx, -1, -1):
        page = page_line_map[pos]
        if page is not None and page > 0:
            return page
    return None


def build_page_boundary_map(full_text: str) -> list[tuple[int, int]]:
    """返回 [(字符偏移, 页码), ...]，按偏移升序。"""
    return [(match.start(), _marker_page(match)) for match in PAGE_MARKER_RE.finditer(full_text)]


def page_at_offset(boundaries: list[tuple[int, int]], offset: int) -> Optional[int]:
    if not boundaries or offset < 0:
        return None
    page: Optional[int] = None
    for pos, page_num in boundaries:
        if pos <= offset:
            page = page_num
        else:
            break
    return page if page and page > 0 else None
