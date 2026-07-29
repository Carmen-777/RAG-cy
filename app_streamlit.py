import hashlib
import html
import http.server
import json
import re
import shutil
import socketserver
import threading
import time
from functools import partial
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyprojroot import here

from src.model_config import (
    AVAILABLE_ANSWERING_MODELS,
    DEFAULT_ANSWERING_MODEL,
    normalize_answering_model,
)
from src.rerank_config import (
    RERANK_MODEL_DASHSCOPE,
    RERANK_MODEL_JINA,
    RERANK_MODEL_LABELS,
)
from src.pipeline import Pipeline, RunConfig
from src.retrieval import HybridRetriever, VectorRetriever, format_page_display
from src.chunk_metadata import ChunkMetadataIndex, resolve_source_label

ROOT_PATH = here() / "data" / "stock_data"
PDF_REPORTS_DIR = ROOT_PATH / "pdf_reports"
STATIC_DIR = here() / "static"
PDF_OSS_BASE = "https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/"
PDF_SERVER_HOST = "127.0.0.1"
PDF_SERVER_PORT = 8765
_pdf_server_port: int | None = None

KIND_MAP = {
    "text": "string",
    "boolean": "boolean",
    "number": "number",
    "name": "name",
}

QUESTION_KIND_LABELS = {
    "text": "text（开放性文本）",
    "boolean": "boolean（是/否）",
    "number": "number（数值）",
    "name": "name（名称）",
}

DEFAULT_RUN_CONFIG = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=False,
    llm_reranking=True,
    llm_reranking_sample_size=20,
    top_n_retrieval=3,
    parallel_requests=1,
    submission_file=False,
    pipeline_details=f"Streamlit UI; {DEFAULT_ANSWERING_MODEL} + DashScope rerank",
    answering_model=DEFAULT_ANSWERING_MODEL,
    config_suffix="_streamlit",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        .rag-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            padding: 28px 24px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
        }
        .rag-header h1 {
            color: white;
            margin: 0 0 8px 0;
            font-size: 1.75rem;
            font-weight: 700;
        }
        .rag-header p {
            color: rgba(255,255,255,0.92);
            margin: 4px 0;
            font-size: 0.95rem;
        }
        .rag-header .features {
            color: rgba(255,255,255,0.85);
            font-size: 0.85rem;
            margin-top: 10px;
        }
        .result-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }
        .result-card h4 {
            color: #4f46e5;
            margin: 0 0 12px 0;
        }
        .result-card .page-link {
            color: #4f46e5;
            font-weight: 600;
            text-decoration: none;
        }
        .result-card .page-link:hover {
            color: #4338ca;
            text-decoration: underline;
        }
        .retrieval-snippet {
            font-size: 0.92rem;
            line-height: 1.75;
            color: #334155;
            white-space: pre-wrap;
            word-break: break-word;
            margin: 10px 0 0 0;
            padding: 12px 14px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }
        .retrieval-snippet p,
        .retrieval-snippet h1,
        .retrieval-snippet h2,
        .retrieval-snippet h3,
        .retrieval-snippet h4,
        .retrieval-snippet h5,
        .retrieval-snippet h6 {
            font-size: 0.92rem !important;
            line-height: 1.75 !important;
            font-weight: 400 !important;
            margin: 0 0 0.5em 0 !important;
        }
        .answer-hero-card {
            background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            border-left: 5px solid #6366f1;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.07);
            margin: 0 0 20px 0;
            overflow: hidden;
        }
        .answer-hero-body {
            padding: 12px 52px 22px 22px;
            font-size: 1.08rem;
            line-height: 1.85;
            color: #334155;
            white-space: pre-wrap;
            margin: 0;
            min-height: 48px;
        }
        .citation-ref {
            display: inline;
            background: #eef2ff;
            color: #4338ca;
            border: 1px solid #c7d2fe;
            border-radius: 6px;
            padding: 1px 7px;
            margin: 0 2px;
            font-size: 0.82em;
            font-weight: 700;
            cursor: pointer;
            line-height: 1.5;
            vertical-align: baseline;
            text-decoration: none;
        }
        .citation-ref:hover {
            background: #4338ca;
            color: #fff;
            border-color: #4338ca;
        }
        .ref-card.citation-highlight {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.28) !important;
            background: #eef2ff !important;
            transition: background 0.25s ease, box-shadow 0.25s ease;
        }
        div[data-testid="stSidebar"] .stSelectbox label,
        div[data-testid="stSidebar"] .stTextArea label {
            font-size: 0.9rem;
        }
        div[data-testid="stSidebar"] {
            background: #fafbff;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_document_pdf_assets(chunked_dir: str) -> dict[str, dict[str, str]]:
    """从 chunked JSON 构建 sha1 / 文档名 -> PDF 静态资源映射。"""
    assets: dict[str, dict[str, str]] = {}
    chunked_path = Path(chunked_dir)
    for json_path in chunked_path.glob("*.json"):
        try:
            document = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metainfo = document.get("metainfo", {})
        file_name = str(metainfo.get("file_name", "")).strip()
        sha1 = str(metainfo.get("sha1", "")).strip()
        if not file_name:
            continue
        if file_name.lower().endswith(".md"):
            source_name = file_name[:-3] + ".pdf"
            display_name = file_name[:-3]
        elif file_name.lower().endswith(".pdf"):
            source_name = file_name
            display_name = file_name[:-4]
        else:
            source_name = f"{file_name}.pdf"
            display_name = file_name

        static_name = (
            f"{sha1}.pdf"
            if sha1
            else f"{hashlib.sha1(source_name.encode('utf-8')).hexdigest()}.pdf"
        )
        info = {
            "source_name": source_name,
            "static_name": static_name,
            "display_name": display_name,
        }
        if sha1:
            assets[sha1] = info
        assets[display_name] = info
        assets[json_path.stem] = info
    return assets


def get_chunk_pdf_asset(
    result: dict, document_pdf_assets: dict[str, dict[str, str]]
) -> dict[str, str] | None:
    """按检索片段的 sha1 / 文档名解析对应 PDF。"""
    sha1 = str(result.get("sha1") or "").strip()
    if sha1 and sha1 in document_pdf_assets:
        info = document_pdf_assets[sha1]
    else:
        display_name = result.get("display_name") or result.get("file_name") or ""
        if str(display_name).lower().endswith(".md"):
            display_name = str(display_name)[:-3]
        info = document_pdf_assets.get(str(display_name).strip())
    if not info:
        return None
    if (STATIC_DIR / info["static_name"]).exists():
        return info
    return None


@st.cache_data
def load_company_pdf_assets(subset_path: str) -> dict[str, dict[str, str]]:
    """company_name -> {source_name, static_name, display_name}。"""
    path = Path(subset_path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    assets: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        company = str(row.get("company_name", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        if not company or not file_name:
            continue
        if file_name.lower().endswith(".md"):
            source_name = file_name[:-3] + ".pdf"
        elif file_name.lower().endswith(".pdf"):
            source_name = file_name
        else:
            source_name = f"{file_name}.pdf"

        sha1 = str(row.get("sha1", "")).strip()
        static_name = (
            f"{sha1}.pdf"
            if sha1
            else f"{hashlib.sha1(source_name.encode('utf-8')).hexdigest()}.pdf"
        )
        assets[company] = {
            "source_name": source_name,
            "static_name": static_name,
            "display_name": source_name,
        }
    return assets


def _try_fetch_pdf(source_name: str) -> Path | None:
    """本地无 PDF 时尝试从 OSS 下载。"""
    PDF_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    local = PDF_REPORTS_DIR / source_name
    if local.exists():
        return local
    try:
        import requests

        url = PDF_OSS_BASE + quote(source_name)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        local.write_bytes(response.content)
        return local
    except Exception:
        return None


class _QuietPDFHandler(http.server.SimpleHTTPRequestHandler):
    """本地 PDF 静态文件服务，供浏览器直接打开并支持 #page=N。"""

    def log_message(self, format, *args):
        return

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def start_pdf_server() -> int | None:
    """启动本地 PDF 文件服务（与 Streamlit 路由隔离，避免 /app/static 冲突）。"""
    global _pdf_server_port
    if _pdf_server_port is not None:
        return _pdf_server_port

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    handler = partial(_QuietPDFHandler, directory=str(STATIC_DIR))

    for port in range(PDF_SERVER_PORT, PDF_SERVER_PORT + 10):
        try:
            httpd = socketserver.TCPServer(
                (PDF_SERVER_HOST, port), handler, bind_and_activate=False
            )
            httpd.allow_reuse_address = True
            httpd.server_bind()
            httpd.server_activate()
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            _pdf_server_port = port
            return port
        except OSError:
            continue
    return None


def ensure_static_pdfs(company_pdf_assets: dict[str, dict[str, str]]):
    """将 PDF 同步到 static/ 目录，供本地 PDF 服务读取。"""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    PDF_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for info in company_pdf_assets.values():
        static_path = STATIC_DIR / info["static_name"]
        source_path = PDF_REPORTS_DIR / info["source_name"]
        if not source_path.exists():
            fetched = _try_fetch_pdf(info["source_name"])
            source_path = fetched if fetched else source_path
        if source_path.exists():
            if not static_path.exists() or source_path.stat().st_mtime > static_path.stat().st_mtime:
                shutil.copy2(source_path, static_path)


def get_company_static_pdf(
    company: str, company_pdf_assets: dict[str, dict[str, str]]
) -> dict[str, str] | None:
    info = company_pdf_assets.get(company)
    if not info:
        return None
    if (STATIC_DIR / info["static_name"]).exists():
        return info
    return None


def build_pdf_page_url(static_name: str, page) -> str | None:
    """构造 PDF 页码跳转链接（本地 HTTP 服务，Chrome 支持 #page=N）。"""
    if not static_name or not (STATIC_DIR / static_name).exists():
        return None
    port = start_pdf_server()
    if not port:
        return None
    url = f"http://{PDF_SERVER_HOST}:{port}/{quote(static_name)}"
    try:
        page_int = int(page)
    except (TypeError, ValueError):
        return url
    if page_int > 0:
        url = f"{url}#page={page_int}"
    return url


def normalize_retrieval_text(text: str, max_len: int = 2000) -> str:
    """去掉行首 Markdown 标题标记，避免被渲染为大号标题。"""
    snippet = text[:max_len] + ("…" if len(text) > max_len else "")
    lines = [re.sub(r"^#{1,6}\s*", "", line) for line in snippet.splitlines()]
    return html.escape("\n".join(lines))


def render_retrieval_snippet(text: str):
    st.markdown(
        f'<div class="retrieval-snippet">{normalize_retrieval_text(text)}</div>',
        unsafe_allow_html=True,
    )


def resolve_result_source_label(
    result: dict,
    index: int,
    document_pdf_assets: dict[str, dict[str, str]] | None = None,
    metadata_index: ChunkMetadataIndex | None = None,
) -> str:
    """解析检索片段来源名称；无法获取时回退为「参考文档 N」。"""
    return resolve_source_label(result, index, metadata_index, document_pdf_assets)


def enrich_retrieval_results(
    results: list[dict],
    document_pdf_assets: dict[str, dict[str, str]] | None = None,
    answer: dict | None = None,
    metadata_index: ChunkMetadataIndex | None = None,
) -> list[dict]:
    """补全检索结果元数据，并与校验后的引用信息对齐。"""
    document_pdf_assets = document_pdf_assets or {}
    citations_by_idx: dict[int, dict] = {}
    if answer:
        for cite in answer.get("verified_citations", []):
            idx = cite.get("chunk_index")
            if idx is not None:
                citations_by_idx[int(idx)] = cite

    enriched: list[dict] = []
    for i, raw in enumerate(results, 1):
        result = dict(raw)
        idx = int(result.get("chunk_index") or i)
        result["chunk_index"] = idx

        if metadata_index:
            result = metadata_index.enrich_result(result, idx)

        sha1 = str(result.get("source_sha1") or result.get("sha1") or "").strip()
        if sha1 and sha1 in document_pdf_assets:
            asset = document_pdf_assets[sha1]
            if not result.get("display_name"):
                result["display_name"] = asset.get("display_name", "")
            if not result.get("source_display_name"):
                result["source_display_name"] = asset.get("display_name", "")
            if not result.get("file_name"):
                result["file_name"] = asset.get("source_name", "")
            if not result.get("source_file"):
                result["source_file"] = asset.get("source_name", "")

        cite = citations_by_idx.get(idx)
        if cite:
            if cite.get("file_name"):
                result["display_name"] = cite["file_name"]
                result["source_display_name"] = cite["file_name"]
            if cite.get("page") is not None:
                result["page"] = cite["page"]
            if cite.get("sha1"):
                result["sha1"] = cite["sha1"]
                result["source_sha1"] = cite["sha1"]

        result["source_label"] = resolve_result_source_label(
            result, idx, document_pdf_assets, metadata_index
        )
        enriched.append(result)
    return enriched


@st.cache_data
def load_chunk_metadata_index(chunked_dir: str) -> dict:
    """加载 chunk 元数据索引（可序列化缓存）。"""
    index = ChunkMetadataIndex.from_chunked_dir(Path(chunked_dir))
    return {
        "by_sha1": index.by_sha1,
        "by_stem": index.by_stem,
        "by_sha1_chunk": {f"{a}|{b}": v for (a, b), v in index.by_sha1_chunk.items()},
    }


def get_chunk_metadata_index(chunked_dir: str) -> ChunkMetadataIndex:
    """从缓存还原 ChunkMetadataIndex 实例。"""
    cached = load_chunk_metadata_index(chunked_dir)
    index = ChunkMetadataIndex()
    index.by_sha1 = cached.get("by_sha1", {})
    index.by_stem = cached.get("by_stem", {})
    index.by_sha1_chunk = {}
    for key, val in cached.get("by_sha1_chunk", {}).items():
        sha1, chunk_id = key.rsplit("|", 1)
        try:
            index.by_sha1_chunk[(sha1, int(chunk_id))] = val
        except ValueError:
            continue
    return index


_CITATION_HANDLER_SCRIPT = r"""
<script>
(function () {
  function getTopWindow() {
    try { return window.top || window.parent || window; } catch (e) { return window; }
  }

  function findCitationTarget(refNum, rootDoc) {
    const selector = '[data-citation-ref="' + refNum + '"]';
    let     target = rootDoc.querySelector(selector);
    if (target) return target;
    target = rootDoc.querySelector('.citation-target-' + refNum);
    if (target) return target;
    target = rootDoc.getElementById('ref-' + refNum);
    if (target) return target;
    const iframes = rootDoc.querySelectorAll('iframe');
    for (let i = 0; i < iframes.length; i++) {
      try {
        const idoc = iframes[i].contentDocument;
        if (!idoc) continue;
        target = idoc.querySelector(selector) || idoc.querySelector('.citation-target-' + refNum)
              || idoc.getElementById('ref-' + refNum);
        if (target) return target;
      } catch (e) { /* cross-origin */ }
    }
    return null;
  }

  function openStreamlitExpander(expanderEl) {
    if (!expanderEl) return;
    const details = expanderEl.querySelector('details');
    if (details) {
      if (!details.open) {
        const summary = details.querySelector('summary');
        if (summary) summary.click();
        else details.open = true;
      }
      return;
    }
    const toggle = expanderEl.querySelector('[data-testid="stExpanderToggleIcon"]');
    if (toggle) {
      const btn = toggle.closest('button') || toggle;
      btn.click();
    }
  }

  function scrollToCitation(refNum) {
    const topWin = getTopWindow();
    const rootDoc = topWin.document;
    const target = findCitationTarget(String(refNum), rootDoc);
    if (!target) {
      console.warn('[RAG] citation target not found: ref-' + refNum);
      return false;
    }
    const expander = target.closest('[data-testid="stExpander"]');
    if (expander) openStreamlitExpander(expander);
    setTimeout(function () {
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      target.classList.add('citation-highlight');
      setTimeout(function () {
        target.classList.remove('citation-highlight');
      }, 2500);
    }, expander ? 350 : 0);
    return false;
  }

  const topWin = getTopWindow();
  topWin.ragScrollToCitation = scrollToCitation;

  function attachDirectHandlers(root) {
    if (!root) return;
    root.querySelectorAll('.citation-ref:not([data-rag-bound])').forEach(function (btn) {
      btn.setAttribute('data-rag-bound', '1');
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        const refNum = btn.getAttribute('data-ref') ||
          (btn.textContent || '').replace(/\D/g, '');
        if (refNum) scrollToCitation(refNum);
      });
    });
  }

  function scanAllDocs() {
    attachDirectHandlers(topWin.document);
    topWin.document.querySelectorAll('iframe').forEach(function (frame) {
      try {
        if (frame.contentDocument) attachDirectHandlers(frame.contentDocument);
      } catch (e) { /* cross-origin */ }
    });
  }

  function bindHandler(doc) {
    if (!doc || !doc.body) return;
    if (doc.body.dataset.ragCitationBound === '1') return;
    doc.body.dataset.ragCitationBound = '1';
    doc.addEventListener('click', function (event) {
      const btn = event.target.closest('.citation-ref');
      if (!btn) return;
      event.preventDefault();
      event.stopPropagation();
      const refNum = btn.getAttribute('data-ref') ||
        (btn.textContent || '').replace(/\D/g, '');
      if (refNum) scrollToCitation(refNum);
    }, true);
  }

  bindHandler(topWin.document);
  bindHandler(document);
  scanAllDocs();
  setInterval(scanAllDocs, 800);
})();
</script>
"""


def inject_citation_click_handler():
    """注册全局引用角标点击与滚动高亮（兼容 Streamlit iframe / expander）。"""
    components.html(_CITATION_HANDLER_SCRIPT, height=0, scrolling=False)


def render_answer_text_with_citations(text: str) -> str:
    """将答案中的 [1]、[2] 角标渲染为可点击高亮按钮（页内锚点跳转）。"""
    if not text:
        return ""

    parts = re.split(r"\[(\d+)\]", str(text))
    html_parts: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            html_parts.append(html.escape(part))
            continue
        ref_num = part
        html_parts.append(
            f'<button type="button" class="citation-ref" data-ref="{ref_num}" '
            f'title="跳转到参考片段 {ref_num}" '
            f'onclick="try{{(window.top||window).ragScrollToCitation(\'{ref_num}\')}}catch(e){{}};return false;">'
            f"[{ref_num}]</button>"
        )
    return "".join(html_parts)


def prepare_answer_for_display(answer: dict | None) -> dict:
    """展示前补全引用角标；未 grounded 时移除 LLM 幻觉角标。"""
    if not answer:
        return {}
    prepared = dict(answer)
    text = str(prepared.get("final_answer") or "").strip()

    if prepared.get("answer_grounded") is False:
        text = re.sub(r"\s*\[\d+\]", "", text).strip()
        prepared["final_answer"] = text or "未在检索资料中找到相关信息"
        return prepared

    indices = prepared.get("source_chunk_indices") or []
    verified = prepared.get("verified_citations") or []
    if not indices and verified:
        indices = [
            cite.get("chunk_index")
            for cite in verified
            if cite.get("chunk_index") is not None
        ]
    if indices and not re.search(r"\[\d+\]", text):
        markers = "".join(f"[{int(idx)}]" for idx in sorted({int(i) for i in indices}))
        prepared["final_answer"] = f"{text} {markers}".strip()
    return prepared


def render_page_display(page_raw) -> str:
    """页码纯文本展示（检索结果区不再打开新窗口）。"""
    page_display = format_page_display(page_raw)
    if page_display == "-":
        return "未知"
    return f"第 {page_display} 页"


def render_page_link(page_raw, pdf_url: str | None) -> str:
    page_display = format_page_display(page_raw)
    if pdf_url and page_display != "-":
        return (
            f'<a class="page-link" href="{html.escape(pdf_url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">第 {html.escape(page_display)} 页</a>'
        )
    return html.escape(page_display)


@st.cache_data
def load_companies(subset_path: str) -> list[str]:
    path = Path(subset_path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    return sorted(df["company_name"].dropna().unique().tolist())


def build_question(company: str, question: str) -> str:
    question = question.strip()
    if not question:
        return ""
    if company in question:
        return question
    return f"{company}，{question}"


def extract_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line)[:80]
    return (text.strip()[:60] + "…") if len(text.strip()) > 60 else text.strip()


def get_similarity(result: dict) -> float:
    if "combined_score" in result:
        return round(float(result["combined_score"]), 3)
    if "relevance_score" in result:
        return round(float(result["relevance_score"]), 3)
    distance = float(result.get("distance", 0))
    return round(1.0 / (1.0 + distance), 3)


# Pipeline 初始化很轻量，不使用 cache_resource，避免热重载后缓存旧类实例导致 AttributeError
def create_pipeline(
    llm_reranking: bool,
    top_n_retrieval: int,
    answering_model: str,
    rerank_model: str = RERANK_MODEL_DASHSCOPE,
    rerank_api_key: str | None = None,
) -> Pipeline:
    model = normalize_answering_model(answering_model)
    rerank_label = RERANK_MODEL_LABELS.get(rerank_model, rerank_model)
    config = RunConfig(
        use_serialized_tables=DEFAULT_RUN_CONFIG.use_serialized_tables,
        parent_document_retrieval=DEFAULT_RUN_CONFIG.parent_document_retrieval,
        llm_reranking=llm_reranking,
        llm_reranking_sample_size=DEFAULT_RUN_CONFIG.llm_reranking_sample_size,
        rerank_model=rerank_model,
        top_n_retrieval=top_n_retrieval,
        parallel_requests=1,
        submission_file=False,
        api_provider=DEFAULT_RUN_CONFIG.api_provider,
        answering_model=model,
        pipeline_details=f"Streamlit UI; {model} + {rerank_label}",
        config_suffix=DEFAULT_RUN_CONFIG.config_suffix,
        rerank_api_key=rerank_api_key or None,
    )
    return Pipeline(ROOT_PATH, run_config=config)


def get_pipeline(
    llm_reranking: bool,
    top_n_retrieval: int,
    answering_model: str,
    rerank_model: str = RERANK_MODEL_DASHSCOPE,
    rerank_api_key: str | None = None,
) -> Pipeline:
    return create_pipeline(
        llm_reranking, top_n_retrieval, answering_model, rerank_model, rerank_api_key
    )


def _get_active_processor(pipeline: Pipeline):
    return getattr(pipeline, "_stream_processor", None)


def read_pipeline_timings(pipeline: Pipeline) -> tuple[float, float, float]:
    """从 QuestionsProcessor 读取计时，兼容 Pipeline 包装方法缺失的情况。"""
    processor = _get_active_processor(pipeline)
    if processor is None:
        return 0.0, 0.0, 0.0

    if hasattr(processor, "get_last_retrieval_elapsed"):
        vector_elapsed = processor.get_last_retrieval_elapsed()
        rerank_elapsed = processor.get_last_rerank_elapsed()
        llm_elapsed = processor.get_last_llm_elapsed()
    else:
        vector_elapsed = getattr(processor, "_last_retrieval_elapsed", 0.0)
        rerank_elapsed = getattr(processor, "_last_rerank_elapsed", 0.0)
        llm_elapsed = getattr(processor, "_last_llm_elapsed", 0.0)
    return vector_elapsed, rerank_elapsed, llm_elapsed


def read_rerank_fallback(pipeline: Pipeline) -> tuple[bool, str | None]:
    """读取重排序是否触发熔断降级。"""
    if hasattr(pipeline, "get_last_rerank_fallback"):
        return pipeline.get_last_rerank_fallback(), pipeline.get_last_rerank_fallback_reason()
    processor = _get_active_processor(pipeline)
    if processor and hasattr(processor, "get_last_rerank_fallback"):
        return processor.get_last_rerank_fallback(), processor.get_last_rerank_fallback_reason()
    return False, None


def show_rerank_fallback_hint(fallback: bool, reason: str | None = None):
    """重排序熔断时在界面角落给出弱提示。"""
    if not fallback:
        return
    message = "重排序超时，已使用快速检索模式"
    if reason and "未配置" in reason:
        message = f"重排序未配置 API Key，已使用快速检索模式"
    st.toast(message, icon="⚡")
    st.markdown(
        f"""
        <div style="position:fixed;bottom:18px;right:18px;z-index:9999;
            background:rgba(251,191,36,0.15);border:1px solid rgba(251,191,36,0.45);
            color:#92400e;padding:8px 14px;border-radius:8px;font-size:0.82rem;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);max-width:320px;">
            ⚡ {html.escape(message)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_pipeline_results(pipeline: Pipeline) -> tuple[dict | None, list[dict]]:
    processor = _get_active_processor(pipeline)
    if processor is None:
        return None, []

    if hasattr(processor, "get_last_stream_answer"):
        answer = processor.get_last_stream_answer()
    else:
        answer = getattr(processor, "_last_stream_answer", None)

    if answer and answer.get("cited_retrieval_results"):
        results = answer["cited_retrieval_results"]
    elif hasattr(processor, "get_last_retrieval_results"):
        results = processor.get_last_retrieval_results()
    else:
        results = getattr(processor, "_last_retrieval_results", [])

    return answer, results


def resolve_display_retrieval_results(
    answer: dict | None, fallback_results: list[dict]
) -> list[dict]:
    """优先展示反向校验后的引用片段。"""
    if answer and answer.get("cited_retrieval_results"):
        return answer["cited_retrieval_results"]
    return fallback_results


def retrieve_documents(
    company: str,
    query: str,
    top_n: int,
    llm_reranking: bool,
    parent_pages: bool = False,
    rerank_model: str = RERANK_MODEL_DASHSCOPE,
    rerank_api_key: str | None = None,
) -> tuple[list[dict], float, float, bool, str | None]:
    """返回 (results, vector_db_elapsed, rerank_elapsed, rerank_fallback, fallback_reason)。"""
    vector_db_dir = ROOT_PATH / "databases" / "vector_dbs"
    documents_dir = ROOT_PATH / "databases" / "chunked_reports"
    rerank_fallback = False
    fallback_reason = None
    if llm_reranking:
        retriever = HybridRetriever(
            vector_db_dir,
            documents_dir,
            rerank_model=rerank_model,
            rerank_api_key=rerank_api_key or None,
        )
        results = retriever.retrieve_by_company_name(
            company_name=company,
            query=query,
            llm_reranking_sample_size=max(top_n * 3, 15),
            top_n=top_n,
            return_parent_pages=parent_pages,
        )
        vector_elapsed = getattr(retriever, "last_vector_db_elapsed", 0.0)
        rerank_elapsed = getattr(retriever, "last_rerank_elapsed", 0.0)
        rerank_fallback = getattr(retriever, "last_rerank_fallback", False)
        fallback_reason = getattr(retriever, "last_rerank_fallback_reason", None)
    else:
        retriever = VectorRetriever(vector_db_dir, documents_dir)
        results = retriever.retrieve_by_company_name(
            company_name=company,
            query=query,
            top_n=top_n,
            return_parent_pages=parent_pages,
        )
        vector_elapsed = getattr(retriever, "last_vector_db_elapsed", 0.0)
        rerank_elapsed = 0.0
    for i, result in enumerate(results, 1):
        result["chunk_index"] = i
    return results, vector_elapsed, rerank_elapsed, rerank_fallback, fallback_reason


def _render_result_cards(
    results: list[dict],
    document_pdf_assets: dict[str, dict[str, str]],
    metadata_index: ChunkMetadataIndex | None = None,
):
    """渲染检索/引用片段卡片列表。"""
    for i, result in enumerate(results, 1):
        heading = extract_heading(result.get("text", ""))
        page_raw = result.get("page")
        chunk_index = result.get("chunk_index", i)
        source_label = result.get("source_label") or resolve_result_source_label(
            result, chunk_index, document_pdf_assets, metadata_index
        )
        page_label = html.escape(render_page_display(page_raw))
        text = result.get("text", "")
        st.markdown(
            f"""
            <div class="result-card ref-card citation-target-{chunk_index}" id="ref-{chunk_index}" data-citation-ref="{chunk_index}">
                <h4>📄 结果 {chunk_index}</h4>
                <ul style="margin:0;padding-left:20px;color:#475569;">
                    <li><b>来源:</b> {html.escape(str(source_label))}</li>
                    <li><b>页码:</b> {page_label}</li>
                    <li><b>内容:</b> {html.escape(heading)}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_retrieval_snippet(text)
        if i < len(results):
            st.divider()


def render_retrieval_results_content(
    results: list[dict],
    company: str,
    question: str,
    elapsed: float,
    rerank_elapsed: float = 0.0,
    company_pdf_assets: dict[str, dict[str, str]] | None = None,
    document_pdf_assets: dict[str, dict[str, str]] | None = None,
    answer: dict | None = None,
    metadata_index: ChunkMetadataIndex | None = None,
    original_results: list[dict] | None = None,
):
    """渲染检索结果内容（供 expander 内部使用）。"""
    company_pdf_assets = company_pdf_assets or {}
    document_pdf_assets = document_pdf_assets or {}
    results = enrich_retrieval_results(
        results, document_pdf_assets, answer, metadata_index
    )

    citation_source = (answer or {}).get("citation_source", "")
    answer_grounded = (answer or {}).get("answer_grounded")

    if citation_source == "reverse_lookup":
        st.caption("✅ 以下片段由**全库反向校验**匹配，与答案内容一致（非原始向量 Top-K）。")
    elif citation_source == "retrieval_match":
        st.caption("✅ 以下片段与答案内容校验一致。")
    elif answer is not None and answer_grounded is False:
        st.warning("⚠️ 未能在语料库中匹配到答案依据，以下为原始向量检索结果（仅供参考）。")

    st.caption(f"🎯 向量检索完成（耗时: {elapsed:.2f} 秒）")
    if rerank_elapsed > 0:
        st.caption(f"🔀 重排序耗时: {rerank_elapsed:.2f} 秒（不计入向量检索）")
    if answer and answer.get("rerank_fallback"):
        st.caption("⚡ 重排序超时或失败，已降级为向量检索顺序。")
    st.markdown(
        f"🏢 **公司:** {company} &nbsp;&nbsp; ❓ **问题:** {question} &nbsp;&nbsp; "
        f"📄 **找到 {len(results)} 个相关文档片段**"
    )
    if answer_grounded is not False:
        st.caption("📎 点击答案中的 [1]、[2] 角标可跳转并高亮对应片段。")
    st.divider()
    _render_result_cards(results, document_pdf_assets, metadata_index)

    if original_results:
        enriched_original = enrich_retrieval_results(
            original_results, document_pdf_assets, None, metadata_index
        )
        with st.expander(
            f"原始向量检索（未引用，{len(enriched_original)} 个片段）",
            expanded=False,
        ):
            st.caption("以下为 LLM 作答前的向量 Top-K 检索结果，未必与最终答案引用一致。")
            _render_result_cards(enriched_original, document_pdf_assets, metadata_index)


def _extract_final_answer_partial(buffer: str) -> str:
    """从流式 JSON 片段中尽量提取 final_answer 字段的文本（支持未闭合字符串）。"""
    match = re.search(r'"final_answer"\s*:\s*"', buffer)
    if not match:
        return ""

    raw = buffer[match.end():]
    chars = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt == "n":
                chars.append("\n")
            elif nxt == "t":
                chars.append("\t")
            elif nxt == '"':
                chars.append('"')
            elif nxt == "\\":
                chars.append("\\")
            else:
                chars.append(nxt)
            i += 2
            continue
        if ch == '"':
            tail = raw[i + 1 :].lstrip()
            if not tail or tail[0] in ",}":
                break
        chars.append(ch)
        i += 1
    return "".join(chars)


def _stream_final_answer_text(pipeline: Pipeline, full_question: str, schema: str):
    """仅流式输出 final_answer 字段文本，过滤 JSON 元数据。"""
    buffer = ""
    emitted_len = 0
    for chunk in pipeline.answer_single_question_stream(full_question, kind=schema):
        buffer += chunk
        partial = _extract_final_answer_partial(buffer)
        if len(partial) > emitted_len:
            yield partial[emitted_len:]
            emitted_len = len(partial)


def _build_answer_card_html(text: str) -> str:
    """生成完整蓝色答案卡片 HTML（单块渲染，避免 Streamlit 打断结构）。"""
    body_html = render_answer_text_with_citations(text)
    return (
        '<div class="answer-hero-card">'
        f'<div class="answer-hero-body" id="final-answer-body">{body_html}</div>'
        "</div>"
    )


def render_answer_card_header():
    """标题与复制图标同一行，复制按钮叠在卡片右上角。"""
    title_col, copy_col = st.columns([11, 1])
    with title_col:
        st.markdown("### 最终答案")
    with copy_col:
        render_copy_icon_button()


def render_copy_icon_button():
    """卡片右上角复制图标，从主文档读取答案正文。"""
    components.html(
        """
        <div style="display:flex; justify-content:flex-end; align-items:center; width:100%; height:36px;">
          <button id="copy-icon-btn" title="复制内容" aria-label="复制内容"
            style="background:#f8fafc; border:1px solid #e2e8f0; cursor:pointer; padding:7px;
                   border-radius:8px; color:#475569; line-height:0; box-shadow:0 1px 2px rgba(15,23,42,0.06);"
            onmouseover="this.style.background='#eef2ff'; this.style.color='#4338ca'; this.style.borderColor='#c7d2fe';"
            onmouseout="this.style.background='#f8fafc'; this.style.color='#475569'; this.style.borderColor='#e2e8f0';">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </button>
        </div>
        <script>
          (function () {
            const btn = document.getElementById('copy-icon-btn');
            if (!btn) return;
            btn.addEventListener('click', function () {
              const doc = window.parent.document;
              const body = doc.getElementById('final-answer-body');
              const text = body ? body.innerText : '';
              if (!text) return;
              navigator.clipboard.writeText(text).then(function () {
                btn.title = '已复制！';
                setTimeout(function () { btn.title = '复制内容'; }, 2000);
              }).catch(function (err) { console.error(err); });
            });
          })();
        </script>
        """,
        height=40,
    )


def render_answer_card_with_copy(text: str):
    """渲染标题、复制图标与蓝色答案卡片。"""
    render_answer_card_header()
    st.markdown(_build_answer_card_html(text), unsafe_allow_html=True)


def stream_answer_card(pipeline: Pipeline, full_question: str, schema: str) -> str:
    """流式更新蓝色卡片内容，返回完整答案文本。"""
    card_placeholder = st.empty()
    accumulated = ""
    for chunk in _stream_final_answer_text(pipeline, full_question, schema):
        accumulated += chunk
        card_placeholder.markdown(_build_answer_card_html(accumulated), unsafe_allow_html=True)
    return accumulated


def render_answer_details(answer: dict, meta: dict | None = None):
    """渲染折叠详情与耗时（不含最终答案卡片）。"""
    step_by_step = answer.get("step_by_step_analysis", "") or "-"
    reasoning_summary = answer.get("reasoning_summary", "") or "-"
    verified_citations = answer.get("verified_citations", [])
    source_chunk_indices = answer.get("source_chunk_indices", [])
    unverified_sentences = answer.get("unverified_sentences", [])

    with st.expander("查看推理过程与引用详情", expanded=False):
        st.markdown("**分步推理**")
        st.info(step_by_step)
        st.markdown("**推理摘要**")
        st.success(reasoning_summary)
        st.markdown("**校验后引用（后端映射）**")
        if verified_citations:
            for cite in verified_citations:
                page_label = format_page_display(cite.get("page"))
                st.markdown(
                    f"- 片段 `[{cite.get('chunk_index')}]` → "
                    f"**{cite.get('file_name')}** 第 {page_label} 页"
                )
        elif source_chunk_indices:
            st.caption(f"片段编号: {source_chunk_indices}（未能映射到具体文档）")
        else:
            st.caption("无已验证引用（可能为模型自行生成）")
        if unverified_sentences:
            st.markdown("**未匹配到检索片段的内容**")
            for sentence in unverified_sentences[:5]:
                st.caption(f"· {sentence[:120]}…" if len(sentence) > 120 else f"· {sentence}")

    if meta:
        timing_parts = []
        if meta.get("elapsed") is not None:
            timing_parts.append(f"向量检索 {meta['elapsed']:.2f}s")
        if meta.get("rerank_elapsed", 0) > 0:
            timing_parts.append(f"重排序 {meta['rerank_elapsed']:.2f}s")
        if meta.get("answer_elapsed"):
            timing_parts.append(f"LLM 生成 {meta['answer_elapsed']:.2f}s")
        if meta.get("answering_model"):
            timing_parts.append(f"模型 {meta['answering_model']}")
        if timing_parts:
            st.caption("⏱️ " + " · ".join(timing_parts))


def render_final_answer_card(final_answer: str):
    """渲染带复制图标的最终答案卡片（静态全文）。"""
    render_answer_card_with_copy(final_answer)


def render_answer_grounding_banner(answer: dict):
    """展示答案是否已在语料中校验 grounding。"""
    if answer.get("answer_grounded") is False:
        st.warning(
            "⚠️ **未找到文档依据**：答案未能在语料库中匹配到对应片段，"
            "可能来自模型自身知识；下方为原始向量检索结果，仅供参考。"
        )
    elif answer.get("citation_source") == "reverse_lookup":
        st.caption(
            "✅ 引用来源已通过**全库反向校验**自动映射；"
            "点击答案中的 [1]、[2] 可跳转至对应片段。"
        )
    elif answer.get("citation_source") == "retrieval_match":
        st.caption(
            "✅ 引用来源已与检索片段校验一致；"
            "点击答案中的 [1]、[2] 可跳转至对应片段。"
        )


def render_answer(answer: dict, meta: dict | None = None):
    """以最终答案卡片优先、详情折叠的方式渲染生成结果（非流式回放）。"""
    prepared = prepare_answer_for_display(answer)
    render_answer_grounding_banner(prepared)
    final_answer = prepared.get("final_answer", "-") or "-"
    render_final_answer_card(final_answer)
    render_answer_details(prepared, meta)


def render_page_header(model_name: str):
    st.markdown(
        f"""
        <div class="rag-header">
            <h1>🚀 RAG Challenge 2 - {html.escape(model_name)} Powered</h1>
            <p>基于获奖 RAG 系统，向量检索 + 可配置重排序 + {html.escape(model_name)}</p>
            <div class="features">
                📖 支持年报智能问答 &nbsp;|&nbsp; ⚡ 向量检索 + DashScope/Jina 重排序 &nbsp;|&nbsp; 🎯 默认开放性文本回答
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    if "retrieval_results" not in st.session_state:
        st.session_state.retrieval_results = None
    if "original_retrieval_results" not in st.session_state:
        st.session_state.original_retrieval_results = None
    if "retrieval_meta" not in st.session_state:
        st.session_state.retrieval_meta = {}
    if "answer_result" not in st.session_state:
        st.session_state.answer_result = None
    if "answering_model" not in st.session_state:
        st.session_state.answering_model = DEFAULT_ANSWERING_MODEL
    else:
        st.session_state.answering_model = normalize_answering_model(
            st.session_state.answering_model
        )


def render_model_selector() -> str:
    current = normalize_answering_model(st.session_state.get("answering_model"))
    model_index = (
        AVAILABLE_ANSWERING_MODELS.index(current)
        if current in AVAILABLE_ANSWERING_MODELS
        else 0
    )
    selected = st.selectbox(
        "选择生成模型",
        options=AVAILABLE_ANSWERING_MODELS,
        index=model_index,
        help="qwen3.6-flash 更快更省；qwen3.7-plus 质量更高",
    )
    st.session_state.answering_model = selected
    return selected


def run_answer_generation(
    pipeline: Pipeline,
    full_question: str,
    schema: str,
    meta: dict | None = None,
    document_pdf_assets: dict[str, dict[str, str]] | None = None,
    metadata_index: ChunkMetadataIndex | None = None,
) -> tuple[dict | None, list[dict], float, float, float]:
    """流式展示 final_answer 文本，返回结构化结果与计时。"""
    render_answer_card_header()
    card_placeholder = st.empty()
    accumulated = ""
    for chunk in _stream_final_answer_text(pipeline, full_question, schema):
        accumulated += chunk
        card_placeholder.markdown(_build_answer_card_html(accumulated), unsafe_allow_html=True)

    answer, results = read_pipeline_results(pipeline)
    vector_elapsed, rerank_elapsed, llm_elapsed = read_pipeline_timings(pipeline)

    prepared_answer = prepare_answer_for_display(answer or {})
    final_text = prepared_answer.get("final_answer") or accumulated
    card_placeholder.markdown(_build_answer_card_html(final_text), unsafe_allow_html=True)

    details_meta = dict(meta or {})
    details_meta.update(
        {
            "elapsed": vector_elapsed,
            "rerank_elapsed": rerank_elapsed,
            "answer_elapsed": llm_elapsed,
        }
    )
    render_answer_grounding_banner(prepared_answer)
    render_answer_details(prepared_answer, details_meta)
    display_results = resolve_display_retrieval_results(prepared_answer, results)
    enriched_results = enrich_retrieval_results(
        display_results, document_pdf_assets or {}, prepared_answer, metadata_index
    )
    return prepared_answer, enriched_results, vector_elapsed, rerank_elapsed, llm_elapsed


def _retrieval_expander_title(answer_obj: dict | None, count: int) -> str:
    if answer_obj and answer_obj.get("answer_grounded") and answer_obj.get("cited_retrieval_results"):
        return f"引用来源（{count} 个已验证片段）"
    if answer_obj and answer_obj.get("answer_grounded") is False:
        return f"原始检索（{count} 个片段，未找到文档依据）"
    return f"检索结果（{count} 个文档片段）"


def main():
    st.set_page_config(
        page_title="RAG Challenge 2 - 年报问答",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()
    companies = load_companies(str(ROOT_PATH / "subset.csv"))
    company_pdf_assets = load_company_pdf_assets(str(ROOT_PATH / "subset.csv"))
    document_pdf_assets = load_document_pdf_assets(
        str(ROOT_PATH / "databases" / "chunked_reports")
    )
    metadata_index = get_chunk_metadata_index(
        str(ROOT_PATH / "databases" / "chunked_reports")
    )
    all_pdf_assets = {**document_pdf_assets, **company_pdf_assets}
    ensure_static_pdfs(all_pdf_assets)
    pdf_port = start_pdf_server()
    if pdf_port:
        st.session_state["pdf_server_port"] = pdf_port

    with st.sidebar:
        st.markdown("### 查询设置")

        answering_model = render_model_selector()

        company = st.selectbox("选择公司", companies, index=0)
        user_question = st.text_area(
            "输入问题",
            "中芯国际在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？",
            height=72,
        )
        question_kind = st.selectbox(
            "问题类型",
            options=list(QUESTION_KIND_LABELS.keys()),
            format_func=lambda x: QUESTION_KIND_LABELS[x],
            index=0,
        )
        llm_reranking = st.checkbox("启用重排序", value=True)
        rerank_model = st.selectbox(
            "重排序模型选择",
            options=[RERANK_MODEL_DASHSCOPE, RERANK_MODEL_JINA],
            format_func=lambda x: RERANK_MODEL_LABELS[x],
            index=0,
            disabled=not llm_reranking,
        )
        with st.expander("高级设置", expanded=False):
            enable_stream = st.checkbox("启用流式输出", value=True)
            top_n = st.slider("检索文档数量", min_value=1, max_value=10, value=3)
            rerank_api_key = st.text_input(
                "Rerank API Key（可选，留空读环境变量）",
                type="password",
                help="DashScope 选 DASHSCOPE_API_KEY，Jina 选 JINA_API_KEY；此处可临时覆盖。",
            )

        col1, col2 = st.columns(2)
        with col1:
            search_btn = st.button("搜索文档", use_container_width=True)
        with col2:
            generate_btn = st.button("生成答案", type="primary", use_container_width=True)

    render_page_header(answering_model)

    full_question = build_question(company, user_question)
    schema = KIND_MAP[question_kind]
    rerank_key = (rerank_api_key or "").strip() or None

    if search_btn:
        if not full_question:
            st.warning("请输入问题。")
        else:
            with st.spinner("正在检索相关文档..."):
                try:
                    results, vector_elapsed, rerank_elapsed, rerank_fallback, fallback_reason = retrieve_documents(
                        company=company,
                        query=full_question,
                        top_n=top_n,
                        llm_reranking=llm_reranking,
                        rerank_model=rerank_model,
                        rerank_api_key=rerank_key,
                    )
                    show_rerank_fallback_hint(rerank_fallback, fallback_reason)
                    st.session_state.retrieval_results = enrich_retrieval_results(
                        results, document_pdf_assets, None, metadata_index
                    )
                    st.session_state.original_retrieval_results = None
                    st.session_state.retrieval_meta = {
                        "company": company,
                        "question": full_question,
                        "elapsed": vector_elapsed,
                        "rerank_elapsed": rerank_elapsed,
                        "rerank_fallback": rerank_fallback,
                    }
                    st.session_state.answer_result = None
                except Exception as e:
                    st.error(f"检索失败: {e}")

    if generate_btn:
        if not full_question:
            st.warning("请输入问题。")
        else:
            try:
                pipeline = get_pipeline(
                    llm_reranking,
                    top_n,
                    answering_model,
                    rerank_model=rerank_model,
                    rerank_api_key=rerank_key,
                )
                if enable_stream:
                    answer, results, retrieve_elapsed, rerank_elapsed, answer_elapsed = run_answer_generation(
                        pipeline=pipeline,
                        full_question=full_question,
                        schema=schema,
                        meta={
                            "company": company,
                            "question": full_question,
                            "answering_model": answering_model,
                        },
                        document_pdf_assets=document_pdf_assets,
                        metadata_index=metadata_index,
                    )
                    st.session_state.answer_rendered_this_run = True
                else:
                    with st.spinner("正在检索并生成答案，请稍候..."):
                        answer = pipeline.answer_single_question(full_question, kind=schema)
                        if isinstance(answer, str):
                            try:
                                answer = json.loads(answer)
                            except json.JSONDecodeError:
                                answer = {"final_answer": answer}
                        answer_from_processor, results = read_pipeline_results(pipeline)
                        if answer_from_processor:
                            answer = answer_from_processor
                        answer = prepare_answer_for_display(answer if isinstance(answer, dict) else {"final_answer": str(answer)})
                        retrieve_elapsed, rerank_elapsed, answer_elapsed = read_pipeline_timings(pipeline)
                        results = enrich_retrieval_results(
                            results, document_pdf_assets, answer, metadata_index
                        )

                rerank_fallback, fallback_reason = read_rerank_fallback(pipeline)
                show_rerank_fallback_hint(rerank_fallback, fallback_reason)
                if isinstance(answer, dict):
                    answer["rerank_fallback"] = rerank_fallback
                st.session_state.answer_result = answer
                st.session_state.retrieval_results = enrich_retrieval_results(
                    resolve_display_retrieval_results(
                        answer if isinstance(answer, dict) else None, results
                    ),
                    document_pdf_assets,
                    answer if isinstance(answer, dict) else None,
                    metadata_index,
                )
                if isinstance(answer, dict):
                    st.session_state.original_retrieval_results = answer.get(
                        "original_retrieval_results"
                    )
                st.session_state.retrieval_meta = {
                    "company": company,
                    "question": full_question,
                    "elapsed": retrieve_elapsed,
                    "rerank_elapsed": rerank_elapsed,
                    "answer_elapsed": answer_elapsed,
                    "answering_model": answering_model,
                    "rerank_fallback": rerank_fallback,
                }
            except Exception as e:
                st.error(f"生成答案时出错: {e}")

    # 1. 优先生成答案（视觉焦点）
    if st.session_state.answer_result is not None:
        if not st.session_state.pop("answer_rendered_this_run", False):
            render_answer(st.session_state.answer_result, st.session_state.retrieval_meta)

    # 2. 检索结果置底，默认折叠
    if st.session_state.retrieval_results is not None:
        meta = st.session_state.retrieval_meta
        answer_obj = st.session_state.answer_result
        count = len(st.session_state.retrieval_results)
        expander_title = _retrieval_expander_title(answer_obj, count)
        show_original = (
            answer_obj
            and answer_obj.get("answer_grounded")
            and st.session_state.get("original_retrieval_results")
        )
        with st.expander(expander_title, expanded=False):
            render_retrieval_results_content(
                st.session_state.retrieval_results,
                meta.get("company", company),
                meta.get("question", full_question),
                meta.get("elapsed", 0),
                meta.get("rerank_elapsed", 0.0),
                company_pdf_assets=company_pdf_assets,
                document_pdf_assets=document_pdf_assets,
                answer=st.session_state.answer_result,
                metadata_index=metadata_index,
                original_results=(
                    st.session_state.original_retrieval_results if show_original else None
                ),
            )

    if st.session_state.answer_result is None and st.session_state.retrieval_results is None:
        st.info("👈 请在左侧输入问题，点击【搜索文档】预览检索结果，或点击【生成答案】获取完整回答。")

    inject_citation_click_handler()


if __name__ == "__main__":
    main()
