"""重排序配置常量（无第三方依赖，避免循环导入）。"""

RERANK_TIMEOUT_SECONDS = 3.0

RERANK_MODEL_DASHSCOPE = "dashscope"
RERANK_MODEL_JINA = "jina"

RERANK_MODEL_LABELS = {
    RERANK_MODEL_DASHSCOPE: "DashScope (国内极速)",
    RERANK_MODEL_JINA: "Jina (高精度)",
}

DASHSCOPE_RERANK_MODEL = "qwen3-vl-rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
