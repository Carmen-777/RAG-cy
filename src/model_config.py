"""DashScope 回答模型配置，供 Streamlit UI 与 API 层共用。"""

# 侧边栏可选模型；后续扩展在此追加即可
AVAILABLE_ANSWERING_MODELS = [
    "qwen3.6-flash",
    "qwen3.7-plus",
]

DEFAULT_ANSWERING_MODEL = "qwen3.6-flash"

# 必须走 MultiModalConversation 接口的模型（走 Generation 会报 url error）
DASHSCOPE_MULTIMODAL_MODELS = frozenset({
    "qwen3.7-plus",
    "qwen3.7-max",
    "qwen3.6-flash",
    "qwen3.6-plus",
})


def dashscope_uses_multimodal_api(model: str) -> bool:
    """Qwen3.5+ 系列均需 MultiModalConversation；旧版 Generation 接口会报 url error。"""
    if model in DASHSCOPE_MULTIMODAL_MODELS:
        return True
    return model.startswith(("qwen3.5-", "qwen3.6-", "qwen3.7-"))


def dashscope_disable_thinking_for_rag(model: str) -> bool:
    """Qwen3.5/3.6/3.7 混合思考模型默认开启 thinking，RAG 结构化回答建议关闭以提速。"""
    return dashscope_uses_multimodal_api(model)


def normalize_answering_model(model: str | None) -> str:
    if model and model in AVAILABLE_ANSWERING_MODELS:
        return model
    return DEFAULT_ANSWERING_MODEL
