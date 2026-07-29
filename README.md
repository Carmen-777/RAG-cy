# RAG-cy — 中文金融研报智能问答

基于 [RAG Challenge 2](https://github.com/IlyaRice/RAG-Challenge-2) 获奖方案深度改造的中文 RAG 系统，面向 A 股公司年报、研报等多文档场景，提供 **Streamlit 交互式问答** 与完整离线数据处理流水线。

## 功能亮点

| 模块 | 说明 |
|------|------|
| **Streamlit 问答 UI** | 公司选择、问题类型、检索预览、一键生成答案 |
| **流式输出** | 最终答案逐字展示，隐藏冗长 JSON 推理过程 |
| **Qwen 多模型** | 默认 `qwen3.6-flash`（更快），可选 `qwen3.7-plus`（更高质量） |
| **混合检索** | 向量 + 词法融合，精确 chunk 检索（非整页 parent 模式） |
| **可配置重排序** | DashScope `qwen3-vl-rerank`（国内默认）/ Jina Reranker，**3 秒超时熔断**自动降级 |
| **引用校验** | 全库反向匹配真实来源，后端注入 `[1][2]` 角标，点击跳转高亮片段 |
| **元数据双保险** | Chunk 级 `source_*` 字段 + 全局索引，来源显示真实文档名 |

## 系统架构（简图）

```
用户问题 → 向量检索 (+ 词法融合) → [可选] Rerank → LLM 生成答案
                ↓                                      ↓
         检索片段 [1][2]…                      引用反向校验 → 展示 & 角标跳转
```

## 环境要求

- Python 3.10+
- Windows / macOS / Linux
- 阿里云 **DashScope API Key**（必需）
- Jina API Key（仅在使用 Jina 重排序时需要）

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd RAG-cy
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 3. 配置环境变量

```bash
# 复制模板（不要提交 .env）
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

编辑 `.env`，至少填写：

```env
DASHSCOPE_API_KEY=你的DashScope密钥
```

可选：使用 Jina 重排序时添加 `JINA_API_KEY`。

> **安全提示**：项目根目录下的 `env` 文件为历史模板，已被 `.gitignore` 忽略。请统一使用 `.env`，**切勿**将含真实密钥的文件提交到 Git。

### 4. 准备数据

数据目录默认：`data/stock_data/`。

1. **公司索引**：复制示例并补全你的公司列表  
   ```bash
   copy data\stock_data\subset.csv.example data\stock_data\subset.csv
   ```
   `subset.csv` 格式：`sha1,file_name,company_name`

2. **预处理产物**（需自行生成或从备份恢复，已在 `.gitignore` 中排除）：
   - `databases/chunked_reports/` — 切块 JSON
   - `databases/vector_dbs/` — FAISS 向量索引
   - `pdf_reports/` — 原始 PDF（可选）

首次构建流水线可参考 `src/pipeline.py` 中的阶段注释，或使用竞赛原版 `main.py` CLI。

**Chunk 元数据补全**（已有切块 JSON 时，无需重新向量化）：

```bash
python -m src.chunk_metadata
```

### 5. 启动 Streamlit

```bash
streamlit run app_streamlit.py
```

浏览器打开 `http://localhost:8501`。

## Streamlit 使用说明

**侧边栏**

- **选择公司** / **输入问题** / **问题类型**（默认开放性文本 `string`）
- **重排序模型选择**：`DashScope (国内极速)` / `Jina (高精度)`
- **启用重排序**：关闭则仅向量检索
- **高级设置**：流式输出、检索数量、可选 Rerank API Key 覆盖

**主界面**

- **搜索文档**：仅检索，底部展开查看片段
- **生成答案**：检索 + LLM 回答；答案区支持 `[1][2]` 点击跳转对应引用卡片
- 重排序超时会在右下角提示「已使用快速检索模式」，不阻塞页面

## 环境变量一览

| 变量 | 必需 | 用途 |
|------|------|------|
| `DASHSCOPE_API_KEY` | ✅ | 问答、Embedding、DashScope Rerank |
| `JINA_API_KEY` | 可选 | Jina 重排序 |
| `OPENAI_API_KEY` | 可选 | OpenAI provider |
| `GEMINI_API_KEY` | 可选 | Gemini provider |
| `IBM_API_KEY` | 可选 | 竞赛遗留 IBM 配置 |
| `MINERU_API_KEY` | 可选 | MinerU PDF 解析 |

完整说明见 [`.env.example`](.env.example)。

## 项目结构

```
RAG-cy/
├── app_streamlit.py          # Streamlit 主入口
├── main.py                   # CLI 入口（竞赛原版）
├── requirements.txt
├── .env.example              # 环境变量模板
├── data/stock_data/          # 默认数据集目录
│   ├── subset.csv.example
│   └── databases/            # 本地生成，不提交 Git
└── src/
    ├── pipeline.py           # 离线流水线
    ├── retrieval.py          # 向量 / 混合检索
    ├── reranking.py          # DashScope / Jina 统一重排
    ├── rerank_config.py      # 重排模型常量
    ├── citation_verification.py  # 引用反向校验
    ├── chunk_metadata.py     # Chunk 来源元数据
    ├── questions_processing.py
    ├── api_requests.py       # DashScope / OpenAI 调用
    └── prompts.py            # RAG Prompt 体系
```

## 离线流水线（可选）

编辑 `src/pipeline.py` 末尾 `RunConfig`，取消注释对应阶段后：

```bash
python src/pipeline.py
```

或使用 `main.py`（需在数据目录下执行，参见竞赛原版文档）。

## 常见问题

**Q: 重排序一直降级为向量模式？**  
A: 检查 `DASHSCOPE_API_KEY` 或 `JINA_API_KEY`；默认 3 秒超时，网络慢时会自动熔断。

**Q: 来源显示「参考文档 N」？**  
A: 运行 `python -m src.chunk_metadata` 补全切块元数据，并重启 Streamlit 清缓存。

**Q: 导入 `RERANK_MODEL_DASHSCOPE` 报错？**  
A: 重启 Streamlit；常量已迁移至 `src/rerank_config.py` 避免热重载循环导入。

## 致谢

本项目基于 [IlyaRice/RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2) 扩展，面向中文研报场景做了检索、引用、UI 与模型接入等多方面优化。

## License

MIT（与上游保持一致）
