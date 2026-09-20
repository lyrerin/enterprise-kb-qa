# 企业知识库智能问答系统（RAG + Agent）

面向企业内部文档的问答系统。上传 PDF / Word / TXT / CSV → 自动解析、按语义单元切分、向量化入库；提问时做**混合路由**（非结构化文档走 RAG，结构化表格走 Text-to-SQL），由大模型生成**带引用来源**的答案。

**技术栈**：Python 3.11+ · FastAPI · LangChain / LangGraph · ChromaDB · SQLite · Streamlit · 本地 HuggingFace 向量模型（无需在线 embedding API）

---

## 核心亮点

### 1. 分块策略路由 —— 按语义单元切，不按字符数切

不再全局一套字符切块器，而是先判文档类型再路由：

| 类型 | 判定方式 | 切法 |
| --- | --- | --- |
| **手册类** | 正文出现 ≥3 个「第X章/节/条」 | 按章节标题切，并**把章节标题拼进块首**（`section_path`），避免 200 字碎片丢掉主语 |
| **表格类** | 扩展名 `.csv` | 整块不切，一行一个块 |
| **其余** | 兜底 | 递归字符切（200 字 / 重叠 30） |

实测把《员工手册》切成 5 个章节块 + 1 个前言块，**每块开头都带 `[第一章：考勤制度]`**。

### 2. 父子块 small-to-big —— 小块检索，大块喂模型

小块（≤400 字）保证命中精准；命中后按 `parent_id` 回表取**整节**喂给模型，保证上下文完整。

- 父块存第二个库 `parent.db`（SQLite），`parent_id` 用**确定性 id** `f"{source}#{i}"` —— 重导同一文件覆盖同一行，不会越积越多
- 回表时按检索相关性顺序**去重**、限制总长 4000 字；无父块的块（CSV 行）用原文兜底
- **实测：喂给模型的 context 从 173 字涨到 580 字**（约 3 倍 token，所以上限不能省）

### 3. CSV → Text-to-SQL 路由

「平均单价最高的类别是哪个」这类**聚合问题，向量检索原理上做不到**。系统把 CSV 读进**内存 SQLite 副本**，让 LLM 生成 SQL 后只读执行：

- **三层安全**：只允许 `SELECT` 开头 + 关键字黑名单（`DROP/DELETE/UPDATE/INSERT/ALTER/...` 与 `;`）+ 在**内存副本**上执行，不碰真实库
- 表名由服务端从目录扫描得出，**不接受用户传表名**
- 实测生成：`SELECT "产品类别" FROM "产品列表" GROUP BY "产品类别" ORDER BY AVG("单价") DESC LIMIT 1` → 正确答出「软件许可」

### 4. Function Calling Agent + 工具安全

LLM 自主决定调哪个工具（检索知识库 / 查数据表 / 算数 / 报时间），**执行在本方进程内**（`TOOL_MAP` 查表即白名单）。

- **工具结果回流**：工具用 `response_format="content_and_artifact"` 返回 `(正文, 来源)` —— 正文喂模型，来源走 artifact 通道给前端显示出处（引用来源、生成的 SQL 都能回到界面）
- **AST 白名单替代 `eval`**：`caculator` 只允许数字字面量和算术运算，执行时清空 `__builtins__`。**12 条注入用例全部拦下**（`__import__('os').system(...)`、`open(...)`、属性访问等）
- **轮数上限**：模型一直要工具时，到 `MAX_TOOL_ROUNDS` 就换成不绑工具的模型收尾 —— 不加这个，`while` 循环没有出口，接口永远不返回

### 5. LangGraph 状态图

把手写的 `while response.tool_calls` 循环重构成**显式状态图**：条件入口 → `retrieve`（完整 RAG）/ `answer`（不查库直接回答）→ END。

- `State` 用 `add_messages` 管消息（节点必须返回 Message 对象，塞字符串会 `InvalidUpdateError`）
- 路由规则是**可读、可改、可打日志**的函数，而不是藏在模型内部
- 引用来源一起带出图

---

## 实测数据

### 检索策略横评（39 题 × 3 策略，脚本 `eval/compare_strategies.py`）

| 策略 | recall@1 | recall@3 | precision@3 | MRR | 平均耗时 |
| --- | --- | --- | --- | --- | --- |
| `vector` | 0.9231 | 1.0000 | 0.9145 | 0.9615 | **33 ms** |
| **`rerank`（当前默认）** | **0.9744** | 1.0000 | **0.9231** | **0.9872** | 1410 ms |
| `hybrid`（向量 + BM25，RRF 融合） | 0.8974 | 1.0000 | 0.8462 | 0.9402 | 28 ms |

**结论**：`rerank` 效果最好但**慢 50 倍**；`hybrid` 反而**最差** —— 在 27 块的小语料上，RRF 会把「关键词像但语义不对」的块拱进前 3（precision@3 掉到 0.8462）。这个结论是跑出来的，不是拍脑袋。完整报告见 `eval/reports/`。

### 其他实跑数字

| 指标 | 数值 |
| --- | --- |
| 知识库规模 | 5 文档 / 27 块 |
| 父块数 | 11 行 |
| 父子块对 context 的影响 | 173 字 → **580 字**（约 3 倍 token） |
| 向量模型 | 本地 `BAAI/bge-small-zh-v1.5`，**512 维，全程离线** |
| 重排模型 | `BAAI/bge-reranker-base`，懒加载单例 |
| 评估集 | 39 题，含 5 道 Text-to-SQL 聚合题 |
| 向量模型缓存效果 | 服务日志里模型加载从"每次问答一次"降到**全局一次** |

---

## 系统架构

```
              ┌────────────────────── 入库链路 ──────────────────────┐
上传 PDF/Word/TXT/CSV
   → 解析（页眉页脚去重 / 表格转 Markdown）
   → 判 doc_type（数「第X章」/ 看扩展名）
   → 分块策略路由
        ├─ handbook → 按章节切 + 存父块 + 子块带 parent_id
        ├─ table    → 整块不切
        └─ default  → 递归字符切
   → 本地向量化（bge-small-zh-v1.5，512 维）
   → 先删同源旧块 → 写入 ChromaDB
              └──────────────────────────────────────────────────┘

              ┌────────────────────── 问答链路 ──────────────────────┐
用户提问  →  四条入口任选：
              ├─ /api/qa/ask         固定 RAG（含 SQL 分叉 + 父块展开）
              ├─ /api/qa/agent       Function Calling，模型自选工具
              ├─ /api/qa/graph       LangGraph 状态图
              └─ /api/qa/ask/stream  流式（SSE）

              ├─ SQL 分叉：命中聚合词 + 表名 → 内存 SQLite 只读执行
              └─ RAG 路径：粗筛 RECALL_K=20 → rerank 精排取 FINAL_K=3
                              → 子块回表换整节父块（≤4000 字）
                              → 拼 prompt → LLM 生成带引用标注的答案
              └──────────────────────────────────────────────────┘
```

---

## 目录结构

```
kbqa/
├── app/
│   ├── __init__.py            # 全局设置：HF 离线开关、stdout 编码
│   ├── main.py                # FastAPI 入口 + 异常分层 handler
│   ├── config.py              # 配置收口（全部走环境变量）
│   ├── api/
│   │   ├── documents.py       # 上传 / 批量上传 / 状态 / 清空
│   │   ├── qa.py              # 6 条问答入口
│   │   └── auth.py            # API Key 认证依赖
│   ├── core/
│   │   ├── parsers.py         # 各格式解析 + 页眉页脚去重
│   │   ├── loader.py          # 按扩展名分发解析器
│   │   ├── splitter.py        # 分块策略路由 + 章节切分 + 父块落库
│   │   ├── parent_store.py    # 父块存取（SQLite）
│   │   ├── embeddings.py      # 本地向量模型（带缓存 + 线程锁）
│   │   ├── vectorstore.py     # ChromaDB 读写
│   │   ├── ingestion.py       # 入库编排（按 source 分组判类型）
│   │   ├── retriever.py       # 四路检索策略统一入口 + 父块展开
│   │   ├── reranker.py        # 交叉编码器重排
│   │   ├── sql_tool.py        # CSV → 内存 SQLite → 生成并只读执行 SQL
│   │   ├── router.py          # RAG / SQL 路由判定
│   │   ├── rag_chain.py       # 固定 RAG 链路
│   │   ├── agent.py           # Function Calling Agent + 工具定义 + AST 安全
│   │   ├── graph.py           # LangGraph 状态图
│   │   ├── memory_manager.py  # 多轮会话（SQLite 持久化）
│   │   └── exception.py       # 业务异常分层
│   ├── models/schemas.py      # Pydantic 模型
│   └── utils/logger.py        # 日志（控制台 + 文件轮转）
├── eval/
│   ├── eval_set.json          # 39 题评估集
│   ├── compare_strategies.py  # 多策略横评（不调 LLM，确定性可重复）
│   ├── evaluate.py            # 端到端评估（含忠实度）
│   └── reports/               # 实测报告落盘
├── streamlit_app.py           # 前端
├── knowledge_docs/            # 待入库语料（.gitignore）
├── uploads/                   # 上传暂存（.gitignore）
├── requirements.txt
├── Dockerfile / docker-compose.yml
```

---

## 快速开始

### 1. 环境要求

- Python 3.11+
- 一个 **DeepSeek API Key**（唯一需要的外部凭证）
- 向量模型与重排模型在**本地运行**，不需要在线 embedding key

### 2. 安装

```bash
git clone https://gitee.com/changlailin/kbqa.git
cd kbqa
python -m venv .venv
.venv\Scripts\activate          # Windows；Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 下载模型（仅首次；国内直连 huggingface.co 不通，用镜像）

```bash
set HF_ENDPOINT=https://hf-mirror.com
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-base')"
```

> 下完之后代码里会强制 `HF_HUB_OFFLINE=1`，全程读本地缓存，不再联网。

### 4. 配置

在项目根目录建 `.env`：

```env
DEEPSEEK_API_KEY=你的Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

### 5. 导入语料

把你的文档放进 `knowledge_docs/`，然后：

```bash
python -m app.core.ingestion
```

### 6. 启动

```bash
# 后端
uvicorn app.main:app --port 8000
# 前端（另开一个终端）
streamlit run streamlit_app.py
```

- Swagger 文档：http://127.0.0.1:8000/docs
- 前端界面：http://127.0.0.1:8501

---

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 + 知识库状态 |
| POST | `/api/documents/upload` | 上传单个文档（增量入库） |
| POST | `/api/documents/upload/batch` | 批量上传 |
| GET | `/api/documents/status` | 文档数 / 块数 / 来源列表 |
| DELETE | `/api/documents/clear` | 清空知识库（同时清父块） |
| POST | `/api/qa/ask` | 单轮问答（**带引用来源**，含 SQL 分叉） |
| POST | `/api/qa/chat` | 多轮对话（SQLite 持久化记忆） |
| POST | `/api/qa/agent` | Function Calling Agent（模型自选工具） |
| POST | `/api/qa/agent/chat` | 带记忆的 Agent 对话 |
| POST | `/api/qa/graph` | LangGraph 状态图问答 |
| POST | `/api/qa/ask/stream` | 流式问答（SSE） |
| GET | `/api/qa/history/{session_id}` | 查看会话历史 |
| DELETE | `/api/qa/history/{session_id}` | 清除会话历史 |
| GET | `/api/auth/me` | 获取当前用户（需 `X-API-Key`） |

---

## 配置说明

| 配置项 | 位置 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | `.env` | 必填 | |
| `DEEPSEEK_MODEL` | `.env` | `deepseek-v4-flash` | |
| `EMBEDDING_MODEL` | `app/config.py` | `BAAI/bge-small-zh-v1.5` | 本地模型，512 维 |
| `RETRIEVAL_STRATEGY` | `app/config.py` | `rerank` | `vector` / `rerank` / `hybrid` / `rewrite` |
| `RECALL_K` / `FINAL_K` | `app/config.py` | `20` / `3` | 粗筛数 / 精排后保留数 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `app/config.py` | `200` / `30` | 兜底字符切块参数 |
| `HISTORY_WINDOW_TURNS` | `app/config.py` | `6` | 多轮窗口（轮） |
| `MAX_TOOL_ROUNDS` | `app/config.py` | `2` | Agent 工具调用轮数上限 |

---

## Docker 部署

```bash
docker compose up -d
```

- 后端 API：http://localhost:8000
- 前端界面：http://localhost:8501

> 模型在容器内跑，建议挂载宿主的 HuggingFace 缓存目录，并设 `HF_HUB_OFFLINE=1`，避免每次起容器重下模型。
