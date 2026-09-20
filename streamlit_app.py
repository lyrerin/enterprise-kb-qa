import streamlit as st
import requests
import uuid
from app.core.parsers import SUPPORTED_EXTENSIONS

API_BASE = "http://localhost:8000"

# 上传框重置计数器（动态 key 用）
if "uploader_reset" not in st.session_state:
    st.session_state.uploader_reset = 0

st.set_page_config(
    page_title="企业知识库问答系统",
    page_icon="📚",
    layout="wide",
)

st.title("📚 企业知识库智能问答系统")
st.caption('基于 RAG + LangChain + FastAPI | 上传文档 → 智能问答')

# ====== 侧边栏：文档管理 ======
with st.sidebar:
    st.header("📁 文档管理")
  
    # 上传文档（key 动态变化，上传完成后强制重建上传框实现清空）
    uploaded_file = st.file_uploader(
        "上传知识文档",
        type=SUPPORTED_EXTENSIONS,
        help="支持 PDF、Word、TXT、CSV 格式",
        key=f"file_uploader_{st.session_state.uploader_reset}",
    )

    if uploaded_file:
        with st.spinner("正在处理文档..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            resp = requests.post(f"{API_BASE}/api/documents/upload", files=files)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ {data['message']}")
            else:
                st.error(f"❌ 上传失败: {resp.text}")
        # 计数器 +1 → key 改变 → 上传框被当作全新组件重建，自动清空
        st.session_state.uploader_reset += 1
        st.rerun()
  
    st.divider()
  
    # 知识库状态
    if st.button("🔄 刷新知识库状态"):
        st.rerun()
  
    try:
        resp = requests.get(f"{API_BASE}/api/documents/status")
        if resp.status_code == 200:
            status = resp.json()
            st.metric("文档数", status["total_documents"])
            st.metric("文本块", status["total_chunks"])
            if status["sources"]:
                st.write("**已导入文档:**")
                for src in status["sources"]:
                    st.write(f"  📄 {src}")
        else:
            st.warning("知识库为空，请上传文档")
    except Exception:
        st.warning("⚠️ 无法连接到后端服务")
  
    st.divider()
  
    # 清除按钮
    if st.button("🗑️ 清空知识库", type="secondary"):
        resp = requests.delete(f"{API_BASE}/api/documents/clear")
        if resp.status_code == 200:
            st.success("知识库已清空")
            st.rerun()
        else:
            st.error(f"❌ 清空失败: {resp.text}")

# ====== 主区域：问答界面 ======

# 初始化会话
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("used_tool"):
            st.caption(f"🔧 工具调用: {', '.join(msg['used_tool'])}")
        if msg.get("sources"):
            with st.expander("📎 引用来源"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"[{i}] {src['source']}")
                    st.text(src["content"][:200])

# 输入框
if question := st.chat_input("输入你的问题，按 Enter 发送..."):
    # 显示用户消息
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})
  
    # 获取回答（Agent 模式：LLM 自主决策是否调用 检索知识库/计算/时间 等工具）
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            resp = requests.post(
                f"{API_BASE}/api/qa/agent/chat",
                json={
                    "question": question,
                    "session_id": st.session_state.session_id,
                },
            )
          
            if resp.status_code == 200:
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                used_tool = data.get("used_tool", [])
              
                st.write(answer)
              
                if used_tool:
                    with st.expander(f"🔧 已调用工具: {', '.join(used_tool)}"):
                        st.caption("LLM 自主决策调用了以上工具来回答问题")
              
                if sources:
                    with st.expander("📎 引用来源"):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"[{i}] 来源: {src['source']}")
                            st.text(src["content"][:200])
              
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "used_tool": used_tool,
                })
            else:
                st.error(f"问答失败: {resp.text}")

# 侧边栏底部：会话管理
with st.sidebar:
    st.divider()
    st.caption(f"会话ID: {st.session_state.session_id}")
    if st.button("🔄 新建会话"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()