"""应用包入口。

这里的设置必须放在包的 __init__ 里——它比 app 下任何子模块都先执行。
放在 config.py 或 embeddings.py 里都太晚：langchain_community.document_loaders
会先一步 import huggingface_hub，而下面这两个离线开关是在那一刻被读进常量的。
"""
import os
import sys

# 1) huggingface.co 在当前网络不可达（443 连不上），而 HF 库默认每次加载模型都会
#    联网 HEAD 核对 config/tokenizer 是否存在，连不上就重试 5 轮再抛超时。
#    强制「只读本地缓存」绕开这一步。要下新模型时再临时置 0 并配 HF_ENDPOINT 镜像。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 2) Windows 控制台默认 GBK，入库脚本里的 emoji（🗑️ ✅ ✂️）会直接抛
#    UnicodeEncodeError，导致 python -m app.core.ingestion 跑不完。
#    输出被重定向/管道时 reconfigure 可能不可用，失败就跳过。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
