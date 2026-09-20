"""检索策略横评：同一批题 × 多策略，落一份报告到 eval/reports/

用法：python eval/compare_strategies.py
只跑检索层，不调 LLM —— 指标是确定性的，可重复对比。
（rewrite 策略每次要调一次 LLM，不在默认跑的范围里，要跑把名字加进 STRATEGIES。）
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.retriever import retrieve

EVAL_SET = Path(__file__).parent / "eval_set.json"
REPORT_DIR = Path(__file__).parent / "reports"
STRATEGIES = ["vector", "rerank", "hybrid"]


def is_hit(doc, item) -> bool:
    """一个检索结果算不算命中该题的黄金答案：source 命中 或 关键词出现在正文里"""
    if doc.metadata.get("source", "") in item.get("expected_sources", []):
        return True
    return any(kw in doc.page_content for kw in item.get("expected_keywords", []))


def run() -> dict:
    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    REPORT_DIR.mkdir(exist_ok=True)
    out = {}

    for s in STRATEGIES:
        r1 = r3 = p3 = mrr = 0.0
        times, rows, errs = [], [], 0
        for it in items:
            t0 = time.time()
            try:
                docs = retrieve(it["question"], strategy=s, k=3)
            except Exception as e:
                errs += 1
                rows.append((it["question"], "ERR " + str(e)[:40], it.get("expected_sources", [])))
                continue
            times.append(time.time() - t0)
            flags = [is_hit(d, it) for d in docs]
            if flags[:1] and flags[0]:
                r1 += 1.0
            if any(flags):
                r3 += 1.0
            p3 += (sum(flags) / len(flags)) if flags else 0.0
            for i, f in enumerate(flags):
                if f:
                    mrr += 1.0 / (i + 1)
                    break
            rows.append((
                it["question"],
                "".join("Y" if f else "." for f in flags),
                sorted({d.metadata.get("source", "?") for d in docs}),
            ))
        n = len(items)
        out[s] = {
            "n": n, "errs": errs,
            "recall1": r1 / n, "recall3": r3 / n, "prec3": p3 / n, "mrr": mrr / n,
            "avg_ms": statistics.mean(times) * 1000 if times else 0.0,
            "rows": rows,
        }
    return out, items


def write_report(out: dict, items: list) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    path = REPORT_DIR / f"{ts}-strategy-compare.md"

    L = []
    L.append(f"# 检索策略横评 {ts}\n")
    L.append(f"题数：{len(items)}（`eval/eval_set.json`）｜k=3｜只跑检索层，不调 LLM\n")
    L.append("**指标口径**：命中 = `source` 落在 `expected_sources` 里 **或** 任一 "
             "`expected_keywords` 出现在块正文里（沿用现版 `evaluate.py` 的「或」口径）。\n")
    L.append("## 总览\n")
    L.append("| 策略 | recall@1 | recall@3 | precision@3 | MRR | 平均耗时 | 报错 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in STRATEGIES:
        m = out[s]
        L.append("| `%s` | %.4f | %.4f | %.4f | %.4f | %.0f ms | %d |" % (
            s, m["recall1"], m["recall3"], m["prec3"], m["mrr"], m["avg_ms"], m["errs"]))
    L.append("\n> 命中标记：`Y` = 该位置命中，`.` = 未命中（位置 1/2/3）\n")

    for s in STRATEGIES:
        L.append(f"\n## 逐题：`{s}`\n")
        L.append("| 问题 | 命中 | 召回的来源 |")
        L.append("| --- | --- | --- |")
        for q, flags, srcs in out[s]["rows"]:
            src = ", ".join(srcs) if isinstance(srcs, list) else str(srcs)
            L.append("| %s | `%s` | %s |" % (q, flags, src))

    path.write_text("\n".join(L), encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    out, items = run()
    print("%-8s %-9s %-9s %-11s %-8s %-10s" % ("策略", "recall@1", "recall@3", "prec@3", "MRR", "平均耗时"))
    for s in STRATEGIES:
        m = out[s]
        print("%-8s %-9.4f %-9.4f %-11.4f %-8.4f %-10.0fms" % (
            s, m["recall1"], m["recall3"], m["prec3"], m["mrr"], m["avg_ms"]))
    p = write_report(out, items)
    print("\n报告已写入:", p)
