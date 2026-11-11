# -*- coding: utf-8 -*-
"""
外部基准测试：拿队友包 v14 里**从未参与我们训练**的音频，检验我们这条 3 类管线。

为什么需要它：
    我们自己只有 CatMeows 的 440 条（21 只猫），cross-cat CV 0.62 是"自己人考自己人"。
    这 41 条来自别人的库（Pixabay / BBC / Freesound 等），是真正的"没见过的声音"，
    才能回答一个残酷但必要的问题：换一批猫、换一个录音环境，我们还剩多少准头？

怎么判：
    A. 在册三类（9 条）：意图 → 情绪 → 我们的 B/F/I，可以直接算对错。
    B. 三类之外（16 条）：求偶/捕猎/疼痛/威胁… 我们**没有**这些类别，
       模型只能硬塞进三类。这里不判对错，只如实记录它塞了什么 —— 这是诚实边界。
    C. 真实环境 16 条（Pixabay 下载）：无真值，量两件事：闸门（猫叫/非猫叫）放行率、
       以及放行后判成什么。

用法：
    python scripts/bench_external.py            # 文本报告
    python scripts/bench_external.py --json     # 机器可读
"""
import os
import io
import sys
import json
import time
import contextlib

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import predict as P  # noqa: E402

EXT_DIR = os.path.join(HERE, "..", "data", "external_test")
INTENT_DIR = os.path.join(EXT_DIR, "intent_test_audio")
REAL_DIR = os.path.join(EXT_DIR, "real_test_audio")

# 意图 → 我们的三类（按**情绪**对齐，不按情境；队友的意图名见其《意图标准手册》）
INTENT_TO_CLASS = {
    # B 满足·放松
    "表达满意": "B",
    "表达舒适": "B",
    "满足低吟": "B",
    # F 期待·索求
    "求食催饭": "F",
    # I 不安·焦虑
    "寻求陪伴": "I",
    "表达不安": "I",
    "分离焦虑": "I",
    "自我安抚": "I",
    "恐惧防御": "I",
}
# 三类之外：我们表达不了的情绪（不判对错，只记录）
OUT_OF_SCOPE = {
    "亲昵互动": "中性/求关注",
    "其他意图": "混合",
    "发现猎物": "兴奋/捕猎",
    "捕猎兴奋": "兴奋/捕猎",
    "呼唤亲人": "亲昵/母性",
    "母子交流": "亲昵/母性",
    "威胁警告": "警告/愤怒",
    "愤怒嚎叫": "警告/愤怒",
    "抱怨不满": "警告/愤怒",
    "打招呼": "中性/求关注",
    "日常交流": "中性/求关注",
    "求偶呼唤": "发情/求偶",
    "求偶嚎叫": "发情/求偶",
    "痛苦哀鸣": "疼痛",
    "表达疼痛": "疼痛",
    "邀请玩耍": "愉悦/开心",
}
CLS_NAME = {"B": "满足·放松", "F": "期待·索求", "I": "不安·焦虑"}


def run_one(path):
    """跑一次完整推理（闸门 + 情绪），把 stdout 噪音吃掉。"""
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = P.translate(path)
    r["_ms"] = int((time.time() - t0) * 1000)
    return r


def intent_of(fname):
    return fname.split("_")[0]


def pred_class(r):
    """取预测出的类别字母（B/F/I）—— 用 alternatives[0].class，别去比中文字符串（空格/标点易错）。"""
    if r.get("status") != "translated":
        return None
    alts = r.get("alternatives") or []
    return alts[0]["class"] if alts else None


def main(as_json=False):
    report = {"a_in_scope": [], "b_out_of_scope": [], "c_real": [], "summary": {}}

    # ---------- A / B：25 条意图音频 ----------
    if os.path.isdir(INTENT_DIR):
        for fname in sorted(os.listdir(INTENT_DIR)):
            if not fname.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
                continue
            intent = intent_of(fname)
            r = run_one(os.path.join(INTENT_DIR, fname))
            passed = r.get("status") == "translated"
            got = pred_class(r)
            item = {
                "file": fname,
                "intent": intent,
                "gate": r.get("status"),
                "pred": got,
                "emotion": r.get("emotion") if passed else None,
                "conf": r.get("confidence") if passed else None,
                "uncertain": r.get("uncertain"),
                "reason": r.get("uncertain_reason"),
                "knn": r.get("knn_dist"),
                "ms": r["_ms"],
            }
            if intent in INTENT_TO_CLASS:
                exp = INTENT_TO_CLASS[intent]
                item["expect"] = exp
                item["correct"] = (passed and got == exp)
                report["a_in_scope"].append(item)
            else:
                item["true_emotion"] = OUT_OF_SCOPE.get(intent, "未知")
                report["b_out_of_scope"].append(item)

    # ---------- C：16 条真实环境音频（无真值） ----------
    if os.path.isdir(REAL_DIR):
        for fname in sorted(os.listdir(REAL_DIR)):
            if not fname.lower().endswith((".mp3", ".wav", ".m4a")):
                continue
            r = run_one(os.path.join(REAL_DIR, fname))
            passed = r.get("status") == "translated"
            got = pred_class(r)
            report["c_real"].append({
                "file": fname,
                "gate": r.get("status"),
                "p_meow": (r.get("gate") or {}).get("p_meow"),
                "pred": got,
                "emotion": r.get("emotion") if passed else None,
                "conf": r.get("confidence") if passed else None,
                "uncertain": r.get("uncertain"),
            })

    # ---------- 汇总 ----------
    a = report["a_in_scope"]
    n_a = len(a)
    n_ok = sum(1 for x in a if x["correct"])
    per_cls = {}
    for exp in ("B", "F", "I"):
        rows = [x for x in a if x["expect"] == exp]
        per_cls[exp] = {
            "n": len(rows),
            "ok": sum(1 for x in rows if x["correct"]),
            "acc": round(sum(1 for x in rows if x["correct"]) / len(rows), 3) if rows else None,
        }
    c = report["c_real"]
    report["summary"] = {
        "in_scope_n": n_a,
        "in_scope_ok": n_ok,
        "in_scope_acc": round(n_ok / n_a, 3) if n_a else None,
        "per_class": per_cls,
        "out_of_scope_n": len(report["b_out_of_scope"]),
        "real_n": len(c),
        "real_gate_pass": sum(1 for x in c if x["gate"] == "translated"),
        "real_uncertain": sum(1 for x in c if x["uncertain"]),
        "avg_ms": round(sum(x.get("ms", 0) for x in a) / max(len(a), 1)),
    }

    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return report

    s = report["summary"]
    print("=" * 66)
    print("听喵说 · 外部基准（数据来自队友包 v14，均未参与我们的训练）")
    print("=" * 66)
    print("\n【A】落在我们三类里的 %d 条 —— 可以判对错" % s["in_scope_n"])
    for x in a:
        mark = "✅" if x["correct"] else "❌"
        detail = ("判成 %s %.0f%%" % (CLS_NAME.get(x["pred"], "—"), (x["conf"] or 0) * 100)
                  if x["pred"] else ("闸门拦下" if x["gate"] != "translated" else "无结果"))
        if x.get("reason") == "unfamiliar":
            detail += "  [不熟 kNN=%.1f]" % (x.get("knn") or 0)
        elif x.get("uncertain"):
            detail += "  [拿不准]"
        print("  %s %-12s 应为 %-8s → %s" % (mark, x["intent"], CLS_NAME[x["expect"]], detail))
    print("\n  → 三类准确率 %d/%d = %.1f%%" % (s["in_scope_ok"], s["in_scope_n"], s["in_scope_acc"] * 100))
    for cls in ("B", "F", "I"):
        p = s["per_class"][cls]
        if p["n"]:
            print("     %s %-8s %d/%d" % (cls, CLS_NAME[cls], p["ok"], p["n"]))

    print("\n【B】三类之外的 %d 条 —— 我们表达不了，只能硬塞（不计分）" % s["out_of_scope_n"])
    for x in report["b_out_of_scope"]:
        detail = ("硬塞成 %s %.0f%%" % (CLS_NAME.get(x["pred"], "—"), (x["conf"] or 0) * 100)
                  if x["pred"] else ("闸门拦下" if x["gate"] != "translated" else "无结果"))
        print("  ○ %-12s（其实是 %-8s）→ %s" % (x["intent"], x["true_emotion"], detail))

    print("\n【C】真实环境 %d 条（Pixabay 下载，无真值）" % s["real_n"])
    print("  闸门放行 %d/%d，其中拿不准 %d 条" % (s["real_gate_pass"], s["real_n"], s["real_uncertain"]))
    for x in c:
        pm = ("%.2f" % x["p_meow"]) if x["p_meow"] is not None else "—"
        if x["gate"] == "translated":
            print("  · %-18s p喵叫=%s → %s %.0f%%%s" % (
                x["file"], pm, CLS_NAME.get(x["pred"], "—"), (x["conf"] or 0) * 100,
                "（拿不准）" if x["uncertain"] else ""))
        else:
            print("  · %-18s p喵叫=%s → 拦下（判为非喵叫）" % (x["file"], pm))

    print("\n平均耗时 %d ms/条（AST+闸门，CPU）" % s["avg_ms"])
    print("=" * 66)
    return report


if __name__ == "__main__":
    main(as_json="--json" in sys.argv)
