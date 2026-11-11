# -*- coding: utf-8 -*-
"""
推理脚本：猫叫音频 → 声学特征 → 闸门检查 → 情境/情绪分类 → 中文"翻译"
用法:
    python predict.py <音频文件>            # 文本输出
    python predict.py <音频文件> --json     # JSON 输出（供本地服务调用）
    python predict.py <音频文件> --no-gate  # 跳过闸门（调试用）

返回结构（--json）：
    通过闸门:
        { ok:true, status:"translated", translation, emotion, context, confidence,
          probabilities, duration_sec, gate:{is_meow:true,p_meow,threshold},
          audio:{peak,max_amp,sr}, note }
    被闸门拦下:
        { ok:true, status:"rejected", message, gate:{...}, duration_sec, note }
"""
import os
import sys
import re
import json
import datetime
import argparse
import numpy as np
import joblib

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "models", "cat_emotion_model.joblib")          # MFCC 版（兜底）
AST_MODEL_PATH = os.environ.get(
    "TMS_AST_MODEL",
    os.path.join(HERE, "..", "models", "cat_emotion_model_ast.joblib"),
)  # AST embedding 版（优先；可用 TMS_AST_MODEL 指向候选模型做 A/B）
GATE_PATH = os.path.join(HERE, "..", "models", "cat_gate.joblib")

from features import load_audio, extract_features_from_signal, SR  # noqa: E402
import quote_bank  # noqa: E402  语录库（情绪作为前置状态 → 语境 → 语录）

try:
    import ast_features  # noqa: E402  AST embedding 提取（torch-free）；不可用则兜底 MFCC
except Exception:
    ast_features = None

# 情境 → 情绪 → 中文翻译文案（可多版，按置信度挑选，让结果更自然）
TRANSLATIONS = {
    "B": {
        "emotion": "满足 · 放松",
        "context": "被抚摸 / 梳毛时",
        "texts": [
            "咕噜噜…好舒服，就是这儿，再摸摸嘛～",
            "舒服～你手别停呀，我眯一会儿～",
        ],
    },
    "F": {
        "emotion": "期待 · 索求",
        "context": "等待食物 / 讨要时",
        "texts": [
            "我饿了！快点给我开饭嘛～",
            "闻到好吃的了，动作快一点呀～",
        ],
    },
    "I": {
        "emotion": "不安 · 焦虑",
        "context": "陌生环境 / 独处时",
        "texts": [
            "这里好陌生…我有点怕，想回熟悉的地方…",
            "放我出去好不好…我不喜欢待在这儿…",
        ],
    },
}

REJECT_MSG = ("这段听起来不像喵叫。如果你录的是呼噜、低吼或者哈气 —— 那确实不是喵叫，"
              "而我目前只会听喵叫，这两种我还分不出来。也可能只是太吵、太短。"
              "换一段它张嘴「喵」的录音再试试？")
TOO_SHORT_MSG = "音频太短了，听不太清楚。录一段 1 秒以上、有连续喵叫的片段再试试～"

# ── 分布外（OOD）护栏 ─────────────────────────────────────────────
# 为什么需要：SVM 在没见过的音频上会给**虚高的置信度**。实测把「痛苦哀鸣」判成
# 「满足·放松」86%、「表达疼痛」70% —— 疼痛说成舒服，这在产品上不能接受。
# 做法：用训练集 embedding 的 kNN 距离衡量"这段声音离我学过的有多远"。
# 实测（训练集内部 kNN 距离 p95=34.4）：
#   判错的四条全在分布外（表达不安 85.8 / 分离焦虑 56.9 / 痛苦哀鸣 42.3 / 表达疼痛 40.5）
#   判对的三条全在分布内（寻求陪伴 17.8 / 求食催饭 25.4 / 表达满意 29.1）
# 所以距离比 SVM 的概率更能说明"我到底有没有资格下这个结论"。
EMB_CACHE = os.path.join(HERE, "..", "data", "_emb_cache", "emb_ast_mean.npy")
_ood = None      # None=未初始化；False=不可用（缓存缺失，静默关闭）


def _ood_state():
    """惰性构建 (scaler, nn, threshold)。任何失败都返回 False —— 护栏是加分项，不能拖垮主流程。"""
    global _ood
    if _ood is not None:
        return _ood
    try:
        import pickle
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        E = np.load(EMB_CACHE)
        sc = StandardScaler().fit(E)
        Es = sc.transform(E)
        nn = NearestNeighbors(n_neighbors=6).fit(Es)
        d, _ = nn.kneighbors(Es)
        thr = float(np.percentile(d[:, 1:].mean(axis=1), 95))   # 排除自身
        _ood = (sc, nn, thr)
    except Exception as e:
        sys.stderr.write("OOD 护栏不可用（不影响主流程）: %s\n" % e)
        _ood = False
    return _ood

# ── 猫格台词库（摘自队友包 v14 的 cat_personas.json：6 种猫格 × 26 种意图）──
# 队友库原样有 441 句，但同一意思在不同猫格下只差一两个字，还混着「人生圆满！」这类太像人的句子。
# 这里做了两件事：① 只取与本 App 三类情境**语义一致**的意图（「恐惧防御/自我安抚」虽也是焦虑，
# 但说的是"怕被靠近""自己扛过去"，情境不符，不取）；② 每类精选 8~9 句，保证猫格语气不同、
# 句子自然、不和原有文案撞车。选句目的是让不同片段/不同猫拿到不同说法，而不是堆数量。
PERSONA_TEXTS = {
    # B 满足·放松（被抚摸 / 梳毛时）← 表达满意 / 表达舒适
    "B": [
        "嗯嗯～舒服舒服～妈妈最好了～",          # 小奶猫
        "就是这里呀～再摸摸嘛～",                # 小奶猫
        "舒服！就是这样！继续！",                # 调皮小子
        "呼噜呼噜～太爽了！",                    # 调皮小子
        "哼，还行吧…也就这样…继续",              # 傲娇小姐
        "不错，本喵勉强认可你的手法",            # 傲娇小姐
        "嗯～舒服，保持这个力度",                # 直爽大哥
        "嗯…这个力道正好…老了就喜欢这样…",      # 温柔奶奶
        "嗯…不错…老夫认可了…",                  # 沉稳大爷
    ],
    # F 期待·索求（等待食物 / 讨要时）← 求食催饭
    "F": [
        "妈妈我饿了嘛～要吃饭饭～",              # 小奶猫
        "碗碗空了嘛～快填满嘛～",                # 小奶猫
        "我要吃！快给我！现在就要！",            # 调皮小子
        "哼，碗都空了你看不到吗？快开罐头",      # 傲娇小姐
        "我都饿瘦了你居然还在玩手机",            # 傲娇小姐
        "喂！饭呢？别磨蹭，老子饿了",            # 直爽大哥
        "哎…胃口不好了，但也得吃…饭呢？",        # 温柔奶奶
        "嗯…老夫饿了…你慢慢来不急",              # 沉稳大爷
    ],
    # I 不安·焦虑（陌生环境 / 独处时）← 寻求陪伴 / 分离焦虑 / 表达不安
    "I": [
        "妈妈你去哪了呀～我害怕～",              # 小奶猫
        "门关上了我好慌呀～你快回来嘛～",        # 小奶猫
        "你去哪了！我到处找你！",                # 调皮小子
        "你去哪了？本喵找你半天了",              # 傲娇小姐
        "哼，不是想你…就是觉得你该在了",          # 傲娇小姐
        "喂！你去哪了？到处找你",                # 直爽大哥
        "哎…你走了我好孤单…",                    # 温柔奶奶
        "嗯…你走了老夫一个人…唉…",              # 沉稳大爷
    ],
}

# 合并进文案池：原有 2 版基础文案在前（本 App 语气），猫格台词在后
for _cls, _extra in PERSONA_TEXTS.items():
    if _cls in TRANSLATIONS and _extra:
        _base = TRANSLATIONS[_cls]["texts"]
        _seen = {re.sub(r"[\s~～.。…!！?？,，、'\"“”‘’:：;；()（）\-]+", "", t) for t in _base}
        TRANSLATIONS[_cls]["texts"] = _base + [
            t for t in _extra
            if re.sub(r"[\s~～.。…!！?？,，、'\"“”‘’:：;；()（）\-]+", "", t) not in _seen
        ]


def run_gate(feat, duration):
    """返回 (是否放行, gate 信息 dict)"""
    if not os.path.exists(GATE_PATH):
        return True, {"is_meow": None, "note": "闸门模型未训练，已跳过"}
    gb = joblib.load(GATE_PATH)
    gpipe, th = gb["pipeline"], float(gb["threshold"])
    p = gpipe.predict_proba(feat)[0]
    classes = list(gpipe.classes_)
    p_meow = float(p[classes.index("meow")])
    return (p_meow >= th), {
        "is_meow": bool(p_meow >= th),
        "p_meow": round(p_meow, 3),
        "threshold": round(th, 3),
    }


def translate(audio_path, as_json=False, use_gate=True):
    mfcc_bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    ast_bundle = joblib.load(AST_MODEL_PATH) if os.path.exists(AST_MODEL_PATH) else None
    if mfcc_bundle is None and ast_bundle is None:
        raise SystemExit(f"模型不存在: {MODEL_PATH} 或 {AST_MODEL_PATH}，请先运行 train.py / train_ast.py")

    # 只加载一次音频，特征也只算一次（闸门与情绪模型共用）
    # 读不出来时不抛异常：视频可能压根没有音轨，这种情况要交给上层降级处理（画面还能看）
    try:
        y = load_audio(audio_path)
        duration = len(y) / SR
        feat = extract_features_from_signal(y, SR).reshape(1, -1)
    except Exception as e:
        # 详细原因（含 ffmpeg 原文）只写进 stderr 供服务端日志排查，不给用户看技术噪音
        sys.stderr.write("decode failed: %s: %s\n" % (type(e).__name__, str(e)[:600]))
        result = {
            "ok": False,
            "status": "audio_failed",
            "message": "这段声音我没能读出来 —— 可能是视频里没有音轨，或者音频格式不受支持。",
            "error": "音频解码失败",
            "note": "趣味解读 · 仅供娱乐，不构成任何诊断",
        }
        if as_json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print("=" * 50)
            print(f"🎧 文件: {os.path.basename(audio_path)}")
            print("🚫 " + result["message"])
            print("   " + result["error"])
            print("=" * 50)
        return result

    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    # ---- 闸门 ----
    passed, gate = (True, {"is_meow": None, "note": "已跳过闸门"})
    if use_gate:
        passed, gate = run_gate(feat, duration)

    if not passed:
        result = {
            "ok": True,
            "status": "rejected",
            "message": TOO_SHORT_MSG if duration < 0.6 else REJECT_MSG,
            "gate": gate,
            "duration_sec": round(duration, 2),
            "note": "趣味解读 · 仅供娱乐，不构成任何诊断",
        }
        if as_json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print("=" * 50)
            print(f"🎧 音频: {os.path.basename(audio_path)}  ({duration:.2f}s)")
            print(f"🚫 未通过检查（P(猫叫)={gate.get('p_meow')} < {gate.get('threshold')}）")
            print(result["message"])
            print("=" * 50)
        return result

    # ---- 情绪分类：优先 AST embedding（准确率更高），不可用/失败则兜底 MFCC ----
    pipe, emo_feat, engine = None, None, None
    if ast_bundle is not None and ast_features is not None and ast_features.available():
        emb = ast_features.extract_ast_embedding(y)
        if emb is not None:
            pipe, emo_feat, engine = ast_bundle["pipeline"], emb.reshape(1, -1), "ast"
    if pipe is None and mfcc_bundle is not None:
        pipe, emo_feat, engine = mfcc_bundle["pipeline"], feat, "mfcc"
    if pipe is None:
        raise SystemExit("情绪模型均不可用")
    sys.stderr.write("emotion engine: %s\n" % engine)

    probs = pipe.predict_proba(emo_feat)[0]
    classes = list(pipe.classes_)
    order = np.argsort(-probs)
    best_i = int(order[0])
    best_cls = classes[best_i]
    conf = float(probs[best_i])
    runner_cls = classes[int(order[1])] if len(order) > 1 else None
    runner_up = float(probs[int(order[1])]) if len(order) > 1 else 0.0
    margin = conf - runner_up

    info = TRANSLATIONS.get(best_cls, {"emotion": "未知", "context": "未知", "texts": ["…"]})
    alt_info = TRANSLATIONS.get(runner_cls, {}) if runner_cls else {}

    # 拿不准就不装确定：置信度低、与次名差距太小、或**离训练分布太远** → 给"最像+备选"。
    # 第三条最关键：SVM 对陌生音频会自信地胡说（实测疼痛叫 → 满足 86%），
    # 而 kNN 距离能诚实地告诉我们"这段我没资格下结论"。
    UNCERTAIN_CONF, UNCERTAIN_MARGIN = 0.65, 0.12
    unfamiliar, knn_dist = False, None
    ood = _ood_state() if engine == "ast" else False
    if ood:
        try:
            sc, nn, thr = ood
            d, _ = nn.kneighbors(sc.transform(emo_feat))
            knn_dist = float(d[0, :5].mean())
            unfamiliar = knn_dist > thr
        except Exception:
            pass
    uncertain = bool(conf < UNCERTAIN_CONF or margin < UNCERTAIN_MARGIN or unfamiliar)

    # ── 娱乐化：情绪是**前置状态**，先定语境，再在语境里出话 ──
    # 时间与声音形状只做微调（不越权决定说什么），所以用户依然只需传一段猫叫。
    state = best_cls if not uncertain else "UNSURE"
    _now = datetime.datetime.now()
    _voiced = quote_bank.count_voiced(y, SR)
    quote, quotes, when, shape = quote_bank.pick(
        state, _now.hour, _now.weekday(), duration, peak, _voiced,
        seed=float(np.sum(emo_feat)),
    )
    text = quote

    alternatives = [
        {
            "class": classes[int(i)],
            "emotion": TRANSLATIONS.get(classes[int(i)], {}).get("emotion", classes[int(i)]),
            "prob": round(float(probs[int(i)]), 3),
        }
        for i in order[:3]
    ]

    result = {
        "ok": True,
        "status": "translated",
        "translation": text,
        "quote": quote,
        "quotes": quotes,
        "state": {
            "key": state,
            "name": quote_bank.STATE_LINES[state]["name"],
            "context": quote_bank.STATE_LINES[state]["context"],
        },
        "when": when,
        "shape": shape,
        "voiced": _voiced,
        "emotion": info["emotion"],
        "context": info["context"],
        "confidence": round(conf, 3),
        "uncertain": uncertain,
        "uncertain_reason": ("unfamiliar" if unfamiliar else ("low_conf" if uncertain else None)),
        "unfamiliar": unfamiliar,
        "knn_dist": round(knn_dist, 1) if knn_dist is not None else None,
        "margin": round(margin, 3),
        "alternatives": alternatives,
        "probabilities": {
            f"{classes[i]}:{TRANSLATIONS.get(classes[i], {}).get('emotion', classes[i])}": round(float(probs[i]), 3)
            for i in range(len(classes))
        },
        "duration_sec": round(duration, 2),
        "gate": gate,
        "engine": engine,
        "audio": {"sr": SR, "peak": round(peak, 3)},
        "note": "趣味解读 · 仅供娱乐，不构成任何诊断",
    }

    if as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print("=" * 50)
        print(f"🎧 音频: {os.path.basename(audio_path)}  ({duration:.2f}s)")
        print(f"💬 它说: {text}")
        print(f"💗 前置情绪: {result['state']['name']}  (语境: {result['state']['context']})"
              f"   [模型最像: {info['emotion']} {conf:.0%}]")
        print(f"🕐 时间档: {when}   声形: {shape}   发声次数: {_voiced}")
        print(f"📊 置信度: {conf:.1%}" + ("   | ⚠ 拿不准（分布外音频，如多猫合唱/嘈杂环境）" if uncertain else "")
              + (f"   | 闸门 P(猫叫)={gate.get('p_meow')}" if gate.get('p_meow') is not None else ""))
        print("   备选:", " / ".join(quotes[1:3]) if len(quotes) > 1 else "—")
        print("=" * 50)
        print("注：趣味解读 · 仅供娱乐")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-gate", action="store_true", help="跳过闸门检查（调试用）")
    args = ap.parse_args()
    translate(args.audio, as_json=args.json, use_gate=not args.no_gate)
