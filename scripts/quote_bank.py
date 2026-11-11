# -*- coding: utf-8 -*-
"""
语录库（猫的嘴替）：情绪作为**前置状态**，先定语境，再在这个语境里出话。

结构：
    情绪状态（B满足 / F期待 / I不安 / UNSURE说不准）
        ↓ 决定语境（被照顾得很好 / 等着什么 / 一个人不太安心 / 它自己也没想清）
    语录池（治愈承接 + 搞笑）
        ↓ 时间与声音形状只做微调（深夜/清晨/周一；短促一声/拖长/连叫/小声/大吼）
    一句话

三条写作硬规矩（改词时请守住）：
    1) 承接，不打气 —— 不写「加油」「明天会更好」，写「先躺下」「头给我摸一下」。
    2) 不复述主人的状态 —— 不写「你很难过吧」。人怕被断言，不怕被接住。
    3) 必须有一个具体的东西 —— 时间、动作、身体部位。越具体越像真的。

为什么情绪仍然是前置状态而不是随机抽句：
    同一个「喵」，在「舒服」和「不安」两个语境里说的话必须不一样，
    否则用户立刻听出是随机文案 —— 这也是我们唯一真正需要的"准"。
"""
import numpy as np

# ══ 情绪 → 语境 → 语录 ═══════════════════════════════════════
# 每个状态：name（展示名）、context（语境说明）、heal（治愈承接）、fun（搞笑）
STATE_LINES = {
    "B": {
        "name": "满足 · 放松",
        "context": "被照顾得很好的时候",
        "heal": [
            "这样就很好。你也不用说话。",
            "我不需要更多了。真的。",
            "你手别停。就停在这儿。",
            "今天我哪儿也不去。你也是。",
        ],
        "fun": [
            "嗯，手法尚可。批准你继续。",
            "摸到这个位置，说明你研究过我。",
            "别停。我说别停。",
            "我允许你把我宠成这样。",
        ],
    },
    "F": {
        "name": "期待 · 索求",
        "context": "等着什么的时候",
        "heal": [
            "我知道你会来。所以我一直在这儿等。",
            "你也饿了吧。先吃你的，我这份不急。",
            "好东西值得等。你也一样。",
        ],
        "fun": [
            "碗空了。你看不见吗。",
            "我等到现在，你最好带点像样的回来。",
            "开饭。现在。立刻。",
            "我不是在催你。我只是在你脚边站着。",
        ],
    },
    "I": {
        "name": "不安 · 焦虑",
        "context": "一个人待着、不太安心的时候",
        "heal": [
            "你不用马上好起来。我就在旁边。",
            "怕就怕吧。我又不会走。",
            "门关着也没关系。我在里面。",
            "你不用解释为什么难受。",
        ],
        "fun": [
            "你走了我怎么办。你怎么这么狠心。",
            "我一个人在家，刚才有只虫子看了我一眼。",
            "别走。就站那儿。别动。",
            "我需要你。但我不想承认。",
        ],
    },
    "UNSURE": {
        "name": "它自己也没想清",
        "context": "说不准 / 这段听着陌生",
        "heal": [
            "这段我没听清。但你要是想有人陪，我在。",
            "它欲言又止。你也一样吧。",
            "说不清也没关系。坐着就行。",
        ],
        "fun": [
            "我说话你听不懂，你说话我也听不懂。扯平了。",
            "它在思考人生。或者在发呆。区别不大。",
            "我暂时没有想说的。但我不想让你走。",
        ],
    },
}

# ══ 时间微调：同一个情绪，在不同的钟点说不一样的话 ══════════════
TIME_LINES = {
    ("B", "night"): ["这么晚了还摸我。行吧。"],
    ("B", "evening"): ["你回来了。这就够了。"],
    ("B", "monday"): ["周一还能这样躺着。挺好。"],
    ("F", "night"): ["半夜饿了。这一点随你。"],
    ("F", "morning"): ["天亮了。开工吧，先给我。"],
    ("F", "evening"): ["闻到味儿了。你今天没偷懒。"],
    ("I", "night"): ["夜里最安静。也最容易想事情。"],
    ("I", "monday"): ["周一。我也不想去。我们都不想去。"],
    ("I", "evening"): ["天黑了还没人回来。我不太喜欢这个时候。"],
}

# ══ 声音形状微调：短促一声 / 拖长 / 连叫 / 小声 / 大吼 ═══════════
SHAPE_LINES = {
    ("F", "short"): ["就一声。你懂的。"],
    ("F", "repeat"): ["喂。喂。喂。喂。"],
    ("I", "long"): ["我把这声拖得很长。就是想让你听见。"],
    ("I", "quiet"): ["我尽量小声。但我还是想你听见。"],
    ("I", "loud"): ["开门。我知道你在里面。"],
    ("B", "long"): ["这声很长。因为很舒服。"],
    ("B", "quiet"): ["小点声。别吵到楼下。"],
    ("UNSURE", "repeat"): ["我说了好几遍。你还是没听懂。"],
    ("UNSURE", "short"): ["就一声。你猜。"],
}


# ══ 信号计算 ═══════════════════════════════════════════════
def time_bucket(hour, weekday):
    """weekday: 0=周一 … 6=周日。返回时间档名。"""
    if weekday == 0 and 5 <= hour < 11:
        return "monday"
    if hour >= 22 or hour < 4:
        return "night"
    if 5 <= hour < 9:
        return "morning"
    if 9 <= hour < 18:
        return "day"
    return "evening"


def count_voiced(y, sr=16000):
    """数这段里有几次发声（能量分段，间隔 ≥150ms 才算新的一声）。
    实测校准：2.28s 梳毛叫=1、1.22s 讨食=1、10s 多猫=1~5。
    早期用包络过峰计数会数到 6 次（把颤动当成了多声），已废弃。"""
    if len(y) < sr // 4:
        return 1
    try:
        import librosa
        segs = librosa.effects.split(y, top_db=25)
    except Exception:
        return 1
    if len(segs) == 0:
        return 1
    n = 1
    for i in range(1, len(segs)):
        if (segs[i][0] - segs[i - 1][1]) / sr >= 0.15:
            n += 1
    return max(1, min(n, 5))


def sound_shape(duration, peak, voiced):
    if voiced >= 3:
        return "repeat"
    if duration >= 1.6:
        return "long"
    if peak is not None and peak < 0.06:
        return "quiet"
    if peak is not None and peak > 0.75:
        return "loud"
    return "short"


def build_pool(state, hour, weekday, duration, peak, voiced):
    """按「情绪 → 时间/形状微调」拼出这一段的候选语（顺序即推荐顺序）。"""
    st = STATE_LINES.get(state) or STATE_LINES["UNSURE"]
    pool = []
    pool.extend(st["heal"][:2])
    # 时间微调插在治愈后、搞笑前：先接住，再逗
    tb = time_bucket(hour, weekday)
    pool.extend(TIME_LINES.get((state, tb), []))
    pool.extend(st["heal"][2:])
    shape = sound_shape(duration, peak, voiced)
    pool.extend(SHAPE_LINES.get((state, shape), []))
    pool.extend(st["fun"][:2])
    pool.extend(st["fun"][2:])
    # 去重保序
    seen, out = set(), []
    for s in pool:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out, tb, shape


def pick(state, hour, weekday, duration, peak, voiced, seed=0.0, n=4):
    """返回 (首选语, 候选列表[含首选，供「换一个」轮换], 时间档, 声音形状)。"""
    pool, tb, shape = build_pool(state, hour, weekday, duration, peak, voiced)
    if not pool:
        pool = ["我在。"]
    # 用音频特征 + seed 做确定性偏移：同一段音频结果稳定，不同片段不同
    start = int(abs(float(seed)) * 97) % len(pool)
    rotated = pool[start:] + pool[:start]
    return rotated[0], rotated[:n], tb, shape
