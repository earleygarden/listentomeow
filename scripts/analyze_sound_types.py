# -*- coding: utf-8 -*-
"""
诊断脚本：量「喵叫 / 呼噜 / 嘶叫」三类的声学差异，为规则型类型检测器定阈值。

为什么需要类型层：
    B 类情境写的是「被抚摸 / 梳毛时」，而真实撸猫最典型的声音是**呼噜**；
    我们现有的闸门判「喵叫 vs 非喵叫」，把呼噜和嘶叫全拒了 —— App 在自己的核心场景失效。
    另外「痛苦哀鸣」会被硬塞成「满足 86%」，需要嘶叫/激烈叫声这一层来兜。

设计取向：**规则型**（不用训练）。原因是我们手里没有成规模的呼噜/嘶叫标注数据
（队友包的 audio_clean 音频没随包发出来），硬训一个模型等于拿 3 个样本训。

用法：python scripts/analyze_sound_types.py
"""
import os
import sys
import glob
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import librosa  # noqa: E402
from features import load_audio, SR  # noqa: E402

EXT = os.path.join(HERE, "..", "data", "external_test")

# 按队友包里的意图标注 + 其《意图标准手册》的类型定义分组
PURR_HINT = ["表达舒适", "自我安抚", "亲昵互动"]          # 呼噜系
HISS_HINT = ["威胁警告", "恐惧防御"]                       # 嘶叫/低吼系
OTHER_TYPES = ["满足低吟", "求偶嚎叫", "分离焦虑", "痛苦哀鸣", "愤怒嚎叫"]  # 嚎叫/哀鸣系


def describe(y, sr=SR):
    """一组与「音色/周期性」相关的描述子。"""
    if len(y) < sr // 10:
        return None
    D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    power = D ** 2
    total = float(power.sum()) + 1e-12

    centroid = float(np.sum(freqs[:, None] * power) / total)
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=D)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))

    # 频段能量占比（低频<400 / 中频 400-2000 / 高频>2000）
    low = float(power[freqs < 400].sum() / total)
    mid = float(power[(freqs >= 400) & (freqs < 2000)].sum() / total)
    high = float(power[freqs >= 2000].sum() / total)

    # 包络调制频率：呼噜的特征是 20~30Hz 的快速颤动
    env = librosa.onset.onset_strength(S=librosa.power_to_db(power + 1e-10), sr=sr, hop_length=256)
    mod_peak, mod_ratio = 0.0, 0.0
    if len(env) >= 8:
        e = env - env.mean()
        spec = np.abs(np.fft.rfft(e))
        f_axis = np.fft.rfftfreq(len(e), d=256 / sr)
        band = (f_axis >= 15) & (f_axis <= 40)
        if band.any():
            mod_peak = float(f_axis[band][np.argmax(spec[band])])
            mod_ratio = float(spec[band].max() / (spec[1:].mean() + 1e-9))

    # 谐波性：谱峰突出程度（tonal 高、噪声低）
    harm = float(np.max(D.mean(axis=1)) / (D.mean() + 1e-9))

    return dict(centroid=centroid, flatness=flatness, zcr=zcr, rms=rms,
                low=low, mid=mid, high=high, mod_peak=mod_peak,
                mod_ratio=mod_ratio, harm=harm, dur=len(y) / sr)


def group_of(fname):
    base = os.path.basename(fname)
    intent = base.split("_")[0]
    if intent in PURR_HINT:
        return "呼噜purr"
    if intent in HISS_HINT:
        return "嘶叫hiss"
    if intent in OTHER_TYPES:
        return "嚎叫howl"
    if base.startswith("cat_hiss"):
        return "嘶叫hiss(pix)"
    if base.startswith("cat_pixabay"):
        return "喵叫meow(pix)"
    return "喵叫meow"


def main():
    files = sorted(glob.glob(os.path.join(EXT, "intent_test_audio", "*.wav"))) + \
            sorted(glob.glob(os.path.join(EXT, "real_test_audio", "*.mp3")))
    rows = {}
    for f in files:
        try:
            y = load_audio(f)
            d = describe(y)
        except Exception as e:
            print("跳过 %s: %s" % (os.path.basename(f), e))
            continue
        if d:
            rows.setdefault(group_of(f), []).append((os.path.basename(f), d))

    keys = ["centroid", "flatness", "zcr", "low", "mid", "high", "mod_peak", "mod_ratio", "harm"]
    print("%-14s %4s " % ("组", "n") + " ".join("%9s" % k for k in keys))
    print("-" * 118)
    for g in sorted(rows):
        arr = rows[g]
        med = [np.median([d[k] for _, d in arr]) for k in keys]
        print("%-14s %4d " % (g, len(arr)) + " ".join("%9.3f" % v for v in med))

    print("\n—— 逐条（用于挑阈值）——")
    for g in sorted(rows):
        print("\n[" + g + "]")
        for name, d in rows[g]:
            print("  %-40s cent=%6.0f flat=%.3f zcr=%.3f low=%.2f mid=%.2f high=%.2f mod=%4.1fHz/%4.1f harm=%5.1f"
                  % (name[:40], d["centroid"], d["flatness"], d["zcr"],
                     d["low"], d["mid"], d["high"], d["mod_peak"], d["mod_ratio"], d["harm"]))


if __name__ == "__main__":
    main()
