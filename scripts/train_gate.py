# -*- coding: utf-8 -*-
"""
训练「闸门」模型：判断一段音频是不是猫的喵叫（meow） vs 其他（other）

为什么需要它：
    情绪模型只在「猫的喵叫」上训练过。用户上传人声、狗叫、噪音、或猫的呼噜/哈气时，
    SVM 依然会给出一个"自信"的三分类结果 —— 这是产品体验上的硬伤。
    闸门先做一道质量检查，拦住明显不是喵叫的输入。

数据（全部本地已有）：
    meow  正样本:
        BarkMeowDB  cat/*.wav   48+ 段（CC BY 4.0，可商用）→ 切成 2s 窗
        CatMeows    440 段喵叫   （CC BY-NC 4.0，仅研究用）→ 原样
    other 负样本:
        BarkMeowDB  dog/*.wav   狗吠（CC BY 4.0）        → 切成 2s 窗
        CatMeows extras/other_vocalizations  13 段非喵叫的猫声（CC BY-NC 4.0）
        synth/negatives         22 个本机合成噪声/静音/纯音/和弦（无版权问题）

评估：
    按「来源 + 个体 / 类型」分组做 GroupKFold，保证同一条录音（或其切窗）不会同时出现在
    训练集和测试集两侧 —— 否则准确率是虚高的。

产物：
    models/cat_gate.joblib  （pipeline + 类别 + 决策阈值 + 评估指标）
"""
import os
import sys
import glob
import json
import numpy as np
import joblib
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "models")
OUT_PATH = os.path.join(OUT_DIR, "cat_gate.joblib")

WIN_SEC = 2.0      # 切窗长度
HOP_SEC = 1.0      # 切窗步长（50% 重叠）
MIN_RMS = 0.004    # 静音窗丢弃阈值

from features import (extract_features, extract_features_from_signal,
                      load_audio, trim_silence, SR, FEATURE_DIM)  # noqa: E402


def slice_windows(y, win_sec=WIN_SEC, hop_sec=HOP_SEC):
    """把长录音切成固定长度窗口，丢弃能量过低的静音窗。"""
    n = int(win_sec * SR)
    h = int(hop_sec * SR)
    if len(y) < n:
        return [y]
    out = []
    for s in range(0, len(y) - n + 1, h):
        seg = y[s:s + n]
        if float(np.sqrt(np.mean(seg ** 2))) >= MIN_RMS:
            out.append(seg)
    return out or [y]

def _cat_id_from_barkmeow(path):
    return os.path.basename(path).replace(".wav", "")


def collect():
    """返回 X, y, groups, sources 以及统计信息"""
    X, y, groups, sources = [], [], [], []
    stats = {}

    def add(sig_or_path, label, group, src):
        try:
            feat = (extract_features(sig_or_path) if isinstance(sig_or_path, str)
                    else extract_features_from_signal(sig_or_path))
        except Exception as e:
            print(f"    ! 跳过 {src}: {e}")
            return
        X.append(feat); y.append(label); groups.append(group); sources.append(src)

    # ---- 正样本 ----
    n0 = len(X)
    bm_cat = sorted(glob.glob(os.path.join(DATA, "BarkMeowDB", "barkmeow", "cat", "*.wav")))
    cm_cnt = 0
    for p in bm_cat:
        y0 = load_audio(p)
        for i, w in enumerate(slice_windows(y0)):
            add(w, "meow", f"bmcat_{_cat_id_from_barkmeow(p)}", f"barkmeow/cat({i})")
            cm_cnt += 1
    stats["BarkMeowDB 猫叫（切窗）"] = cm_cnt

    n1 = len(X)
    for p in sorted(glob.glob(os.path.join(DATA, "CatMeows", "dataset", "*.wav"))):
        base = os.path.basename(p)
        add(p, "meow", f"cm_{base.split('_')[1]}", "catmeows/dataset")
    stats["CatMeows 喵叫（原样）"] = len(X) - n1
    stats["meow 合计"] = len(X) - n0

    # ---- 负样本 ----
    n2 = len(X)
    bm_dog = sorted(glob.glob(os.path.join(DATA, "BarkMeowDB", "barkmeow", "dog", "*.wav")))
    d_cnt = 0
    for p in bm_dog:
        y0 = load_audio(p)
        for i, w in enumerate(slice_windows(y0)):
            add(w, "other", f"bmdog_{_cat_id_from_barkmeow(p)}", f"barkmeow/dog({i})")
            d_cnt += 1
    stats["BarkMeowDB 狗吠（切窗）"] = d_cnt

    n3 = len(X)
    for p in sorted(glob.glob(os.path.join(DATA, "CatMeows_extras", "other_vocalizations", "*.wav"))):
        base = os.path.basename(p)
        add(p, "other", f"cmo_{base.split('_')[1]}", "catmeows/other_vocalizations")
    stats["CatMeows 非喵叫猫声"] = len(X) - n3

    n4 = len(X)
    for p in sorted(glob.glob(os.path.join(DATA, "synth", "negatives", "*.wav"))):
        kind = os.path.basename(p).replace(".wav", "").split("_")[0]
        add(p, "other", f"syn_{kind}", "synth/negatives")
    stats["合成负样本"] = len(X) - n4
    stats["other 合计"] = len(X) - n2

    return (np.array(X), np.array(y), np.array(groups), np.array(sources)), stats


def pick_threshold(probs_meow, y_true, target_recall=0.95):
    """选一个偏宽松的阈值：优先保证喵叫不被误拒（召回 >= target_recall），
    在此基础上尽量多拦掉非喵叫。"""
    m = (y_true == "meow")
    best_th, best_rej = 0.5, -1.0
    for th in np.arange(0.02, 0.99, 0.01):
        rec = float((probs_meow[m] >= th).mean())
        if rec < target_recall:
            continue
        rej = float((probs_meow[~m] < th).mean())
        if rej > best_rej:
            best_rej, best_th = rej, float(th)
    return best_th, best_rej


def main():
    print("=" * 62)
    print(" 闸门模型（是猫叫 / 不是猫叫）· 训练")
    print("=" * 62)

    (X, y, groups, sources), stats = collect()
    print("\n数据构成：")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    print(f"\n样本总数 {len(X)}  特征维度 {X.shape[1]}  分组数 {len(set(groups))}")
    u, c = np.unique(y, return_counts=True)
    print("类别分布:", {a: int(b) for a, b in zip(u, c)})

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                    class_weight="balanced", random_state=42)),
    ])

    n_groups = len(set(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    proba = cross_val_predict(pipe, X, y, groups=groups, cv=gkf, method="predict_proba")
    # cross_val_predict 的列顺序 = 标签编码顺序（sorted），即 ["meow", "other"]
    classes = sorted(set(y))
    mi = classes.index("meow")
    p_meow = proba[:, mi]

    pred = np.where(p_meow >= 0.5, "meow", "other")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score((y == "meow").astype(int), p_meow)
    print(f"\n—— 分组交叉验证（阈值 0.5）——")
    print(f"准确率 {acc:.3f}   AUC {auc:.3f}\n")
    print(classification_report(y, pred, labels=["meow", "other"],
                                target_names=["meow 是喵叫", "other 不是喵叫"], digits=3))
    print("混淆矩阵 (行=真实 meow/other, 列=预测 meow/other):")
    print(confusion_matrix(y, pred, labels=["meow", "other"]))

    th, rej = pick_threshold(p_meow, y, target_recall=0.95)
    m = (y == "meow")
    print(f"\n—— 产品推荐工作点（优先不误拒真猫叫）——")
    print(f"阈值 P(meow) >= {th:.2f}")
    print(f"  拦下非喵叫比例（other 被拒）: {rej:.1%}")
    print(f"  喵叫通过率（meow 保留）    : {(p_meow[m] >= th).mean():.1%}")

    # 各类来源在推荐工作点下的表现（看闸门对哪类最容易漏）
    print("\n按来源看「被拦下的比例」（越高越好）：")
    for s in sorted(set(sources)):
        sel = sources == s
        if not sel.any():
            continue
        rate = float((p_meow[sel] < th).mean())
        print(f"  {s:<34} n={sel.sum():<5} 拦截率 {rate:6.1%}")

    pipe.fit(X, y)
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "classes": list(pipe.classes_),
        "threshold": float(th),
        "feature_dim": FEATURE_DIM,
        "cv_accuracy": float(acc),
        "cv_auc": float(auc),
        "target_recall": 0.95,
        "note": "闸门：判断是否为猫的喵叫。阈值已按「召回>=95%」调优，偏宽松。",
    }, OUT_PATH)
    print(f"\n模型已保存: {os.path.abspath(OUT_PATH)}")

    meta = {"samples": int(len(X)), "groups": int(n_groups),
            "cv_accuracy": round(float(acc), 4), "cv_auc": round(float(auc), 4),
            "threshold": round(float(th), 3), "stats": stats}
    with open(os.path.join(OUT_DIR, "cat_gate_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"评估元数据: {os.path.abspath(os.path.join(OUT_DIR, 'cat_gate_meta.json'))}")


if __name__ == "__main__":
    main()
