# -*- coding: utf-8 -*-
"""
训练猫叫情境/情绪分类器（MVP）
- 数据: CatMeows（440段，三类情境 B/F/I）
- 特征: MFCC + delta + 统计池化（见 features.py）
- 模型: 标准化 + SVM(RBF, probability=True)
- 评估: 按「猫 ID」分组交叉验证（GroupKFold），避免同一只猫的样本泄漏到训练/测试两侧
- 产物: models/cat_emotion_model.joblib（内含 scaler + svm + 标签映射）
"""
import os
import sys
import glob
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "CatMeows", "dataset")
MODEL_DIR = os.path.join(HERE, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cat_emotion_model.joblib")

# 情境 → 中文语义标签
CONTEXT_MAP = {
    "B": ("brushing", "被抚摸/梳毛", "满足、放松"),
    "F": ("food", "等待食物", "期待、索求"),
    "I": ("isolation", "陌生环境独处", "不安、焦虑"),
}

from features import extract_features, FEATURE_DIM  # noqa: E402


def load_dataset():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    if not files:
        raise SystemExit(f"未找到音频文件: {DATA_DIR}")

    X, y, groups, names = [], [], [], []
    for f in files:
        base = os.path.basename(f)
        parts = base.split("_")
        ctx = parts[0]            # B / F / I
        cat_id = parts[1]         # 猫的唯一ID，用于分组
        if ctx not in CONTEXT_MAP:
            continue
        X.append(extract_features(f))
        y.append(ctx)
        groups.append(cat_id)
        names.append(base)

    return np.array(X), np.array(y), np.array(groups), names


def main():
    print("=" * 56)
    print(" 猫叫情境分类器 · 训练")
    print("=" * 56)

    X, y, groups, names = load_dataset()
    print(f"样本数: {len(X)}  特征维度: {X.shape[1]}  猫数(分组): {len(set(groups))}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("类别分布:", {c: int(n) for c, n in zip(uniq, cnt)})

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, class_weight="balanced",
                    random_state=42)),
    ])

    # 分组交叉验证：同一只猫要么在训练集、要么在测试集
    n_groups = len(set(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=gkf)

    acc = accuracy_score(y, y_pred)
    print(f"\n—— 分组交叉验证准确率: {acc:.3f} ——\n")
    print(classification_report(
        y, y_pred,
        labels=["B", "F", "I"],
        target_names=[f"{c}({CONTEXT_MAP[c][1]})" for c in ["B", "F", "I"]],
        digits=3))

    cm = confusion_matrix(y, y_pred, labels=["B", "F", "I"])
    print("混淆矩阵 (行=真实 B/F/I, 列=预测 B/F/I):")
    print(cm)

    # 用全部数据训练最终模型并保存
    pipe.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "classes": list(pipe.classes_),
        "context_map": CONTEXT_MAP,
        "feature_dim": FEATURE_DIM,
        "cv_accuracy": float(acc),
    }, MODEL_PATH)
    print(f"\n模型已保存: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
