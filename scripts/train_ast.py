# -*- coding: utf-8 -*-
"""
在 AST 768 维 embedding 上训练情绪分类 SVM，另存为 cat_emotion_model_ast.joblib。
embedding 来自 experiment_transfer_cls.py 用官方 ASTFeatureExtractor 抽取并缓存的
data/_emb_cache/emb_ast_mean.npy（与 ast_features.py 的 numpy 预处理逐元素一致，已验证）。
产物与 cat_emotion_model.joblib 同结构，predict.py 无缝切换。
"""
import os
import sys
import pickle
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CACHE = os.path.join(ROOT, "data", "_emb_cache")
MODEL_DIR = os.path.join(ROOT, "models")
OUT = os.path.join(MODEL_DIR, "cat_emotion_model_ast.joblib")

import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

CONTEXT_MAP = {
    "B": ("brushing", "被抚摸/梳毛", "满足、放松"),
    "F": ("food", "等待食物", "期待、索求"),
    "I": ("isolation", "陌生环境独处", "不安、焦虑"),
}


def main():
    E = np.load(os.path.join(CACHE, "emb_ast_mean.npy"))
    meta = pickle.load(open(os.path.join(CACHE, "meta_ast_cls.pkl"), "rb"))
    y, groups = meta["y"], meta["groups"]
    print(f"样本 {len(E)}  维度 {E.shape[1]}  猫数 {len(set(groups))}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, class_weight="balanced", random_state=42)),
    ])
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    y_pred = cross_val_predict(pipe, E, y, groups=groups, cv=gkf)
    acc = accuracy_score(y, y_pred)
    print(f"\n—— AST embedding 分组CV准确率: {acc:.3f} ——\n")
    print(classification_report(y, y_pred, labels=["B", "F", "I"],
                                target_names=["B满足", "F期待", "I不安"], digits=3))

    pipe.fit(E, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "classes": list(pipe.classes_),
        "context_map": CONTEXT_MAP,
        "feature_dim": int(E.shape[1]),
        "feature_type": "ast_embedding_768_meanpool",
        "cv_accuracy": float(acc),
        "note": "AST(AudioSet) 768维时间平均隐藏态 + SVM；对照 MFCC 版 0.564",
    }, OUT)
    print(f"模型已保存: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
