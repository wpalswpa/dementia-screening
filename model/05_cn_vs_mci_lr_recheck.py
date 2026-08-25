"""
Q4 추가 확인: 주 모델(model/01)의 분류기를 RandomForest에서 LogisticRegression으로
바꾼 뒤, "치매를 뺀 CN vs MCI에서는 신호가 없다"는 Q4 결론(model/02·03, RandomForest
기반 Nested CV)이 새 분류기에서도 그대로인지 재확인한다.

프로토콜은 model/01과 완전히 동일하다(매 fold 학습 데이터로만 RandomForest 중요도
상위 10개 피처 선택 → StandardScaler+LogisticRegression 학습 → 평가 데이터로 측정).
대상만 CN+MCI 162명(치매 12명 제외)으로 좁혔다.

결과(실행 로그 및 reports/q5_lr_recheck.json):
  기본 44개 피처, 시계열 포함 50개 피처 모두 정확도가 베이스라인(68.5%)에 못 미치고
  AUC도 0.52~0.54 수준 — 분류기를 바꿔도 Q4 결론(MCI 단독 조기 신호는 현재 피처로는
  못 찾음)은 그대로다.
"""
import json
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")
V2_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table_v2.csv")
OUT_JSON = os.path.join(BASE_DIR, "reports", "q5_lr_recheck.json")

NON_FEATURE_COLS = ["EMAIL", "split", "DIAG_NM", "diag2class"]
TOP_K = 10
N_SPLITS, N_REPEATS = 5, 10


def make_scout():
    return RandomForestClassifier(random_state=42, class_weight="balanced")


def make_classifier():
    # model/01의 최종 분류기와 동일한 구성
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )


def run(path, label):
    df = pd.read_csv(path)
    df = df[df["DIAG_NM"].isin(["CN", "MCI"])].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["DIAG_NM"]
    print(f"\n=== {label}: {len(df)}명 (CN {(y == 'CN').sum()} / MCI {(y == 'MCI').sum()}), 피처 {len(feature_cols)}개 ===")

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    rows = []
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scout = make_scout()
        scout.fit(X_train, y_train)
        importances = pd.Series(scout.feature_importances_, index=feature_cols)
        top_cols = importances.sort_values(ascending=False).head(TOP_K).index.tolist()

        model = make_classifier()
        model.fit(X_train[top_cols], y_train)
        pred = model.predict(X_test[top_cols])
        proba = model.predict_proba(X_test[top_cols])[:, list(model.classes_).index("MCI")]

        rows.append({
            "accuracy": accuracy_score(y_test, pred),
            "baseline_accuracy": (y_test == "CN").mean(),
            "mci_recall": recall_score(y_test, pred, pos_label="MCI", zero_division=0),
            "roc_auc": roc_auc_score((y_test == "MCI").astype(int), proba),
        })

    results = pd.DataFrame(rows)
    summary = {}
    for col in results.columns:
        print(f"  {col}: {results[col].mean():.3f} ± {results[col].std():.3f}")
        summary[col] = {"mean": round(results[col].mean(), 4), "std": round(results[col].std(), 4)}
    return summary


def main():
    out = {
        "base_44_features": run(BASE_PATH, "기본 44개 피처 + LogisticRegression"),
        "v2_50_features": run(V2_PATH, "시계열 포함 50개 피처 + LogisticRegression"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {OUT_JSON}")


if __name__ == "__main__":
    main()
