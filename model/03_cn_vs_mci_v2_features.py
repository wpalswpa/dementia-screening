"""
model/02와 완전히 동일한 방법론(Nested CV, K는 안쪽에서 선택, 피처 안정성 추적)을
새로 만든 시계열 피처(feature_table_v2.csv: 요일차이/추세/자기상관/구성비변화)까지
포함한 데이터에 다시 적용한다. 기존 44개 피처로는 CN vs MCI 신호가 베이스라인보다
못했는데, 새 피처를 추가하면 달라지는지 확인한다.

출력: reports/model_metrics_cn_vs_mci_v2.json
      reports/figures/10_feature_stability_cn_vs_mci_v2.png
"""
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table_v2.csv")
FIG_PATH = os.path.join(BASE_DIR, "reports", "figures", "10_feature_stability_cn_vs_mci_v2.png")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "model_metrics_cn_vs_mci_v2.json")

NON_FEATURE_COLS = ["EMAIL", "split", "DIAG_NM", "diag2class"]
POS_LABEL = "MCI"
K_CANDIDATES = [5, 8, 10, 15, 20, 30, 50]
N_OUTER_SPLITS, N_OUTER_REPEATS = 5, 10
N_INNER_SPLITS = 4

NEW_FEATURES = [
    "weekday_weekend_steps_diff", "steps_trend_slope", "sleep_efficiency_trend_slope",
    "steps_autocorr_lag1", "sleep_efficiency_autocorr_lag1", "activity_composition_shift",
]


def make_model():
    return RandomForestClassifier(random_state=42, class_weight="balanced")


def inner_select_k_and_features(X_train, y_train):
    inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=1)
    importances = pd.DataFrame(index=X_train.columns)
    for i, (tr_idx, _) in enumerate(inner_cv.split(X_train, y_train)):
        m = make_model()
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        importances[f"f{i}"] = m.feature_importances_
    ranked = importances.mean(axis=1).sort_values(ascending=False)

    best_k, best_score = None, -1
    for k in K_CANDIDATES:
        cols = ranked.head(k).index.tolist()
        scores = []
        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            m = make_model()
            m.fit(X_train.iloc[tr_idx][cols], y_train.iloc[tr_idx])
            pred = m.predict(X_train.iloc[val_idx][cols])
            scores.append(accuracy_score(y_train.iloc[val_idx], pred))
        mean_score = sum(scores) / len(scores)
        if mean_score > best_score:
            best_score, best_k = mean_score, k
    return ranked.head(best_k).index.tolist(), best_k


def main():
    df = pd.read_csv(FEATURE_PATH)
    df = df[df["DIAG_NM"] != "Dem"].copy()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["DIAG_NM"]

    print(f"CN vs MCI 재검증 v2 (44개 기존 + {len(NEW_FEATURES)}개 신규 = {len(feature_cols)}개 피처): {y.value_counts().to_dict()}")

    outer_cv = RepeatedStratifiedKFold(n_splits=N_OUTER_SPLITS, n_repeats=N_OUTER_REPEATS, random_state=42)
    rows, chosen_k_list, feature_hits = [], [], Counter()

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cols, k = inner_select_k_and_features(X_train, y_train)
        chosen_k_list.append(k)
        feature_hits.update(cols)

        model = make_model()
        model.fit(X_train[cols], y_train)
        pred = model.predict(X_test[cols])

        rows.append({
            "accuracy": accuracy_score(y_test, pred),
            "baseline_accuracy": (y_test == "CN").mean(),
            "mci_recall": recall_score(y_test, pred, pos_label=POS_LABEL, zero_division=0),
            "mci_precision": precision_score(y_test, pred, pos_label=POS_LABEL, zero_division=0),
        })
        if (fold_i + 1) % 10 == 0:
            print(f"  outer fold {fold_i + 1}/{N_OUTER_SPLITS * N_OUTER_REPEATS} 완료")

    results = pd.DataFrame(rows)
    n_folds = len(results)

    print(f"\n[CN vs MCI v2, Nested CV, {n_folds}회 outer fold]")
    for col in ["accuracy", "baseline_accuracy", "mci_recall", "mci_precision"]:
        print(f"  {col}: {results[col].mean():.3f} ± {results[col].std():.3f}")
    print(f"\n안쪽 루프가 고른 K 분포: {Counter(chosen_k_list)}")

    stability = (pd.Series(feature_hits) / n_folds).sort_values(ascending=False)
    print(f"\n피처 선택 빈도 TOP 15:")
    print(stability.head(15))
    new_in_top15 = [c for c in stability.head(15).index if c in NEW_FEATURES]
    print(f"\n신규 피처 중 TOP 15에 든 것: {new_in_top15}")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = ["#DD8452" if c in NEW_FEATURES else "#8172B2" for c in stability.head(15).sort_values().index]
    stability.head(15).sort_values().plot.barh(ax=ax, color=colors)
    ax.set_title(f"CN vs MCI(v2) 피처 선택 빈도 TOP 15 (주황=신규 피처)")
    ax.set_xlabel("선택된 비율")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)

    metrics = {
        "comparison": "CN vs MCI v2 (Dem excluded, + time-series features)",
        "n_cn": int((y == "CN").sum()),
        "n_mci": int((y == "MCI").sum()),
        "n_features_total": len(feature_cols),
        "performance": {c: {"mean": round(results[c].mean(), 4), "std": round(results[c].std(), 4)}
                        for c in ["accuracy", "baseline_accuracy", "mci_recall", "mci_precision"]},
        "chosen_k_distribution": {str(k): v for k, v in Counter(chosen_k_list).items()},
        "feature_selection_frequency": stability.round(3).to_dict(),
        "new_features_in_top15": new_in_top15,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {METRICS_PATH}, {FIG_PATH}")


if __name__ == "__main__":
    main()
