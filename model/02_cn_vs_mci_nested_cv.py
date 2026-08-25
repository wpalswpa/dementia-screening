"""
검증 항목 ②+③+④를 한 번에 다룬다.

② CN vs MCI 단독 재검증
   지금까지의 "CN vs CI(=MCI+Dem)" 결과는 Dem 12명의 극단적 차이(MMSE 16.6 vs
   CN 27.7)가 신호를 떠받치고 있을 수 있다. 이 프로젝트가 필요한 건 "이미 뚜렷한
   Dem을 찾아내는 것"이 아니라 "아직 애매한 MCI를 조기에 걸러내는 것"이므로,
   Dem을 완전히 제외하고 CN vs MCI만으로 같은 파이프라인을 재실행해서 신호가
   살아있는지 확인한다.

③ Nested CV로 K 선택 누수 제거
   기존 TOP_K=10은 예전에 테스트셋 정확도를 보고 고른 값이 상수로 박혀있어
   오염이 남아있었다. 여기서는 바깥 루프(성능 추정)와 안쪽 루프(K 선택)를
   완전히 분리해서, 바깥 테스트 fold는 K 선택 과정에 전혀 관여하지 않는다.

④ 피처 안정성 추적
   바깥 fold(50회)마다 안쪽에서 뽑힌 top 피처 목록을 전부 기록해서,
   "각 피처가 50번 중 몇 번이나 선택됐는지" 비율을 낸다. 순위가 아니라
   선택 빈도로 안정성을 본다 — 매번 다른 피처가 뽑히면 근거로 쓸 수 없다.

출력: reports/model_metrics_cn_vs_mci.json
      reports/figures/09_feature_stability_cn_vs_mci.png
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
FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")
FIG_PATH = os.path.join(BASE_DIR, "reports", "figures", "09_feature_stability_cn_vs_mci.png")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "model_metrics_cn_vs_mci.json")

NON_FEATURE_COLS = ["EMAIL", "split", "DIAG_NM", "diag2class"]
POS_LABEL = "MCI"
K_CANDIDATES = [5, 8, 10, 15, 20, 30, 44]  # 안쪽 루프가 이 중에서 고른다
N_OUTER_SPLITS, N_OUTER_REPEATS = 5, 10
N_INNER_SPLITS = 4


def make_model():
    return RandomForestClassifier(random_state=42, class_weight="balanced")


def inner_select_k_and_features(X_train, y_train):
    """안쪽 CV로 K 후보 중 최적을 고르고, 그 K에서의 피처 목록을 반환한다.
    바깥 테스트 fold는 이 함수 안 어디에도 들어오지 않는다."""
    inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=1)

    # 1) 안쪽 fold들의 학습 부분에서만 피처 중요도를 평균 내 순위를 만든다.
    importances = pd.DataFrame(index=X_train.columns)
    for i, (tr_idx, _) in enumerate(inner_cv.split(X_train, y_train)):
        m = make_model()
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        importances[f"f{i}"] = m.feature_importances_
    ranked = importances.mean(axis=1).sort_values(ascending=False)

    # 2) 각 K 후보를 안쪽 CV 정확도로 비교해서 제일 좋은 K를 고른다.
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
    df = df[df["DIAG_NM"] != "Dem"].copy()  # ② Dem 완전 제외
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["DIAG_NM"]  # CN vs MCI

    print(f"CN vs MCI 재검증 (Dem 제외): {y.value_counts().to_dict()}")

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

    print(f"\n[CN vs MCI, Nested CV, {n_folds}회 outer fold]")
    for col in ["accuracy", "baseline_accuracy", "mci_recall", "mci_precision"]:
        print(f"  {col}: {results[col].mean():.3f} ± {results[col].std():.3f}")
    print(f"\n안쪽 루프가 고른 K 분포: {Counter(chosen_k_list)}")

    stability = pd.Series(feature_hits) / n_folds
    stability = stability.sort_values(ascending=False)
    print(f"\n피처 선택 빈도 TOP 15 (전체 {n_folds}회 outer fold 중 몇 번 뽑혔는지):")
    print(stability.head(15))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    stability.head(15).sort_values().plot.barh(ax=ax, color="#8172B2")
    ax.set_title(f"CN vs MCI 피처 선택 빈도 TOP 15 ({n_folds}회 outer fold 중)")
    ax.set_xlabel("선택된 비율")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)

    metrics = {
        "comparison": "CN vs MCI (Dem excluded)",
        "n_cn": int((y == "CN").sum()),
        "n_mci": int((y == "MCI").sum()),
        "nested_cv": f"outer {N_OUTER_SPLITS}-fold x {N_OUTER_REPEATS} repeats, inner {N_INNER_SPLITS}-fold",
        "performance": {c: {"mean": round(results[c].mean(), 4), "std": round(results[c].std(), 4)}
                        for c in ["accuracy", "baseline_accuracy", "mci_recall", "mci_precision"]},
        "chosen_k_distribution": {str(k): v for k, v in Counter(chosen_k_list).items()},
        "feature_selection_frequency": stability.round(3).to_dict(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {METRICS_PATH}, {FIG_PATH}")


if __name__ == "__main__":
    main()
