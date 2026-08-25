"""
feature_table.csv로 CN vs CI(MCI+Dem) 스크리닝 모델을 학습·평가한다.

원칙은 하나뿐이다: **매 반복마다 학습 데이터로만 피처를 고르고, 따로 떼어둔
평가 데이터로만 성능을 잰다.** 그 외의 기교(K 후보를 여러 개 놓고 안쪽에서
탐색하는 것 등)는 쓰지 않는다 — 그래야 어디서 무슨 정보가 새는지 코드만 보고도
바로 알 수 있다.

절차 (fold 50개 = 5-fold x 10회 반복, 매번 동일):
  1) 학습 데이터(X_train)로 RandomForest를 한 번 학습해서 44개 피처 중요도를 얻는다.
     (RandomForest는 "어떤 피처가 중요한가"를 보는 용도로만 쓴다 — feature_importances_가
     해석하기 쉽기 때문)
  2) 그 중요도로 상위 10개만 고른다. (테스트 데이터는 이 단계 어디에도 안 들어간다)
  3) 상위 10개 피처를 표준화(StandardScaler, 학습 데이터 기준으로만 fit)한 뒤
     LogisticRegression으로 다시 학습해서, 따로 떼어둔 평가 데이터(X_test)로 성능을 잰다.
     (RandomForest를 최종 분류기로도 써봤는데 174명짜리 작은 표본+상관된 피처 조합에서
     LogisticRegression보다 확연히 불안정했다 — AUC 0.549 vs 0.601, 재현율 38.7% vs 52.7%.
     "더 복잡한 모델이 항상 낫다"는 가정 자체가 틀렸던 경우라 더 단순한 모델로 바꿨다.)
  4) 이번 fold에서 뽑힌 10개 피처를 기록해둔다 — 나중에 "50번 중 몇 번 뽑혔는지"로
     안정성을 보고한다(Q5 검증과 같은 방식).

정확도(accuracy)는 주 지표로 안 쓴다: class_weight='balanced'를 쓰면 "위험군을 놓치지
않는 대신 정상을 위험군으로 더 잘못 찍는" 쪽으로 일부러 기운 모델이 되어 정확도가
구조적으로 낮게 나오기 때문이다. 대신 임계값에 안 묶이고 "위험군일수록 점수를 높게
매기는가"만 보는 AUC(ROC-AUC)를 스크리닝 성능의 주 지표로 삼는다. 정확도/재현율/정밀도는
참고 지표로 함께 보고한다.

출력: reports/figures/04_feature_importance.png
      data/processed/model_cn_ci.pkl (배포용, 전체 174명 + 가장 많이 뽑힌 10개 피처로 재학습)
"""
import json
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "model_cn_ci.pkl")
FIG_PATH = os.path.join(BASE_DIR, "reports", "figures", "04_feature_importance.png")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "model_metrics.json")

NON_FEATURE_COLS = ["EMAIL", "split", "DIAG_NM", "diag2class"]
TOP_K = 10
N_SPLITS, N_REPEATS = 5, 10  # 5-fold x 10회 = 50번 반복


def make_scout():
    # 피처 중요도만 보는 용도. CN 111 : CI 63으로 쏠려 있어 class_weight='balanced' 사용.
    return RandomForestClassifier(random_state=42, class_weight="balanced")


def make_classifier():
    # 최종 판정용. 표본이 작아(174명) 규제가 강하고 단순한 모델이 더 안정적으로 잘 됐다.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )


def main():
    df = pd.read_csv(FEATURE_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["diag2class"]

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    rows, feature_hits, importance_sums = [], Counter(), pd.Series(0.0, index=feature_cols)

    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 1) 학습 데이터만으로 44개 피처 전체 중요도를 본다.
        scout = make_scout()
        scout.fit(X_train, y_train)
        importances = pd.Series(scout.feature_importances_, index=feature_cols)
        importance_sums += importances

        # 2) 상위 10개만 고른다. 테스트 데이터는 아직 한 번도 쓰이지 않았다.
        top_cols = importances.sort_values(ascending=False).head(TOP_K).index.tolist()
        feature_hits.update(top_cols)

        # 3) 상위 10개로 LogisticRegression을 다시 학습해서, 따로 떼어둔 평가 데이터로만 성능을 잰다.
        #    StandardScaler도 파이프라인 안에서 X_train에만 fit되므로 누수가 없다.
        model = make_classifier()
        model.fit(X_train[top_cols], y_train)
        pred = model.predict(X_test[top_cols])
        proba_ci = model.predict_proba(X_test[top_cols])[:, list(model.classes_).index("CI")]

        rows.append({
            "accuracy": accuracy_score(y_test, pred),
            "baseline_accuracy": (y_test == "CN").mean(),
            "ci_recall": recall_score(y_test, pred, pos_label="CI", zero_division=0),
            "ci_precision": precision_score(y_test, pred, pos_label="CI", zero_division=0),
            "roc_auc": roc_auc_score((y_test == "CI").astype(int), proba_ci),
        })

    results = pd.DataFrame(rows)
    n_folds = len(results)
    print(f"[CN vs CI, {n_folds}회 반복] 매 fold마다 학습 데이터로 10개 피처를 새로 고르고 평가함")
    for col in ["accuracy", "baseline_accuracy", "ci_recall", "ci_precision", "roc_auc"]:
        print(f"  {col}: {results[col].mean():.3f} ± {results[col].std():.3f}")

    # 50번 중 가장 자주 뽑힌 순서로 피처를 정리한다. 이 순위가 "왜 위험군인지"의 근거가 된다.
    stability = (pd.Series(feature_hits) / n_folds).sort_values(ascending=False)
    top_stable_cols = stability.head(TOP_K).index.tolist()
    print(f"\n{TOP_K}개 피처 선택 빈도 (50번 중 몇 번 뽑혔는지):")
    print(stability.head(TOP_K))

    fig, ax = plt.subplots(figsize=(6, 5))
    stability.head(TOP_K).sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"위험군(CI) 판별에 중요한 라이프로그 지표 TOP {TOP_K}")
    ax.set_xlabel(f"{n_folds}번 반복 중 선택된 비율")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)

    # 배포용 모델은 가장 안정적으로 뽑힌 10개 피처로, 174명 전체를 다시 학습한다.
    final_model = make_classifier()
    final_model.fit(X[top_stable_cols], y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": top_stable_cols}, f)
    print(f"\n배포용 모델 저장 완료(174명 전체로 재학습): {MODEL_PATH}")

    metrics = {
        "n_people": len(df),
        "diag_nm_counts": df["DIAG_NM"].value_counts().to_dict(),
        "diag2class_counts": df["diag2class"].value_counts().to_dict(),
        "model": "LogisticRegression(class_weight='balanced', StandardScaler 포함) "
                 "— 피처 중요도 산출용 RandomForest는 별도로 씀",
        "cv_setting": f"RepeatedStratifiedKFold({N_SPLITS}-fold x {N_REPEATS} repeats), "
                       "매 fold마다 학습 데이터로만 상위 10개 피처 선택 후 평가 데이터로 검증",
        "performance": {c: {"mean": round(results[c].mean(), 4), "std": round(results[c].std(), 4)}
                        for c in ["accuracy", "baseline_accuracy", "ci_recall", "ci_precision", "roc_auc"]},
        "selected_features": {name: round(val, 3) for name, val in stability.head(TOP_K).items()},
        "average_importance_all_features": (importance_sums / n_folds).round(5).to_dict(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"평가 지표 저장 완료: {METRICS_PATH}")


if __name__ == "__main__":
    main()
