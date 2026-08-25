"""
Q5: CN vs CI(MCI+Dem) 결과가 우연이 아닌지 통계적으로 확인한다 (Permutation Test).

절차: 라벨(diag2class)을 무작위로 섞은 뒤, 이미 확정된 동일 피처 세트로 동일한
반복교차검증을 돌려 "우연히 나올 수 있는 성능"의 분포를 만든다. 이걸 500번
반복해서 우연 분포를 만들고, 그 안에서 우리가 실제로 관측한 성능이 상위 몇 %에
위치하는지로 p-value를 계산한다. p < 0.05면 "우연이라고 보기 어렵다"고 말할 수 있다.

주의: model/01이 먼저 실행되어 reports/model_metrics.json(선택된 피처 목록)이
있어야 동작한다.

출력: reports/permutation_test_result.json
      reports/figures/11_permutation_test.png
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "model_metrics.json")
OUT_JSON = os.path.join(BASE_DIR, "reports", "permutation_test_result.json")
OUT_FIG = os.path.join(BASE_DIR, "reports", "figures", "11_permutation_test.png")

N_PERMUTATIONS = 500
N_SPLITS, N_REPEATS_PER_PERMUTATION = 5, 3  # 실측 관측치는 10회였지만, 500번 반복해야 하니 3회로 축소(총 1,500 CV 평가는 동일 급)


def make_model():
    # model/01의 최종 분류기와 동일한 구성 — 검증 대상 모델과 검증에 쓰는 모델이 같아야 한다.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )


def mean_cv_score(X, y, n_splits, n_repeats, seed):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    accs, recalls = [], []
    for train_idx, test_idx in rskf.split(X, y):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]
        accs.append(accuracy_score(y_test, pred))
        recalls.append(recall_score(y_test, pred, pos_label="CI", zero_division=0))
    return np.mean(accs), np.mean(recalls)


def main():
    df = pd.read_csv(FEATURE_PATH)
    with open(METRICS_PATH, encoding="utf-8") as f:
        model_metrics = json.load(f)
    top_cols = list(model_metrics["selected_features"].keys())
    print(f"이미 확정된 TOP {len(top_cols)} 피처로 검증: {top_cols}")

    X = df[top_cols]
    y = df["diag2class"]

    # 1) 실제 라벨로 관측한 성능(참값, 시드 고정)
    observed_acc, observed_recall = mean_cv_score(X, y, N_SPLITS, N_REPEATS_PER_PERMUTATION, seed=42)
    print(f"\n실제 라벨 기준 관측 성능: 정확도 {observed_acc:.3f}, CI재현율 {observed_recall:.3f}")

    # 2) 라벨을 무작위로 섞어 "우연 분포"를 500번 쌓는다
    null_accs, null_recalls = [], []
    for i in range(N_PERMUTATIONS):
        y_shuffled = y.sample(frac=1, random_state=i).reset_index(drop=True)
        acc, rec = mean_cv_score(X, y_shuffled, N_SPLITS, N_REPEATS_PER_PERMUTATION, seed=100 + i)
        null_accs.append(acc)
        null_recalls.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  permutation {i + 1}/{N_PERMUTATIONS} 완료")

    null_accs = np.array(null_accs)
    null_recalls = np.array(null_recalls)

    p_acc = (np.sum(null_accs >= observed_acc) + 1) / (N_PERMUTATIONS + 1)
    p_recall = (np.sum(null_recalls >= observed_recall) + 1) / (N_PERMUTATIONS + 1)

    print(f"\n[Permutation Test 결과, {N_PERMUTATIONS}회]")
    print(f"  우연 분포 정확도: 평균 {null_accs.mean():.3f} ± {null_accs.std():.3f}")
    print(f"  실제 관측 정확도: {observed_acc:.3f}  →  p = {p_acc:.4f}")
    print(f"  우연 분포 CI재현율: 평균 {null_recalls.mean():.3f} ± {null_recalls.std():.3f}")
    print(f"  실제 관측 CI재현율: {observed_recall:.3f}  →  p = {p_recall:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, null_dist, obs, p, title in [
        (axes[0], null_accs, observed_acc, p_acc, "정확도"),
        (axes[1], null_recalls, observed_recall, p_recall, "CI 재현율"),
    ]:
        ax.hist(null_dist, bins=30, color="#8172B2", alpha=0.75, label="라벨을 섞었을 때(우연)")
        ax.axvline(obs, color="#C44E52", linewidth=2.5, label=f"실제 관측값 (p={p:.3f})")
        ax.set_title(f"{title} — Permutation Test")
        ax.set_xlabel(title)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)

    result = {
        "n_permutations": N_PERMUTATIONS,
        "features_used": top_cols,
        "observed": {"accuracy": round(float(observed_acc), 4), "ci_recall": round(float(observed_recall), 4)},
        "null_distribution": {
            "accuracy": {"mean": round(float(null_accs.mean()), 4), "std": round(float(null_accs.std()), 4)},
            "ci_recall": {"mean": round(float(null_recalls.mean()), 4), "std": round(float(null_recalls.std()), 4)},
        },
        "p_value": {"accuracy": round(float(p_acc), 4), "ci_recall": round(float(p_recall), 4)},
        "significant_at_0.05": {"accuracy": bool(p_acc < 0.05), "ci_recall": bool(p_recall < 0.05)},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {OUT_JSON}, {OUT_FIG}")


if __name__ == "__main__":
    main()
