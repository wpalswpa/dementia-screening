"""
================================================================================
 치매 위험 조기 스크리닝 — 전체 분석 파이프라인
 팀명: 찾아조 (치매 위험 조기 스크리닝 서비스)
================================================================================

무엇을 하는 코드인가
--------------------
AI Hub "치매 고위험군 웨어러블 라이프로그" 데이터(174명 — 정상 111 / 경도인지장애 51 /
치매 12)를 입력으로 받아, 아래 순서를 그대로 실행한다.

  1단계) 전처리 — 활동/수면 원본을 사람×일 단위로 합치고, 반지를 오래 벗고 있던
         믿을 수 없는 날을 제외한다.
  2단계) 피처 생성 — 사람마다 "평소 수준(평균)"과 "날마다의 들쭉날쭉함(표준편차)"을
         계산해 한 사람당 한 줄짜리 피처 테이블(44개 피처)을 만든다.
  3단계) 탐색 분석(EDA) — 진단군(CN/MCI/Dem)별로 활동량·수면효율·MMSE 평균을 비교한다.
  4단계) 핵심 모델(Q3·Q4) — 정상군 vs 위험군(경도인지장애+치매)을 구분하는 모델을
         학습하고, 반복 교차검증으로 정직하게 성능을 재고, 어떤 피처가 왜 중요한지 확인한다.
  5단계) Q5 검증 — "④번 결과가 사실은 이미 뚜렷한 치매 12명 덕분 아닌가?"라는 의심을
         풀기 위해, 치매를 완전히 빼고 정상군 vs 경도인지장애만으로 같은 방식을 재검증한다.
  6단계) Q6 검증 — "④번 성능 개선이 우연 아닌가?"를 확인하기 위해, 라벨을 무작위로
         섞은 가짜 데이터 500벌과 비교하는 순열 검정(Permutation Test)을 수행한다.

이 파일 하나로 전체 분석 절차(Q1~Q6)를 처음부터 끝까지 재현할 수 있다. 원본 프로젝트
저장소에서는 이 단계들이 preprocessing/·model/ 폴더 아래 독립 스크립트로 나뉘어 있는데,
제출용으로 하나로 합치면서 각 단계가 "왜 그렇게 했는지"를 주석으로 자세히 남겼다.

핵심 설계 원칙 — 데이터 누수(leakage) 방지
------------------------------------------
이 프로젝트에서 가장 신경 쓴 부분이다. 모델 성능을 측정할 때 아래 두 가지를 항상 지킨다.
  - "어떤 피처를 쓸지 고르는 단계"와 "성능을 재는 단계"는 반드시 서로 다른 사람들의
    데이터를 쓴다. 같은 사람이 두 단계에 겹쳐 들어가면, 답안지를 미리 보고 시험을
    치는 것과 같아서 성능이 실제보다 부풀려진다.
  - 성능은 단 한 번의 데이터 분할이 아니라, 여러 번 다르게 나눠서 반복 평가한
    평균±표준편차로만 보고한다. 한 번의 운 좋은(또는 나쁜) 분할에 좌우되지 않기 위함이다.

실행 방법
---------
    pip install pandas numpy scikit-learn matplotlib
    python 소스코드.py

이 스크립트와 같은 폴더에 raw_data/01_aihub_wearable/ 원본 데이터가 있어야 한다.
결과물은 원본 프로젝트 결과와 섞이지 않도록 submission_output/data/, submission_output/reports/
아래에 별도로 저장된다.
================================================================================
"""
import json
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 깨짐 방지 (Windows 기준)
plt.rcParams["axes.unicode_minus"] = False


# ==============================================================================
# 0. 공통 설정 — 경로와 상수
# ==============================================================================

# 이 스크립트 파일이 있는 폴더를 기준으로 모든 경로를 잡는다(컴퓨터가 바뀌어도 그대로 동작).
# 출력은 원본 프로젝트의 data/processed·reports와 완전히 분리된 submission_output/ 아래에
# 저장한다 — 이 스크립트를 실행해도 팀이 이미 검증해둔 결과 파일을 덮어쓰지 않기 위해서다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_BASE = os.path.join(BASE_DIR, "raw_data", "01_aihub_wearable")
OUTPUT_DIR = os.path.join(BASE_DIR, "submission_output")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "data")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
for d in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

DAILY_CLEAN_PATH = os.path.join(PROCESSED_DIR, "lifelog_daily_clean.csv")
FEATURE_TABLE_PATH = os.path.join(PROCESSED_DIR, "feature_table.csv")
MODEL_PATH = os.path.join(PROCESSED_DIR, "model_cn_ci.pkl")

# 반지를 180분(3시간) 이상 벗고 있었던 날은 그날 기록을 믿을 수 없어 제외한다.
NON_WEAR_LIMIT_MIN = 180

# 사람 단위로 평균/표준편차를 낼 활동·수면 원본 지표 18개
AGG_COLS = [
    "activity_steps", "activity_score", "activity_total",
    "activity_high", "activity_medium", "activity_low", "activity_inactive",
    "activity_cal_total", "activity_average_met",
    "sleep_score", "sleep_efficiency", "sleep_total",
    "sleep_deep", "sleep_light", "sleep_rem",
    "sleep_onset_latency", "sleep_restless", "sleep_hr_average",
]

NON_FEATURE_COLS = ["EMAIL", "split", "DIAG_NM", "diag2class"]
TOP_K = 10                      # 최종 모델이 쓰는 피처 개수
N_SPLITS, N_REPEATS = 5, 10     # 반복 교차검증: 5조각 x 10회 = 50번 평가
RANDOM_STATE = 42


# ==============================================================================
# 1단계. 전처리 — 활동/수면 원본을 사람×일 단위로 정제
# ==============================================================================

def _load_split(folder_name, file_name, cols):
    """Training/Validation 폴더 하나에서 활동 또는 수면 CSV를 읽는다."""
    sub = "1.걸음걸이" if "activity" in file_name else "2.수면"
    path = os.path.join(RAW_BASE, folder_name, "원천데이터", sub, file_name)
    df = pd.read_csv(path, usecols=cols)
    df["split"] = "train" if folder_name.startswith("1") else "val"
    return df


def clean_lifelog():
    """
    1단계: raw_data의 활동·수면 CSV를 사람×일 단위로 병합하고,
    믿을 수 없는 날(반지 장시간 미착용)을 제거한다.

    출력: data/processed/lifelog_daily_clean.csv
    """
    print("\n" + "=" * 70)
    print("1단계. 전처리 — 라이프로그 정제")
    print("=" * 70)

    activity_cols = [
        "EMAIL", "activity_day_start",
        "activity_steps", "activity_score", "activity_total",
        "activity_high", "activity_medium", "activity_low", "activity_inactive",
        "activity_cal_total", "activity_average_met", "activity_non_wear",
    ]
    sleep_cols = [
        "EMAIL", "sleep_bedtime_start", "sleep_bedtime_end",
        "sleep_score", "sleep_efficiency", "sleep_total",
        "sleep_deep", "sleep_light", "sleep_rem",
        "sleep_onset_latency", "sleep_restless", "sleep_hr_average",
    ]

    activity = pd.concat([
        _load_split("1.Training", "train_activity.csv", activity_cols),
        _load_split("2.Validation", "val_activity.csv", activity_cols),
    ], ignore_index=True)
    sleep = pd.concat([
        _load_split("1.Training", "train_sleep.csv", sleep_cols),
        _load_split("2.Validation", "val_sleep.csv", sleep_cols),
    ], ignore_index=True)

    # activity_day_start는 매일 04:00 고정값이라 그 날짜를 join key로 쓴다.
    # 수면은 "잠에서 깬 시각(bedtime_end)"이 같은 날의 활동 기록과 대응된다
    # (자정을 넘겨 자는 경우가 많아, 잠든 시각이 아니라 깬 시각 기준으로 날짜를 맞춘다).
    activity["date"] = pd.to_datetime(activity["activity_day_start"]).dt.date
    sleep["date"] = pd.to_datetime(sleep["sleep_bedtime_end"]).dt.date

    before = len(activity)
    activity = activity[activity["activity_non_wear"] < NON_WEAR_LIMIT_MIN]
    print(f"미착용 3시간 이상으로 제외된 날: {before - len(activity)} / {before}")

    daily = pd.merge(activity, sleep, on=["EMAIL", "date", "split"], how="inner")
    print(f"활동+수면 매칭된 사람×일 기록 수: {len(daily)} (사람 수: {daily['EMAIL'].nunique()})")

    daily.to_csv(DAILY_CLEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {DAILY_CLEAN_PATH}")
    return daily


# ==============================================================================
# 2단계. 피처 생성 — 사람 단위 "평균 + 변동성" 지표 만들기
# ==============================================================================

def _clock_to_hour(series):
    """자정을 넘나드는 취침/기상 시각을 하루 흐름 순서로 맞춘다.
    예: 00:30 -> 24.5시, 22:00 -> 22.0시. 이래야 표준편차가 "몇 시쯤 자는지"의
    불규칙성을 제대로 반영한다(자정 근처에서 값이 튀는 것을 방지)."""
    dt = pd.to_datetime(series)
    hour = dt.dt.hour + dt.dt.minute / 60
    return hour.where(hour >= 12, hour + 24)


def _load_labels():
    """의사가 매긴 진단 라벨(CN/MCI/Dem)을 이메일 기준으로 불러온다."""
    def read_label(folder_name, file_name):
        path = os.path.join(RAW_BASE, folder_name, "라벨링데이터", "3.인지기능", file_name)
        return pd.read_csv(path)[["SAMPLE_EMAIL", "DIAG_NM"]]

    labels = pd.concat([
        read_label("1.Training", "training_label.csv"),
        read_label("2.Validation", "val_label.csv"),
    ], ignore_index=True)
    return labels.rename(columns={"SAMPLE_EMAIL": "EMAIL"})


def build_features(daily):
    """
    2단계: 일 단위 기록을 사람 단위로 요약한다. 18개 지표마다 평균(_mean)과
    표준편차(_std, 날마다의 들쭉날쭉함)를 함께 계산해 44개 피처를 만든다.
    MMSE 점수는 진단에 이미 쓰인 정답이므로 입력 피처에 넣지 않는다(순환 논리 방지).

    출력: data/processed/feature_table.csv (174명 × 44피처 + 라벨)
    """
    print("\n" + "=" * 70)
    print("2단계. 전처리 — 사람 단위 피처 생성")
    print("=" * 70)

    daily = daily.copy()
    daily["bedtime_hour"] = _clock_to_hour(daily["sleep_bedtime_start"])
    daily["wake_hour"] = _clock_to_hour(daily["sleep_bedtime_end"])

    agg = daily.groupby(["EMAIL", "split"])[AGG_COLS].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]

    rhythm = daily.groupby(["EMAIL", "split"])[["bedtime_hour", "wake_hour"]].std()
    rhythm.columns = ["bedtime_irregularity", "wake_irregularity"]

    day_count = daily.groupby(["EMAIL", "split"]).size().rename("day_count")

    features = pd.concat([agg, rhythm, day_count], axis=1).reset_index()
    features["steps_cv"] = features["activity_steps_std"] / features["activity_steps_mean"]

    labels = _load_labels()
    table = pd.merge(features, labels, on="EMAIL", how="inner")
    # 위험군(diag2class="CI") = 경도인지장애(MCI) + 치매(Dem) 통합
    table["diag2class"] = table["DIAG_NM"].map({"CN": "CN", "MCI": "CI", "Dem": "CI"})

    print(f"사람 수: {len(table)}")
    print(table["DIAG_NM"].value_counts().to_string())
    print(table["diag2class"].value_counts().to_string())
    print(f"사람당 평균 유효 일수: {table['day_count'].mean():.1f}일")

    table.to_csv(FEATURE_TABLE_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {FEATURE_TABLE_PATH}")
    return table


# ==============================================================================
# 3단계. 탐색 분석(EDA) — Q1·Q2: 진단군별 라이프로그 차이가 있는가
# ==============================================================================

def eda_summary(table):
    """
    3단계: 정상/경도인지장애/치매 그룹별로 활동량·수면효율 평균을 비교한다.
    (MMSE는 모델 입력이 아니라 여기 참고용 비교에만 사용한다.)
    """
    print("\n" + "=" * 70)
    print("3단계. 탐색 분석(EDA) — 진단군별 라이프로그 비교")
    print("=" * 70)

    summary = table.groupby("DIAG_NM").agg(
        걸음수_평균=("activity_steps_mean", "mean"),
        수면효율_평균=("sleep_efficiency_mean", "mean"),
    ).reindex(["CN", "MCI", "Dem"])
    print(summary.round(1).to_string())
    print(
        "\n주의: 걸음수는 MCI가 CN보다 오히려 많게 나온다 — "
        "'활동량이 줄면 위험군'이라는 단순 가설이 이 데이터에서는 성립하지 않는다.\n"
        "그래서 4단계 모델은 평균 활동량보다 '날마다의 들쭉날쭉함(변동성)'에 주목한다."
    )


# ==============================================================================
# 4단계. 핵심 모델(Q3·Q4) — 정상군 vs 위험군(CN vs CI) 스크리닝
# ==============================================================================
#
# 절차 (fold 50개 = 5조각 x 10회 반복, 매번 동일):
#   1) 학습 데이터로만 RandomForest를 학습해 44개 피처의 중요도를 구한다.
#      (RandomForest는 "어떤 피처가 중요한가"를 보는 용도로만 쓴다.
#       feature_importances_가 해석하기 쉽기 때문이다.)
#   2) 그 중요도로 상위 10개 피처를 고른다 — 평가 데이터는 이 단계 어디에도
#      들어가지 않는다(누수 방지의 핵심).
#   3) 상위 10개 피처를 표준화(StandardScaler, 학습 데이터에만 fit)한 뒤
#      LogisticRegression으로 다시 학습해서, 따로 떼어둔 평가 데이터로만 성능을 잰다.
#      (RandomForest를 최종 분류기로도 시도했지만, 174명짜리 작은 표본에서는
#       LogisticRegression이 모든 지표에서 더 안정적이었다 — 복잡한 모델이
#       항상 좋은 게 아니라는 걸 데이터로 확인한 사례다.)
#   4) 이번 fold에서 뽑힌 10개 피처를 기록해, "50번 중 몇 번 뽑혔는지"로
#      안정성(=신뢰할 수 있는 근거인지)을 판단한다.
#
# 정확도(accuracy)를 주 지표로 쓰지 않는 이유: class_weight='balanced'는
# "위험군을 놓치지 않는 대신 정상을 위험군으로 조금 더 잘못 보는" 쪽으로 일부러
# 기운 설정이라, 정확도가 구조적으로 낮아진다. 그래서 임계값에 안 묶이고
# "위험군일수록 점수를 높게 매기는가"만 보는 AUC를 주 지표로 삼는다.
# ==============================================================================

def _make_scout():
    """피처 중요도 산출 전용 모델."""
    return RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")


def _make_classifier():
    """최종 판정 모델. 표준화 + 로지스틱 회귀."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    )


def train_screening_model(table):
    """
    4단계: CN vs CI(MCI+Dem 통합) 스크리닝 모델을 학습·평가한다.

    출력: reports/figures/04_feature_importance.png
          reports/model_metrics.json
          data/processed/model_cn_ci.pkl (배포용 최종 모델)
    반환: (평가 결과 DataFrame, 최종 선택된 상위 10개 피처 리스트)
    """
    print("\n" + "=" * 70)
    print("4단계. 핵심 모델(Q3·Q4) — CN vs CI 스크리닝")
    print("=" * 70)

    feature_cols = [c for c in table.columns if c not in NON_FEATURE_COLS]
    X, y = table[feature_cols], table["diag2class"]

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows, feature_hits, importance_sums = [], Counter(), pd.Series(0.0, index=feature_cols)

    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # (1) 학습 데이터만으로 피처 중요도를 본다.
        scout = _make_scout()
        scout.fit(X_train, y_train)
        importances = pd.Series(scout.feature_importances_, index=feature_cols)
        importance_sums += importances

        # (2) 상위 10개만 고른다 — 테스트 데이터는 여기 관여하지 않는다.
        top_cols = importances.sort_values(ascending=False).head(TOP_K).index.tolist()
        feature_hits.update(top_cols)

        # (3) 상위 10개로 다시 학습해서, 따로 떼어둔 평가 데이터로만 성능을 잰다.
        model = _make_classifier()
        model.fit(X_train[top_cols], y_train)
        pred = model.predict(X_test[top_cols])
        proba_ci = model.predict_proba(X_test[top_cols])[:, list(model.classes_).index("CI")]

        rows.append({
            "accuracy": accuracy_score(y_test, pred),
            "baseline_accuracy": (y_test == "CN").mean(),  # 무조건 "정상" 예측 시 정확도
            "ci_recall": recall_score(y_test, pred, pos_label="CI", zero_division=0),
            "ci_precision": precision_score(y_test, pred, pos_label="CI", zero_division=0),
            "roc_auc": roc_auc_score((y_test == "CI").astype(int), proba_ci),
        })

    results = pd.DataFrame(rows)
    n_folds = len(results)
    print(f"[{n_folds}회 반복 교차검증 — 매 fold마다 학습 데이터로만 피처 재선택]")
    for col in ["accuracy", "baseline_accuracy", "ci_recall", "ci_precision", "roc_auc"]:
        print(f"  {col}: {results[col].mean():.3f} ± {results[col].std():.3f}")

    # 50번 중 가장 자주 뽑힌 피처 순서 = "왜 위험군인지"의 근거
    stability = (pd.Series(feature_hits) / n_folds).sort_values(ascending=False)
    top_stable_cols = stability.head(TOP_K).index.tolist()
    print(f"\nTOP {TOP_K} 피처 선택 빈도 (50번 중 몇 번 뽑혔는지):")
    print(stability.head(TOP_K).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    stability.head(TOP_K).sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"위험군(CI) 판별에 중요한 라이프로그 지표 TOP {TOP_K}")
    ax.set_xlabel(f"{n_folds}번 반복 중 선택된 비율")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "04_feature_importance.png"), dpi=150)
    plt.close(fig)

    # 배포용 모델: 가장 안정적으로 뽑힌 10개 피처로 174명 전체를 다시 학습
    final_model = _make_classifier()
    final_model.fit(X[top_stable_cols], y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": top_stable_cols}, f)
    print(f"\n배포용 모델 저장 완료: {MODEL_PATH}")

    metrics = {
        "n_people": len(table),
        "diag_nm_counts": table["DIAG_NM"].value_counts().to_dict(),
        "diag2class_counts": table["diag2class"].value_counts().to_dict(),
        "model": "LogisticRegression(class_weight='balanced') — 피처 선택은 RandomForest",
        "cv_setting": f"RepeatedStratifiedKFold({N_SPLITS}-fold x {N_REPEATS} repeats)",
        "performance": {c: {"mean": round(results[c].mean(), 4), "std": round(results[c].std(), 4)}
                        for c in ["accuracy", "baseline_accuracy", "ci_recall", "ci_precision", "roc_auc"]},
        "selected_features": {name: round(val, 3) for name, val in stability.head(TOP_K).items()},
    }
    with open(os.path.join(REPORTS_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return results, top_stable_cols


# ==============================================================================
# 5단계. Q5 검증 — 치매를 빼도 신호가 남는가 (CN vs MCI 단독, Nested CV)
# ==============================================================================
#
# 왜 필요한가: 4단계 결과가 사실은 "이미 증상이 뚜렷한 치매 12명"의 극단적 차이
# 덕분일 수 있다는 우려가 있다. 그래서 치매 사례를 완전히 제외하고, 정상군 vs
# 경도인지장애군만으로 같은 검증을 다시 한다.
#
# Nested CV를 쓰는 이유: 바깥 루프(성능 평가)와 안쪽 루프(몇 개 피처를 쓸지 고르는 것)를
# 완전히 분리해서, 바깥 테스트 fold가 피처 선택 과정에 절대 관여하지 않게 한다.
# ==============================================================================

def _inner_select_features(X_train, y_train, k_candidates, n_inner_splits=4):
    """안쪽 CV로 최적 K를 고르고 그 피처 목록을 반환한다. 바깥 테스트 fold는 관여하지 않는다."""
    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=1)

    importances = pd.DataFrame(index=X_train.columns)
    for i, (tr_idx, _) in enumerate(inner_cv.split(X_train, y_train)):
        m = _make_scout()
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        importances[f"f{i}"] = m.feature_importances_
    ranked = importances.mean(axis=1).sort_values(ascending=False)

    best_k, best_score = None, -1
    for k in k_candidates:
        cols = ranked.head(k).index.tolist()
        scores = []
        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            m = _make_scout()
            m.fit(X_train.iloc[tr_idx][cols], y_train.iloc[tr_idx])
            pred = m.predict(X_train.iloc[val_idx][cols])
            scores.append(accuracy_score(y_train.iloc[val_idx], pred))
        mean_score = sum(scores) / len(scores)
        if mean_score > best_score:
            best_score, best_k = mean_score, k

    return ranked.head(best_k).index.tolist()


def verify_cn_vs_mci(table):
    """
    5단계(Q5): 치매를 제외한 CN vs MCI만으로 재검증한다.

    출력: reports/model_metrics_cn_vs_mci.json
    """
    print("\n" + "=" * 70)
    print("5단계. Q5 검증 — 치매 제외, CN vs MCI 단독 재검증 (Nested CV)")
    print("=" * 70)

    df = table[table["DIAG_NM"] != "Dem"].copy()  # 치매 완전 제외
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["DIAG_NM"]
    print(f"대상: {y.value_counts().to_dict()}")

    outer_cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    k_candidates = [5, 8, 10, 15, 20, 30, len(feature_cols)]
    rows = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cols = _inner_select_features(X_train, y_train, k_candidates)

        model = _make_scout()
        model.fit(X_train[cols], y_train)
        pred = model.predict(X_test[cols])

        rows.append({
            "accuracy": accuracy_score(y_test, pred),
            "baseline_accuracy": (y_test == "CN").mean(),
            "mci_recall": recall_score(y_test, pred, pos_label="MCI", zero_division=0),
        })

    results = pd.DataFrame(rows)
    print(f"[{len(results)}회 outer fold]")
    for col in results.columns:
        print(f"  {col}: {results[col].mean():.3f} ± {results[col].std():.3f}")
    print(
        "\n결론: 정확도가 베이스라인을 못 넘으면, 4단계의 성능 개선은 주로 치매 단계의 "
        "뚜렷한 차이에서 온 것이지 경도인지장애 단독 조기 신호는 아니라는 뜻이다."
    )

    with open(os.path.join(REPORTS_DIR, "model_metrics_cn_vs_mci.json"), "w", encoding="utf-8") as f:
        json.dump({
            "comparison": "CN vs MCI (Dem excluded)",
            "performance": {c: {"mean": round(results[c].mean(), 4), "std": round(results[c].std(), 4)}
                            for c in results.columns},
        }, f, ensure_ascii=False, indent=2)


# ==============================================================================
# 6단계. Q6 검증 — 이 결과가 우연이 아닌지 통계로 확인 (Permutation Test)
# ==============================================================================
#
# 방법: 라벨(누가 위험군인지)을 무작위로 뒤섞은 가짜 데이터에 똑같은 파이프라인을
# 500번 반복 적용해 "우연히 나올 수 있는 성능"의 분포(귀무분포)를 만든다. 우리가
# 실제로 얻은 성능이 이 분포의 상위 몇 %에 위치하는지로 p-value를 계산한다.
# p < 0.05면 "이 결과가 우연히 나왔을 가능성은 낮다"고 말할 수 있다.
# ==============================================================================

def _mean_cv_score(X, y, n_repeats, seed):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=seed)
    accs, recalls = [], []
    for train_idx, test_idx in rskf.split(X, y):
        model = _make_classifier()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]
        accs.append(accuracy_score(y_test, pred))
        recalls.append(recall_score(y_test, pred, pos_label="CI", zero_division=0))
    return np.mean(accs), np.mean(recalls)


def permutation_test(table, selected_features, n_permutations=500, n_repeats_per_permutation=3):
    """
    6단계(Q6): 4단계에서 확정된 상위 10개 피처로, 실제 라벨 성능과
    500번의 "라벨 무작위 셔플" 성능을 비교해 p-value를 계산한다.

    출력: reports/permutation_test_result.json
          reports/figures/11_permutation_test.png
    """
    print("\n" + "=" * 70)
    print("6단계. Q6 검증 — 순열 검정(Permutation Test)")
    print("=" * 70)

    X = table[selected_features]
    y = table["diag2class"]

    observed_acc, observed_recall = _mean_cv_score(X, y, n_repeats_per_permutation, seed=RANDOM_STATE)
    print(f"실제 라벨 기준 관측 성능: 정확도 {observed_acc:.3f}, CI재현율 {observed_recall:.3f}")

    null_accs, null_recalls = [], []
    for i in range(n_permutations):
        y_shuffled = y.sample(frac=1, random_state=i).reset_index(drop=True)
        acc, rec = _mean_cv_score(X, y_shuffled, n_repeats_per_permutation, seed=100 + i)
        null_accs.append(acc)
        null_recalls.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  permutation {i + 1}/{n_permutations} 완료")

    null_accs, null_recalls = np.array(null_accs), np.array(null_recalls)
    # (우연이 관측치보다 크거나 같은 횟수 + 1) / (전체 + 1) — p=0이 되는 것을 방지하는 관례적 공식
    p_acc = (np.sum(null_accs >= observed_acc) + 1) / (n_permutations + 1)
    p_recall = (np.sum(null_recalls >= observed_recall) + 1) / (n_permutations + 1)

    print(f"\n[Permutation Test 결과, {n_permutations}회]")
    print(f"  정확도: 우연 분포 평균 {null_accs.mean():.3f} vs 실제 {observed_acc:.3f}  →  p = {p_acc:.4f}")
    print(f"  CI재현율: 우연 분포 평균 {null_recalls.mean():.3f} vs 실제 {observed_recall:.3f}  →  p = {p_recall:.4f}")
    print("  (p < 0.05면 '우연이라고 보기 어렵다'고 판단)")

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
    fig.savefig(os.path.join(FIGURES_DIR, "11_permutation_test.png"), dpi=150)
    plt.close(fig)

    result = {
        "n_permutations": n_permutations,
        "features_used": selected_features,
        "observed": {"accuracy": round(float(observed_acc), 4), "ci_recall": round(float(observed_recall), 4)},
        "p_value": {"accuracy": round(float(p_acc), 4), "ci_recall": round(float(p_recall), 4)},
        "significant_at_0.05": {"accuracy": bool(p_acc < 0.05), "ci_recall": bool(p_recall < 0.05)},
    }
    with open(os.path.join(REPORTS_DIR, "permutation_test_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ==============================================================================
# 전체 실행
# ==============================================================================

def main():
    daily = clean_lifelog()                              # 1단계
    table = build_features(daily)                         # 2단계
    eda_summary(table)                                     # 3단계 (Q1, Q2)
    _, top_features = train_screening_model(table)         # 4단계 (Q3, Q4)
    verify_cn_vs_mci(table)                                 # 5단계 (Q5)
    permutation_test(table, top_features)                   # 6단계 (Q6)

    print("\n" + "=" * 70)
    print("전체 파이프라인 완료. 결과는 reports/ 폴더를 확인하세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()
