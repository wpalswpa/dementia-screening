"""
적재 데이터 재학습 파이프라인 — 상용화 데이터 루프의 마지막 단계.

서비스에 업로드되어 DB(data/service.db)에 적재된 기록 중 **검진 확진 라벨이
입력된 것만** 골라, 기존 학습 데이터(feature_table.csv, 174명)에 합쳐서
model/01과 동일한 절차(매 fold 학습 데이터로만 피처 선택 → 로지스틱 회귀 평가)로
재학습한다. 라벨 없는 업로드는 학습에 쓰지 않는다 — 정답 없이 학습하면
모델이 자기 예측을 정답처럼 배우는 오염이 생기기 때문이다.

절차:
  1) DB에서 라벨 붙은 업로드 로드 → preprocessing/02와 같은 방식으로 사람 단위 피처 계산
     (18개 지표 mean/std + 취침/기상 불규칙성 + day_count + steps_cv = 44개)
  2) 기존 feature_table과 결합 → 50회 반복 교차검증으로 새 성능 측정
  3) 기존 모델(pkl)을 백업한 뒤 전체 데이터로 재학습해 교체
     → 서비스는 재시작만 하면 새 모델을 쓴다

실행: python model\\06_retrain_from_uploads.py [--min-new N]
  --min-new: 재학습에 필요한 최소 신규 라벨 수 (기본 10 — 몇 건으로는 성능이 안 바뀌므로)

출력: data/processed/model_cn_ci.pkl (교체), data/processed/model_cn_ci_backup.pkl (이전본)
      reports/retrain_report.json (기존 vs 재학습 성능 비교)
"""
import argparse
import json
import os
import pickle
import shutil
import sys
from collections import Counter
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from service import storage  # noqa: E402  (DB 접근 모듈 재사용)

FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "model_cn_ci.pkl")
BACKUP_PATH = os.path.join(BASE_DIR, "data", "processed", "model_cn_ci_backup.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "retrain_report.json")

NON_FEATURE_COLS = ["EMAIL", "split", "DIAG_NM", "diag2class"]
AGG_COLS = [  # preprocessing/02와 동일
    "activity_steps", "activity_score", "activity_total",
    "activity_high", "activity_medium", "activity_low", "activity_inactive",
    "activity_cal_total", "activity_average_met",
    "sleep_score", "sleep_efficiency", "sleep_total",
    "sleep_deep", "sleep_light", "sleep_rem",
    "sleep_onset_latency", "sleep_restless", "sleep_hr_average",
]
TOP_K = 10
N_SPLITS, N_REPEATS = 5, 10


def make_scout():
    return RandomForestClassifier(random_state=42, class_weight="balanced")


def make_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )


def clock_to_hour(series):
    dt = pd.to_datetime(series, errors="coerce")
    hour = dt.dt.hour + dt.dt.minute / 60
    return hour.where(hour >= 12, hour + 24)


def uploads_to_features(labels: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """적재된 일 단위 기록 → 사람(업로드) 단위 44개 피처. preprocessing/02와 동일한 계산."""
    rows = []
    for _, lab in labels.iterrows():
        g = daily[daily["upload_id"] == lab["upload_id"]]
        if g[AGG_COLS].isna().any().any():
            print(f"  [건너뜀] {lab['upload_id']}: 일부 지표 컬럼이 비어 있어 재학습에서 제외")
            continue
        row = {"EMAIL": f"upload_{lab['upload_id']}", "split": "service_upload",
               "DIAG_NM": lab["diag_label"],
               "diag2class": "CN" if lab["diag_label"] == "CN" else "CI"}
        for c in AGG_COLS:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std()
        row["bedtime_irregularity"] = clock_to_hour(g["sleep_bedtime_start"]).std()
        row["wake_irregularity"] = clock_to_hour(g["sleep_bedtime_end"]).std()
        row["day_count"] = len(g)
        row["steps_cv"] = row["activity_steps_std"] / row["activity_steps_mean"] if row["activity_steps_mean"] else None
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame) -> dict:
    """model/01과 동일한 무누수 프로토콜로 성능을 잰다."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["diag2class"]
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    accs, recalls, aucs, hits = [], [], [], Counter()

    for tr, te in rskf.split(X, y):
        scout = make_scout()
        scout.fit(X.iloc[tr], y.iloc[tr])
        top = pd.Series(scout.feature_importances_, index=feature_cols).nlargest(TOP_K).index.tolist()
        hits.update(top)
        model = make_classifier()
        model.fit(X.iloc[tr][top], y.iloc[tr])
        pred = model.predict(X.iloc[te][top])
        proba = model.predict_proba(X.iloc[te][top])[:, list(model.classes_).index("CI")]
        accs.append(accuracy_score(y.iloc[te], pred))
        recalls.append(recall_score(y.iloc[te], pred, pos_label="CI", zero_division=0))
        aucs.append(roc_auc_score((y.iloc[te] == "CI").astype(int), proba))

    n = len(accs)
    top_stable = [name for name, _ in hits.most_common(TOP_K)]
    return {
        "n_people": len(df),
        "accuracy": round(sum(accs) / n, 4),
        "ci_recall": round(sum(recalls) / n, 4),
        "roc_auc": round(sum(aucs) / n, 4),
        "top_features": top_stable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-new", type=int, default=10,
                        help="재학습에 필요한 최소 신규 라벨 수 (기본 10)")
    args = parser.parse_args()

    labels, daily = storage.load_labeled_uploads()
    print(f"DB 적재 현황: {storage.stats()}")
    if len(labels) < args.min_new:
        print(f"\n라벨 붙은 업로드가 {len(labels)}건뿐입니다 (최소 {args.min_new}건 필요). "
              f"재학습하지 않고 종료합니다 — 몇 건으로는 성능이 의미 있게 바뀌지 않습니다.")
        return

    new_features = uploads_to_features(labels, daily)
    if new_features.empty:
        print("재학습에 쓸 수 있는 완전한 업로드가 없습니다. 종료합니다.")
        return

    base = pd.read_csv(FEATURE_PATH)
    combined = pd.concat([base, new_features[base.columns]], ignore_index=True)
    print(f"\n학습 데이터: 기존 {len(base)}명 + 신규 라벨 {len(new_features)}명 = {len(combined)}명")

    print("\n[기존 데이터만] 성능 측정 중...")
    before = evaluate(base)
    print(f"  정확도 {before['accuracy']} | 재현율 {before['ci_recall']} | AUC {before['roc_auc']}")
    print("[기존+신규] 성능 측정 중...")
    after = evaluate(combined)
    print(f"  정확도 {after['accuracy']} | 재현율 {after['ci_recall']} | AUC {after['roc_auc']}")

    # 기존 모델 백업 후, 합친 데이터 전체로 최종 모델 재학습·교체
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
    feature_cols = [c for c in combined.columns if c not in NON_FEATURE_COLS]
    final = make_classifier()
    final.fit(combined[after["top_features"]], combined["diag2class"])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final, "feature_cols": after["top_features"]}, f)
    print(f"\n새 모델 저장 완료: {MODEL_PATH} (이전 모델 백업: {BACKUP_PATH})")
    print("서비스에 반영하려면 uvicorn을 재시작하세요.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "retrained_at": datetime.now().isoformat(timespec="seconds"),
            "n_base": len(base), "n_new_labeled": len(new_features),
            "before": before, "after": after,
        }, f, ensure_ascii=False, indent=2)
    print(f"재학습 리포트 저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
