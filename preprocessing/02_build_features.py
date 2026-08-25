"""
사람×일 단위로 정리된 라이프로그(lifelog_daily_clean.csv)를 사람 단위로 집계해
모델 입력용 피처 테이블을 만들고, 진단 라벨(DIAG_NM)을 붙인다.

입력: data/processed/lifelog_daily_clean.csv, raw_data의 진단 라벨 csv
출력: data/processed/feature_table.csv (174명 × 피처)
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_PATH = os.path.join(BASE_DIR, "data", "processed", "lifelog_daily_clean.csv")
RAW_BASE = os.path.join(BASE_DIR, "raw_data", "01_aihub_wearable")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")

# 평균(mean)과 변동성(std)을 함께 보는 라이프로그 피처
AGG_COLS = [
    "activity_steps", "activity_score", "activity_total",
    "activity_high", "activity_medium", "activity_low", "activity_inactive",
    "activity_cal_total", "activity_average_met",
    "sleep_score", "sleep_efficiency", "sleep_total",
    "sleep_deep", "sleep_light", "sleep_rem",
    "sleep_onset_latency", "sleep_restless", "sleep_hr_average",
]


def clock_to_hour(series):
    # 자정을 넘나드는 취침/기상 시각을 하루 흐름 순서(저녁~다음날 아침)로 맞춘다.
    # 예: 00:30 -> 24.5시, 22:00 -> 22.0시. 이래야 표준편차가 "몇 시쯤 자는지"의
    # 불규칙성을 제대로 반영한다(자정 근처에서 값이 널뛰는 것을 막아준다).
    dt = pd.to_datetime(series)
    hour = dt.dt.hour + dt.dt.minute / 60
    return hour.where(hour >= 12, hour + 24)


def load_labels():
    def read_label(folder_name, file_name):
        path = fr"{RAW_BASE}\{folder_name}\라벨링데이터\3.인지기능\{file_name}"
        return pd.read_csv(path)[["SAMPLE_EMAIL", "DIAG_NM"]]

    labels = pd.concat([
        read_label("1.Training", "training_label.csv"),
        read_label("2.Validation", "val_label.csv"),
    ], ignore_index=True)
    return labels.rename(columns={"SAMPLE_EMAIL": "EMAIL"})


def main():
    daily = pd.read_csv(DAILY_PATH)
    daily["bedtime_hour"] = clock_to_hour(daily["sleep_bedtime_start"])
    daily["wake_hour"] = clock_to_hour(daily["sleep_bedtime_end"])

    agg = daily.groupby(["EMAIL", "split"])[AGG_COLS].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]

    rhythm = daily.groupby(["EMAIL", "split"])[["bedtime_hour", "wake_hour"]].std()
    rhythm.columns = ["bedtime_irregularity", "wake_irregularity"]

    day_count = daily.groupby(["EMAIL", "split"]).size().rename("day_count")

    features = pd.concat([agg, rhythm, day_count], axis=1).reset_index()
    features["steps_cv"] = features["activity_steps_std"] / features["activity_steps_mean"]

    labels = load_labels()
    table = pd.merge(features, labels, on="EMAIL", how="inner")
    table["diag2class"] = table["DIAG_NM"].map({"CN": "CN", "MCI": "CI", "Dem": "CI"})

    print(f"사람 수: {len(table)}")
    print(table["DIAG_NM"].value_counts())
    print(table["diag2class"].value_counts())
    print(f"사람당 평균 유효 일수: {table['day_count'].mean():.1f} (최소 {table['day_count'].min()}, 최대 {table['day_count'].max()})")

    table.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
