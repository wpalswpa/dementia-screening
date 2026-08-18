"""
raw_data의 활동(activity)·수면(sleep) 파일을 사람×일 단위로 합치고,
믿을 수 없는 날짜(반지를 오래 벗고 있었던 날)를 걸러낸다.

입력: raw_data/01_aihub_wearable/{1.Training,2.Validation}/원천데이터/{1.걸음걸이,2.수면}/*.csv
출력: data/processed/lifelog_daily_clean.csv
"""
import os

import pandas as pd

# 이 스크립트 파일 위치 기준으로 프로젝트 루트를 찾는다 — 압축을 어느 폴더에 풀든
# (드라이브 문자나 폴더명이 달라져도) 그대로 동작한다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_BASE = os.path.join(BASE_DIR, "raw_data", "01_aihub_wearable")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "lifelog_daily_clean.csv")

# 3시간(180분) 이상 반지를 벗고 있었던 날은 그날의 활동/수면 기록을 믿기 어려워 제외한다.
# (activity_non_wear는 실제로 0~300분 사이 값을 가지며, 300분=5시간이 상한이다.)
NON_WEAR_LIMIT_MIN = 180

ACTIVITY_COLS = [
    "EMAIL", "activity_day_start",
    "activity_steps", "activity_score", "activity_total",
    "activity_high", "activity_medium", "activity_low", "activity_inactive",
    "activity_cal_total", "activity_average_met", "activity_non_wear",
]

SLEEP_COLS = [
    "EMAIL", "sleep_bedtime_start", "sleep_bedtime_end",
    "sleep_score", "sleep_efficiency", "sleep_total",
    "sleep_deep", "sleep_light", "sleep_rem",
    "sleep_onset_latency", "sleep_restless", "sleep_hr_average",
]


def load_split(folder_name, file_name, cols):
    path = fr"{RAW_BASE}\{folder_name}\원천데이터\{'1.걸음걸이' if 'activity' in file_name else '2.수면'}\{file_name}"
    df = pd.read_csv(path, usecols=cols)
    df["split"] = "train" if folder_name.startswith("1") else "val"
    return df


def to_date(series):
    return pd.to_datetime(series).dt.date


def main():
    activity = pd.concat([
        load_split("1.Training", "train_activity.csv", ACTIVITY_COLS),
        load_split("2.Validation", "val_activity.csv", ACTIVITY_COLS),
    ], ignore_index=True)
    sleep = pd.concat([
        load_split("1.Training", "train_sleep.csv", SLEEP_COLS),
        load_split("2.Validation", "val_sleep.csv", SLEEP_COLS),
    ], ignore_index=True)

    # activity_day_start는 매일 04:00 고정값이라 그 날짜만 join key로 쓴다.
    # sleep은 잠에서 깬 시각(bedtime_end)이 같은 날짜의 활동 기록과 대응된다.
    activity["date"] = to_date(activity["activity_day_start"])
    sleep["date"] = to_date(sleep["sleep_bedtime_end"])

    before = len(activity)
    activity = activity[activity["activity_non_wear"] < NON_WEAR_LIMIT_MIN]
    print(f"미착용 3시간 이상으로 제외된 날: {before - len(activity)} / {before}")

    daily = pd.merge(
        activity, sleep,
        on=["EMAIL", "date", "split"],
        how="inner",
    )
    print(f"활동+수면 매칭된 사람×일 기록 수: {len(daily)} (사람 수: {daily['EMAIL'].nunique()})")

    daily.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
