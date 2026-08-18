"""
CN vs MCI 재검증(model/02)에서 기존 평균/표준편차 피처로는 신호가 없었다.
평균·산포가 아니라 "시간 흐름 자체"에서 나오는 피처를 추가로 만들어본다.

추가 피처:
- weekday_weekend_steps_diff : 주중-주말 평균 걸음수 차이 (요일 루틴 유지력)
- steps_trend_slope / sleep_efficiency_trend_slope
    : 관측 기간 동안 활동량·수면효율이 늘었는지 줄었는지 추세(평균 대비 상대 기울기)
- steps_autocorr_lag1 / sleep_efficiency_autocorr_lag1
    : 오늘이 어제와 얼마나 비슷한지(자기상관) — 표준편차(산포)와는 다른 개념.
      산포가 커도 예측 가능한 패턴(자기상관 높음)일 수 있고, 산포가 작아도
      매일 무작위로 오르내리는(자기상관 낮음) 경우가 있어 이 둘은 별개 정보다.
- activity_composition_shift
    : 저/중/고강도·비활동 시간의 "구성비"가 하루하루 얼마나 바뀌는지
      (총 변동량이 아니라 강도 구성 패턴 자체의 변화)

입력: data/processed/lifelog_daily_clean.csv, data/processed/feature_table.csv
출력: data/processed/feature_table_v2.csv (기존 44컬럼 + 신규 6컬럼)
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_PATH = os.path.join(BASE_DIR, "data", "processed", "lifelog_daily_clean.csv")
BASE_FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table_v2.csv")

COMPOSITION_COLS = ["activity_high", "activity_medium", "activity_low", "activity_inactive"]


def trend_slope(series):
    if len(series) < 5 or series.mean() == 0:
        return np.nan
    x = np.arange(len(series))
    slope = np.polyfit(x, series.values, 1)[0]
    return slope / series.mean()  # 평균 대비 상대 기울기 (하루당 % 변화)


def composition_shift(day_df):
    comp = day_df[COMPOSITION_COLS]
    total = comp.sum(axis=1).replace(0, np.nan)
    prop = comp.div(total, axis=0)
    return prop.diff().abs().sum(axis=1).mean()


def per_person(g):
    g = g.sort_values("date").reset_index(drop=True)
    g["dow"] = pd.to_datetime(g["date"]).dt.dayofweek
    weekday_steps = g.loc[g["dow"] < 5, "activity_steps"].mean()
    weekend_steps = g.loc[g["dow"] >= 5, "activity_steps"].mean()

    return pd.Series({
        "weekday_weekend_steps_diff": weekday_steps - weekend_steps,
        "steps_trend_slope": trend_slope(g["activity_steps"]),
        "sleep_efficiency_trend_slope": trend_slope(g["sleep_efficiency"]),
        "steps_autocorr_lag1": g["activity_steps"].autocorr(lag=1),
        "sleep_efficiency_autocorr_lag1": g["sleep_efficiency"].autocorr(lag=1),
        "activity_composition_shift": composition_shift(g),
    })


def main():
    daily = pd.read_csv(DAILY_PATH)
    advanced = daily.groupby("EMAIL").apply(per_person, include_groups=False).reset_index()

    base = pd.read_csv(BASE_FEATURE_PATH)
    merged = pd.merge(base, advanced, on="EMAIL", how="left")

    new_cols = [c for c in advanced.columns if c != "EMAIL"]
    print("신규 피처 결측 개수:")
    print(merged[new_cols].isna().sum())

    merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH} ({merged.shape[0]}명 x {merged.shape[1]}컬럼)")


if __name__ == "__main__":
    main()
