"""
치매 위험 조기 스크리닝 서비스 (FastAPI) — 기획안 09~10절의 1단계(스크리닝) 구현.

흐름은 세 단계뿐이다:
  1) 사용자가 일 단위 생활 기록 CSV를 업로드한다 (웨어러블 기기에서 내보낸 형식,
     data/processed/lifelog_daily_clean.csv 와 같은 컬럼 구조).
  2) 서버가 학습 때와 똑같은 방식으로 "사람 단위 지표"(평균/변동성)를 계산해서
     학습된 모델(model_cn_ci.pkl)에 넣는다.
  3) 위험 점수 + 판단 근거(정상군 평균과의 비교) + 치매안심센터 안내 + 한계 고지를
     돌려준다. 진단이 아니라 "병원에 가볼 이유가 있는지"까지만 말한다.

실행: uvicorn service.main:app --host 127.0.0.1 --port 8002  (프로젝트 루트에서)
"""
import os
import pickle
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "model_cn_ci.pkl")
FEATURE_TABLE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")
DAILY_CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "lifelog_daily_clean.csv")
INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")

MIN_VALID_DAYS = 14  # 기획안 09절: 최소 데이터 기간 기준 — 유효 기록 2주 미만이면 판정하지 않는다

# 결과 화면에서 사람이 읽을 수 있는 지표 이름 (모델 피처명 → 설명)
FEATURE_LABELS = {
    "activity_low_std": "저강도 활동 시간의 들쭉날쭉함",
    "sleep_restless_std": "수면 중 뒤척임의 들쭉날쭉함",
    "sleep_light_mean": "얕은 수면 시간(평균)",
    "activity_total_std": "총 활동 시간의 들쭉날쭉함",
    "activity_inactive_mean": "비활동 시간(평균)",
    "activity_score_std": "활동 점수의 들쭉날쭉함",
    "sleep_restless_mean": "수면 중 뒤척임(평균)",
    "activity_high_mean": "고강도 활동 시간(평균)",
    "sleep_efficiency_mean": "수면 효율(평균)",
    "activity_high_std": "고강도 활동 시간의 들쭉날쭉함",
}

# 원본 컬럼별 단위와 환산 배율 — 수면 시간류는 원본이 초 단위라 분으로 바꿔서 보여준다.
# (모델 계산에는 영향 없음, 결과 화면 표시용)
BASE_UNITS = {
    "activity_steps": ("걸음", 1), "activity_score": ("점", 1),
    "activity_total": ("분", 1), "activity_high": ("분", 1), "activity_medium": ("분", 1),
    "activity_low": ("분", 1), "activity_inactive": ("분", 1),
    "activity_cal_total": ("kcal", 1), "activity_average_met": ("MET", 1),
    "sleep_score": ("점", 1), "sleep_efficiency": ("%", 1), "sleep_restless": ("%", 1),
    "sleep_total": ("분", 1 / 60), "sleep_deep": ("분", 1 / 60), "sleep_light": ("분", 1 / 60),
    "sleep_rem": ("분", 1 / 60), "sleep_onset_latency": ("분", 1 / 60),
    "sleep_hr_average": ("회/분", 1),
}


def feature_unit(feature_name):
    """피처명("sleep_light_mean")에서 표시용 (단위, 환산배율)을 찾는다."""
    base = feature_name.rsplit("_", 1)[0]
    return BASE_UNITS.get(base, ("", 1))

app = FastAPI(title="치매 위험 조기 스크리닝 (찾아조)", docs_url="/docs")


def load_model_and_reference():
    """학습된 모델과, 비교 기준이 될 정상군(CN) 평균/표준편차를 불러온다."""
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    ref = pd.read_csv(FEATURE_TABLE_PATH)
    cn = ref[ref["diag2class"] == "CN"]
    cn_mean = cn[feature_cols].mean()
    cn_std = cn[feature_cols].std()
    # 화면 표시용 정상군 평균 (모델 피처 여부와 무관하게 필요) — 추이 그래프 기준선, 요약 카드
    cn_display = {
        "sleep_efficiency": round(float(cn["sleep_efficiency_mean"].mean()), 1),
        "activity_total": round(float(cn["activity_total_mean"].mean()), 0),
        "steps": round(float(cn["activity_steps_mean"].mean()), 0),
    }
    return model, feature_cols, cn_mean, cn_std, cn_display


MODEL, FEATURE_COLS, CN_MEAN, CN_STD, CN_DISPLAY = load_model_and_reference()

# 치매안심센터 데이터 (공공데이터포털 「전국치매센터표준데이터」 317곳).
# service/fetch_centers.py를 실행하면 최신 데이터로 갱신된다.
CENTERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dementia_centers.csv")
CENTERS = pd.read_csv(CENTERS_PATH)
CENTERS["위도"] = pd.to_numeric(CENTERS["위도"], errors="coerce")
CENTERS["경도"] = pd.to_numeric(CENTERS["경도"], errors="coerce")
CENTERS = CENTERS.dropna(subset=["위도", "경도"]).reset_index(drop=True)

# 모델 피처명("activity_low_std")에서 원본 일 단위 컬럼명("activity_low")을 뽑아,
# 업로드 CSV에 꼭 있어야 하는 컬럼 목록을 만든다.
REQUIRED_DAILY_COLS = sorted({c.rsplit("_", 1)[0] for c in FEATURE_COLS})


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """일 단위 기록 → 사람 단위 지표 1행. preprocessing/02와 같은 계산(평균/표준편차)."""
    row = {}
    for col in FEATURE_COLS:
        base, stat = col.rsplit("_", 1)          # 예: "activity_low_std" → ("activity_low", "std")
        row[col] = daily[base].mean() if stat == "mean" else daily[base].std()
    return pd.DataFrame([row], columns=FEATURE_COLS)


def clock_to_hour(series):
    """취침/기상 시각을 '저녁부터 다음날 아침까지' 흐름의 시(hour) 값으로 바꾼다.
    예: 22:30 → 22.5,  00:30 → 24.5 (자정 넘김).  preprocessing/02와 동일한 계산."""
    dt = pd.to_datetime(series, errors="coerce")
    hour = dt.dt.hour + dt.dt.minute / 60
    return hour.where(hour >= 12, hour + 24)


@app.get("/")
def index():
    """데모용 웹 화면."""
    return FileResponse(INDEX_HTML)


@app.get("/sample.csv")
def sample_csv(kind: str = "normal"):
    """시연용 샘플 데이터 — 학습 데이터에서 한 사람의 일 단위 기록을 익명으로 내려준다.
    kind=normal 이면 정상군, kind=risk 면 위험군(CI) 사례를 준다 (시연에서 두 경우 비교용)."""
    daily = pd.read_csv(DAILY_CLEAN_PATH)
    ref = pd.read_csv(FEATURE_TABLE_PATH)
    target_class = "CI" if kind == "risk" else "CN"
    emails = ref[ref["diag2class"] == target_class]["EMAIL"]
    one_email = emails.iloc[0]
    sample = daily[daily["EMAIL"] == one_email].drop(columns=["EMAIL", "split"], errors="ignore")
    buf = sample.to_csv(index=False, encoding="utf-8-sig")
    return Response(content=buf, media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=sample_lifelog_{kind}.csv"})


@app.get("/centers")
def centers(lat: float = None, lon: float = None, region: str = None, limit: int = 5):
    """가까운 치매안심센터를 찾아준다.

    - lat/lon(사용자 위치)이 오면: 직선거리(하버사인 공식) 기준 가까운 순 limit개
    - region(예: "서울", "경기")이 오면: 관할지자체·주소에 그 지역명이 들어간 센터 목록
    - 둘 다 없으면: 400
    """
    df = CENTERS
    if region:
        mask = df["관할지자체"].str.contains(region, na=False) | df["도로명주소"].str.contains(region, na=False)
        df = df[mask]
        if df.empty:
            raise HTTPException(404, f"'{region}' 지역의 센터를 찾지 못했습니다. 시도 이름(예: 서울, 경남)으로 검색해주세요.")

    if lat is not None and lon is not None:
        # 하버사인 공식으로 km 단위 직선거리 계산
        import numpy as np
        lat1, lon1 = np.radians(lat), np.radians(lon)
        lat2, lon2 = np.radians(df["위도"].values), np.radians(df["경도"].values)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        dist_km = 6371 * 2 * np.arcsin(np.sqrt(a))
        df = df.assign(distance_km=np.round(dist_km, 1)).sort_values("distance_km")
    elif not region:
        raise HTTPException(400, "lat/lon(내 위치) 또는 region(지역명) 중 하나는 필요합니다.")

    rows = []
    for _, r in df.head(limit).iterrows():
        rows.append({
            "name": r["센터명"],
            "type": r["센터유형"],
            "address": r["도로명주소"] if isinstance(r["도로명주소"], str) and r["도로명주소"] else r.get("지번주소", ""),
            "phone": r["관리기관전화번호"] if isinstance(r["관리기관전화번호"], str) and r["관리기관전화번호"] else r.get("운영기관전화번호", ""),
            "lat": float(r["위도"]),
            "lon": float(r["경도"]),
            "distance_km": float(r["distance_km"]) if "distance_km" in df.columns else None,
            "programs": (r["주요프로그램"][:80] + "…") if isinstance(r["주요프로그램"], str) and len(r["주요프로그램"]) > 80 else r["주요프로그램"],
        })
    return {
        "total_centers": int(len(CENTERS)),
        "source": "공공데이터포털 전국치매센터표준데이터 (data.go.kr/data/15021138)",
        "results": rows,
        "note": "치매상담콜센터 1899-9988 (24시간, 무료)",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """CSV 업로드 → 위험 점수와 근거를 돌려준다."""
    try:
        daily = pd.read_csv(BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "CSV 파일을 읽을 수 없습니다. 쉼표로 구분된 CSV인지 확인해주세요.")

    missing = [c for c in REQUIRED_DAILY_COLS if c not in daily.columns]
    if missing:
        raise HTTPException(400, f"필수 컬럼이 없습니다: {', '.join(missing)}")

    # 필수 컬럼에 결측이 있는 날은 제외하고, 유효 일수가 기준(2주) 미만이면 판정하지 않는다.
    valid = daily.dropna(subset=REQUIRED_DAILY_COLS)
    if len(valid) < MIN_VALID_DAYS:
        raise HTTPException(422, f"유효한 기록이 {len(valid)}일뿐입니다. "
                                 f"신뢰할 수 있는 판정을 위해 최소 {MIN_VALID_DAYS}일 이상의 기록이 필요합니다.")

    features = build_features(valid)
    proba_ci = float(MODEL.predict_proba(features)[0][list(MODEL.classes_).index("CI")])
    is_risk = proba_ci >= 0.5

    # ── 각 지표가 이번 판정에서 "위험 쪽"으로 작용했는지 "정상 쪽"으로 작용했는지 ──
    # 최종 분류기가 로지스틱 회귀라서, (표준화된 지표값 × 가중치)의 부호로
    # 그 지표가 위험 점수를 올렸는지 내렸는지를 모델 계산 그대로 알 수 있다.
    # 이게 없으면 "정상군 범위를 벗어났는데 왜 위험이 아니냐"는 질문에 답할 수 없다 —
    # 벗어나도 '건강한 방향'으로 벗어난 지표는 오히려 점수를 내리기 때문이다.
    scaler = MODEL.named_steps["standardscaler"]
    lr = MODEL.named_steps["logisticregression"]
    x_scaled = scaler.transform(features[FEATURE_COLS])[0]
    toward_ci = 1.0 if list(lr.classes_)[1] == "CI" else -1.0  # 계수의 방향을 "위험(CI) 쪽" 기준으로 통일
    contributions = dict(zip(FEATURE_COLS, toward_ci * lr.coef_[0] * x_scaled))

    # 판단 근거: 정상군 평균에서 몇 표준편차(σ) 벗어났는지로 상위 3개를 고르고,
    # "벗어난 정도"와 "벗어난 방향(위험/건강)"을 합쳐 하나의 종합 판정으로 보여준다.
    #   양호      = 위험 쪽으로 작용하지 않음 (많이 벗어났어도 건강한 방향이면 양호)
    #   주의      = 위험 쪽으로 작용 (벗어난 정도 2σ 미만)
    #   위험 신호 = 위험 쪽으로 작용 + 정상군 범위를 크게 벗어남 (2σ 이상)
    z_scores = (features.iloc[0] - CN_MEAN) / CN_STD
    reasons = []
    for name in z_scores.abs().sort_values(ascending=False).index[:3]:
        unit, scale = feature_unit(name)
        user_val = float(features.iloc[0][name]) * scale
        cn_val = float(CN_MEAN[name]) * scale
        cn_sd = float(CN_STD[name]) * scale
        abs_z = abs(float(z_scores[name]))
        out_text = "정상군 범위를 크게 벗어났" if abs_z >= 2 else "정상군 범위를 벗어났"
        pushed_risk = contributions.get(name, 0.0) > 0

        # 종합 판정 (한눈에 보는 한 단어) + 이유 설명 (한 문장)
        if not pushed_risk:
            verdict, verdict_level = "양호", "ok"
            detail = (f"{out_text}지만 건강한 방향이라, 위험 점수를 오히려 내렸습니다" if abs_z >= 1
                      else "정상군 범위 안이고, 위험 점수를 내리는 쪽으로 작용했습니다")
        elif abs_z < 2:
            verdict, verdict_level = "주의", "warn"
            detail = (f"{out_text}고, 위험 점수를 올리는 쪽으로 작용했습니다" if abs_z >= 1
                      else "정상군 범위 안이지만, 위험 쪽으로 약간 기울어 있습니다")
        else:
            verdict, verdict_level = "위험 신호", "high"
            detail = "정상군 범위를 크게 벗어났고, 위험 점수를 올리는 쪽으로 작용했습니다"

        reasons.append({
            "indicator": FEATURE_LABELS.get(name, name),
            "unit": unit,
            "user_value": round(user_val, 1),
            "cn_average": round(cn_val, 1),
            "cn_range": [round(cn_val - cn_sd, 1), round(cn_val + cn_sd, 1)],  # 정상군 대부분이 드는 구간
            "direction": "높음" if user_val > cn_val else "낮음",
            "verdict": verdict,              # 양호 / 주의 / 위험 신호
            "verdict_level": verdict_level,  # 화면 색상용 (ok/warn/high)
            "detail": detail,                # 판정 이유 한 문장
            "deviation_sigma": round(abs_z, 1),
        })

    return {
        "valid_days": int(len(valid)),
        "risk_score": round(proba_ci, 3),                     # 0~1, 참고용 점수
        "risk_threshold": 0.5,                                # 이 값 이상이면 위험군 추정 (화면 기준선용)
        "screening_result": "인지저하 위험군 추정 — 전문기관 상담을 권장합니다" if is_risk
                            else "뚜렷한 위험 신호 없음 — 정기적인 확인을 권장합니다",
        "reasons": reasons,
        "criteria_note": "판정 기준: 정상군(111명) 평균에서 얼마나 벗어났는지(범위)와 어느 방향으로 벗어났는지"
                         "(위험 쪽/건강한 쪽)를 합쳐 양호·주의·위험 신호로 표시합니다. 건강한 방향으로 벗어난 것은 "
                         "많이 벗어나도 '양호'입니다. '정상군 범위'는 정상군 3명 중 2명이 드는 구간(평균±1 표준편차)입니다.",
        # 화면의 추이 그래프용 — 하루당 1개 값(일 단위). 정상군 평균을 기준선으로 함께 준다.
        "daily_series": {
            "dates": (pd.to_datetime(valid["date"], errors="coerce").dt.strftime("%m/%d").tolist()
                      if "date" in valid.columns else None),
            "sleep_efficiency": [round(float(v), 1) for v in valid["sleep_efficiency"].tolist()],
            "activity_total": [round(float(v), 1) for v in valid["activity_total"].tolist()],
            # 매일 몇 시에 자고 일어나는지 (시 단위, 24 이상이면 자정 넘김 = 다음날)
            "bedtime_hour": ([round(float(v), 1) if pd.notna(v) else None for v in clock_to_hour(valid["sleep_bedtime_start"])]
                             if "sleep_bedtime_start" in valid.columns else None),
            "wake_hour": ([round(float(v), 1) if pd.notna(v) else None for v in clock_to_hour(valid["sleep_bedtime_end"])]
                          if "sleep_bedtime_end" in valid.columns else None),
            "cn_reference": CN_DISPLAY,                        # 정상군 평균 기준선 (수면효율 %, 활동시간 분, 걸음)
        },
        # 상단 요약 카드용 — 내 평균 vs 정상군 평균
        "summary": {
            "avg_sleep_efficiency": round(float(valid["sleep_efficiency"].mean()), 1),
            "avg_activity_total": round(float(valid["activity_total"].mean()), 0),
            "avg_steps": round(float(valid["activity_steps"].mean()), 0) if "activity_steps" in valid.columns else None,
            "cn_reference": CN_DISPLAY,
        },
        "center_guide": {
            "note": "위험군으로 추정될 경우 가까운 치매안심센터에서 무료 검진을 받을 수 있습니다.",
            "call_center": "치매상담콜센터 1899-9988 (24시간, 무료)",
            "find_center_url": "https://www.nid.or.kr/support/hi_list.aspx",
        },
        "disclaimer": "본 결과는 의료 진단이 아닌 참고용 스크리닝입니다. "
                      "학습 데이터의 표본이 작아(174명, 치매 확진 12명) 결과 해석에 주의가 필요합니다.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
