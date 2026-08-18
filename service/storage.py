"""
업로드 적재 저장소 — 서비스에 올라온 라이프로그를 DB에 자동 저장한다.

상용화 시나리오의 데이터 루프:
  ① 사용자가 CSV 업로드 → 예측과 동시에 이 모듈이 DB에 적재 (라벨 없음)
  ② 위험군 안내를 받은 사용자가 검진을 받고, 확진 결과(CN/MCI/Dem)가 입력되면
     해당 업로드에 라벨이 붙는다 (POST /uploads/{id}/label — 기관 연계용)
  ③ 라벨이 쌓이면 model/06_retrain_from_uploads.py 가 기존 학습 데이터에
     합쳐서 재학습 → 서비스는 재시작만 하면 새 모델을 쓴다

DB는 파이썬 내장 SQLite를 쓴다(별도 설치 불필요 — 시연 어디서든 동작).
Oracle로 옮길 때는 이 파일의 SQL만 database/ 폴더 스크립트 패턴으로 바꾸면 된다.

저장 위치: data/service.db (개인 생활기록이 담기므로 git에 올리지 않는다)
"""
import os
import sqlite3
import uuid
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "service.db")

# 일 단위 레코드에서 적재할 컬럼 (업로드에 없는 컬럼은 NULL로 저장)
DAILY_COLS = [
    "date",
    "activity_steps", "activity_score", "activity_total",
    "activity_high", "activity_medium", "activity_low", "activity_inactive",
    "activity_cal_total", "activity_average_met", "activity_non_wear",
    "sleep_score", "sleep_efficiency", "sleep_total",
    "sleep_deep", "sleep_light", "sleep_rem",
    "sleep_onset_latency", "sleep_restless", "sleep_hr_average",
    "sleep_bedtime_start", "sleep_bedtime_end",
]


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """테이블이 없으면 만든다. 서비스 시작 시 한 번 호출."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                upload_id        TEXT PRIMARY KEY,   -- 익명 식별자 (uuid)
                created_at       TEXT NOT NULL,      -- 업로드 시각
                valid_days       INTEGER,            -- 유효 기록 일수
                risk_score       REAL,               -- 예측 당시 위험 점수
                screening_result TEXT,               -- 예측 당시 판정 문구
                diag_label       TEXT,               -- 검진 확진 라벨 (CN/MCI/Dem, 입력 전엔 NULL)
                labeled_at       TEXT                -- 라벨 입력 시각
            )""")
        cols_sql = ", ".join(f'"{c}" TEXT' if c in ("date", "sleep_bedtime_start", "sleep_bedtime_end")
                             else f'"{c}" REAL' for c in DAILY_COLS)
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS upload_daily (
                upload_id TEXT NOT NULL REFERENCES uploads(upload_id),
                {cols_sql}
            )""")


def save_upload(daily: pd.DataFrame, valid_days: int, risk_score: float, screening_result: str) -> str:
    """업로드 1건(일 단위 기록 전체 + 예측 결과)을 적재하고 익명 ID를 돌려준다."""
    upload_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat(timespec="seconds")

    rows = []
    for _, r in daily.iterrows():
        rows.append([upload_id] + [
            (str(r[c]) if c in ("date", "sleep_bedtime_start", "sleep_bedtime_end") else float(r[c]))
            if c in daily.columns and pd.notna(r[c]) else None
            for c in DAILY_COLS
        ])

    with _conn() as con:
        con.execute("INSERT INTO uploads VALUES (?, ?, ?, ?, ?, NULL, NULL)",
                    (upload_id, now, valid_days, risk_score, screening_result))
        placeholders = ", ".join("?" * (1 + len(DAILY_COLS)))
        con.executemany(f"INSERT INTO upload_daily VALUES ({placeholders})", rows)
    return upload_id


def set_label(upload_id: str, diagnosis: str) -> bool:
    """검진 확진 결과를 기록한다. 해당 업로드가 없으면 False."""
    with _conn() as con:
        cur = con.execute(
            "UPDATE uploads SET diag_label = ?, labeled_at = ? WHERE upload_id = ?",
            (diagnosis, datetime.now().isoformat(timespec="seconds"), upload_id))
        return cur.rowcount > 0


def stats() -> dict:
    """적재 현황 — 시연·모니터링용."""
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        labeled = con.execute("SELECT COUNT(*) FROM uploads WHERE diag_label IS NOT NULL").fetchone()[0]
        days = con.execute("SELECT COUNT(*) FROM upload_daily").fetchone()[0]
        by_label = dict(con.execute(
            "SELECT diag_label, COUNT(*) FROM uploads WHERE diag_label IS NOT NULL GROUP BY diag_label").fetchall())
    return {"total_uploads": total, "labeled_uploads": labeled,
            "total_daily_records": days, "label_counts": by_label}


def load_labeled_uploads() -> tuple:
    """재학습용 — 라벨이 붙은 업로드의 (라벨 목록, 일 단위 기록)을 돌려준다."""
    with _conn() as con:
        labels = pd.read_sql_query(
            "SELECT upload_id, diag_label FROM uploads WHERE diag_label IS NOT NULL", con)
        if labels.empty:
            return labels, pd.DataFrame()
        ids = ",".join(f"'{i}'" for i in labels["upload_id"])
        daily = pd.read_sql_query(f"SELECT * FROM upload_daily WHERE upload_id IN ({ids})", con)
    return labels, daily
