"""
AI Hub 원본(raw) CSV를 Oracle에 "있는 그대로" 임시 적재한다 (스테이징 방식).

  TBL_RAW_ACTIVITY  활동 원본 (train+val, 31컬럼 + SRC)
  TBL_RAW_SLEEP     수면 원본 (train+val, 36컬럼 + SRC)
  TBL_RAW_MMSE      MMSE 검사 원본 (문항별 점수 포함)
  TBL_RAW_LABEL     진단 라벨 원본

CSV 컬럼을 자동 분석해서 테이블을 생성한다:
  - 컬럼명: Oracle 규칙에 맞게 정리 (특수문자→_, 대문자, 최대 30자)
  - 숫자 컬럼 → NUMBER, 짧은 문자열 → VARCHAR2(n CHAR), 4000바이트 넘는 문자열
    (5분/1분 단위 시계열 등) → CLOB
※ 이 데이터는 AI Hub 이용약관상 재배포 금지 — 로컬 DB에만 적재할 것.

실행: python database\\04_load_rawdata.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import connect_db  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "raw_data", "01_aihub_wearable")

DATASETS = [
    ("TBL_RAW_ACTIVITY", [
        (os.path.join(BASE, "1.Training", "원천데이터", "1.걸음걸이", "train_activity.csv"), "train"),
        (os.path.join(BASE, "2.Validation", "원천데이터", "1.걸음걸이", "val_activity.csv"), "val"),
    ]),
    ("TBL_RAW_SLEEP", [
        (os.path.join(BASE, "1.Training", "원천데이터", "2.수면", "train_sleep.csv"), "train"),
        (os.path.join(BASE, "2.Validation", "원천데이터", "2.수면", "val_sleep.csv"), "val"),
    ]),
    ("TBL_RAW_MMSE", [
        (os.path.join(BASE, "1.Training", "원천데이터", "3.인지기능", "train_mmse.csv"), "train"),
        (os.path.join(BASE, "2.Validation", "원천데이터", "3.인지기능", "val_mmse.csv"), "val"),
    ]),
    ("TBL_RAW_LABEL", [
        (os.path.join(BASE, "1.Training", "라벨링데이터", "3.인지기능", "training_label.csv"), "train"),
        (os.path.join(BASE, "2.Validation", "라벨링데이터", "3.인지기능", "val_label.csv"), "val"),
    ]),
]

CHUNK = 500  # 한 번에 INSERT할 행 수


def oracle_col_name(name, used):
    """CSV 컬럼명 → Oracle 컬럼명 (특수문자 제거, 대문자, 30자 제한, 중복 방지)."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_").upper()[:30]
    base = clean
    i = 1
    while clean in used:
        clean = f"{base[:27]}_{i}"
        i += 1
    used.add(clean)
    return clean


def column_ddl(series, col):
    """값을 보고 컬럼 타입 결정: 숫자 → NUMBER, 문자열은 길이에 따라 VARCHAR2/CLOB."""
    if pd.api.types.is_numeric_dtype(series):
        return f"{col} NUMBER", "num"
    max_len = series.dropna().astype(str).str.len().max()
    max_len = 0 if pd.isna(max_len) else int(max_len)
    if max_len > 1300:  # 한글 포함 4000바이트 한도 고려
        return f"{col} CLOB", "clob"
    size = max(50, min(1300, max_len * 2))  # 여유 2배
    return f"{col} VARCHAR2({size} CHAR)", "str"


def load_table(conn, table, sources):
    frames = []
    for path, src in sources:
        df = pd.read_csv(path)
        df["SRC"] = src
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    used = set()
    col_map = {c: oracle_col_name(c, used) for c in df.columns}
    df = df.rename(columns=col_map)

    ddl_parts, types = [], {}
    for c in df.columns:
        ddl, t = column_ddl(df[c], c)
        ddl_parts.append(ddl)
        types[c] = t

    cur = conn.cursor()
    try:
        cur.execute(f"DROP TABLE {table}")
    except Exception:
        pass
    cur.execute(f"CREATE TABLE {table} (\n  " + ",\n  ".join(ddl_parts) + "\n)")

    cols = list(df.columns)
    placeholders = ", ".join(f":{i + 1}" for i in range(len(cols)))
    insert = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"

    # CLOB 컬럼은 입력 크기를 미리 지정해야 executemany가 안전하다
    import oracledb
    sizes = [oracledb.DB_TYPE_CLOB if types[c] == "clob" else None for c in cols]

    rows = []
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                row.append(None)
            elif types[c] == "num":
                row.append(float(v) if not isinstance(v, (int, np.integer)) else int(v))
            else:
                row.append(str(v))
        rows.append(tuple(row))

    for i in range(0, len(rows), CHUNK):
        cur.setinputsizes(*sizes)
        cur.executemany(insert, rows[i:i + CHUNK])
    conn.commit()

    cur.execute(f"SELECT COUNT(*) FROM {table}")
    n = cur.fetchone()[0]
    clob_cols = [c for c, t in types.items() if t == "clob"]
    print(f"  {table}: {n}행 적재 (컬럼 {len(cols)}개, CLOB {len(clob_cols)}개: {clob_cols})")
    cur.close()


def main():
    with connect_db() as conn:
        for table, sources in DATASETS:
            print(f"[{table}]")
            load_table(conn, table, sources)

        # 검증: 참가자 수 교차 확인
        cur = conn.cursor()
        for t, col in [("TBL_RAW_ACTIVITY", "EMAIL"), ("TBL_RAW_SLEEP", "EMAIL"),
                       ("TBL_RAW_MMSE", "SAMPLE_EMAIL"), ("TBL_RAW_LABEL", "SAMPLE_EMAIL")]:
            cur.execute(f"SELECT COUNT(DISTINCT {col}) FROM {t}")
            print(f"검증 — {t} 고유 참가자: {cur.fetchone()[0]}명")
        cur.close()


if __name__ == "__main__":
    main()
