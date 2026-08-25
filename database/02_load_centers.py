"""
service/data/dementia_centers.csv(공공데이터포털 연동 결과)를 Oracle DB의
TBL_DEMENTIA_CENTER 테이블에 적재한다.

사전 준비:
  1) pip install oracledb
  2) database/01_create_centers_table.sql 실행 (테이블 생성 — 00_setup_and_load.py가 대신 해줌)
  3) database/database.py의 접속 정보를 자기 환경에 맞게 수정

참고: 웹 서비스(service/main.py)는 시연 안정성을 위해 CSV를 직접 읽는다.
이 스크립트는 "외부 데이터 → DB 적재" 파이프라인을 보여주는 선택 구성요소로,
DB 조회로 전환하려면 main.py의 CENTERS 로딩 부분만 바꾸면 된다.
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import connect_db  # noqa: E402

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "service", "data", "dementia_centers.csv")

INSERT_SQL = """
INSERT INTO TBL_DEMENTIA_CENTER
    (CENTER_ID, CNTER_NM, CNTER_SE, RDNMADR, LNMADR, LATITUDE, LONGITUDE,
     PHONE_NUMBER, OPER_PHONE, INSTT_NM, PROGRAMS, REFERENCE_DATE)
VALUES
    (SEQ_DEMENTIA_CENTER.NEXTVAL, :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11)
"""


def cut(value, limit):
    """결측은 None, 문자열은 글자 수 제한 (테이블 CHAR 단위 길이에 맞춤)."""
    if pd.isna(value):
        return None
    return str(value)[:limit]


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"CSV 로드: {len(df)}건")

    rows = []
    for _, r in df.iterrows():
        rows.append((
            cut(r.get("센터명"), 200),
            cut(r.get("센터유형"), 50),
            cut(r.get("도로명주소"), 300),
            cut(r.get("지번주소"), 300),
            float(r["위도"]) if pd.notna(r["위도"]) else None,
            float(r["경도"]) if pd.notna(r["경도"]) else None,
            cut(r.get("관리기관전화번호"), 50),
            cut(r.get("운영기관전화번호"), 50),
            cut(r.get("관할지자체"), 100),
            cut(r.get("주요프로그램"), 1300),
            cut(r.get("데이터기준일자"), 20),
        ))

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)
            conn.commit()
            cur.execute("SELECT COUNT(*) FROM TBL_DEMENTIA_CENTER")
            count = cur.fetchone()[0]
    print(f"{count}건 저장 완료")


if __name__ == "__main__":
    main()
