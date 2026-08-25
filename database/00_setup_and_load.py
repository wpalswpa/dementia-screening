"""
치매안심센터 Oracle 적재 — 한 번에 실행하는 러너.

01_create_centers_table.sql(테이블·시퀀스 생성)을 실행한 뒤,
02_load_centers.py(CSV → INSERT)를 호출하고, 03의 검증 쿼리로 결과를 확인한다.

실행: python database\\00_setup_and_load.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import connect_db  # noqa: E402

SQL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_create_centers_table.sql")


def run_sql_file(conn, path):
    """SQL 파일 실행 — PL/SQL 블록은 '/' 줄로, 일반 문장은 ';'로 구분한다."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # 주석 줄 제거
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    text = "\n".join(lines)

    statements = []
    buf = []
    for ln in text.splitlines():
        if ln.strip() == "/":          # PL/SQL 블록 종료
            statements.append("\n".join(buf).strip())
            buf = []
        else:
            buf.append(ln)
    rest = "\n".join(buf)
    statements += [s.strip() for s in rest.split(";") if s.strip()]

    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    conn.commit()


def main():
    print("① 테이블·시퀀스 생성")
    with connect_db() as conn:
        run_sql_file(conn, SQL_PATH)
        print("   TBL_DEMENTIA_CENTER, SEQ_DEMENTIA_CENTER 준비 완료")

    print("② CSV 적재")
    import importlib
    loader = importlib.import_module("02_load_centers")
    loader.main()

    print("③ 검증")
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM TBL_DEMENTIA_CENTER")
            print(f"   전체 건수: {cur.fetchone()[0]}")
            cur.execute("SELECT CNTER_SE, COUNT(*) FROM TBL_DEMENTIA_CENTER GROUP BY CNTER_SE ORDER BY 2 DESC")
            for se, cnt in cur.fetchall():
                print(f"   {se}: {cnt}")
            cur.execute("SELECT COUNT(*) FROM TBL_DEMENTIA_CENTER WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL")
            print(f"   좌표 보유: {cur.fetchone()[0]}")
            cur.execute("""
                SELECT CNTER_NM, PHONE_NUMBER FROM (
                    SELECT CNTER_NM, PHONE_NUMBER FROM TBL_DEMENTIA_CENTER ORDER BY CENTER_ID
                ) WHERE ROWNUM <= 3""")
            for nm, ph in cur.fetchall():
                print(f"   예시: {nm} ({ph})")


if __name__ == "__main__":
    main()
