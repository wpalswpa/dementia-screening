"""
공공데이터포털 「전국치매센터표준데이터」 연동 스크립트.

공공데이터포털(data.go.kr)이 표준데이터셋에 공개로 제공하는 다운로드 엔드포인트
(/download/columList.json → /download/standard.json)를 그대로 사용한다.
serviceKey 발급 없이 동작하며, 실행할 때마다 최신 데이터(전국 317개 센터,
위경도·전화번호·프로그램 정보 포함)를 받아 CSV로 저장한다.

서비스(main.py)는 이 CSV를 읽어서 쓴다 — 발표 시연 때 인터넷이 끊겨도
서비스가 죽지 않도록, 실시간 호출 대신 "연동 스크립트 + 로컬 캐시" 구조를 택했다.
데이터를 갱신하고 싶을 때만 이 스크립트를 다시 실행하면 된다.

출처: 공공데이터포털 전국치매센터표준데이터 (https://www.data.go.kr/data/15021138/standard.do)
출력: service/data/dementia_centers.csv
"""
import csv
import json
import os
import urllib.parse
import urllib.request

PUBLIC_DATA_PK = "15021138"  # 전국치매센터표준데이터
PORTAL = "https://www.data.go.kr"
REFERER = f"{PORTAL}/data/{PUBLIC_DATA_PK}/standard.do"

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dementia_centers.csv")

# 서비스에서 실제로 쓰는 컬럼만 남긴다 (영문코드 → 한글 이름)
KEEP_COLS = {
    "CNTER_NM": "센터명",
    "CNTER_SE": "센터유형",
    "RDNMADR": "도로명주소",
    "LNMADR": "지번주소",
    "LATITUDE": "위도",
    "LONGITUDE": "경도",
    "PHONE_NUMBER": "관리기관전화번호",
    "OPER_PHONE_NUMBER": "운영기관전화번호",
    "INSTT_NM": "관할지자체",
    "IMBCLTY_INTRCN": "주요프로그램",
    "REFERENCE_DATE": "데이터기준일자",
}


def _get_json(url):
    req = urllib.request.Request(url, headers={"Referer": REFERER, "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as res:
        raw = res.read().decode("utf-8-sig")
    return json.loads(raw)


def main():
    # 1) 메타데이터(컬럼 목록, 전체 건수, 내부 테이블명) 조회
    header = _get_json(f"{PORTAL}/download/columList.json?pk={PUBLIC_DATA_PK}&ext=CSV")
    total = header["totalCount"]
    print(f"전국치매센터표준데이터: 총 {total}건 (기준 컬럼 {len(header['tableVO']['colNmList'])}개)")

    # 2) 데이터 본문 조회 (perPage=10000이라 한 번에 전부 받아진다)
    params = [
        ("totalCount", total),
        ("svcTableNm", header["tableVO"]["svcTableNm"]),
        ("perPage", 10000),
        ("page", 1),
    ] + [("colNmList", c) for c in header["tableVO"]["colNmList"]]
    url = f"{PORTAL}/download/standard.json?publicDataPk={PUBLIC_DATA_PK}&" + urllib.parse.urlencode(params)
    rows = _get_json(url)
    print(f"수신 완료: {len(rows)}건")

    # 3) 필요한 컬럼만 남겨 CSV 저장
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(KEEP_COLS.values())
        for r in rows:
            writer.writerow([r.get(code, "") for code in KEEP_COLS])
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
