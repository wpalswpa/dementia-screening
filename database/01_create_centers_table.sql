-- 치매안심센터 테이블 생성 (공공데이터포털 「전국치매센터표준데이터」 적재용)
-- 여러 번 재실행해도 안전하도록, 기존 테이블/시퀀스가 있으면 먼저 DROP한다.
-- Oracle 11g은 VARCHAR2 길이가 기본 "바이트" 단위라(한글 1자 = 3바이트),
-- 잘림 방지를 위해 글자 수 기준(CHAR semantics)으로 선언한다.

BEGIN
   EXECUTE IMMEDIATE 'DROP TABLE TBL_DEMENTIA_CENTER';
EXCEPTION WHEN OTHERS THEN NULL;
END;
/

BEGIN
   EXECUTE IMMEDIATE 'DROP SEQUENCE SEQ_DEMENTIA_CENTER';
EXCEPTION WHEN OTHERS THEN NULL;
END;
/

CREATE TABLE TBL_DEMENTIA_CENTER (
    CENTER_ID       NUMBER PRIMARY KEY,          -- 내부 일련번호 (시퀀스)
    CNTER_NM        VARCHAR2(200 CHAR),          -- 센터명
    CNTER_SE        VARCHAR2(50 CHAR),           -- 센터유형 (치매안심센터/광역치매센터 등)
    RDNMADR         VARCHAR2(300 CHAR),          -- 도로명주소
    LNMADR          VARCHAR2(300 CHAR),          -- 지번주소
    LATITUDE        NUMBER,                      -- 위도
    LONGITUDE       NUMBER,                      -- 경도
    PHONE_NUMBER    VARCHAR2(50 CHAR),           -- 관리기관전화번호
    OPER_PHONE      VARCHAR2(50 CHAR),           -- 운영기관전화번호
    INSTT_NM        VARCHAR2(100 CHAR),          -- 관할지자체
    PROGRAMS        VARCHAR2(1300 CHAR),         -- 주요 치매관리 프로그램 (11g 4000바이트 한도 내)
    REFERENCE_DATE  VARCHAR2(20 CHAR)            -- 데이터기준일자
);

CREATE SEQUENCE SEQ_DEMENTIA_CENTER START WITH 1 INCREMENT BY 1;
