-- 치매안심센터 테이블 생성 (공공데이터포털 「전국치매센터표준데이터」 적재용)
-- 여러 번 재실행해도 안전하도록, 기존 테이블/시퀀스가 있으면 먼저 DROP한다.

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
    CNTER_NM        VARCHAR2(200),               -- 센터명
    CNTER_SE        VARCHAR2(50),                -- 센터유형 (치매안심센터/광역치매센터 등)
    RDNMADR         VARCHAR2(300),               -- 도로명주소
    LNMADR          VARCHAR2(300),               -- 지번주소
    LATITUDE        NUMBER,                      -- 위도
    LONGITUDE       NUMBER,                      -- 경도
    PHONE_NUMBER    VARCHAR2(50),                -- 관리기관전화번호
    OPER_PHONE      VARCHAR2(50),                -- 운영기관전화번호
    INSTT_NM        VARCHAR2(100),               -- 관할지자체
    PROGRAMS        VARCHAR2(2000),              -- 주요 치매관리 프로그램
    REFERENCE_DATE  VARCHAR2(20)                 -- 데이터기준일자
);

CREATE SEQUENCE SEQ_DEMENTIA_CENTER START WITH 1 INCREMENT BY 1;
