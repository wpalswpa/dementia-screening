"""
Oracle 접속 정보 — 로컬 Oracle XE 11.2 (C:\\OracleXE112_Win64) 기준.

본인 계정이 다르면 아래 값만 수정하면 됩니다.
Oracle 11g은 python-oracledb의 Thin 모드가 지원되지 않아
Instant Client 기반 Thick 모드를 사용합니다.
"""
import oracledb

DB_USER = "system"
DB_PASSWORD = "본인_비밀번호"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE_NAME = "xe"

INSTANT_CLIENT_DIR = r"C:\instantclient-basic-windows.x64-23.26.2.0.0\instantclient_23_0"

_thick_mode_initialized = False


def connect_db():
    global _thick_mode_initialized

    if not _thick_mode_initialized:
        oracledb.init_oracle_client(lib_dir=INSTANT_CLIENT_DIR)
        _thick_mode_initialized = True

    return oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        service_name=DB_SERVICE_NAME,
    )
