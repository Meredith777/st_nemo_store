import requests
import pandas as pd
import os
import json
import sqlite3

def _serialize_for_sqlite(df):
    """list/dict 타입 컬럼을 JSON 문자열로 변환"""
    df_db = df.copy()
    for col in df_db.columns:
        if df_db[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_db[col] = df_db[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )
    return df_db


def collect_nemo_data():
    # API 요청 URL 및 기본 파라미터 설정
    url = "https://www.nemoapp.kr/api/store/search-list"
    base_params = {
        "CompletedOnly": "false",
        "NELat": "37.588566955900646",
        "NELng": "127.00629522585783",
        "SWLat": "37.55948110239206",
        "SWLng": "126.97876753347148",
        "Zoom": "15",
        "SortBy": "29"
    }

    # HTTP 헤더 설정 (Scraping Prompt 참고)
    headers = {
        "referer": "https://www.nemoapp.kr/store",
        "sec-ch-ua": '"Not:A-Brand";v="99", "Microsoft Edge";v="145", "Chromium";v="145"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
    }

    # SQLite DB 연결
    db_path = "data/nemo_stores.db"
    conn = sqlite3.connect(db_path)

    total_count = 0
    page_index = 0

    while True:
        params = base_params.copy()
        params["PageIndex"] = str(page_index)
        
        print(f"API 요청 중 (페이지 {page_index}): {url}")
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # 'items' 추출
            items = data.get("items", [])
            if not items:
                print(f"페이지 {page_index}에 데이터가 없습니다. 수집을 종료합니다.")
                break

            # 즉시 SQLite DB에 저장 (첫 페이지는 replace, 이후는 append)
            df_page = pd.DataFrame(items)
            df_page = _serialize_for_sqlite(df_page)
            if_exists = "replace" if page_index == 0 else "append"
            df_page.to_sql("stores", conn, if_exists=if_exists, index=False)
            conn.commit()

            total_count += len(items)
            print(f"페이지 {page_index} 수집 및 DB 저장 완료: {len(items)}개 아이템 (누적: {total_count})")
            
            page_index += 1
            
        except Exception as e:
            print(f"오류 발생 (페이지 {page_index}): {e}")
            break

    conn.close()

    if total_count == 0:
        print("수집된 데이터가 없습니다.")
        return

    # CSV도 함께 저장 (DB에서 읽어서 저장)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM stores", conn)
    conn.close()

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "nemo_stores.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 데이터 요약 출력
    print(f"\n전체 수집 완료: 총 {total_count}개 아이템")
    print(f"SQLite DB 저장: {db_path} (테이블: stores)")
    print(f"CSV 저장: {csv_path}")
    print(f"컬럼 수: {len(df.columns)}")

if __name__ == "__main__":
    collect_nemo_data()
