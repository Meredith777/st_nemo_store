import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# 페이지 설정
st.set_page_config(page_title="네모 부동산 매물 대시보드", layout="wide")

# ── 경로 및 설정 ──────────────────────────────────────────
# 현재 파일(src/dashboard.py)의 상위 폴더(src/)의 상위 폴더(root)를 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "nemo_stores.db")

# 지하철역 좌표 매핑
SUBWAY_COORDS = {
    "종로5가역": [37.5709, 127.0019],
    "을지로입구역": [37.5660, 126.9822],
    "안국역": [37.5765, 126.9855],
    "종각역": [37.5702, 126.9831],
    "종로3가역": [37.5704, 126.9921],
    "광화문역": [37.5714, 126.9765],
    "광화문(세종문화회관)역": [37.5714, 126.9765],
    "명동역": [37.5609, 126.9862],
    "을지로3가역": [37.5663, 126.9918],
    "혜화역": [37.5821, 127.0018],
    "동대문역사문화공원역": [37.5651, 127.0079],
    "종로4가역": [37.5704, 126.9921], # Approximate
    "을지로4가역": [37.5668, 126.9981],
    "충무로역": [37.5612, 126.9942],
    "동대입구역": [37.5591, 127.0052],
    "시청역": [37.5657, 126.9769],
    "한성대입구(삼선교)역": [37.5884, 127.0062],
    "한성대입구역": [37.5884, 127.0062],
    "회현(남대문시장)역": [37.5585, 126.9782],
}

# ── 데이터 로드 함수 ──────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        st.error(f"데이터베이스 파일을 찾을 수 없습니다: {DB_PATH}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM stores", conn)
    conn.close()
    
    # 지하철역 이름 추출 및 좌표 매핑
    def get_coords(station_info):
        if not station_info:
            return None, None
        station_name = station_info.split(',')[0].strip()
        coords = SUBWAY_COORDS.get(station_name)
        if coords:
            return coords[0], coords[1]
        return None, None

    df['lat'], df['lon'] = zip(*df['nearSubwayStation'].apply(get_coords))
    
    # 수치형 변수 변환
    num_cols = ["deposit", "monthlyRent", "premium", "maintenanceFee", "size", "floor"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 월세 평당가 계산 (만원/㎡) - 0으로 나누기 방지
    df['rentPerSize'] = df.apply(lambda x: x['monthlyRent'] / x['size'] if x['size'] > 0 else 0, axis=1)
    
    return df

# ── 메인 UI ──────────────────────────────────────────────
st.title("🏢 네모 부동산 매물 탐색 대시보드")
st.markdown("---")

df = load_data()

if df.empty:
    st.stop()

# ── 사이드바 필터 ────────────────────────────────────────
st.sidebar.header("🔍 검색 필터")

# 업종 필터
biz_large = st.sidebar.multiselect("업종 대분류", options=sorted(df['businessLargeCodeName'].unique()), default=[])
if not biz_large:
    filtered_df = df.copy()
else:
    filtered_df = df[df['businessLargeCodeName'].isin(biz_large)]

biz_middle = st.sidebar.multiselect("업종 중분류", options=sorted(filtered_df['businessMiddleCodeName'].unique()), default=[])
if biz_middle:
    filtered_df = filtered_df[filtered_df['businessMiddleCodeName'].isin(biz_middle)]

# 가격 필터 (만원 단위)
st.sidebar.subheader("💰 가격 조건 (만원)")
dep_min, dep_max = int(df['deposit'].min()), int(df['deposit'].max())
deposit_range = st.sidebar.slider("보증금 범위", dep_min, dep_max, (dep_min, dep_max), step=500)

rent_min, rent_max = int(df['monthlyRent'].min()), int(df['monthlyRent'].max())
rent_range = st.sidebar.slider("월세 범위", rent_min, rent_max, (rent_min, rent_max), step=10)

pre_min, pre_max = int(df['premium'].min()), int(df['premium'].max())
premium_range = st.sidebar.slider("권리금 범위", pre_min, pre_max, (pre_min, pre_max), step=500)

# 면적 및 층수 필터
st.sidebar.subheader("📐 공간 조건")
size_min, size_max = float(df['size'].min()), float(df['size'].max())
size_range = st.sidebar.slider("전용면적 (㎡)", size_min, size_max, (size_min, size_max), step=1.0)

floor_options = sorted(df['floor'].unique())
selected_floors = st.sidebar.multiselect("층수", options=floor_options, default=floor_options)

# 평당가 필터 (만원/㎡)
st.sidebar.subheader("📊 가성비 조건")
rps_min, rps_max = float(df['rentPerSize'].min()), float(df['rentPerSize'].max())
rps_range = st.sidebar.slider("월세 평당가 (만원/㎡)", rps_min, rps_max, (rps_min, rps_max), step=0.1)

# 필터 적용
filtered_df = filtered_df[
    (filtered_df['deposit'].between(deposit_range[0], deposit_range[1])) &
    (filtered_df['monthlyRent'].between(rent_range[0], rent_range[1])) &
    (filtered_df['premium'].between(premium_range[0], premium_range[1])) &
    (filtered_df['size'].between(size_range[0], size_range[1])) &
    (filtered_df['floor'].isin(selected_floors)) &
    (filtered_df['rentPerSize'].between(rps_range[0], rps_range[1]))
]

# ── 메인 화면 레이아웃 (Side-by-Side: Map & Details) ────────────────
main_col1, main_col2 = st.columns([1.5, 1])

with main_col1:
    col1_header, col1_toggle = st.columns([1, 1])
    with col1_header:
        st.subheader("📍 매물 위치")
    with col1_toggle:
        map_mode = st.radio("지도 모드", ["개별 매물", "매물 밀도 (히트맵)"], horizontal=True, label_visibility="collapsed")
    
    map_df = filtered_df.dropna(subset=['lat', 'lon'])
    
    if not map_df.empty:
        # 지터링 추가 (겹침 방지)
        map_df['lat_j'] = map_df['lat'] + (pd.Series(range(len(map_df))) % 10 - 5) * 0.0001
        map_df['lon_j'] = map_df['lon'] + (pd.Series(range(len(map_df))) % 10 - 5) * 0.0001
        
        if map_mode == "개별 매물":
            fig_map = px.scatter_map(
                map_df,
                lat="lat_j",
                lon="lon_j",
                hover_name="title",
                hover_data={
                    "lat_j": False, "lon_j": False, 
                    "deposit": ":,", "monthlyRent": ":,", "premium": ":,",
                    "businessMiddleCodeName": True, "size": ":.1f",
                    "rentPerSize": ":.1f",
                    "id": False
                },
                custom_data=["id"],
                color="businessLargeCodeName",
                zoom=13,
                height=650,
                labels={
                    "deposit": "보증금", "monthlyRent": "월세", "premium": "권리금",
                    "businessMiddleCodeName": "업종", "size": "면적(㎡)", "rentPerSize": "평당가(㎡)"
                }
            )
        else:
            fig_map = px.density_map(
                map_df,
                lat="lat",
                lon="lon",
                z="monthlyRent", # 월세 기준 밀도
                radius=20,
                hover_name="nearSubwayStation",
                zoom=13,
                height=650,
                color_continuous_scale="Viridis",
            )
        
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        # 지도 선택 이벤트 캡처
        selection = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", selection_mode="points", key="map_layout_final")
        
        # 선택된 포인트가 있으면 세션 상태 업데이트
        if selection and selection.get("selection") and selection["selection"].get("points"):
            selected_id_map = selection["selection"]["points"][0]["customdata"][0]
            if st.session_state.get('selected_id') != selected_id_map:
                st.session_state['selected_id'] = selected_id_map
                # 셀렉트박스(제목 기준)와 동기화를 위해 해당 ID의 제목을 찾음
                selected_title_map = map_df[map_df['id'] == selected_id_map]['title'].iloc[0]
                st.session_state['property_selector_id'] = selected_title_map
                st.rerun()
    else:
        st.info("선택한 조건에 맞는 위치 데이터가 없습니다.")

with main_col2:
    st.subheader("🔍 매물 상세 정보")
    if not filtered_df.empty:
        # 세션 상태 초기화 (ID 기준)
        if 'selected_id' not in st.session_state:
            st.session_state['selected_id'] = filtered_df['id'].iloc[0]
        
        # 현재 선택된 ID가 필터링된 결과에 없으면 첫 번째로 변경
        if st.session_state['selected_id'] not in filtered_df['id'].values:
            st.session_state['selected_id'] = filtered_df['id'].iloc[0]

        # 셀렉트박스 연동 로직
        item_list = filtered_df[['id', 'title']].values.tolist()
        ids = [x[0] for x in item_list]
        titles = [x[1] for x in item_list]
        curr_index = ids.index(st.session_state['selected_id'])
        
        def on_selectbox_change():
            new_title = st.session_state['property_selector_id']
            new_id = filtered_df[filtered_df['title'] == new_title]['id'].iloc[0]
            st.session_state['selected_id'] = new_id

        selected_title = st.selectbox(
            "🔎 매물 선택 (지도 클릭 시 자동 연동)", 
            options=titles, 
            index=curr_index,
            on_change=on_selectbox_change,
            key="property_selector_id"
        )
        
        item = filtered_df[filtered_df['id'] == st.session_state['selected_id']].iloc[0]
        
        with st.container(border=True):
            st.markdown(f"### 🏠 {item['title']}")
            
            img_urls = json.loads(item['smallPhotoUrls']) if isinstance(item['smallPhotoUrls'], str) else []
            if img_urls:
                st.image(img_urls[0], use_container_width=True)
            else:
                st.info("🖼️ 이미지가 제공되지 않는 매물입니다.")
            
            # 메트릭스
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("보증금", f"{item['deposit']:,}만")
            m_col2.metric("월세", f"{item['monthlyRent']:,}만")
            m_col3.metric("권리금", f"{item['premium']:,}만")
            
            st.markdown("---")
            st.write(f"**업종**: {item['businessLargeCodeName']} > {item['businessMiddleCodeName']}")
            st.write(f"**공간**: {item['size']}㎡ ({item['size']/3.3058:.1f}평) / {item['floor']}층")
            st.write(f"**평당가**: 월 {item['rentPerSize']:.1f}만원/㎡ (평당 {item['rentPerSize']*3.3058:.1f}만원)")
            st.write(f"**지하철**: {item['nearSubwayStation']}")
            st.write(f"**조회/관심**: 👀 {item['viewCount']}개 / ❤️ {item['favoriteCount']}개")
            

            if len(img_urls) > 1:
                with st.expander("📸 추가 사진 보기"):
                    img_cols = st.columns(2)
                    for i, url in enumerate(img_urls[1:5]): # 상위 몇 개만
                        img_cols[i % 2].image(url, use_container_width=True)
    else:
        st.info("데이터가 없습니다.")


st.markdown("---")
st.subheader("📊 필터링 데이터 분석 및 목록")

sub_col1, sub_col2 = st.columns([1, 1])

with sub_col1:
    if not filtered_df.empty:
        biz_counts = filtered_df['businessLargeCodeName'].value_counts().reset_index()
        biz_counts.columns = ['업종', '개수']
        fig_pie = px.pie(biz_counts, values='개수', names='업종', title='업종 대분류 비중', hole=0.3)
        fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("통계 데이터가 없습니다.")

with sub_col2:
    if not filtered_df.empty:
        fig_hist = px.histogram(filtered_df, x="monthlyRent", nbins=20, title="월세 가격대 분포 (만원)", 
                               labels={'monthlyRent': '월세'}, color_discrete_sequence=['#4361ee'])
        fig_hist.update_layout(bargap=0.1, margin=dict(t=40, b=0, l=0, r=0), height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("#### 📝 필터링된 매물 목록")
display_cols = ["title", "businessMiddleCodeName", "deposit", "monthlyRent", "premium", "size", "floor", "nearSubwayStation"]
st.dataframe(filtered_df[display_cols], use_container_width=True)
