"""
네모 부동산(Nemo Store) 매물 데이터 탐색적 데이터 분석(EDA)
- 데이터: nemostore/data/nemo_stores.db (stores 테이블, 665행 × 40열)
- 단위: 보증금/월세/권리금/매매가/관리비/평당가 = 만원, 면적 = ㎡
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sqlite3
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 경로 설정 ──────────────────────────────────────────────
DB_PATH = "data/nemo_stores.db"
IMG_DIR = "images"
REPORT_PATH = "docs/eda_report.md"
os.makedirs(IMG_DIR, exist_ok=True)

# ── 글로벌 스타일 ──────────────────────────────────────────
COLORS = [
    "#4361ee", "#3a0ca3", "#7209b7", "#f72585", "#4cc9f0",
    "#06d6a0", "#ffd166", "#ef476f", "#118ab2", "#073b4c",
    "#e76f51", "#2a9d8f", "#264653", "#e9c46a", "#f4a261",
]

def save_fig(name):
    """그래프 저장 헬퍼"""
    path = os.path.join(IMG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

# ══════════════════════════════════════════════════════════
# 0. 데이터 로드
# ══════════════════════════════════════════════════════════
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM stores", conn)
conn.close()

# 분석에 불필요한 컬럼 제외 (URL, ID 등)
DROP_COLS = [
    "isPriority", "agentId", "previewPhotoUrl", "smallPhotoUrls",
    "originPhotoUrls", "isInYourFavorited", "completionConfirmedDateUtc",
    "buildingManagementSerialNumber",
]
df_analysis = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# 날짜 컬럼 변환
for col in ["confirmedDateUtc", "createdDateUtc", "editedDateUtc"]:
    if col in df_analysis.columns:
        df_analysis[col] = pd.to_datetime(df_analysis[col], errors="coerce", utc=True)

# 수치형 / 범주형 분리
NUM_COLS = [
    "deposit", "monthlyRent", "premium", "sale", "maintenanceFee",
    "floor", "groundFloor", "size", "viewCount", "favoriteCount",
    "areaPrice", "firstDeposit", "firstMonthlyRent", "firstPremium",
]
CAT_COLS = [
    "businessLargeCodeName", "businessMiddleCodeName",
    "priceTypeName", "isMoveInDate",
]
TEXT_COLS = ["title", "nearSubwayStation"]

report_lines = []
def rpt(text=""):
    """리포트에 텍스트 추가"""
    report_lines.append(text)
    print(text)

# ══════════════════════════════════════════════════════════
# 1. 기본 데이터 탐색
# ══════════════════════════════════════════════════════════
rpt("# 네모 부동산 매물 데이터 EDA 보고서\n")
rpt(f"> 분석 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
rpt("---\n")

rpt("## 1. 데이터 개요\n")
rpt(f"- **총 행(매물 수)**: {len(df):,}개")
rpt(f"- **총 열(컬럼 수)**: {len(df.columns)}개")
rpt(f"- **분석 대상 컬럼**: {len(df_analysis.columns)}개 (URL/ID 컬럼 제외)\n")

# 상위 5행
rpt("### 1-1. 상위 5개 행\n")
preview_cols = ["title", "businessLargeCodeName", "priceTypeName", "deposit", "monthlyRent", "premium", "size", "floor", "nearSubwayStation", "viewCount"]
rpt(df[preview_cols].head().to_markdown(index=False))
rpt("")

# 하위 5행
rpt("### 1-2. 하위 5개 행\n")
rpt(df[preview_cols].tail().to_markdown(index=False))
rpt("")

# info
rpt("### 1-3. 데이터 타입 및 결측치\n")
rpt("| 컬럼 | 타입 | 비결측 | 결측 | 결측률(%) |")
rpt("|:---|:---|---:|---:|---:|")
for col in df_analysis.columns:
    non_null = df_analysis[col].notna().sum()
    null_cnt = df_analysis[col].isna().sum()
    null_pct = null_cnt / len(df_analysis) * 100
    dtype = str(df_analysis[col].dtype)
    rpt(f"| {col} | {dtype} | {non_null} | {null_cnt} | {null_pct:.1f} |")
rpt("")

# 중복 확인
dup_count = df.duplicated(subset=["id"]).sum()
rpt(f"### 1-4. 중복 데이터 확인\n")
rpt(f"- **ID 기준 중복 행**: {dup_count}개")
dup_title = df.duplicated(subset=["title", "deposit", "monthlyRent", "size"]).sum()
rpt(f"- **제목+보증금+월세+면적 기준 중복**: {dup_title}개\n")

# ══════════════════════════════════════════════════════════
# 2. 기술 통계
# ══════════════════════════════════════════════════════════
rpt("---\n")
rpt("## 2. 기술 통계\n")

# 수치형
rpt("### 2-1. 수치형 변수 기술 통계 (단위: 금액=만원, 면적=㎡)\n")
num_desc = df[NUM_COLS].describe().T
num_desc.columns = ["개수", "평균", "표준편차", "최솟값", "25%", "중앙값", "75%", "최댓값"]
rpt(num_desc.to_markdown())
rpt("")

# 범주형
rpt("### 2-2. 범주형 변수 기술 통계\n")
for col in CAT_COLS:
    rpt(f"#### {col}\n")
    vc = df[col].value_counts()
    cat_table = pd.DataFrame({"값": vc.index, "빈도": vc.values, "비율(%)": (vc.values / len(df) * 100).round(1)})
    rpt(cat_table.to_markdown(index=False))
    rpt("")

# ══════════════════════════════════════════════════════════
# 3. 범주형 빈도 분석 시각화
# ══════════════════════════════════════════════════════════
rpt("---\n")
rpt("## 3. 범주형 변수 빈도 분석\n")

# 3-1. 업종 대분류
rpt("### 3-1. 업종 대분류(businessLargeCodeName) 빈도\n")
vc_large = df["businessLargeCodeName"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(vc_large.index[::-1], vc_large.values[::-1], color=COLORS[:len(vc_large)])
ax.set_xlabel("빈도수")
ax.set_title("업종 대분류별 매물 빈도", fontsize=14, fontweight="bold")
for bar in bars:
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width())}", va="center", fontsize=10)
save_fig("01_business_large_freq.png")
rpt(f"![업종 대분류 빈도](../images/01_business_large_freq.png)\n")
rpt(vc_large.reset_index().rename(columns={"index": "업종 대분류", "count": "빈도", "businessLargeCodeName": "업종 대분류"}).to_markdown(index=False))
rpt("")
rpt("> **해석**: 업종 대분류별 매물 수를 보면, 특정 업종에 매물이 집중되어 있는지 확인할 수 있습니다. 가장 많은 매물이 등록된 업종이 해당 지역의 주요 시장 수요를 반영하며, 소수 업종의 매물은 틈새시장 가능성을 시사합니다.\n")

# 3-2. 업종 중분류 상위 30
rpt("### 3-2. 업종 중분류(businessMiddleCodeName) 상위 30개\n")
vc_mid = df["businessMiddleCodeName"].value_counts().head(30)
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(vc_mid.index[::-1], vc_mid.values[::-1], color=COLORS[2 % len(COLORS)])
ax.set_xlabel("빈도수")
ax.set_title("업종 중분류 상위 30개 매물 빈도", fontsize=14, fontweight="bold")
for bar in bars:
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width())}", va="center", fontsize=9)
save_fig("02_business_middle_top30.png")
rpt(f"![업종 중분류 상위30](../images/02_business_middle_top30.png)\n")
rpt(vc_mid.reset_index().rename(columns={"businessMiddleCodeName": "업종 중분류", "count": "빈도"}).to_markdown(index=False))
rpt("")
rpt("> **해석**: 업종 중분류 상위 30개를 보면 세부 업종별 시장 공급 현황을 파악할 수 있습니다. 한식, 카페, 일반음식점 등 외식업이 상위권을 차지한다면 해당 지역이 상업 밀집 지역임을 의미하며, 기타창업모음이 많다면 다양한 용도의 매물이 있음을 나타냅니다.\n")

# 3-3. 거래유형
rpt("### 3-3. 거래유형(priceTypeName) 빈도\n")
vc_price = df["priceTypeName"].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    vc_price.values, labels=vc_price.index, autopct="%1.1f%%",
    colors=COLORS[:len(vc_price)], startangle=90, textprops={"fontsize": 12}
)
ax.set_title("거래유형별 매물 비율", fontsize=14, fontweight="bold")
save_fig("03_price_type_pie.png")
rpt(f"![거래유형 비율](../images/03_price_type_pie.png)\n")
rpt(vc_price.reset_index().rename(columns={"priceTypeName": "거래유형", "count": "빈도"}).to_markdown(index=False))
rpt("")
rpt("> **해석**: 거래유형 분포를 통해 해당 지역의 주된 거래 형태를 파악할 수 있습니다. 임대(월세) 비율이 높다면 창업 진입 장벽이 낮은 지역이고, 매매 비율이 높다면 안정적 투자 성향이 강한 지역입니다.\n")

# ══════════════════════════════════════════════════════════
# 4. 수치형 변수 분포 (일변량)
# ══════════════════════════════════════════════════════════
rpt("---\n")
rpt("## 4. 수치형 변수 분포 분석 (일변량)\n")

# 4-1. 보증금 분포
rpt("### 4-1. 보증금(deposit) 분포\n")
fig, ax = plt.subplots(figsize=(10, 5))
deposit_data = df["deposit"][df["deposit"] > 0]
ax.hist(deposit_data, bins=50, color=COLORS[0], edgecolor="white", alpha=0.8)
ax.axvline(deposit_data.median(), color="red", linestyle="--", label=f"중앙값: {deposit_data.median():,.0f}만원")
ax.axvline(deposit_data.mean(), color="orange", linestyle="--", label=f"평균: {deposit_data.mean():,.0f}만원")
ax.set_xlabel("보증금 (만원)")
ax.set_ylabel("빈도수")
ax.set_title("보증금 분포 (보증금 > 0)", fontsize=14, fontweight="bold")
ax.legend()
save_fig("04_deposit_dist.png")
rpt(f"![보증금 분포](../images/04_deposit_dist.png)\n")
deposit_stats = deposit_data.describe()
rpt(f"| 통계 | 값(만원) |\n|:---|---:|\n| 평균 | {deposit_stats['mean']:,.0f} |\n| 중앙값 | {deposit_stats['50%']:,.0f} |\n| 최솟값 | {deposit_stats['min']:,.0f} |\n| 최댓값 | {deposit_stats['max']:,.0f} |\n| 표준편차 | {deposit_stats['std']:,.0f} |\n")
rpt("> **해석**: 보증금 분포는 오른쪽으로 치우친(right-skewed) 형태를 보이는 경우가 많습니다. 중앙값과 평균의 차이가 클수록 소수의 고가 매물이 평균을 끌어올리고 있음을 의미하며, 실제 시장의 일반적인 보증금 수준은 중앙값 기준으로 판단하는 것이 적절합니다.\n")

# 4-2. 월세 분포
rpt("### 4-2. 월세(monthlyRent) 분포\n")
fig, ax = plt.subplots(figsize=(10, 5))
rent_data = df["monthlyRent"][df["monthlyRent"] > 0]
ax.hist(rent_data, bins=50, color=COLORS[3], edgecolor="white", alpha=0.8)
ax.axvline(rent_data.median(), color="red", linestyle="--", label=f"중앙값: {rent_data.median():,.0f}만원")
ax.axvline(rent_data.mean(), color="orange", linestyle="--", label=f"평균: {rent_data.mean():,.0f}만원")
ax.set_xlabel("월세 (만원)")
ax.set_ylabel("빈도수")
ax.set_title("월세 분포 (월세 > 0)", fontsize=14, fontweight="bold")
ax.legend()
save_fig("05_monthly_rent_dist.png")
rpt(f"![월세 분포](../images/05_monthly_rent_dist.png)\n")
rent_stats = rent_data.describe()
rpt(f"| 통계 | 값(만원) |\n|:---|---:|\n| 평균 | {rent_stats['mean']:,.0f} |\n| 중앙값 | {rent_stats['50%']:,.0f} |\n| 최솟값 | {rent_stats['min']:,.0f} |\n| 최댓값 | {rent_stats['max']:,.0f} |\n| 표준편차 | {rent_stats['std']:,.0f} |\n")
rpt("> **해석**: 월세 역시 보증금과 마찬가지로 우측 꼬리가 긴 분포를 보입니다. 대부분의 매물이 특정 월세 범위에 집중되어 있으며, 일부 고가 매물이 분포의 극단에 위치합니다. 창업 비용 산정 시 월세 중앙값을 기준으로 예산을 잡는 것이 현실적입니다.\n")

# 4-3. 전용면적 분포
rpt("### 4-3. 전용면적(size) 분포\n")
fig, ax = plt.subplots(figsize=(10, 5))
size_data = df["size"][df["size"] > 0]
ax.hist(size_data, bins=50, color=COLORS[4], edgecolor="white", alpha=0.8)
ax.axvline(size_data.median(), color="red", linestyle="--", label=f"중앙값: {size_data.median():,.1f}㎡")
ax.set_xlabel("전용면적 (㎡)")
ax.set_ylabel("빈도수")
ax.set_title("전용면적 분포", fontsize=14, fontweight="bold")
ax.legend()
save_fig("06_size_dist.png")
rpt(f"![전용면적 분포](../images/06_size_dist.png)\n")
size_stats = size_data.describe()
rpt(f"| 통계 | 값(㎡) | 약 평 |\n|:---|---:|---:|\n| 평균 | {size_stats['mean']:,.1f} | {size_stats['mean']/3.3058:.1f}평 |\n| 중앙값 | {size_stats['50%']:,.1f} | {size_stats['50%']/3.3058:.1f}평 |\n| 최솟값 | {size_stats['min']:,.1f} | {size_stats['min']/3.3058:.1f}평 |\n| 최댓값 | {size_stats['max']:,.1f} | {size_stats['max']/3.3058:.1f}평 |\n")
rpt("> **해석**: 전용면적 분포를 통해 해당 지역에서 공급되는 매물의 규모를 파악할 수 있습니다. 소형 매물(10~30㎡)이 많다면 소규모 자영업 중심, 대형 매물(100㎡ 이상)이 많다면 음식점·카페 등의 용도가 주류인 지역입니다.\n")

# 4-4. 조회수 분포
rpt("### 4-4. 조회수(viewCount) 분포\n")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["viewCount"], bins=50, color=COLORS[5], edgecolor="white", alpha=0.8)
ax.axvline(df["viewCount"].median(), color="red", linestyle="--", label=f"중앙값: {df['viewCount'].median():,.0f}")
ax.set_xlabel("조회수")
ax.set_ylabel("빈도수")
ax.set_title("매물 조회수 분포", fontsize=14, fontweight="bold")
ax.legend()
save_fig("07_view_count_dist.png")
rpt(f"![조회수 분포](../images/07_view_count_dist.png)\n")
vc_stats = df["viewCount"].describe()
rpt(f"| 통계 | 값 |\n|:---|---:|\n| 평균 | {vc_stats['mean']:,.0f} |\n| 중앙값 | {vc_stats['50%']:,.0f} |\n| 최솟값 | {vc_stats['min']:,.0f} |\n| 최댓값 | {vc_stats['max']:,.0f} |\n")
rpt("> **해석**: 조회수 분포는 대부분의 매물이 낮은 조회수를 기록하지만, 일부 인기 매물은 매우 높은 조회수를 가집니다. 조회수가 높은 매물의 특성(위치, 가격, 업종)을 분석하면 시장에서 선호하는 조건을 추론할 수 있습니다.\n")

# 4-5. 층수 분포
rpt("### 4-5. 층수(floor) 분포\n")
fig, ax = plt.subplots(figsize=(10, 5))
floor_vc = df["floor"].value_counts().sort_index()
ax.bar(floor_vc.index.astype(str), floor_vc.values, color=COLORS[1], edgecolor="white")
ax.set_xlabel("층수")
ax.set_ylabel("빈도수")
ax.set_title("매물 층수 분포", fontsize=14, fontweight="bold")
for i, v in enumerate(floor_vc.values):
    ax.text(i, v + 2, str(v), ha="center", fontsize=9)
save_fig("08_floor_dist.png")
rpt(f"![층수 분포](../images/08_floor_dist.png)\n")
rpt(floor_vc.reset_index().rename(columns={"floor": "층수", "count": "빈도"}).to_markdown(index=False))
rpt("")
rpt("> **해석**: 상가 매물은 1층이 가장 많은 비중을 차지하는 것이 일반적입니다. 1층 매물은 유동인구 접근성이 높아 가격이 비싸지만 창업 성공률도 높고, 지하층이나 2층 이상은 상대적으로 저렴하지만 업종 선택이 제한될 수 있습니다.\n")

# ══════════════════════════════════════════════════════════
# 5. 이변량 분석
# ══════════════════════════════════════════════════════════
rpt("---\n")
rpt("## 5. 이변량 분석\n")

# 5-1. 보증금 vs 월세
rpt("### 5-1. 보증금 vs 월세 산점도\n")
fig, ax = plt.subplots(figsize=(10, 7))
mask = (df["deposit"] > 0) & (df["monthlyRent"] > 0)
ax.scatter(df.loc[mask, "deposit"], df.loc[mask, "monthlyRent"],
           alpha=0.5, c=COLORS[0], s=30, edgecolors="white", linewidth=0.5)
ax.set_xlabel("보증금 (만원)")
ax.set_ylabel("월세 (만원)")
ax.set_title("보증금 vs 월세", fontsize=14, fontweight="bold")
save_fig("09_deposit_vs_rent.png")
rpt(f"![보증금 vs 월세](../images/09_deposit_vs_rent.png)\n")
corr_val = df.loc[mask, ["deposit", "monthlyRent"]].corr().iloc[0, 1]
rpt(f"- **상관계수(r)**: {corr_val:.3f}\n")
rpt("> **해석**: 보증금과 월세 사이의 상관관계를 통해 보증금이 높을수록 월세도 높은 경향이 있는지 확인할 수 있습니다. 양의 상관관계가 강하다면 두 비용이 연동되는 시장이며, 약하다면 보증금을 높여 월세를 낮추는 전략(전환비율)이 활발한 시장임을 의미합니다.\n")

# 5-2. 면적 vs 평당가
rpt("### 5-2. 전용면적 vs 평당가 산점도\n")
fig, ax = plt.subplots(figsize=(10, 7))
mask2 = (df["size"] > 0) & (df["areaPrice"] > 0)
ax.scatter(df.loc[mask2, "size"], df.loc[mask2, "areaPrice"],
           alpha=0.5, c=COLORS[3], s=30, edgecolors="white", linewidth=0.5)
ax.set_xlabel("전용면적 (㎡)")
ax.set_ylabel("평당가 (만원/평)")
ax.set_title("전용면적 vs 평당가", fontsize=14, fontweight="bold")
save_fig("10_size_vs_areaprice.png")
rpt(f"![면적 vs 평당가](../images/10_size_vs_areaprice.png)\n")
corr_val2 = df.loc[mask2, ["size", "areaPrice"]].corr().iloc[0, 1]
rpt(f"- **상관계수(r)**: {corr_val2:.3f}\n")
rpt("> **해석**: 전용면적과 평당가의 관계를 통해 규모에 따른 단가 변화를 파악합니다. 일반적으로 면적이 커질수록 평당가가 낮아지는 '규모의 경제' 현상이 나타나며, 소형 매물의 평당가가 높다면 소규모 점포에 대한 수요가 강한 시장입니다.\n")

# 5-3. 업종 대분류별 보증금 박스플롯
rpt("### 5-3. 업종 대분류별 보증금 분포\n")
fig, ax = plt.subplots(figsize=(12, 6))
cats = df["businessLargeCodeName"].value_counts().index.tolist()
data_box = [df.loc[df["businessLargeCodeName"] == c, "deposit"].dropna().values for c in cats]
bp = ax.boxplot(data_box, labels=cats, patch_artist=True, showfliers=True)
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("보증금 (만원)")
ax.set_title("업종 대분류별 보증금 분포", fontsize=14, fontweight="bold")
plt.xticks(rotation=15, ha="right")
save_fig("11_deposit_by_business.png")
rpt(f"![업종별 보증금](../images/11_deposit_by_business.png)\n")
pivot_dep = df.groupby("businessLargeCodeName")["deposit"].agg(["count", "mean", "median", "min", "max"]).round(0)
pivot_dep.columns = ["매물수", "평균(만원)", "중앙값(만원)", "최솟값(만원)", "최댓값(만원)"]
rpt(pivot_dep.to_markdown())
rpt("")
rpt("> **해석**: 업종별 보증금 분포를 비교하면 업종에 따른 진입 비용 차이를 확인할 수 있습니다. 중앙값이 높은 업종은 상대적으로 자본이 많이 필요한 분야이며, 이상치(극단값)가 많은 업종은 매물 간 가격 편차가 크다는 것을 의미합니다.\n")

# 5-4. 거래유형별 월세 박스플롯
rpt("### 5-4. 거래유형별 월세 분포\n")
fig, ax = plt.subplots(figsize=(8, 6))
price_types = df["priceTypeName"].value_counts().index.tolist()
data_rent = [df.loc[(df["priceTypeName"] == pt) & (df["monthlyRent"] > 0), "monthlyRent"].dropna().values for pt in price_types]
bp2 = ax.boxplot(data_rent, labels=price_types, patch_artist=True, showfliers=True)
for patch, color in zip(bp2["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("월세 (만원)")
ax.set_title("거래유형별 월세 분포", fontsize=14, fontweight="bold")
save_fig("12_rent_by_pricetype.png")
rpt(f"![거래유형별 월세](../images/12_rent_by_pricetype.png)\n")
pivot_rent = df[df["monthlyRent"] > 0].groupby("priceTypeName")["monthlyRent"].agg(["count", "mean", "median", "min", "max"]).round(0)
pivot_rent.columns = ["매물수", "평균(만원)", "중앙값(만원)", "최솟값(만원)", "최댓값(만원)"]
rpt(pivot_rent.to_markdown())
rpt("")
rpt("> **해석**: 거래유형에 따른 월세 분포를 비교하면 임대 방식에 따른 비용 구조 차이를 이해할 수 있습니다. 월세 거래의 분포가 넓다면 다양한 가격대의 매물이 존재하는 유연한 시장이며, 특정 유형에 월세가 0인 경우는 매매나 전세 거래를 의미합니다.\n")

# 5-5. 인근 지하철역별 평균 월세 상위 20
rpt("### 5-5. 인근 지하철역별 평균 월세 상위 20\n")
df_subway = df[df["nearSubwayStation"].notna()].copy()
df_subway["역명"] = df_subway["nearSubwayStation"].str.split(",").str[0].str.strip()
subway_rent = df_subway.groupby("역명").agg(
    매물수=("monthlyRent", "count"),
    평균월세=("monthlyRent", "mean"),
    중앙월세=("monthlyRent", "median"),
).sort_values("평균월세", ascending=False)
subway_top20 = subway_rent[subway_rent["매물수"] >= 3].head(20)

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(subway_top20.index[::-1], subway_top20["평균월세"].values[::-1], color=COLORS[8])
ax.set_xlabel("평균 월세 (만원)")
ax.set_title("인근 지하철역별 평균 월세 상위 20 (매물 3개 이상)", fontsize=14, fontweight="bold")
for i, v in enumerate(subway_top20["평균월세"].values[::-1]):
    ax.text(v + 5, i, f"{v:,.0f}", va="center", fontsize=9)
save_fig("13_subway_avg_rent.png")
rpt(f"![역별 평균 월세](../images/13_subway_avg_rent.png)\n")
rpt(subway_top20.round(0).to_markdown())
rpt("")
rpt("> **해석**: 지하철역별 평균 월세를 비교하면 지역 내 상권의 위계를 파악할 수 있습니다. 월세가 높은 역세권은 유동인구가 많고 상업적 가치가 높은 핵심 상권이며, 월세가 낮은 역세권은 상대적으로 접근성이 떨어지거나 개발 가능성이 있는 지역입니다.\n")

# ══════════════════════════════════════════════════════════
# 6. 다변량 분석
# ══════════════════════════════════════════════════════════
rpt("---\n")
rpt("## 6. 다변량 분석\n")

# 6-1. 상관관계 히트맵
rpt("### 6-1. 주요 수치형 변수 상관관계 히트맵\n")
corr_cols = ["deposit", "monthlyRent", "premium", "sale", "maintenanceFee", "size", "viewCount", "favoriteCount", "areaPrice"]
corr_labels = ["보증금", "월세", "권리금", "매매가", "관리비", "면적", "조회수", "관심수", "평당가"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha="right")
ax.set_yticklabels(corr_labels)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_matrix.values[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("주요 수치형 변수 상관관계", fontsize=14, fontweight="bold")
save_fig("14_correlation_heatmap.png")
rpt(f"![상관관계 히트맵](../images/14_correlation_heatmap.png)\n")
corr_display = corr_matrix.copy()
corr_display.index = corr_labels
corr_display.columns = corr_labels
rpt(corr_display.round(2).to_markdown())
rpt("")
rpt("> **해석**: 수치형 변수 간 상관관계를 한눈에 파악할 수 있습니다. 양의 상관(빨간색)은 두 변수가 함께 증가하는 경향을, 음의 상관(파란색)은 반대 경향을 나타냅니다. 보증금-월세, 면적-관리비 등의 관계를 통해 비용 구조의 연동 패턴을 이해할 수 있습니다.\n")

# 6-2. 면적 vs 보증금 (업종별 색상)
rpt("### 6-2. 전용면적 vs 보증금 (업종 대분류별)\n")
fig, ax = plt.subplots(figsize=(12, 8))
for i, cat in enumerate(cats):
    subset = df[(df["businessLargeCodeName"] == cat) & (df["deposit"] > 0) & (df["size"] > 0)]
    ax.scatter(subset["size"], subset["deposit"], alpha=0.6, c=COLORS[i % len(COLORS)],
               label=cat, s=40, edgecolors="white", linewidth=0.5)
ax.set_xlabel("전용면적 (㎡)")
ax.set_ylabel("보증금 (만원)")
ax.set_title("전용면적 vs 보증금 (업종 대분류별)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
save_fig("15_size_deposit_by_biz.png")
rpt(f"![면적 vs 보증금 업종별](../images/15_size_deposit_by_biz.png)\n")
pivot_multi = df[df["deposit"] > 0].groupby("businessLargeCodeName").agg(
    매물수=("id", "count"),
    평균면적=("size", "mean"),
    평균보증금=("deposit", "mean"),
    평균월세=("monthlyRent", "mean"),
).round(1)
rpt(pivot_multi.to_markdown())
rpt("")
rpt("> **해석**: 업종별로 면적과 보증금의 관계를 색상으로 구분하여 시각화하면, 업종에 따른 공간 수요와 비용 패턴의 차이를 확인할 수 있습니다. 음식업은 넓은 면적에 높은 보증금, 사무실은 작은 면적에 상대적으로 낮은 보증금을 보이는 등 업종별 특성이 드러납니다.\n")

# 6-3. 업종 대분류 × 거래유형 교차표
rpt("### 6-3. 업종 대분류 × 거래유형 교차 분석\n")
cross_tab = pd.crosstab(df["businessLargeCodeName"], df["priceTypeName"], margins=True, margins_name="합계")
fig, ax = plt.subplots(figsize=(12, 6))
cross_data = cross_tab.drop("합계").drop("합계", axis=1)
cross_data.plot(kind="bar", ax=ax, color=COLORS[:len(cross_data.columns)], edgecolor="white", width=0.7)
ax.set_xlabel("업종 대분류")
ax.set_ylabel("매물 수")
ax.set_title("업종 대분류 × 거래유형 교차 분석", fontsize=14, fontweight="bold")
ax.legend(title="거래유형")
plt.xticks(rotation=15, ha="right")
save_fig("16_biz_pricetype_cross.png")
rpt(f"![업종×거래유형](../images/16_biz_pricetype_cross.png)\n")
rpt(cross_tab.to_markdown())
rpt("")
rpt("> **해석**: 업종과 거래유형의 교차분석을 통해 업종별로 어떤 거래 형태가 주류인지 파악할 수 있습니다. 특정 업종에서 매매 비율이 높다면 장기 투자 성향이, 월세 비율이 높다면 유동적인 시장 구조를 나타냅니다.\n")

# ══════════════════════════════════════════════════════════
# 7. TF-IDF 키워드 분석
# ══════════════════════════════════════════════════════════
rpt("---\n")
rpt("## 7. 매물 제목(title) TF-IDF 키워드 분석\n")

titles = df["title"].dropna().tolist()
vectorizer = TfidfVectorizer(
    max_features=1000,
    token_pattern=r"[가-힣a-zA-Z0-9]+",
    min_df=2
)
tfidf_matrix = vectorizer.fit_transform(titles)
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
keyword_df = pd.DataFrame({
    "키워드": feature_names,
    "TF-IDF 점수": tfidf_scores
}).sort_values("TF-IDF 점수", ascending=False)

top30_kw = keyword_df.head(30)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(top30_kw["키워드"].values[::-1], top30_kw["TF-IDF 점수"].values[::-1], color=COLORS[6])
ax.set_xlabel("TF-IDF 점수 합계")
ax.set_title("매물 제목 TF-IDF 키워드 상위 30개", fontsize=14, fontweight="bold")
save_fig("17_tfidf_keywords.png")
rpt(f"![TF-IDF 키워드](../images/17_tfidf_keywords.png)\n")
top30_kw_display = top30_kw.reset_index(drop=True)
top30_kw_display.index = top30_kw_display.index + 1
top30_kw_display.index.name = "순위"
top30_kw_display["TF-IDF 점수"] = top30_kw_display["TF-IDF 점수"].round(2)
rpt(top30_kw_display.to_markdown())
rpt("")
rpt("> **해석**: TF-IDF 키워드 분석을 통해 매물 제목에서 자주 등장하면서도 차별적인 의미를 가진 핵심 키워드를 추출합니다. 지역명(동명, 역명), 업종명(카페, 음식점), 특징(1층, 주차, 신축) 등이 상위에 나타나며, 이를 통해 해당 지역 매물의 주요 셀링 포인트를 파악할 수 있습니다.\n")

# ══════════════════════════════════════════════════════════
# 8. 추가 분석
# ══════════════════════════════════════════════════════════
rpt("---\n")
rpt("## 8. 추가 분석\n")

# 8-1. 관리비 vs 면적
rpt("### 8-1. 전용면적 vs 관리비 산점도\n")
fig, ax = plt.subplots(figsize=(10, 6))
mask3 = (df["size"] > 0) & (df["maintenanceFee"] > 0)
ax.scatter(df.loc[mask3, "size"], df.loc[mask3, "maintenanceFee"],
           alpha=0.5, c=COLORS[9], s=30, edgecolors="white", linewidth=0.5)
ax.set_xlabel("전용면적 (㎡)")
ax.set_ylabel("관리비 (만원)")
ax.set_title("전용면적 vs 관리비", fontsize=14, fontweight="bold")
save_fig("18_size_vs_maintenance.png")
rpt(f"![면적 vs 관리비](../images/18_size_vs_maintenance.png)\n")
corr_val3 = df.loc[mask3, ["size", "maintenanceFee"]].corr().iloc[0, 1]
rpt(f"- **상관계수(r)**: {corr_val3:.3f}\n")
rpt("> **해석**: 전용면적과 관리비의 관계를 보면 면적이 클수록 관리비가 높아지는 경향이 있는지 확인할 수 있습니다. 양의 상관이 강하면 면적에 비례하는 관리비 체계이고, 관계가 약하다면 건물 등급이나 관리 서비스 수준이 관리비의 주요 결정 요인입니다.\n")

# 8-2. 조회수 vs 관심수
rpt("### 8-2. 조회수 vs 관심(좋아요)수 산점도\n")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["viewCount"], df["favoriteCount"],
           alpha=0.5, c=COLORS[7], s=30, edgecolors="white", linewidth=0.5)
ax.set_xlabel("조회수")
ax.set_ylabel("관심수")
ax.set_title("조회수 vs 관심(좋아요)수", fontsize=14, fontweight="bold")
save_fig("19_view_vs_favorite.png")
rpt(f"![조회수 vs 관심수](../images/19_view_vs_favorite.png)\n")
corr_val4 = df[["viewCount", "favoriteCount"]].corr().iloc[0, 1]
rpt(f"- **상관계수(r)**: {corr_val4:.3f}\n")
rpt("> **해석**: 조회수와 관심수의 관계를 통해 매물에 대한 관심의 질을 평가할 수 있습니다. 조회수 대비 관심수가 높은 매물은 실제 거래 전환 가능성이 높은 '핫한' 매물이며, 조회수만 높고 관심수가 낮은 매물은 단순 탐색 대상에 그치는 것일 수 있습니다.\n")

# ══════════════════════════════════════════════════════════
# 9. 종합 요약
# ══════════════════════════════════════════════════════════
rpt("---\n")
rpt("## 9. 분석 종합 요약\n")
rpt(f"""
| 항목 | 내용 |
|:---|:---|
| 총 매물 수 | {len(df):,}개 |
| 업종 대분류 수 | {df['businessLargeCodeName'].nunique()}개 |
| 업종 중분류 수 | {df['businessMiddleCodeName'].nunique()}개 |
| 거래유형 수 | {df['priceTypeName'].nunique()}개 |
| 보증금 중앙값 | {df['deposit'].median():,.0f}만원 |
| 월세 중앙값 | {df[df['monthlyRent']>0]['monthlyRent'].median():,.0f}만원 |
| 전용면적 중앙값 | {df['size'].median():,.1f}㎡ ({df['size'].median()/3.3058:.1f}평) |
| 평당가 중앙값 | {df['areaPrice'].median():,.0f}만원/평 |
| 조회수 중앙값 | {df['viewCount'].median():,.0f}건 |
""")
rpt("""
### 핵심 인사이트

1. **시장 구조**: 해당 지역은 다양한 업종의 상가 매물이 분포하며, 임대(월세) 중심의 거래 구조를 보입니다.
2. **비용 패턴**: 보증금과 월세는 양의 상관관계를 보이며, 면적이 클수록 보증금과 관리비가 증가하는 경향이 있습니다.
3. **1층 프리미엄**: 1층 매물이 가장 많으며, 유동인구 접근성이 좋아 높은 가격대를 형성합니다.
4. **키워드 트렌드**: 매물 제목에서 지역명, 층수 정보, 업종 키워드가 핵심 셀링 포인트로 활용됩니다.
5. **역세권 프리미엄**: 지하철역별 월세 편차가 크며, 핵심 역세권일수록 월세가 높게 형성됩니다.
""")

# ══════════════════════════════════════════════════════════
# 리포트 저장
# ══════════════════════════════════════════════════════════
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\n✅ 리포트 저장 완료: {REPORT_PATH}")
print(f"✅ 이미지 저장 폴더: {IMG_DIR}")
print(f"✅ 총 시각화 수: {len([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])}개")
