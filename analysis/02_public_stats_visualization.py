"""
문제인식(01절) 발표용 공공 통계 시각화. AI Hub raw_data(모델링용)와는 별개로,
보건복지부/HIRA 공개 통계로 "치매 문제가 왜 심각한가"를 보여준다.
(원래 팀원이 만든 버전을 검수 후 수정: 근거 약한 차트 제거, 라벨 겹침/기준선 보완)

원본 CSV 출처: raw_data_stats/ (보건복지부, HIRA, 중앙치매센터)
출력: ex/_시각화/02~06_*.png (01번 유병률 추이 차트는 삭제 — 근거 부족, 기획안과 중복)
"""
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

COLOR_MAIN = '#2E5EAA'
COLOR_SUB = '#E8743B'
COLOR_ACCENT = '#5FAD56'
plt.rcParams['axes.facecolor'] = '#FAFAFA'
plt.rcParams['figure.facecolor'] = 'white'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(BASE_DIR, "raw_data_stats")  # 최종 PNG 추출 후 삭제됨 — 재실행하려면 출처 URL로 재수집 필요
OUT = os.path.join(BASE_DIR, "intro") + os.sep

# ============================================================
# Chart 2: 성별 등록 치매환자수 추이 (2015-2024)
# ============================================================
df2 = pd.read_csv(BASE + r"\보건복지부\보건복지부_치매환자 등록 현황_20241231.csv", encoding='cp949')
gender_trend = df2.groupby('연도')[['남', '여']].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(gender_trend['연도'], gender_trend['여'], label='여성', color=COLOR_SUB, width=0.6)
ax.bar(gender_trend['연도'], gender_trend['남'], bottom=gender_trend['여'], label='남성', color=COLOR_MAIN, width=0.6)
total = gender_trend['남'] + gender_trend['여']
for x, y in zip(gender_trend['연도'], total):
    ax.annotate(f'{y:,.0f}', (x, y), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=9)
ax.set_title('전국 등록 치매환자수 추이 (성별, 2015~2024)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('연도'); ax.set_ylabel('등록 환자수 (명)')
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, -0.02, '※ 치매안심센터 등록 기준 통계 — 유병률 기반 추정 인구수와는 집계 기준이 다름',
          ha='center', fontsize=9, color='#666666')
plt.tight_layout()
plt.savefig(OUT + '02_gender_registration_trend.png', bbox_inches='tight')
plt.close()
gender_trend['여성비율'] = gender_trend['여'] / total * 100
print('Chart 2 done'); print(gender_trend[['연도', '여성비율']])

# ============================================================
# Chart 3: 시도별 치매 진료인원 (HIRA, 2024)
# ============================================================
df3 = pd.read_csv(BASE + r"\건강보험심사평가원\건강보험심사평가원_시군구별 성별 치매질환 통계 2024.csv", encoding='cp949')
prov = df3.groupby('시도')['환자수'].sum().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(10, 7))
colors = [COLOR_MAIN if i > 2 else COLOR_SUB for i in range(len(prov))]
bars = ax.barh(prov['시도'][::-1], prov['환자수'][::-1], color=colors[::-1])
for bar, val in zip(bars, prov['환자수'][::-1]):
    ax.text(val + 500, bar.get_y() + bar.get_height() / 2, f'{val:,}', va='center', fontsize=9)
ax.set_title('시도별 치매 진료인원 (2024년) — 인구 규모 영향 있음, ⑤번과 함께 볼 것', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('진료인원 (명)')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT + '03_region_patients_2024.png', bbox_inches='tight')
plt.close()
print('Chart 3 done')

# ============================================================
# Chart 4: 연령대·성별 치매 진료인원 분포 (HIRA, 2024)
# ============================================================
df4 = pd.read_csv(BASE + r"\건강보험심사평가원\건강보험심사평가원_시군구별 성별 연령별 치매질환 통계 2024.csv", encoding='cp949')
age_order = ['0~9세', '10~19세', '20~29세', '30~39세', '40~49세', '50~59세',
             '60~69세', '70~79세', '80~89세', '90~99세', '100세이상']
age_gender = df4.groupby(['연령구분', '성별'])['환자수'].sum().unstack().reindex(age_order)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(age_order))
w = 0.38
ax.bar(x - w / 2, age_gender['남'], width=w, label='남성', color=COLOR_MAIN)
ax.bar(x + w / 2, age_gender['여'], width=w, label='여성', color=COLOR_SUB)
ax.set_xticks(x); ax.set_xticklabels(age_order, rotation=30, ha='right')
ax.set_title('연령대·성별 치매 진료인원 분포 (2024년)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('진료인원 (명)')
ax.legend(fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT + '04_age_gender_distribution.png', bbox_inches='tight')
plt.close()
print('Chart 4 done')

# ============================================================
# Chart 5: 시군구 유병률 상위 10개 지역 (2024) + 전국 평균 기준선
# ============================================================
df5 = pd.read_csv(BASE + r"\보건복지부\보건복지부_시군구별 치매현황_20251231.csv", encoding='cp949', low_memory=False)
df5['연령별_norm'] = df5['연령별'].str.replace(' - ', '~').str.replace(' ', '')
gu = df5[(df5['연도'] == 2024) & (df5['시도'] != '전국') & (df5['시군구'] != df5['시도']) &
         (df5['성별'] == '전체') & (df5['연령별_norm'] == '65세이상')].copy()
gu['추정치매환자유병률'] = pd.to_numeric(gu['추정치매환자유병률'].astype(str).str.replace('%', ''), errors='coerce')
gu['지역명'] = gu['시도'].str[:2] + ' ' + gu['시군구']
top10 = gu.nlargest(10, '추정치매환자유병률')[['지역명', '추정치매환자유병률', '노인인구수']].sort_values('추정치매환자유병률')

nat_row = df5[(df5['연도'] == 2024) & (df5['시도'] == '전국') & (df5['시군구'] == '전국') &
              (df5['성별'] == '전체') & (df5['연령별_norm'] == '65세이상')]
nat_avg = float(str(nat_row['추정치매환자유병률'].iloc[0]).replace('%', ''))

fig, ax = plt.subplots(figsize=(10, 6.5))
bars = ax.barh(top10['지역명'], top10['추정치매환자유병률'], color=COLOR_ACCENT)
for bar, val in zip(bars, top10['추정치매환자유병률']):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', fontsize=9)
ax.axvline(nat_avg, color='#555555', linestyle='--', linewidth=1.5, zorder=1)
ax.text(nat_avg + 0.15, len(top10) - 0.4, f'전국 평균 {nat_avg:.1f}%', color='#555555', fontsize=9.5,
        va='top', ha='left', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
ax.set_title('65세 이상 치매 유병률 상위 10개 시군구 (2024년)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('유병률 (%)')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT + '05_top10_prevalence_regions.png', bbox_inches='tight')
plt.close()
print('Chart 5 done, 전국평균=', nat_avg)

# ============================================================
# Chart 6: 독거노인 수 vs 치매 등록환자수 (시도별, 2024) — 참고용
# ============================================================
df6a = pd.read_csv(BASE + r"\보건복지부\보건복지부_독거노인 수_연령별_시도별_20241231.csv", encoding='cp949')
df6a = df6a[df6a['연도'] == 2024].copy()
df6a['독거노인수'] = df6a[['65-69세', '70-74세', '75-79세', '80-84세', '85세이상']].sum(axis=1)
df6a['시도명'] = df6a['시도'].str.split(' ').str[0]

df6b = pd.read_csv(BASE + r"\보건복지부\보건복지부_치매환자 등록 현황_20241231.csv", encoding='cp949')
df6b = df6b[df6b['연도'] == 2024].copy()
df6b['등록환자수'] = df6b['남'] + df6b['여']
df6b['시도명'] = df6b['시도'].str.split(' ').str[0]

merged = pd.merge(df6a[['시도명', '독거노인수']], df6b[['시도명', '등록환자수']], on='시도명')

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(merged['독거노인수'], merged['등록환자수'], s=110, color=COLOR_MAIN, alpha=0.75,
           edgecolor='white', linewidth=1, zorder=3)

crowded = (merged['독거노인수'] < 150000) & (merged['등록환자수'] < 30000)
fan_offsets = [(10, 10), (10, -16), (-55, 10), (-55, -16)]
crowded_idx = merged[crowded].sort_values('독거노인수').index.tolist()

for _, row in merged.iterrows():
    if row.name in crowded_idx:
        dx, dy = fan_offsets[crowded_idx.index(row.name) % len(fan_offsets)]
    else:
        dx, dy = (8, 8)
    ax.annotate(row['시도명'], (row['독거노인수'], row['등록환자수']),
                textcoords='offset points', xytext=(dx, dy), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75), zorder=4)

z = np.polyfit(merged['독거노인수'], merged['등록환자수'], 1)
xs = np.linspace(merged['독거노인수'].min(), merged['독거노인수'].max(), 100)
ax.plot(xs, np.polyval(z, xs), '--', color=COLOR_SUB, alpha=0.7, label='추세선', zorder=2)
corr = merged['독거노인수'].corr(merged['등록환자수'])

ax.set_title('시도별 독거노인 수 - 치매 등록환자수 관계 (2024년)', fontsize=13, fontweight='bold', pad=15)
ax.text(0.02, 0.98, f'상관계수 r={corr:.2f} (참고용 — 두 값 모두 인구 규모에 영향받아\n인과관계를 의미하지 않음)',
        transform=ax.transAxes, fontsize=9, color='#666666', va='top', ha='left')
ax.set_xlabel('독거노인 수 (65세 이상, 명)')
ax.set_ylabel('치매 등록환자수 (명)')
ax.legend(fontsize=10, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT + '06_solo_elderly_vs_dementia.png', bbox_inches='tight')
plt.close()
print('Chart 6 done, corr =', corr)

# ============================================================
# Chart 7: 치매 고위험군 — 인구·사회학적 요인별 유병률 (2023년 치매역학조사)
# 출처: 보건복지부·중앙치매센터 [별첨]2023년 치매역학조사 및 실태조사 주요 결과, p.4
# (해당 보고서는 원표를 그대로 인용 — raw_data_stats/보건복지부/ 에 원문 보관)
# 기획안 01절 "고령·여성·농촌·독거·저학력일수록 유병률 높음" 통계 조사 리스트 6번 항목을
# 채워주는 자료. CSV가 아니라 보고서 표를 직접 인용하므로 하드코딩한다.
# ============================================================
factors = {
    '성별': {'남성': 8.85, '여성': 9.57},
    '지역(동 vs 읍면)': {'동(도시)': 5.5, '읍·면(농어촌)': 9.4},
    '가구유형': {'배우자와 거주': 4.9, '배우자 외 동거인': 5.2, '독거가구': 10.0},
    '교육수준': {'대학교 이상': 1.4, '고등학교 졸업': 2.6, '무학': 21.3},
}

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (title, data) in zip(axes.flat, factors.items()):
    labels = list(data.keys())
    values = list(data.values())
    bars = ax.bar(labels, values, color=COLOR_MAIN, width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(axis='x', labelsize=9)

fig.suptitle('치매 고위험군 — 인구·사회학적 요인별 유병률 (2023년 치매역학조사)', fontsize=14, fontweight='bold', y=1.00)
fig.text(0.5, -0.01, '출처: 보건복지부·중앙치매센터, 2023년 치매역학조사 및 실태조사(65세 이상 기준)',
          ha='center', fontsize=9, color='#666666')
plt.tight_layout()
plt.savefig(OUT + '07_risk_factors_by_demographic.png', bbox_inches='tight')
plt.close()
print('Chart 7 done')
