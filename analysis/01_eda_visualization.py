"""
feature_table.csv를 진단군(CN/MCI/Dem)별로 비교하는 EDA 시각화.
03절 Q1(활동량 차이), Q2(수면 패턴 차이)에 답하고, MMSE는 참고용으로 함께 본다.

출력: reports/figures/01_activity_by_group.png
      reports/figures/02_sleep_by_group.png
      reports/figures/03_mmse_by_group.png
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_BASE = os.path.join(BASE_DIR, "raw_data", "01_aihub_wearable")
FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_table.csv")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
GROUP_ORDER = ["CN", "MCI", "Dem"]

# CN→MCI→Dem은 임의의 범주가 아니라 위험도가 커지는 순서(ordinal)이므로,
# 서로 무관한 색 3개가 아니라 하나의 파란색 계열을 옅음→짙음으로 쓴다.
# (dataviz 가이드의 sequential/ordinal 팔레트 step 250/450/650)
GROUP_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def style_axes(ax):
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.spines["bottom"].set_linewidth(1)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED, length=0)


def load_mmse():
    # 모델 입력에는 MMSE를 쓰지 않지만(순환 논리 방지), 참고용 시각화에는 사용한다.
    def read(folder, file):
        path = fr"{RAW_BASE}\{folder}\원천데이터\3.인지기능\{file}"
        return pd.read_csv(path)[["SAMPLE_EMAIL", "TOTAL"]]

    mmse = pd.concat([
        read("1.Training", "train_mmse.csv"),
        read("2.Validation", "val_mmse.csv"),
    ], ignore_index=True)
    return mmse.rename(columns={"SAMPLE_EMAIL": "EMAIL", "TOTAL": "mmse_total"})


def bar_by_group(df, value_col, title, ylabel, out_name, unit=""):
    means = df.groupby("DIAG_NM")[value_col].mean().reindex(GROUP_ORDER)
    counts = df.groupby("DIAG_NM")[value_col].count().reindex(GROUP_ORDER)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(GROUP_ORDER, means, width=0.5, color=GROUP_COLORS, zorder=3)

    # 라벨이 그래프 위 테두리에 잘리지 않도록 y축 상한에 18% 여유를 미리 확보한다.
    ax.set_ylim(0, means.max() * 1.18)
    for bar, mean, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + means.max() * 0.02,
                 f"{mean:.1f}{unit}", ha="center", va="bottom", fontsize=11, color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + means.max() * 0.08,
                 f"n={n}", ha="center", va="bottom", fontsize=8.5, color=MUTED)

    ax.set_title(title, color=INK, fontsize=13, pad=12)
    ax.set_ylabel(ylabel, color="#52514e")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, out_name), dpi=150)
    plt.close(fig)
    print(f"{title}: " + ", ".join(f"{g}={means[g]:.1f}" for g in GROUP_ORDER))


def main():
    df = pd.read_csv(FEATURE_PATH)
    mmse = load_mmse().groupby("EMAIL", as_index=False)["mmse_total"].mean()
    df = pd.merge(df, mmse, on="EMAIL", how="left")

    bar_by_group(df, "activity_steps_mean", "진단군별 일평균 걸음수", "걸음수", "01_activity_by_group.png")
    bar_by_group(df, "sleep_efficiency_mean", "진단군별 평균 수면효율", "수면효율(%)", "02_sleep_by_group.png")
    bar_by_group(df, "mmse_total", "진단군별 MMSE 평균 (참고용, 모델 입력 아님)", "MMSE 총점", "03_mmse_by_group.png")


if __name__ == "__main__":
    main()
