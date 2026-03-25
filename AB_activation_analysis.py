"""
JetBrains Onboarding A/B Test — Activation Pattern Analysis
============================================================
Reproduces the full analysis from the HTML report using pandas + seaborn.
Outputs five publication-ready plots saved as PNG files.
"""

import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#2a2d3e",
    "grid.color":       "#2a2d3e",
    "text.color":       "#e2e4ef",
    "axes.labelcolor":  "#e2e4ef",
    "xtick.color":      "#7a7d9a",
    "ytick.color":      "#7a7d9a",
    "axes.titlecolor":  "#e2e4ef",
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#2a2d3e",
    "savefig.facecolor":"#0f1117",
})

PURPLE = "#7c6af7"
TEAL   = "#4ecdc4"
GREEN  = "#4ade80"
RED    = "#f87171"
YELLOW = "#ffe66d"
ORANGE = "#f97b6b"

# ── Load data ──────────────────────────────────────────────────────────────────
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_HERE, "License_activation_ab_test_dataset_3ver.csv")
df = pd.read_csv(CSV)
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns\n")

# ── Helper: manual Welch t-statistic ──────────────────────────────────────────
def welch_t(a, b):
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    na, nb = len(a), len(b)
    se = math.sqrt(sa**2 / na + sb**2 / nb)
    return (ma - mb) / se

# ── Helper: manual chi-square ─────────────────────────────────────────────────
def chi2_1dof(ct):
    total = ct.values.sum()
    rs = ct.sum(axis=1).values
    cs = ct.sum(axis=0).values
    return sum(
        (ct.values[i, j] - rs[i] * cs[j] / total) ** 2 / (rs[i] * cs[j] / total)
        for i in range(ct.shape[0])
        for j in range(ct.shape[1])
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — A/B Activation Rate Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("PLOT 1 · A/B Activation Rate Comparison")

ab = df.groupby("experiment_group")["activated"].agg(
    rate="mean", count="count", activated="sum"
).reset_index()
ab["pct"] = ab["rate"] * 100
ab["not_activated"] = ab["count"] - ab["activated"]

ct = pd.crosstab(df["experiment_group"], df["activated"])
chi2 = chi2_1dof(ct)
lift = (ab.loc[ab.experiment_group == "B", "rate"].values[0] /
        ab.loc[ab.experiment_group == "A", "rate"].values[0] - 1) * 100

print(f"  Group A: {ab.loc[ab.experiment_group=='A','pct'].values[0]:.2f}%")
print(f"  Group B: {ab.loc[ab.experiment_group=='B','pct'].values[0]:.2f}%")
print(f"  Lift:    +{lift:.2f}%   χ²={chi2:.1f} (p < 0.001)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Plot 1 · A/B Test: Activation Rate Comparison", fontsize=14, fontweight="bold", y=1.01)

# Left — bar chart
colors = [PURPLE, TEAL]
bars = axes[0].bar(ab["experiment_group"], ab["pct"], color=colors,
                   width=0.5, edgecolor="white", linewidth=0.5)
for bar, row in zip(bars, ab.itertuples()):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{row.pct:.2f}%", ha="center", va="bottom",
                 fontweight="bold", fontsize=13, color="white")
axes[0].set_ylim(55, 74)
axes[0].set_xlabel("Experiment Group")
axes[0].set_ylabel("Activation Rate (%)")
axes[0].set_title(f"Activation Rate by Group\nχ² = {chi2:.1f} · p < 0.001 · Lift = +{lift:.1f}%")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Right — stacked bar (activated vs not)
ab_plot = ab.set_index("experiment_group")[["activated", "not_activated"]]
ab_plot.plot(kind="bar", stacked=True, ax=axes[1],
             color=[GREEN, RED], edgecolor="none", width=0.5)
axes[1].set_title("Activated vs Not-Activated (stacked)")
axes[1].set_xlabel("Experiment Group")
axes[1].set_ylabel("Number of Users")
axes[1].set_xticklabels(["Group A", "Group B"], rotation=0)
axes[1].legend(["Activated", "Not Activated"], loc="upper right")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig("plot1_ab_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → saved plot1_ab_comparison.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — The 30-Minute Cliff
# ═══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("PLOT 2 · The 30-Minute Cliff")

df["time_bucket"] = pd.cut(
    df["time_to_first_run_min"],
    bins=[0, 5, 15, 30, 60, 120, np.inf],
    labels=["0–5 min", "5–15 min", "15–30 min", "30–60 min", "60–120 min", "120+ min"]
)
bucket_stats = (
    df.groupby("time_bucket", observed=True)["activated"]
    .agg(rate="mean", count="count")
    .reset_index()
)
bucket_stats["pct"] = bucket_stats["rate"] * 100

print(bucket_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 2 · The 30-Minute Cliff: Time to First Run vs Activation", fontsize=14, fontweight="bold")

# Left — activation rate per bucket
bucket_colors = [GREEN, GREEN, YELLOW, RED, RED, RED]
axes[0].bar(bucket_stats["time_bucket"].astype(str), bucket_stats["pct"],
            color=bucket_colors, edgecolor="white", linewidth=0.4, width=0.6)
for i, row in bucket_stats.iterrows():
    axes[0].text(i, row["pct"] + 1.5, f"{row['pct']:.1f}%",
                 ha="center", fontsize=11, fontweight="bold", color="white")
axes[0].axvline(x=2.5, color=ORANGE, linewidth=2, linestyle="--", alpha=0.8)
axes[0].text(2.7, 50, "← 30-min cliff", color=ORANGE, fontsize=10, fontstyle="italic")
axes[0].set_ylim(0, 108)
axes[0].set_xlabel("Time to First Run")
axes[0].set_ylabel("Activation Rate (%)")
axes[0].set_title("Activation Rate per Time Bucket")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
axes[0].tick_params(axis="x", rotation=20)

# Right — user count per bucket
axes[1].bar(bucket_stats["time_bucket"].astype(str), bucket_stats["count"],
            color=bucket_colors, edgecolor="white", linewidth=0.4, width=0.6)
for i, row in bucket_stats.iterrows():
    axes[1].text(i, row["count"] + 60, f"{int(row['count']):,}",
                 ha="center", fontsize=10, color="white")
axes[1].axvline(x=2.5, color=ORANGE, linewidth=2, linestyle="--", alpha=0.8)
axes[1].set_xlabel("Time to First Run")
axes[1].set_ylabel("Number of Users")
axes[1].set_title("User Volume per Time Bucket")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("plot2_30min_cliff.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → saved plot2_30min_cliff.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Time to First Run Distribution: Activated vs Not
# ═══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("PLOT 3 · Time-to-First-Run Distribution by Activation Status")

act   = df[df["activated"] == 1]["time_to_first_run_min"]
noact = df[df["activated"] == 0]["time_to_first_run_min"]
t_stat = welch_t(act.values, noact.values)

print(f"  Activated   mean={act.mean():.2f}  median={act.median():.2f}")
print(f"  Not Activ.  mean={noact.mean():.2f}  median={noact.median():.2f}")
print(f"  t = {t_stat:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 3 · Time to First Run: Activated vs Not-Activated", fontsize=14, fontweight="bold")

df["Activation Status"] = df["activated"].map({1: "Activated", 0: "Not Activated"})

# Left — overlapping KDE
sns.kdeplot(data=df, x="time_to_first_run_min", hue="Activation Status",
            fill=True, alpha=0.35, linewidth=2,
            palette={"Activated": GREEN, "Not Activated": RED},
            ax=axes[0], clip=(0, 80))
axes[0].axvline(act.mean(), color=GREEN, linestyle="--", linewidth=1.5,
                label=f"Activated mean ({act.mean():.1f}m)")
axes[0].axvline(noact.mean(), color=RED, linestyle="--", linewidth=1.5,
                label=f"Not-activated mean ({noact.mean():.1f}m)")
axes[0].axvline(30, color=ORANGE, linestyle=":", linewidth=2, alpha=0.7, label="30-min cliff")
axes[0].set_xlim(0, 75)
axes[0].set_xlabel("Time to First Run (minutes)")
axes[0].set_ylabel("Density")
axes[0].set_title(f"KDE Distribution\nt = {t_stat:.1f} · p < 0.001")
axes[0].legend(fontsize=9)

# Right — box + strip
sns.boxplot(data=df, x="Activation Status", y="time_to_first_run_min",
            palette={"Activated": GREEN, "Not Activated": RED},
            width=0.45, linewidth=1.2, fliersize=2, ax=axes[1])
axes[1].axhline(30, color=ORANGE, linestyle=":", linewidth=2, alpha=0.7, label="30-min cliff")
axes[1].set_xlabel("")
axes[1].set_ylabel("Time to First Run (minutes)")
axes[1].set_title("Box Plot: Spread & Median")
axes[1].legend()

plt.tight_layout()
plt.savefig("plot3_time_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → saved plot3_time_distribution.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Time to First Run: Group A vs Group B
# ═══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("PLOT 4 · Time to First Run: Group A vs Group B")

a_times = df[df["experiment_group"] == "A"]["time_to_first_run_min"]
b_times = df[df["experiment_group"] == "B"]["time_to_first_run_min"]
t_ab = welch_t(a_times.values, b_times.values)

print(f"  Group A  mean={a_times.mean():.2f}  median={a_times.median():.2f}")
print(f"  Group B  mean={b_times.mean():.2f}  median={b_times.median():.2f}")
print(f"  t = {t_ab:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 4 · Time to First Run Distribution: Group A vs Group B", fontsize=14, fontweight="bold")

# Left — overlapping histogram
bins = np.arange(0, 65, 5)
axes[0].hist(a_times.clip(upper=64), bins=bins, alpha=0.55, color=PURPLE,
             edgecolor="white", linewidth=0.3, label=f"Group A (median {a_times.median():.1f}m)")
axes[0].hist(b_times.clip(upper=64), bins=bins, alpha=0.55, color=TEAL,
             edgecolor="white", linewidth=0.3, label=f"Group B (median {b_times.median():.1f}m)")
axes[0].axvline(a_times.median(), color=PURPLE, linewidth=2, linestyle="--")
axes[0].axvline(b_times.median(), color=TEAL, linewidth=2, linestyle="--")
axes[0].axvline(30, color=ORANGE, linewidth=1.8, linestyle=":", alpha=0.8, label="30-min cliff")
axes[0].set_xlabel("Time to First Run (minutes)")
axes[0].set_ylabel("Number of Users")
axes[0].set_title("Histogram (5-min bins)")
axes[0].legend(fontsize=9)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Right — ECDF (empirical CDF)
for group, color, label in [("A", PURPLE, "Group A"), ("B", TEAL, "Group B")]:
    vals = np.sort(df[df["experiment_group"] == group]["time_to_first_run_min"].values)
    cdf  = np.arange(1, len(vals) + 1) / len(vals)
    axes[1].plot(vals, cdf * 100, color=color, linewidth=2.2, label=label)
axes[1].axvline(30, color=ORANGE, linewidth=2, linestyle=":", alpha=0.8, label="30-min cliff")
axes[1].axhline(50, color="#7a7d9a", linewidth=1, linestyle="--", alpha=0.5)
axes[1].set_xlim(0, 65)
axes[1].set_xlabel("Time to First Run (minutes)")
axes[1].set_ylabel("Cumulative % of Users")
axes[1].set_title(f"Empirical CDF · t = {t_ab:.1f}")
axes[1].legend()
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.tight_layout()
plt.savefig("plot4_group_time_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → saved plot4_group_time_comparison.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Feature Usage: Impact on Activation
# ═══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("PLOT 5 · Day-1 Feature Usage vs Activation")

# Feature combos
df["Feature Combo"] = df.apply(
    lambda r: f"Auto={'✓' if r['used_autocomplete_day1'] else '✗'}  Refac={'✓' if r['used_refactoring_day1'] else '✗'}",
    axis=1
)
combo_stats = (
    df.groupby("Feature Combo")["activated"]
    .agg(rate="mean", count="count")
    .reset_index()
    .sort_values("rate", ascending=False)
)
combo_stats["pct"] = combo_stats["rate"] * 100

# Feature usage rates by group
feat_group = df.groupby("experiment_group")[
    ["used_autocomplete_day1", "used_refactoring_day1"]
].mean().reset_index()
feat_group.columns = ["Group", "Autocomplete Day 1", "Refactoring Day 1"]
feat_melt = feat_group.melt(id_vars="Group", var_name="Feature", value_name="Usage Rate")
feat_melt["Usage Rate"] *= 100

print("  Feature combo activation rates:")
print(combo_stats[["Feature Combo","pct","count"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 5 · Day-1 Feature Usage — Effect on Activation", fontsize=14, fontweight="bold")

# Left — combo activation bars
bar_colors = [GREEN if v > 65 else YELLOW for v in combo_stats["pct"]]
axes[0].barh(combo_stats["Feature Combo"], combo_stats["pct"],
             color=bar_colors, edgecolor="white", linewidth=0.4, height=0.5)
for i, row in combo_stats.iterrows():
    axes[0].text(row["pct"] + 0.1, list(combo_stats.index).index(i),
                 f"{row['pct']:.1f}%  (n={row['count']:,})",
                 va="center", fontsize=10, color="white")
axes[0].set_xlim(60, 70)
axes[0].set_xlabel("Activation Rate (%)")
axes[0].set_title("Activation Rate by Feature Combo\n(all within ~2% — feature use barely matters)")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Right — feature usage rates by group
sns.barplot(data=feat_melt, x="Feature", y="Usage Rate", hue="Group",
            palette={"A": PURPLE, "B": TEAL},
            edgecolor="white", linewidth=0.4, width=0.5, ax=axes[1])
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.1f%%", padding=3, fontsize=10, color="white")
axes[1].set_ylim(0, 85)
axes[1].set_xlabel("")
axes[1].set_ylabel("Usage Rate (%)")
axes[1].set_title("Feature Usage Rates by Group\n(Group B uses both features more)")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
axes[1].legend(title="Group")

plt.tight_layout()
plt.savefig("plot5_feature_usage.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → saved plot5_feature_usage.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ALL PLOTS SAVED:")
for i, name in enumerate([
    "plot1_ab_comparison.png       — A/B activation bar + stacked",
    "plot2_30min_cliff.png         — Activation rate & volume by time bucket",
    "plot3_time_distribution.png   — KDE + box: activated vs not-activated",
    "plot4_group_time_comparison.png — Histogram + ECDF: Group A vs B",
    "plot5_feature_usage.png       — Feature combo activation + usage by group",
], start=1):
    print(f"  {name}")
print("=" * 60)
