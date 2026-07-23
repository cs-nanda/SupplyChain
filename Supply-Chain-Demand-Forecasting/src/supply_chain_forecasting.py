"""
Supply Chain Analytics and Demand Forecasting
==============================================
Dataset: M5 Forecasting (Walmart retail sales)
Models:  MSTL (Multiple Seasonal-Trend via Loess) + Naive Seasonal Baseline
Metrics: MAE and SMAPE
Split:   80% training / 20% test (chronological)
"""

# ─────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os, time

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("=" * 60)
print("  Supply Chain Analytics & Demand Forecasting")
print("=" * 60)
print("\n[1/7] Loading datasets...")

SALES   = "/mnt/user-data/uploads/1777908220194_sales_train_validation.csv"
EVAL    = "/mnt/user-data/uploads/1777908220194_sales_train_evaluation.csv"
CAL     = "/mnt/user-data/uploads/1777908220193_calendar.csv"
PRICES  = "/mnt/user-data/uploads/1777908220194_sell_prices.csv"
OUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

sales_val  = pd.read_csv(SALES)
sales_eval = pd.read_csv(EVAL)
calendar   = pd.read_csv(CAL, parse_dates=["date"])
prices     = pd.read_csv(PRICES)

print(f"  Validation sales  : {sales_val.shape}  (rows × cols)")
print(f"  Evaluation sales  : {sales_eval.shape}")
print(f"  Calendar          : {calendar.shape}")
print(f"  Sell prices       : {prices.shape}")

# ─────────────────────────────────────────────
# 2. Melt to long format (use validation file — 1913 days)
# ─────────────────────────────────────────────
print("\n[2/7] Reshaping to long format + merging calendar...")

meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols  = [c for c in sales_val.columns if c.startswith("d_")]

# BUG FIX 1 & 3: Aggregate in wide format BEFORE melting to avoid ~58M row explosion.
# Instead of melting all 30,490 × 1,913 rows, compute group-level daily sums directly.
cal_sub = calendar[["d", "date", "wm_yr_wk", "event_name_1",
                     "event_type_1", "snap_CA", "snap_TX", "snap_WI"]]

def agg_group(df, group_col):
    """Sum day columns per group, melt, then merge calendar."""
    grouped = df.groupby(group_col)[day_cols].sum().reset_index()
    melted  = grouped.melt(id_vars=group_col, value_vars=day_cols,
                           var_name="d", value_name="sales")
    melted  = melted.merge(cal_sub, on="d", how="left")
    melted["date"] = pd.to_datetime(melted["date"])
    return melted.sort_values([group_col, "date"]).reset_index(drop=True)

# Three slim aggregations — each is only (n_groups × 1913) rows
sales_by_cat   = agg_group(sales_val, "cat_id")
sales_by_state = agg_group(sales_val, "state_id")
sales_daily    = (sales_val[day_cols].sum()             # all SKUs summed
                  .reset_index().rename(columns={"index": "d", 0: "sales"})
                  .merge(cal_sub, on="d", how="left"))
sales_daily["date"] = pd.to_datetime(sales_daily["date"])

# BUG FIX 2: row count was off because wc -l counts newlines (including header).
n_skus = len(sales_val)   # correct — len() counts data rows only
print(f"  SKU-store series  : {n_skus:,}")
print(f"  Day columns       : {len(day_cols)}")
print(f"  Date range        : {cal_sub['date'].min()} → {cal_sub['date'].max()}")
print(f"  Categories        : {sorted(sales_val['cat_id'].unique().tolist())}")
print(f"  States            : {sorted(sales_val['state_id'].unique().tolist())}")

# ─────────────────────────────────────────────
# 3. EDA  (aggregate daily total sales across all series)
# ─────────────────────────────────────────────
print("\n[3/7] Exploratory Data Analysis...")

daily_total = sales_daily.rename(columns={"sales": "sales"})

# Stats
print(f"  Avg daily units sold : {daily_total['sales'].mean():.0f}")
print(f"  Max daily units sold : {daily_total['sales'].max():.0f}")
zero_frac = (sales_val[day_cols] == 0).values.mean()
print(f"  Zero-sales fraction  : {zero_frac:.2%}")

# ── Plot 1: Total daily sales trend
fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.suptitle("Supply Chain Analytics — Exploratory Data Analysis", fontsize=15, fontweight="bold", y=0.98)

axes[0].plot(daily_total["date"], daily_total["sales"], color="#2563EB", linewidth=0.8, alpha=0.9)
axes[0].set_title("Total Daily Sales (All SKUs & Stores)", fontsize=12)
axes[0].set_xlabel("Date"); axes[0].set_ylabel("Units Sold")
axes[0].grid(alpha=0.3)

# ── Plot 2: Sales by category (uses pre-aggregated sales_by_cat)
cat_daily = sales_by_cat.groupby(["date", "cat_id"])["sales"].sum().reset_index()
palette = {"HOBBIES": "#2563EB", "FOODS": "#16A34A", "HOUSEHOLD": "#DC2626"}
for cat, grp in cat_daily.groupby("cat_id"):
    axes[1].plot(grp["date"], grp["sales"], label=cat, linewidth=0.8,
                 color=palette.get(cat, "gray"), alpha=0.85)
axes[1].set_title("Daily Sales by Category", fontsize=12)
axes[1].set_xlabel("Date"); axes[1].set_ylabel("Units Sold")
axes[1].legend(); axes[1].grid(alpha=0.3)

# ── Plot 3: Sales by state (uses pre-aggregated sales_by_state)
state_daily = sales_by_state.groupby(["date", "state_id"])["sales"].sum().reset_index()
state_pal = {"CA": "#7C3AED", "TX": "#D97706", "WI": "#0891B2"}
for state, grp in state_daily.groupby("state_id"):
    axes[2].plot(grp["date"], grp["sales"], label=state, linewidth=0.8,
                 color=state_pal.get(state, "gray"), alpha=0.85)
axes[2].set_title("Daily Sales by State", fontsize=12)
axes[2].set_xlabel("Date"); axes[2].set_ylabel("Units Sold")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/01_eda_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_eda_trends.png")

# ── Plot 4: Monthly seasonality heatmap (category × month)
sales_by_cat["month"]   = sales_by_cat["date"].dt.month
sales_by_cat["weekday"] = sales_by_cat["date"].dt.dayofweek
monthly_cat = sales_by_cat.groupby(["cat_id", "month"])["sales"].mean().unstack()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Seasonality Patterns", fontsize=13, fontweight="bold")

sns.heatmap(monthly_cat, ax=axes[0], cmap="YlOrRd", annot=True, fmt=".0f",
            linewidths=0.5, cbar_kws={"label": "Avg Daily Sales"})
axes[0].set_title("Avg Daily Sales by Category × Month")
axes[0].set_xlabel("Month"); axes[0].set_ylabel("Category")

dow_cat = sales_by_cat.groupby(["cat_id", "weekday"])["sales"].mean().unstack()
dow_cat.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
sns.heatmap(dow_cat, ax=axes[1], cmap="Blues", annot=True, fmt=".0f",
            linewidths=0.5, cbar_kws={"label": "Avg Daily Sales"})
axes[1].set_title("Avg Daily Sales by Category × Day of Week")
axes[1].set_xlabel("Day"); axes[1].set_ylabel("Category")

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/02_eda_seasonality.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_eda_seasonality.png")

# ─────────────────────────────────────────────
# 4. Prepare aggregate weekly series for modelling
#    (aggregate = total across all stores & items to keep runtime reasonable)
# ─────────────────────────────────────────────
print("\n[4/7] Preparing weekly aggregate series (80/20 split)...")

weekly = (sales_daily
          .groupby(pd.Grouper(key="date", freq="W-SAT"))["sales"]
          .sum()
          .reset_index()
          .rename(columns={"date": "ds", "sales": "y"}))

# Drop incomplete last week if present
weekly = weekly[weekly["y"] > 0].reset_index(drop=True)

n_total = len(weekly)
n_train = int(n_total * 0.80)
n_test  = n_total - n_train

train_df = weekly.iloc[:n_train].copy()
test_df  = weekly.iloc[n_train:].copy()

print(f"  Total weekly obs  : {n_total}")
print(f"  Train weeks       : {n_train}  ({train_df['ds'].min().date()} → {train_df['ds'].max().date()})")
print(f"  Test  weeks       : {n_test}   ({test_df['ds'].min().date()} → {test_df['ds'].max().date()})")

train_series = train_df.set_index("ds")["y"]
test_series  = test_df.set_index("ds")["y"]

# ─────────────────────────────────────────────
# 5. Modelling
# ─────────────────────────────────────────────
print("\n[5/7] Fitting models...")

# ── Metric helpers ──
def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def smape(actual, predicted):
    num   = np.abs(actual - predicted)
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(num / denom) * 100

# ────────────────────────────────────
# Model A: Naive Seasonal (last-year same week) — simple baseline
# ────────────────────────────────────
print("  [A] Naive Seasonal baseline...")
naive_preds = []
for i in range(n_test):
    # For each test step i, look back 52 weeks from the corresponding position in train
    lag_idx = len(train_series) - 52 + (i % 52)
    if 0 <= lag_idx < len(train_series):
        naive_preds.append(float(train_series.iloc[lag_idx]))
    else:
        naive_preds.append(float(train_series.mean()))

naive_preds = np.array(naive_preds)
naive_mae   = mae(test_series.values, naive_preds)
naive_smape = smape(test_series.values, naive_preds)
print(f"     MAE={naive_mae:.1f}   SMAPE={naive_smape:.2f}%")

# ────────────────────────────────────
# Model B: MSTL + ETS residuals
# ────────────────────────────────────
print("  [B] MSTL decomposition + ETS on residuals...")

# MSTL with weekly (period=52) and partial annual (period=13) seasonality
mstl = MSTL(train_series, periods=[52, 13])
mstl_result = mstl.fit()

trend     = mstl_result.trend
seasonal  = mstl_result.seasonal.sum(axis=1)
residual  = mstl_result.resid

# Fit ETS on residuals to forecast them forward
ets_resid = ExponentialSmoothing(residual, trend=None, seasonal=None).fit()
resid_fc  = ets_resid.forecast(n_test)

# Project trend (linear extrapolation from last N points)
N = 26  # 6 months of trend slope
trend_slope = (trend.iloc[-1] - trend.iloc[-N]) / N
trend_fc = np.array([trend.iloc[-1] + trend_slope * (i + 1) for i in range(n_test)])

# Project seasonal (repeat last 52-week seasonal cycle)
seas_52 = mstl_result.seasonal.iloc[:, 0]  # first seasonal component (period 52)
seas_fc = np.array([float(seas_52.iloc[(-52 + i % 52)]) for i in range(n_test)])

mstl_preds = trend_fc + seas_fc + resid_fc.values
mstl_preds = np.maximum(mstl_preds, 0)  # clip negatives

mstl_mae   = mae(test_series.values, mstl_preds)
mstl_smape = smape(test_series.values, mstl_preds)
print(f"     MAE={mstl_mae:.1f}   SMAPE={mstl_smape:.2f}%")

# ────────────────────────────────────
# Model C: Holt-Winters ETS (triple exponential smoothing)
# ────────────────────────────────────
print("  [C] Holt-Winters ETS (triple exponential smoothing)...")
hw = ExponentialSmoothing(
    train_series,
    trend="add",
    seasonal="add",
    seasonal_periods=52,
    damped_trend=True
).fit()
hw_preds = hw.forecast(n_test).values
hw_preds = np.maximum(hw_preds, 0)

hw_mae   = mae(test_series.values, hw_preds)
hw_smape = smape(test_series.values, hw_preds)
print(f"     MAE={hw_mae:.1f}   SMAPE={hw_smape:.2f}%")

# ─────────────────────────────────────────────
# 6. Evaluation & Comparison
# ─────────────────────────────────────────────
print("\n[6/7] Generating evaluation charts...")

results = pd.DataFrame({
    "Model": ["Naive Seasonal", "MSTL + ETS", "Holt-Winters ETS"],
    "MAE":   [naive_mae, mstl_mae, hw_mae],
    "SMAPE": [naive_smape, mstl_smape, hw_smape]
}).sort_values("MAE").reset_index(drop=True)

print("\n  ╔══════════════════════╦═══════════╦═══════════╗")
print("  ║ Model                ║    MAE    ║  SMAPE %  ║")
print("  ╠══════════════════════╬═══════════╬═══════════╣")
for _, row in results.iterrows():
    print(f"  ║ {row['Model']:<20} ║ {row['MAE']:>9.1f} ║ {row['SMAPE']:>9.2f} ║")
print("  ╚══════════════════════╩═══════════╩═══════════╝")

best_model = results.iloc[0]["Model"]
print(f"\n  ✓ Best model (lowest MAE): {best_model}")

# ── Plot 3: Forecast vs Actual
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle("Demand Forecasting — Model Comparison", fontsize=14, fontweight="bold")

# Full view
axes[0].plot(train_series.index, train_series.values, color="#94A3B8",
             linewidth=0.9, label="Train", alpha=0.7)
axes[0].plot(test_series.index, test_series.values, color="#0F172A",
             linewidth=2.0, label="Actual (test)")
axes[0].plot(test_series.index, naive_preds, color="#F59E0B",
             linewidth=1.5, linestyle="--", label=f"Naive Seasonal  MAE={naive_mae:.0f}")
axes[0].plot(test_series.index, mstl_preds,  color="#2563EB",
             linewidth=2.0, linestyle="-.", label=f"MSTL+ETS        MAE={mstl_mae:.0f}")
axes[0].plot(test_series.index, hw_preds,    color="#16A34A",
             linewidth=1.8, linestyle=":",  label=f"Holt-Winters    MAE={hw_mae:.0f}")
axes[0].axvline(x=test_series.index[0], color="red", linestyle="--", alpha=0.5, linewidth=1.2)
axes[0].set_title("Full Series + Forecasts")
axes[0].set_ylabel("Weekly Units Sold")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Zoom into test period
axes[1].plot(test_series.index, test_series.values, color="#0F172A",
             linewidth=2.5, label="Actual")
axes[1].plot(test_series.index, naive_preds, color="#F59E0B",
             linewidth=1.8, linestyle="--", label=f"Naive Seasonal  SMAPE={naive_smape:.1f}%")
axes[1].plot(test_series.index, mstl_preds,  color="#2563EB",
             linewidth=2.2, linestyle="-.", label=f"MSTL+ETS        SMAPE={mstl_smape:.1f}%")
axes[1].plot(test_series.index, hw_preds,    color="#16A34A",
             linewidth=2.0, linestyle=":",  label=f"Holt-Winters    SMAPE={hw_smape:.1f}%")
axes[1].fill_between(test_series.index,
                     mstl_preds * 0.92, mstl_preds * 1.08,
                     color="#2563EB", alpha=0.12, label="MSTL ±8% band")
axes[1].set_title("Zoom — Test Period Only (20%)")
axes[1].set_ylabel("Weekly Units Sold"); axes[1].set_xlabel("Date")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/03_forecast_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_forecast_comparison.png")

# ── Plot 4: Error distribution
errors_naive = test_series.values - naive_preds
errors_mstl  = test_series.values - mstl_preds
errors_hw    = test_series.values - hw_preds

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Forecast Error Distributions (Actual − Predicted)", fontsize=13, fontweight="bold")

for ax, err, label, color in zip(
    axes,
    [errors_naive, errors_mstl, errors_hw],
    ["Naive Seasonal", "MSTL + ETS", "Holt-Winters ETS"],
    ["#F59E0B", "#2563EB", "#16A34A"]
):
    ax.hist(err, bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(np.mean(err), color="black", linewidth=1.5, linestyle=":", label=f"mean={np.mean(err):.0f}")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Error (units)"); ax.set_ylabel("Count")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/04_error_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_error_distributions.png")

# ── Plot 5: Model metric bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Evaluation Summary", fontsize=13, fontweight="bold")

colors = ["#2563EB", "#16A34A", "#F59E0B"]
bar_order = results["Model"].tolist()
bar_mae   = results["MAE"].tolist()
bar_smape = results["SMAPE"].tolist()
x = np.arange(len(bar_order))

b1 = axes[0].bar(x, bar_mae, color=colors[:len(bar_order)], edgecolor="white", width=0.5)
axes[0].bar_label(b1, fmt="%.0f", padding=3, fontsize=10)
axes[0].set_xticks(x); axes[0].set_xticklabels(bar_order, fontsize=9)
axes[0].set_title("Mean Absolute Error (MAE) ↓ lower is better")
axes[0].set_ylabel("MAE (units)"); axes[0].grid(axis="y", alpha=0.3)

b2 = axes[1].bar(x, bar_smape, color=colors[:len(bar_order)], edgecolor="white", width=0.5)
axes[1].bar_label(b2, fmt="%.1f%%", padding=3, fontsize=10)
axes[1].set_xticks(x); axes[1].set_xticklabels(bar_order, fontsize=9)
axes[1].set_title("SMAPE (%) ↓ lower is better")
axes[1].set_ylabel("SMAPE (%)"); axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/05_model_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_model_metrics.png")

# ── Plot 6: MSTL decomposition
fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle("MSTL Decomposition — Aggregate Weekly Sales", fontsize=13, fontweight="bold")

axes[0].plot(train_series.index, train_series.values, color="#0F172A", linewidth=0.9)
axes[0].set_title("Original Series"); axes[0].set_ylabel("Sales"); axes[0].grid(alpha=0.3)

axes[1].plot(train_series.index, trend, color="#2563EB", linewidth=1.5)
axes[1].set_title("Trend Component"); axes[1].set_ylabel("Trend"); axes[1].grid(alpha=0.3)

axes[2].plot(train_series.index, seasonal, color="#16A34A", linewidth=0.9)
axes[2].set_title("Seasonal Components (sum)"); axes[2].set_ylabel("Seasonal"); axes[2].grid(alpha=0.3)

axes[3].plot(train_series.index, residual, color="#DC2626", linewidth=0.8, alpha=0.7)
axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[3].set_title("Residuals"); axes[3].set_ylabel("Residual"); axes[3].set_xlabel("Date")
axes[3].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/06_mstl_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 06_mstl_decomposition.png")

# ─────────────────────────────────────────────
# 7. Future Forecast (next 4 weeks / 1 month)
# ─────────────────────────────────────────────
print("\n[7/7] Generating 4-week ahead forecast (upcoming month)...")

full_weekly = weekly.set_index("ds")["y"]

# Refit MSTL on full data
mstl_full   = MSTL(full_weekly, periods=[52, 13]).fit()
trend_full  = mstl_full.trend
seas_full   = mstl_full.seasonal.iloc[:, 0]
resid_full  = mstl_full.resid

# Refit ETS on full residuals
ets_full  = ExponentialSmoothing(resid_full, trend=None, seasonal=None).fit()
resid_fc4 = ets_full.forecast(4).values

trend_slope4 = (trend_full.iloc[-1] - trend_full.iloc[-26]) / 26
trend_fc4    = np.array([trend_full.iloc[-1] + trend_slope4 * (i + 1) for i in range(4)])
seas_fc4     = np.array([float(seas_full.iloc[(-52 + i % 52)]) for i in range(4)])

future_preds = np.maximum(trend_fc4 + seas_fc4 + resid_fc4, 0)
last_date    = full_weekly.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=4, freq="W-SAT")

future_df = pd.DataFrame({
    "Week": future_dates.strftime("%Y-%m-%d"),
    "Forecast_Units": np.round(future_preds).astype(int),
    "Lower_Bound (−8%)": np.round(future_preds * 0.92).astype(int),
    "Upper_Bound (+8%)": np.round(future_preds * 1.08).astype(int)
})

print("\n  4-Week Demand Forecast (MSTL + ETS):")
print(future_df.to_string(index=False))
future_df.to_csv(f"{OUT_DIR}/07_4week_forecast.csv", index=False)
print("\n  Saved: 07_4week_forecast.csv")

# ── Final forecast chart
fig, ax = plt.subplots(figsize=(14, 6))
history_plot = full_weekly.iloc[-52:]
ax.plot(history_plot.index, history_plot.values, color="#94A3B8", linewidth=1.2,
        label="Historical (last 52 wks)", alpha=0.9)
ax.plot(future_dates, future_preds, color="#2563EB", linewidth=2.5,
        marker="o", markersize=8, label="4-week MSTL forecast")
ax.fill_between(future_dates, future_preds * 0.92, future_preds * 1.08,
                color="#2563EB", alpha=0.18, label="±8% confidence band")
for i, (d, v) in enumerate(zip(future_dates, future_preds)):
    ax.annotate(f"{int(v):,}", xy=(d, v), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=10, color="#1E3A8A", fontweight="bold")
ax.axvline(x=future_dates[0] - pd.Timedelta(days=3), color="red",
           linestyle="--", alpha=0.5, linewidth=1.5, label="Forecast start")
ax.set_title("Upcoming Month Demand Forecast (MSTL + ETS)", fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Weekly Units Sold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/08_future_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 08_future_forecast.png")

# ─────────────────────────────────────────────
# Save results table
# ─────────────────────────────────────────────
results.to_csv(f"{OUT_DIR}/09_model_comparison.csv", index=False)
print("  Saved: 09_model_comparison.csv")

print("\n" + "=" * 60)
print("  ✅  COMPLETE — all outputs saved to /mnt/user-data/outputs/")
print("=" * 60)
