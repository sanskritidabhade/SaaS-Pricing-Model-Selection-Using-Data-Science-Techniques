"""
PROJECT: CAC & LTV MODEL

EXECUTIVE SUMMARY:

This project models Customer Acquisition Cost (CAC), Lifetime Value (LTV), and key metrics
for a fictional SaaS business selling software subscriptions. The model includes:

- CAC calculations by acquisition channel
- LTV calculations based on ARPU, gross margin, and churn
- LTV:CAC ratios
- Cohort analysis and retention trends
- Visualizations including cohort heatmaps, bar charts for LTV/CAC, and ARPU by region

Insights:
- Organic Search has the highest LTV:CAC ratio (~87x), driven by low CAC and solid retention.
- Paid channels like Google Ads and Meta Ads yield decent LTV but higher acquisition costs.
- ARPU is relatively consistent across global regions, ranging from ~$163–$176.
- Cohorts show typical churn over the first 12 months, with retention stabilizing around 50%.

Note:
This is a synthetic dataset. CAC values are likely lower than real-world SaaS benchmarks
because marketing spend was randomized in a narrow range. In practice, CAC can vary
significantly and may be 3–10x higher depending on industry, region, and product type.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV
df = pd.read_csv("cac_ltv_model.csv")

# Check basic info
print(df.shape)
print(df.columns)
print(df.head())

# Exploratory Data Analysis (EDA) (lines 13-25)

# Data types
print("\nData types:")
print(df.dtypes)

# Unique values in categorical columns
cat_cols = ['acquisition_channel', 'signup_source', 'region', 'customer_tier', 'contract_length_months']
for col in cat_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())

# Quick stats
print("\nNumerical columns summary:")
print(df.describe())

#  Calculate Average CAC
cac_by_channel = df.groupby("acquisition_channel")["marketing_spend"].mean().reset_index()
print(cac_by_channel)

#  LTV Calculation (lines 32-48)
# Replace churn rates of 0 with small number
df["churn_rate_adj"] = df["churn_rate"].replace(0, 0.0001)

# Calculate LTV
df["ltv"] = (df["arpu"] * df["gross_margin"]) / df["churn_rate_adj"]

# Show some sample rows
print(df[["customer_id", "arpu", "gross_margin", "churn_rate", "ltv"]].head())

# Calculate average LTV overall
avg_ltv = df["ltv"].mean()
print("\nAverage LTV across all customers: ${:,.2f}".format(avg_ltv))

# Calculate average LTV by acquisition channel
ltv_by_channel = df.groupby("acquisition_channel")["ltv"].mean().reset_index()
print("\nAverage LTV by acquisition channel:")
print(ltv_by_channel)

# Calculate LTV:CAC ratio (lines 52-64)

# Merge CAC and LTV tables
ltv_cac = pd.merge(
    ltv_by_channel,
    cac_by_channel,
    on="acquisition_channel"
)

# Calculate LTV:CAC ratio
ltv_cac["ltv_cac_ratio"] = ltv_cac["ltv"] / ltv_cac["marketing_spend"]

# Display results
print("\nLTV:CAC Ratios by Acquisition Channel:")
print(ltv_cac)

# Cohorts (lines 68-103)

# Create cohort_month from year and month
df["cohort_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
print(df[["customer_id", "cohort_month"]].head())

# Calculate Customers per Cohort
cohort_sizes = df.groupby("cohort_month")["customer_id"].nunique().reset_index()
cohort_sizes.rename(columns={"customer_id": "num_customers"}, inplace=True)

print("\nCohort sizes:")
print(cohort_sizes)

# Simulate Retention Over Time

# Simulate retention curve for each cohort
retention_records = []

for cohort, group in df.groupby("cohort_month"):
    initial_customers = group.shape[0]
    
    avg_churn = group["churn_rate"].mean()
    
    # Simulate retention up to 12 months
    for month in range(13):
        retention_rate = (1 - avg_churn) ** month
        retained_customers = int(initial_customers * retention_rate)
        
        retention_records.append({
            "cohort_month": cohort,
            "months_since_acquisition": month,
            "customers_remaining": retained_customers
        })

retention_df = pd.DataFrame(retention_records)

print("\nRetention DataFrame sample:")
print(retention_df.head(20))

# Plot 1 (Pivot Retention and plot a Heatmap)

# Pivot retention_df
cohort_matrix = retention_df.pivot(
    index="cohort_month",
    columns="months_since_acquisition",
    values="customers_remaining"
)

print("\nCohort Matrix:")
print(cohort_matrix.head())

# Plot heatmap
plt.figure(figsize=(12,6))
sns.heatmap(cohort_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Customer Retention by Cohort Month")
plt.xlabel("Months Since Acquisition")
plt.ylabel("Cohort Month")
plt.tight_layout()
plt.savefig("plot1_cohort_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2

# Data
channels = ltv_cac["acquisition_channel"]
ltv_values = ltv_cac["ltv"]
cac_values = ltv_cac["marketing_spend"]

# Positioning for bars
x = np.arange(len(channels))
bar_width = 0.35

# Plot
plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, ltv_values, width=bar_width, color='teal', label="LTV")
plt.bar(x + bar_width/2, cac_values, width=bar_width, color='orange', label="CAC")

# Add data labels above bars
for i, val in enumerate(ltv_values):
    plt.text(x[i] - bar_width/2, val + 50, f"${val:,.0f}", ha='center', color='black')

for i, val in enumerate(cac_values):
    plt.text(x[i] + bar_width/2, val + 20, f"${val:,.0f}", ha='center', color='black')

# Add labels & title
plt.xticks(x, channels)
plt.ylabel("Amount ($)")
plt.title("LTV vs CAC by Acquisition Channel")
plt.legend()
plt.tight_layout()
plt.savefig("plot2_ltv_vs_cac.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: LTV:CAC Ratio by Acquisition Channel
plt.figure(figsize=(8,5))
sns.barplot(
    x="acquisition_channel",
    y="ltv_cac_ratio",
    data=ltv_cac,
    hue="acquisition_channel",
    dodge=False,
    palette="viridis",
    legend=False
)

# Add data labels on top of bars
for i, val in enumerate(ltv_cac["ltv_cac_ratio"]):
    plt.text(
        i,
        val + 1,
        f"{val:.1f}x",
        ha='center',
        color='black'
    )

plt.title("LTV:CAC Ratio by Acquisition Channel")
plt.ylabel("LTV:CAC Ratio")
plt.xlabel("Acquisition Channel")
plt.tight_layout()
plt.savefig("plot3_ltv_cac_ratio.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Average ARPU by Region
arpu_region = df.groupby("region")["arpu"].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(
    x="region",
    y="arpu",
    data=arpu_region,
    hue="region",
    dodge=False,
    palette="coolwarm",
    legend=False
)

# Add data labels above bars
for i, val in enumerate(arpu_region["arpu"]):
    plt.text(
        i,
        val + 2,
        f"${val:,.0f}",
        ha='center',
        color='black'
    )

plt.title("Average ARPU by Region")
plt.ylabel("Average ARPU ($)")
plt.xlabel("Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot4_arpu_by_region.png", dpi=300, bbox_inches='tight')
plt.close()