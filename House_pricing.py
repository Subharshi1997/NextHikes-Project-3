# ======================================================
# EXPLORATORY DATA ANALYSIS – REAL ESTATE PRICING
# Fully Aligned with Project Guideline (Page 4 & 5)
# Interactive | Clean | Submission Ready
# ======================================================

# =========================
# IMPORT NECESSARY PACKAGES
# =========================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_style("whitegrid")

# =========================
# 1. LOADING THE DATA
# =========================
df = pd.read_excel("housing_data.xlsx")
df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

# =========================
# 2. CLEANING THE DATA
# =========================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df.drop_duplicates(inplace=True)

# =========================
# 5. FEATURE ENGINEERING
# =========================
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["PricePerSqFt"] = df["SalePrice"] / df["GrLivArea"]

# =========================
# FIGURE SETUP (INTERACTIVE)
# =========================
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.22, top=0.82)
current_plot = 0

# ======================================================
# 3. UNIVARIATE ANALYSIS
# ======================================================
def plot_price_distribution():
    ax.clear()
    sns.histplot(df["SalePrice"], kde=True, ax=ax)
    ax.set_title(
        "Univariate Analysis: Sale Price Distribution\n\n"
        "Insight:\n"
        "• Majority of houses fall in the mid-price range\n"
        "• Right-skewed distribution highlights luxury properties",
        fontsize=12
    )
    ax.set_xlabel("Sale Price")
    ax.set_ylabel("Frequency")

def plot_outlier_detection():
    ax.clear()
    sns.boxplot(x=df["SalePrice"], ax=ax)
    ax.set_title(
        "Univariate Analysis: Outlier Detection\n\n"
        "Insight:\n"
        "• Most homes lie within a stable price range\n"
        "• Extreme values represent premium market properties",
        fontsize=12
    )
    ax.set_xlabel("Sale Price")

# ======================================================
# 4. MULTIVARIATE ANALYSIS
# ======================================================
def plot_size_vs_price():
    ax.clear()
    sns.scatterplot(
        x="GrLivArea",
        y="SalePrice",
        data=df,
        alpha=0.6,
        label="Houses",
        ax=ax
    )
    sns.regplot(
        x="GrLivArea",
        y="SalePrice",
        data=df,
        scatter=False,
        color="red",
        label="Trend Line",
        ax=ax
    )
    ax.legend()
    ax.set_title(
        "Multivariate Analysis: Living Area vs Sale Price\n\n"
        "Insight:\n"
        "• Strong positive relationship\n"
        "• Larger homes consistently command higher prices",
        fontsize=12
    )
    ax.set_xlabel("Living Area (Sq Ft)")
    ax.set_ylabel("Sale Price")

def plot_correlation_matrix():
    ax.clear()
    corr = df[["SalePrice", "GrLivArea", "OverallQual", "GarageCars", "HouseAge"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(
        "Multivariate Analysis: Correlation Matrix\n\n"
        "Insight:\n"
        "• SalePrice strongly correlates with size and quality\n"
        "• House age has a weaker negative influence",
        fontsize=12
    )

# ======================================================
# 6. FEATURE ENGINEERING & SIZE IMPACT
# ======================================================
def plot_price_per_sqft():
    ax.clear()
    sns.histplot(df["PricePerSqFt"], kde=True, ax=ax)
    ax.set_title(
        "Feature Engineering: Price per Square Foot\n\n"
        "Insight:\n"
        "• Normalizes house prices across sizes\n"
        "• Useful for fair market comparison",
        fontsize=12
    )
    ax.set_xlabel("Price per Sq Ft")

def plot_quality_impact():
    ax.clear()
    sns.boxplot(x="OverallQual", y="SalePrice", data=df, ax=ax)
    ax.set_title(
        "Feature Impact: Construction Quality vs Price\n\n"
        "Insight:\n"
        "• Higher quality leads to significantly higher valuation\n"
        "• Quality is a key pricing driver",
        fontsize=12
    )
    ax.set_xlabel("Overall Quality")
    ax.set_ylabel("Sale Price")

# ======================================================
# 7. MARKET TRENDS & HISTORICAL PRICING
# ======================================================
def plot_market_trend():
    ax.clear()
    trend = df.groupby("YrSold")["SalePrice"].median()
    ax.plot(trend.index, trend.values, marker="o")
    ax.set_title(
        "Market Trends: Median Sale Price Over Time\n\n"
        "Insight:\n"
        "• Price growth until 2007\n"
        "• Decline post-2008 reflects economic slowdown",
        fontsize=12
    )
    ax.set_xlabel("Year Sold")
    ax.set_ylabel("Median Sale Price")

# ======================================================
# 8. CUSTOMER PREFERENCES & AMENITIES
# ======================================================
def plot_amenities_impact():
    ax.clear()
    sns.boxplot(x="GarageCars", y="SalePrice", data=df, ax=ax)
    ax.set_title(
        "Customer Preferences: Garage Capacity Impact\n\n"
        "Insight:\n"
        "• Houses with more garage space command higher prices\n"
        "• Parking is a valued amenity",
        fontsize=12
    )
    ax.set_xlabel("Garage Capacity")
    ax.set_ylabel("Sale Price")

def plot_market_segmentation():
    ax.clear()

    features = df[["SalePrice", "GrLivArea", "OverallQual", "GarageCars"]]
    scaled = StandardScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["MarketSegment"] = kmeans.fit_predict(scaled)

    sns.scatterplot(
        x="GrLivArea",
        y="SalePrice",
        hue="MarketSegment",
        palette="Set1",
        data=df,
        ax=ax
    )

    ax.legend(title="Market Segment")
    ax.set_title(
        "Market Segmentation using K-Means Clustering\n\n"
        "Insight:\n"
        "• Segment 0: Affordable homes\n"
        "• Segment 1: Mid-range family homes\n"
        "• Segment 2: Premium luxury properties",
        fontsize=12
    )
    ax.set_xlabel("Living Area")
    ax.set_ylabel("Sale Price")

# ======================================================
# VISUALIZATION ORDER (GUIDELINE FLOW)
# ======================================================
plots = [
    plot_price_distribution,
    plot_outlier_detection,
    plot_size_vs_price,
    plot_correlation_matrix,
    plot_price_per_sqft,
    plot_quality_impact,
    plot_market_trend,
    plot_amenities_impact,
    plot_market_segmentation
]

# ======================================================
# BUTTON FUNCTION
# ======================================================
def next_plot(event):
    global current_plot
    current_plot = (current_plot + 1) % len(plots)
    plots[current_plot]()
    plt.draw()

# ======================================================
# SMALLER BUTTON SETUP (UI IMPROVED)
# ======================================================
ax_button = plt.axes([0.44, 0.06, 0.12, 0.045])
btn = Button(ax_button, "Next ▶")
btn.on_clicked(next_plot)

# ======================================================
# INITIAL PLOT
# ======================================================
plots[current_plot]()
plt.show()
