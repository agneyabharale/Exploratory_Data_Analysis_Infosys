# Exploratory Data Analysis & Financial Performance Prediction of Infosys Ltd.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-388E3C.svg)](https://seaborn.pydata.org/)
[![License](https://img.shields.io/badge/Project-Academic%20%2F%20Enterprise-success.svg)](#)

An enterprise-grade financial analytics, exploratory data analysis (EDA), and machine learning pipeline evaluating the operational trajectory, capital efficiency, and profitability of **Infosys Ltd.** (FY 2022–2025) benchmarked against an industry cross-section of over **4,600 publicly listed companies**.

---

## Table of Contents
1. [Executive Overview & Core Findings](#executive-overview--core-findings)
2. [Project Architecture & Directory Structure](#project-architecture--directory-structure)
3. [The Financial Analytics Pipeline](#the-financial-analytics-pipeline)
   - [Phase 1: Ingestion, Cleaning & Outlier Treatment](#phase-1-ingestion-cleaning--outlier-treatment)
   - [Phase 2: Univariate, Bivariate & Multivariate EDA](#phase-2-univariate-bivariate--multivariate-eda)
   - [Phase 3: Statistical Hypothesis Testing](#phase-3-statistical-hypothesis-testing)
   - [Phase 4: Machine Learning Suite & Infosys Benchmarking](#phase-4-machine-learning-suite--infosys-benchmarking)
   - [Phase 5: DuPont Decomposition & Financial Doctor Diagnostics](#phase-5-dupont-decomposition--financial-doctor-diagnostics)
4. [The Financial Storyteller: What Each Visual Signifies](#the-financial-storyteller-what-each-visual-signifies)
5. [Setup & Step-by-Step Execution](#setup--step-by-step-execution)
6. [Tech Stack & Dependencies](#tech-stack--dependencies)
7. [Patentability & Innovation Roadmap](#patentability--innovation-roadmap)

---

## Executive Overview & Core Findings

This project bridges the gap between raw corporate financial statements and strategic business intelligence. By decomposing multi-year balance sheets, income statements, and cash flows alongside industry peers, the system answers three vital corporate finance questions:

```text
 1. How scalable is Infosys's business model?
    └── Revenue surged from ₹1,21,641 Cr to ₹1,62,900 Cr (10.2% CAGR), while Net Profit expanded 
        from ₹22,110 Cr to ₹30,371 Cr (11.2% CAGR), demonstrating positive operating leverage.

 2. What is the quality of reported earnings?
    └── Cumulative Operating Cash Flow to Net Profit (OCF/PAT) stands at 1.02x, proving accounting 
        profits translate 100% into liquid cash without working capital drag or artificial inflation.

 3. Why did Return on Equity (ROE) nearly double from 40.0% to 79.2%?
    └── 3-Stage DuPont Decomposition reveals margins (~18%) and asset turnover (1.65x) were steady. 
        The primary engine was the Equity Multiplier (+61%), driven by aggressive cash return via 
        share buybacks and dividends, dramatically maximizing returns on retained shareholder capital.
```

---

## Project Architecture & Directory Structure

The repository follows a clean modular pattern separating reusable statistical modules (`src/`) from sequential execution workflows (`scripts/`), data partitions (`data/`), and deliverables (`reports/`, `visualizations/`):

```text
DEVL_Mini_Project/
├── data/
│   ├── raw/
│   │   └── Project.xlsx                         # Master Excel workbook with financial statements
│   └── processed/
│       ├── income_statement.csv                 # Transposed 4-year P&L (Revenue, EBIT, Net Profit)
│       ├── balance_sheet.csv                    # Assets, liabilities, and shareholder equity
│       ├── cash_flow.csv                        # Operating, investing, and financing cash flows
│       ├── financial_ratios.csv                 # Standard liquidity, leverage, and margin ratios
│       └── industry_data.csv                    # 4,668-record corporate cross-sectional benchmark
├── src/                                         # Reusable Core Python Modules
│   ├── data_cleaning.py                         # Schema normalization, IQR/Z-score outlier filters
│   ├── eda.py                                   # Univariate, bivariate, and multivariate visualizers
│   ├── stats_analysis.py                        # Pearson correlations, p-value statistical tests
│   ├── regression.py                            # Linear Regression, Decision Tree & KNN models
│   ├── dupont_analysis.py                       # 3-Stage DuPont decomposition engine & charts
│   └── financial_doctor.py                      # Automated corporate finance diagnostic scorecard
├── scripts/                                     # Sequential Pipeline Workflows
│   ├── 01_data_acquisition_cleaning.py          # Data ingestion, schema audit, and CSV export
│   ├── 02_eda.py                                # Univariate, bivariate, and multivariate analysis
│   ├── 03_statistical_analysis.py               # Hypothesis testing & industry correlation heatmap
│   ├── 04_regression.py                         # ML model training, evaluation & Infosys benchmarking
│   └── 05_final_insights.py                     # DuPont run, diagnostic generation, and dashboard build
├── visualizations/                              # High-Resolution Exported Artifacts (300 DPI)
│   ├── revenue/                                 # revenue_trend.png
│   ├── profitability/                           # pat_trend.png
│   ├── cash_flow/                               # ocf_dist.png
│   ├── bivariate/                               # revenue_vs_net_profit.png, ocf_vs_net_profit.png,
│   │                                            # opm_vs_roe_industry.png, roe_by_top_industries.png
│   ├── multivariate/                            # financial_metrics_pairplot.png, bubble chart
│   ├── ratios/                                  # dupont_decomposition_trend.png
│   ├── industry/                                # correlation_heatmap.png
│   └── regression/                              # model_comparison_actual_vs_predicted.png,
│                                                # decision_tree_feature_importance.png,
│                                                # residuals_comparison.png, regression_line.png
├── reports/
│   ├── dashboard/
│   │   └── final_dashboard.png                  # Executive 4-panel visual master dashboard
│   └── executive_financial_onboarding_guide.md  # Plain-English onboarding manual for new analysts
├── requirements.txt                             # Python dependencies
└── README.md                                    # Comprehensive system documentation
```

---

## The Financial Analytics Pipeline

```text
 ┌───────────────────────┐      ┌───────────────────────┐      ┌───────────────────────┐
 │ 01. Ingest & Clean    │ ───► │ 02. Multi-Level EDA   │ ───► │ 03. Hypothesis Test   │
 │ • Transpose P&L/BS/CF │      │ • Univariate Trends   │      │ • Correlation Matrix  │
 │ • IQR Outlier Checks  │      │ • Bivariate Pairs     │      │ • Pearson p-value Test│
 └───────────────────────┘      │ • Multivariate Bubble │      └──────────┬────────────┘
                                └───────────────────────┘                 │
 ┌───────────────────────┐      ┌───────────────────────┐                 │
 │ 05. Executive Intel   │ ◄─── │ 04. Machine Learning  │ ◄───────────────┘
 │ • DuPont Decomposition│      │ • Linear Regression   │
 │ • Financial Doctor    │      │ • Decision Tree       │
 │ • Master Dashboard    │      │ • K-Nearest Neighbors │
 └───────────────────────┘      └───────────────────────┘
```

### Phase 1: Ingestion, Cleaning & Outlier Treatment
* **Module**: [`src/data_cleaning.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/src/data_cleaning.py) | **Script**: [`scripts/01_data_acquisition_cleaning.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/scripts/01_data_acquisition_cleaning.py)
* Transposes horizontal multi-year reporting periods into standardized, time-indexed tabular frames.
* Identifies missing values across datasets, strips string formatting, and normalizes column headers.
* Applies Interquartile Range (IQR) and Z-score outlier detection on financial variables to prevent distorted statistical estimates.

### Phase 2: Univariate, Bivariate & Multivariate EDA
* **Module**: [`src/eda.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/src/eda.py) | **Script**: [`scripts/02_eda.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/scripts/02_eda.py)
* **Univariate**: Line charts tracing 4-year Revenue and PAT trajectories; KDE histograms mapping Operating Cash Flow distribution.
* **Bivariate**: 
  - Revenue vs. Net Profit (conversion efficiency).
  - Operating Cash Flow vs. Net Profit (accounting earnings quality).
  - 5-Year Operating Profit Margin (OPM) vs. 5-Year ROE across industry peers.
  - Sector-wise ROE boxplots comparing top 8 Indian corporate industries.
* **Multivariate**: 
  - 4-way pairplot matrix assessing pairwise distributions across `opm_5year`, `return_on_assets_5years`, `average_return_on_capital_employed_5years`, and `average_return_on_equity_5years`.
  - Multivariate Bubble Chart sizing companies by Market Capitalization to analyze scale effects.

### Phase 3: Statistical Hypothesis Testing
* **Module**: [`src/stats_analysis.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/src/stats_analysis.py) | **Script**: [`scripts/03_statistical_analysis.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/scripts/03_statistical_analysis.py)
* Evaluates Pearson correlation coefficient between Operating Profit Margin and Return on Equity:
  * **Null Hypothesis ($H_0$)**: There is no statistically significant linear relationship between OPM and ROE.
  * **Alternative Hypothesis ($H_1$)**: There is a statistically significant linear relationship between OPM and ROE.
* **Result**: Pearson $r = 0.0761$, $p < 0.0001$. We reject $H_0$. While statistically significant due to the large sample size ($N = 4,668$), the low correlation coefficient proves that **operating margin alone does not dictate shareholder return**.

### Phase 4: Machine Learning Suite & Infosys Benchmarking
* **Module**: [`src/regression.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/src/regression.py) | **Script**: [`scripts/04_regression.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/scripts/04_regression.py)
* Preprocesses 3,300+ clean industry records using 1st/99th percentile quantile clipping and `StandardScaler` feature normalization.
* **Target ($Y$)**: `average_return_on_equity_5years`
* **Features ($X$)**: `opm_5year`, `return_on_assets_5years`, `average_return_on_capital_employed_5years`, `historical_pe_5years`
* Evaluates three competitive machine learning paradigms:
  1. **Multiple Linear Regression**: Evaluates baseline linear relationships and feature $p$-values.
  2. **Decision Tree Regressor** (`max_depth=5`, `min_samples_leaf=10`): Captures non-linear ratio interactions and ranks feature importances.
  3. **K-Nearest Neighbors Regressor** ($K=7$, distance-weighted): Predicts company returns based on similarity clustering in multi-ratio space.
* **Infosys Benchmarking**: Passes Infosys's actual operating metrics through all three models to compare **Predicted Expected ROE (~22%) vs. Actual Reported ROE (~60%)**, quantifying Infosys's distinct economic moat.

### Phase 5: DuPont Decomposition & Financial Doctor Diagnostics
* **Modules**: [`src/dupont_analysis.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/src/dupont_analysis.py), [`src/financial_doctor.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/src/financial_doctor.py) | **Script**: [`scripts/05_final_insights.py`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/scripts/05_final_insights.py)
* Breaks down ROE into **Net Profit Margin $\times$ Asset Turnover $\times$ Equity Multiplier**.
* Computes health scorecards (Green Flags vs. Amber Alerts) across liquidity, solvency, earnings quality, and peer percentiles.
* Compiles the 4-panel executive master dashboard ([`final_dashboard.png`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/reports/dashboard/final_dashboard.png)) and generates the complete onboarding manual ([`executive_financial_onboarding_guide.md`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/reports/executive_financial_onboarding_guide.md)).

---

## The Financial Storyteller: What Each Visual Signifies

*(Use this section to onboard new employees joining the team)*

| Visualization | Primary Metric | What It Shows | What It Signifies in Business Terms |
| :--- | :--- | :--- | :--- |
| **`revenue_trend.png`** | Total Revenue | Consistent 4-year upward slope (₹1.21L Cr → ₹1.63L Cr) | **Scalable Demand**: Enterprise digital transformation spending remained robust across North America and Europe. |
| **`pat_trend.png`** | Profit After Tax | Net profit expanding from ₹22.1k Cr → ₹30.4k Cr | **Cost Discipline**: Operating leverage worked; profits grew at 11.2% CAGR, outpacing revenue growth (10.2%). |
| **`ocf_vs_net_profit.png`** | OCF vs. PAT | Operating cash flow closely mirrors net income each year | **High Earnings Quality**: Customers pay on time; reported accounting profit translates 100% into bank cash. |
| **`roe_by_top_industries.png`** | Industry ROE | Tech & Healthcare have high, compact distributions | **Asset-Light Economics**: Software requires talent and IP, not steel mills, yielding superior return on equity. |
| **`dupont_decomposition_trend.png`** | DuPont 3 Drivers | Margin flat (~18%), Turnover steady (~1.65x), Leverage up (+61%) | **Capital Strategy**: ROE surged to 79% because excess cash was returned via buybacks/dividends, slimming equity. |
| **`model_comparison_actual_vs_predicted.png`** | Actual vs. Predicted | Linear fit is flat; Decision Tree and KNN adapt to clusters | **Non-Linear Financial Reality**: Financial markets cluster by peer group; Linear regression oversimplifies reality. |

---

## Setup & Step-by-Step Execution

### 1. Prerequisites & Environment Setup
Clone the repository and install required packages inside a virtual environment:

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
```

### 2. Running the Complete Pipeline
Execute the scripts sequentially from within the `scripts/` directory:

```powershell
cd scripts

# Step 1: Clean raw Project.xlsx and export normalized CSV datasets
python 01_data_acquisition_cleaning.py

# Step 2: Run Univariate, Bivariate, and Multivariate Exploratory Data Analysis
python 02_eda.py

# Step 3: Run Statistical Correlation Analysis & Hypothesis Testing
python 03_statistical_analysis.py

# Step 4: Train Linear Regression, Decision Tree & KNN Models and Benchmark Infosys
python 04_regression.py

# Step 5: Execute DuPont Decomposition, Financial Diagnostics, Dashboard & Onboarding Guide
python 05_final_insights.py
```

All generated charts will be populated in [`visualizations/`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/visualizations/) and final deliverables in [`reports/`](file:///c:/Users/Agneya%20Bharale/OneDrive/Learning/Pandas/DEVL_Mini_Project/reports/).

---

## Tech Stack & Dependencies

| Category | Library | Purpose |
| :--- | :--- | :--- |
| **Data Ingestion & Wrangling** | `pandas`, `openpyxl` | Excel multi-sheet parsing, statement transpositions, missing value handling |
| **Numerical Processing** | `numpy` | Mathematical matrix operations, quantile clipping, RMSE computations |
| **Statistical Analysis** | `scipy`, `statsmodels` | Pearson correlation, hypothesis testing ($p$-values), OLS model summaries |
| **Data Visualization** | `matplotlib`, `seaborn` | High-res line charts, KDE histograms, pairplots, boxplots, and bubble charts |
| **Machine Learning** | `scikit-learn` | Multi-linear regression, Decision Trees, KNN, feature scaling, evaluation metrics |
| **Dashboard Compilation** | `PIL` (Pillow) | Assembling individual visualization outputs into multi-panel executive dashboards |

---

## Patentability & Innovation Roadmap

To advance this pipeline from an explanatory data science project into **patentable Intellectual Property (IP)**, consider these strategic technical extensions:

1. **Multi-Modal Earnings Discrepancy Engine**:
   - Fuse structured 10-K financial ratios with unstructured earnings call audio/transcripts via cross-attention transformers.
   - Automatically compute a novel **"Divergence Score"** flagging when management tone conflicts with underlying cash flow quality.
2. **Dynamic Knowledge Graph Shock Propagation**:
   - Construct a heterogeneous knowledge graph of client-vendor dependencies across the IT services industry.
   - Use Graph Neural Networks (GNNs) to model how foreign exchange shocks or client budget freezes propagate through peer networks.
3. **Causal AI Counterfactual Simulator**:
   - Implement Structural Causal Models (SCMs) using Directed Acyclic Graphs (DAGs) and do-calculus.
   - Formulate a decision-support engine answering: *"What minimum capital reallocation from SG&A to R&D causally guarantees ROE $> 20\%$ during economic downturns?"*
