# Customer Behavior Prediction Platform

Vantara Retail Solutions case study: predict churn, customer lifetime value (CLV) and segments from
the Online Retail II transactions, explain every prediction, and serve the results through an API
and a dashboard. Built by a single contributor in a 4-week sprint, following the project PRD.

Status: **Week 1 (data foundation and feature engineering) complete.** Weeks 2-4 pending.

## Setup (Windows, PowerShell)

Python 3.12 is recommended (the PRD needs 3.10+; several ML libraries lag behind the newest Python).

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Without `uv`: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.
On macOS/Linux activate with `source .venv/bin/activate`.

## Get the data

Download the zip from https://archive.ics.uci.edu/dataset/502/online+retail+ii (about 43 MB, no
login), unzip it and place `online_retail_II.xlsx` in `data/raw/` (outside the zip). The PRD's
"Option B" (`ucimlrepo`) does **not** work for this dataset: the UCI repository lists it
(id 502) but marks it as not available for import, so the loader falls back to the manual file.
Raw and interim data are not committed (see `.gitignore`); this step re-creates them.

Dataset: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository
(CC BY 4.0). Do not substitute the older single-year "Online Retail" dataset.

## Run the pipeline (single command)

```powershell
python -m src.pipeline
```

This loads and validates the raw data, cleans it, builds the customer feature table, and writes:

| File | Content |
|---|---|
| `data/interim/transactions_clean.csv.gz` | Cleaned transaction lines with flags (not committed) |
| `data/processed/customer_features.csv` | One row per customer: 14 base features, 9 category shares (the "other" share is left out as the reference category because shares sum to 1), 3 targets (`churn_90d`, `clv_target`, `next_category`) |
| `data/processed/pipeline_report.json` | Row counts for every cleaning rule, churn rate, cutoff date |

Loading the Excel file takes a few minutes (about 1 minute on the development laptop for the
whole pipeline after the file is read). Tests and code style:

```powershell
python -m pytest --cov=src
ruff check src tests
black src tests
```

Coverage on the source package (PRD target: at least 70%):

```powershell
python -m pytest --cov=src --cov-report=term-missing
```

Notebooks (run the pipeline first): `notebooks/01_eda.ipynb` (EDA, correlations, VIF, hypotheses) and
`notebooks/02_feature_engineering.ipynb` (feature dictionary, leakage check, class balance).

## Results of the Week 1 pipeline run (real data)

| Step | Rows | Note |
|---|---|---|
| Raw transactions (both sheets) | 1,067,371 | Matches the PRD (about 1.07 million) |
| Exact duplicates removed | 34,335 | |
| Price <= 0 removed | 6,019 | Adjustments and manual corrections |
| Data-entry errors removed | 70 | |
| **Cleaned transactions** | **1,026,947** | |
| Lines without Customer ID (flagged) | 229,162 | About 22% of lines; excluded from customer features |
| Return lines (kept) | 19,058 | Used for `return_rate` and net CLV |
| Non-product lines (flagged) | 5,702 | POST, DOT, M, bank charges, gift vouchers, etc. |
| Bulk quantity flagged | 53,054 | Above the IQR fence of 28.5 units (kept) |
| High price flagged | 66,130 | Above the IQR fence of 8.5 (kept) |
| **Customers in the feature table** | **5,256** | Customers with at least one purchase before 2011-09-10 |
| Churn rate (`churn_90d = 1`) | 56.5% | See the note under known limitations |

The full report is in `data/processed/pipeline_report.json`.

## EDA highlights (real data, from `notebooks/01_eda.ipynb`)

- Revenue peaks every November (about 1.1 million per month in both 2010 and 2011); December 2011 is
  incomplete because the data ends on 9 December.
- The UK has 5,334 customers and 14.0 million revenue; Ireland (EIRE) has only 3 customers but 0.59 million
  revenue, so a few wholesale accounts carry very large spend.
- Recency is bimodal: a large group bought in the last 30 days before the cutoff, and a second group last
  bought about 300 days before it (holiday 2010 shoppers who did not return). Median recency is 163 days.
- Frequency is very skewed (median 3 orders, maximum 284). Monetary total is roughly log-normal and has
  extreme outliers (maximum about 457,000), so log transforms will help linear models.
- Median feature values of retained versus churned customers: recency 66 versus 285 days, frequency 5 versus 2
  orders, total spend 1,583 versus 476.
- Strongest correlations with `churn_90d`: `engagement_score` -0.49, `recency_days` +0.43,
  `seasonal_concentration` +0.40, `frequency` -0.27. `return_rate` and `frequency_trend` are almost
  uncorrelated (about 0.02). `seasonal_concentration` is high mostly for one-order customers, whose spend sits in a
  single month, so it partly duplicates `frequency`.
- `gap_variance` is 0 for customers with fewer than three orders (no gap history), so it is confounded with
  `frequency`.

## Design decisions and assumptions

The PRD leaves several definitions open. These are the choices made, all configurable in
`config/config.yaml`:

| Topic | Decision |
|---|---|
| Churn label | `churn_90d = 1` if a customer makes no purchase in the 90 days after the cutoff date |
| Cutoff date | 2011-09-10. The window 2011-09-10 to 2011-12-09 is fully observed (the data ends on 2011-12-09). Features use only data strictly before the cutoff |
| Customers included | Customers with at least one purchase before the cutoff |
| CLV | `clv_historical` = sales minus returns before the cutoff (a feature); `clv_target` = sales revenue in the 90 days after the cutoff (the regression target) |
| Returns | Kept (not dropped). Used for `return_rate` and net CLV. Invoices starting with "C" are cancellations |
| Missing Customer ID | Rows are flagged and excluded from customer-level features (about 22% of all lines) |
| Exact duplicate rows | Dropped. A genuine repeated line with identical values cannot be told apart from a duplicate |
| Price <= 0 | Dropped (adjustments and manual corrections) |
| Data-entry errors | |quantity| above 10,000 or price above 5,000 dropped. Bulk quantities and high prices below that are only flagged (IQR fences) |
| Non-product codes | POST, DOT, M, BANK CHARGES, AMAZONFEE, gift vouchers and similar are flagged and excluded from customer features |
| Product category | The data has no category field, so categories come from description keywords (`config.yaml`) |
| Discount sensitivity | The data has no discount field, so the proxy is the share of units bought below 90% of the product's median price |
| Seasonal concentration | Herfindahl index of a customer's spend over calendar months |
| Scaling and encoding | `FeaturePreprocessor` fits a one-hot Country encoder (top 5 countries plus "Other") and a StandardScaler on training data only and can be saved to `models_artifacts/`. Tree models use the unscaled matrix (`scale=False`) |
| High-cardinality encoding | `FrequencyEncoder` turns StockCode into its relative frequency; the feature `product_popularity` is the mean frequency of the products a customer bought |
| Next purchase category | `next_category` is the category with the highest spend in the label window (missing for customers who did not buy). It is a target, not a model input |
| Random seed | Fixed seed 42 in `config.yaml`. Week 1 contains no random steps, so it is first used for the train/test split in Week 2 |

## Differences from the PRD

The PRD's Option B for loading the data (`ucimlrepo`) does not work for Online Retail II, so the manual
download is required. See also the design decisions above for definitions the PRD leaves open.

## Tests

`tests/test_features.py` contains the leakage tests required by the PRD: deleting all post-cutoff
rows must not change any feature, adding future purchases must not change any feature, and labels
use only the label window. It also checks hand-calculated feature values. `tests/test_data.py` covers
validation, each cleaning rule and an end-to-end pipeline run on synthetic data.

## Differences from the PRD folder structure

Added `tests/test_data.py`, `tests/helpers.py`, `tests/__init__.py` and `pyproject.toml` (ruff, black and
pytest settings). `.gitkeep` files keep empty PRD folders in Git.

## Known limitations (Week 1)

- Churn is defined relative to one cutoff date. The observed churn rate is 56.5%, higher than first
  expected for a window that covers the holiday season. A large share of "churned" customers probably
  stopped buying long before the cutoff, which can make the task easier than it looks (recency alone
  may separate them). The EDA notebook shows churn by recency bucket; if needed, the label can be
  restricted to customers active in the last 180 days before the cutoff.
- Keyword-based categories and the price-based discount proxy are approximations.
- Duplicate removal may drop a few genuine repeated lines.
 
 ## Week 2: Models and segmentation

Split: stratified 70/15/15 (seed 42). Hyperparameters tuned with 5-fold CV on the training split. Models compared on the validation split; the test split was used once, for the two selected models.

### Churn models (validation, all customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.736 | 0.712 | 0.897 | 0.794 | 0.814 |
| Random Forest | 0.749 | 0.787 | 0.762 | 0.774 | 0.813 |
| LightGBM | 0.752 | 0.753 | 0.834 | 0.791 | 0.807 |
| Logistic Regression | 0.725 | 0.781 | 0.713 | 0.746 | 0.796 |
| Decision Tree | 0.736 | 0.778 | 0.747 | 0.762 | 0.793 |
| KNN | 0.730 | 0.742 | 0.800 | 0.770 | 0.782 |

Selection rule (fixed before looking at the test split): highest validation ROC-AUC; models within 0.005 AUC are tied, and the tie is broken by recall on recently active customers (recency <= 180 days). XGBoost and Random Forest were tied; XGBoost had the better recent-customer recall (0.716 vs 0.445).

### Final test results (789 customers, used once)

| Model | Subset | Precision | Recall | ROC-AUC |
| --- | --- | --- | --- | --- |
| XGBoost (churn) | all customers | 0.721 | 0.906 | 0.804 |
| XGBoost (churn) | recency <= 180 | 0.617 | 0.748 | 0.802 |

CLV (two-stage model, original scale, all customers): R-squared 0.134, MAE 690, RMSE 5,766.

| PRD target | Result |
| --- | --- |
| Churn ROC-AUC >= 0.80 | Met (0.804, a thin margin) |
| Churn recall >= 0.70 | Met (0.906) |
| CLV R-squared >= 0.60 | Not met (0.134) |

### CLV

About 56% of customers have a CLV target of 0. A single regressor reached R-squared 0.33 on validation (original scale). A two-stage model (P(active) from the churn model x expected spend of active customers) reached 0.42 on validation but only 0.134 on the test split, where a few very large customers dominate the error (RMSE 5,766 vs MAE 690). The target of 0.60 was not reached.

### Segments

One-time buyers are split by recency; repeat customers are clustered with K-Means (k=3, silhouette 0.34). A Gaussian Mixture Model was the second algorithm and was not preferred (BIC kept falling; silhouette 0.03).

| Segment | Customers | Churn rate | Mean CLV |
| --- | --- | --- | --- |
| Lapsed low-value | 2,154 | 61% | 216 |
| Loyal high-value | 1,485 | 25% | 1,457 |
| One-time buyer (lapsed) | 1,179 | 85% | 73 |
| One-time buyer (recent) | 396 | 64% | 201 |
| High-return buyer | 42 | 64% | 298 |

### Limitations

- One cutoff date (2011-09-10) and one 90-day window; no test across time periods.
- Recall is lower for recently active customers in most models.
- Recall is easy to reach when 56% of customers churn; precision should be read alongside it.
- No clear seasonal segment was found in this data.
- Next purchase category is in the feature table but is not modeled.