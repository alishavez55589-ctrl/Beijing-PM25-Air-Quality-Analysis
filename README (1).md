# Beijing PM2.5 Air Quality — End-to-End Data Science Pipeline

**Course:** Data Science and Analytics (MCT-341L)  
**Dataset:** Beijing PM2.5 Data (2010–2014) — UCI Machine Learning Repository  
**License:** CC BY 4.0  

---

## 📌 Project Overview

This project builds a complete data science pipeline on Beijing's hourly air quality data collected at the US Embassy monitoring station from January 2010 to December 2014. The pipeline covers everything from raw data loading to trained machine learning models ready for deployment.

Two prediction tasks are solved:
- **Regression** — predict the exact PM2.5 concentration (µg/m³)
- **Classification** — predict the air quality category (Good → Hazardous)

---

## 📁 Project Structure

```
project/
│
├── beijing_pm25_clean.csv        # Raw input dataset
├── beijing_pm25_final.csv        # Cleaned + feature-engineered dataset
├── FinalProject.ipynb            # Main notebook (all 8 phases)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
└── dashboard/
    ├── models/
    │   ├── regression_model.pkl      # Trained Ridge Regression
    │   ├── classification_model.pkl  # Trained Random Forest
    │   ├── scaler.pkl                # Fitted StandardScaler
    │   ├── feature_names.pkl         # Feature column order
    │   ├── kmeans_model.pkl          # Trained K-Means (k=6)
    │   └── pca_model.pkl             # Fitted PCA (2 components)
    └── data/
        ├── cleaned_data.csv          # Full cleaned dataset
        └── cluster_data.csv          # Test set with cluster labels
```

---

## 📊 Dataset Information

| Field | Details |
|---|---|
| **Source** | https://archive.uci.edu/dataset/381/beijing+pm2+5+data |
| **Collection Period** | January 1, 2010 — December 31, 2014 |
| **Total Records** | 43,824 hourly readings |
| **Features** | 13 original + 5 engineered = 18 total |
| **Missing Values** | 2,067 in pm2.5 column (~4.7%) |
| **Target (Regression)** | `pm2.5` — particulate matter in µg/m³ |
| **Target (Classification)** | `air_quality` — 6 categories based on China MEP standard |

### Feature Descriptions

| Column | Unit | Description |
|---|---|---|
| `year`, `month`, `day`, `hour` | — | Timestamp components |
| `pm2.5` | µg/m³ | Fine particulate matter (**Regression Target**) |
| `DEWP` | °C | Dew point temperature |
| `TEMP` | °C | Air temperature |
| `PRES` | hPa | Atmospheric pressure |
| `cbwd` | — | Wind direction (NW, SE, cv, NE) |
| `Iws` | m/s | Cumulative wind speed |
| `Is` | hours | Cumulative snow hours |
| `Ir` | hours | Cumulative rain hours |

---

## ⚙️ Installation & Setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd beijing-pm25-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Make sure `beijing_pm25_clean.csv` is in the root project folder.

### 4. Run the notebook

```bash
jupyter notebook FinalProject.ipynb
```

---

## 🔄 Pipeline Phases

### Phase 1 — Dataset Acquisition & Initial Exploration
- Load CSV and inspect shape, dtypes, memory usage
- Statistical summary (mean, median, std, skewness)
- Define regression and classification targets
- Class balance check with imbalance ratio

### Phase 2 — Exploratory Data Analysis (EDA)
- Univariate distributions with skewness and kurtosis
- Categorical feature analysis (wind direction, season, time of day)
- Pearson correlation matrix with multicollinearity check
- Bivariate scatter plots and box plots
- Temporal patterns: monthly, seasonal, and hourly PM2.5 trends

### Phase 3 — Data Cleaning & Preprocessing
- Missing value audit and median imputation for pm2.5
- IQR-based outlier detection and Winsorisation (no rows dropped)
- One-hot encoding for wind direction, ordinal encoding for season/time
- Stratified 80/20 train/test split

### Phase 4 — Data Visualization & Storytelling
- Distribution dashboard (2×3 grid)
- Annotated correlation heatmap with top-3 pairs highlighted
- Feature-target correlation bar chart
- Joint density hexbin plot (DEWP vs PM2.5)
- Domain insight: wind direction vs PM2.5

### Phase 5 — Feature Engineering
- `log_pm25` — log transform to reduce skewness
- `temp_dewp_gap` — dew point depression (humidity indicator)
- `wind_pressure` — Iws × PRES interaction term
- `precip_total` — combined rain + snow hours
- `is_heating_season` — binary flag for Beijing's coal heating months
- StandardScaler fitted on training data only (no data leakage)
- Feature selection: Pearson filter vs Lasso embedded method

### Phase 6 — Regression Modelling
| Model | R² | RMSE (µg/m³) | MAE (µg/m³) |
|---|---|---|---|
| Baseline (Mean) | 0.0000 | 76.80 | 61.40 |
| Linear Regression | 0.3281 | 62.96 | 49.48 |
| Ridge Regression | 0.3281 | 62.96 | 49.48 |

- Residual analysis: Residuals vs Fitted, Q-Q plot
- Ridge coefficient interpretation (top 10 features)

### Phase 7 — Classification Modelling
| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.47 | ~0.32 | ~0.82 |
| Random Forest | ~0.67 | ~0.52 | ~0.91 |

- Confusion matrix with counts and percentages
- Classification report (Precision, Recall, F1 per class)
- ROC-AUC curves (One-vs-Rest for all 6 classes)
- Precision-Recall curves
- Error analysis: 10 misclassified samples

### Phase 8 — Unsupervised Analysis (K-Means Clustering)
- Elbow curve + Silhouette scores for k = 2 to 10
- Optimal k = 6 (matches 6 air quality categories)
- PCA visualization (2D scatter with centroids)
- Cluster profiling by mean feature values
- Cluster vs air quality cross-tabulation

---

## 🧪 Key Findings

1. **Winter is worst** — coal-based heating (Nov–Mar) is the dominant PM2.5 source
2. **Wind disperses pollution** — Iws has the strongest negative correlation with PM2.5
3. **Humidity traps pollution** — high dew point correlates with higher PM2.5
4. **PM2.5 is right-skewed** — log transform required for linear models
5. **Calm wind = worst air** — cv (calm) wind direction produces highest PM2.5 averages

---

## 📦 Dependencies

```
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.8
seaborn==0.13.2
scipy==1.17.1
scikit-learn==1.8.0
joblib==1.5.3
```

---

## 👥 Authors

Shave Nasir
Basil Iftikhar

**Institution:** [University of Engineering and Technology lahore]  

