# 🇺🇸 Yield Curve Inversion — An Early Warning System

### 📘 Overview  
This project builds an **early-warning system** for detecting **U.S. Treasury yield curve inversions** up to **12 weeks ahead**.  
Yield-curve inversions are historically reliable early indicators of potential economic slowdowns.  
We combine **time-series econometrics**, **unsupervised learning**, and **supervised machine learning** to create an interpretable, data-driven forecasting pipeline.

---

## 🎯 Objective  
> Predict whether the **10-year minus 3-month Treasury spread (10Y – 3M)** will **invert** within the next **12 weeks**, using historical yield-curve structure and regime dynamics.

---

## 🗂️ Data Source  
All data comes from **FRED (Federal Reserve Economic Data)**  

- Weekly constant-maturity Treasury yields  
  - 3 Month (3M), 2 Year (2Y), 5 Year (5Y), 10 Year (10Y)  
- Frequency     : Weekly (Friday close)  
- Time Span     : 1990 – 2025  
- Source        : [https://fred.stlouisfed.org](https://fred.stlouisfed.org)

---

## ⚙️ Methodology

### 🅰️ Phase A – Data Collection  
- Pulled weekly Treasury yields via FRED API  
- Cleaned, aligned, and forward-filled missing values  
- Exported processed tables under `/Data/processed`

### 🅱️ Phase B – EDA + Regime Identification  
- Created weekly yield-curve “snapshots”  
- Applied **K-Means clustering** to detect curve shapes → *Steep*, *Flat*, *Inverted*  
- Conducted **ADF / KPSS** tests → checked stationarity  
- Performed **Johansen cointegration** → derived **Error Correction Terms (ECTs)**  
- Used **Principal Component Analysis (PCA)** → summarized *Level*, *Slope*, *Curvature*  

### 🅲️ Phase C – Modeling  
#### Models  
- **ARIMA / VECM** → time-series forecasting of long-term yields  
- **Logistic Regression (+ state features)** → interpretable inversion classifier  
- **HistGradientBoosting** → non-linear baseline model  
- **Rolling-origin backtesting** → realistic temporal validation  

### 🅳️ Phase D – Evaluation  
Metrics  : AUC | PR-AUC | Brier | Precision | Recall | F1  
- Performed threshold sweep → optimal cutoff = **0.40** (recall-focused)  
- Produced diagnostic plots (ROC, PR, Reliability, Confusion, Hero charts)

---

## 🔍 Key Insights  
✅ Yield levels are non-stationary; spreads are stable and mean-reverting  
✅ ECT captures short-term adjustments to long-run equilibrium  
✅ Final Logit + state model achieved:  
- **AUC:** 0.91  
- **PR-AUC:** 0.76  
- **Brier:** 0.12  
- **Recall:** 98 % **Precision:** 63 % (at thr = 0.40)  
✅ Inversion ≠ Recession → It signals risk build-up, not guaranteed downturn  

---

## 🎨 Visualization Highlights  

**Hero Chart (last 5 years)**  
- Line    = Predicted inversion probability  
- Shaded = Model calls inversion (> 0.40)  
- Dots    = Actual inversions  
- Latest probability ≈ **0.94 → “Red Alert” signal**

![Hero Chart](Data/processed/figs/hero_inv_h12.png)

---

## 🚦 Traffic-Light System  

| Probability Range | Signal | Interpretation |
|------------------:|:-------|:----------------|
| < 0.25   | 🟢 Normal | Curve healthy |
| 0.25 – 0.40 | 🟡 Caution | Flattening trend |
| ≥ 0.40   | 🔴 Alert | Inversion likely (≈ 12 weeks ahead) |

---

## 💻 Streamlit App  

Run interactive dashboard locally:  
```bash
streamlit run app.py
```
## Author
1. Karan Badlani | MS in Data Science | Northeastern Universtity
2. Sajan Arora | MS in Data Science | Northeastern Universtity
