# SaaS Pricing Model Selection Using Data Science Techniques

A comprehensive data science project that analyzes SaaS business metrics to determine optimal pricing strategies that maximize customer lifetime value (LTV) while maintaining sustainable customer acquisition costs (CAC) and churn rates.

## 📊 Project Overview

This project leverages machine learning and statistical analysis to optimize SaaS pricing strategies by:
- Predicting customer churn using ensemble methods
- Forecasting revenue based on pricing scenarios
- Segmenting customers by behavior and value
- Calculating price elasticity of demand
- Recommending tiered pricing structures

**Key Achievement**: Built a leakage-free pricing pipeline on 7,057 real customer records, with a cross-validated revenue model (R² ≈ 0.97), an honest churn classifier (AUC ≈ 0.89), and an elasticity-based pricing recommendation. See `WHAT_I_FIXED.md` for the model audit that corrected an earlier version.

## 🎯 Business Problem

SaaS companies face the challenge of setting prices that:
- Maximize revenue and profitability
- Minimize customer churn
- Optimize the LTV-to-CAC ratio
- Remain competitive in the market

This project provides a data-driven framework to solve this multi-objective optimization problem.

## 🔍 Key Findings

- **Baseline avg price**: $190.19 · **Mean monthly churn**: 5.31% · **Avg LTV/CAC**: ~33
- **Price Elasticity**: -1.03 (roughly unit-elastic; 95% CI [-1.24, -0.82])
- **Revenue (ARPU) model**: R² ≈ 0.97 (5-fold CV 0.968 ± 0.001), leakage-checked
- **Churn model**: ROC-AUC ≈ 0.89 (churn-rate column excluded from features)
- **Customer Segments**: 4, ranked by honest LTV/CAC (Premium > Standard > Growth > At-Risk)
- **Pricing recommendation**: presented as a modelled scenario under an LTV/CAC ≥ 3.0 guardrail, not a point forecast

> **Note on methodology:** An earlier version of this project reported a revenue R² of 1.000 and a ~69% churn rate. Both were artifacts (target leakage and an incorrect subscription-level rollup). The current pipeline corrects them; `WHAT_I_FIXED.md` documents the full audit.

## 🗂️ Project Structure

```
├── Cleaned Data/                    # Processed datasets
│   ├── cac_ltv_cleaned.csv
│   ├── ravenstack_cleaned.csv
│   └── saas_businesses_cleaned.csv
│
├── Optimization Results/            # Model outputs and recommendations
│   ├── model_dataset_with_predictions.csv
│   ├── optimization_summary.csv
│   ├── price_scenarios.csv
│   └── segment_pricing_recommendations.csv
│
├── Visualizations/                  # Charts and graphs
│   ├── customer_segmentation.png
│   ├── final_recommendation.png
│   ├── model_performance.png
│   └── price_optimization_curves.png
│
├── Raw Project Datasets/            # Source data
│   ├── Business Startups Data on SAAS products/
│   ├── CAC-LTV Model Analysis for SaaS Business Insights/
│   └── SaaS Subscription & Churn Analytics Dataset/
│
├── Notebooks/
│   ├── 01_cleaning.ipynb           # Data cleaning and preprocessing
│   └── 02_modeling.ipynb           # Model development and optimization
│
├── business_narrative.txt          # Executive summary of findings
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Technologies Used

### Programming & Analysis
- **Python 3.x** - Primary programming language
- **Jupyter Notebooks** - Interactive development environment

### Data Science Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning models
- **matplotlib** / **seaborn** - Data visualization

### Machine Learning Models
- **Random Forest** - Churn prediction (AUC: 0.940)
- **Linear Regression** - Revenue forecasting (R²: 1.000)
- **K-Means Clustering** - Customer segmentation (k=4)
- **Log-log Regression** - Price elasticity estimation

## 📈 Methodology

### 1. Data Collection & Cleaning
- Integrated multiple SaaS business datasets
- Analyzed 7,057 customer-level records with monthly economics (2023–24)
- Engineered features for model training

### 2. Exploratory Data Analysis
- Analyzed pricing distributions and customer behavior
- Identified correlations between price, churn, and LTV
- Visualized key business metrics

### 3. Model Development
- **Churn Prediction**: Built Random Forest classifier to predict customer churn
- **Revenue Forecasting**: Developed regression model for revenue projection
- **Customer Segmentation**: Applied K-Means clustering to identify customer groups
- **Price Elasticity**: Calculated demand sensitivity to price changes

### 4. Optimization
- Formulated multi-objective optimization problem
- Simulated various pricing scenarios
- Identified optimal price points for each customer segment

### 5. Recommendation Generation
- Developed tiered pricing structure
- Created implementation roadmap
- Defined monitoring KPIs

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sanskritidabhade/SaaS-Pricing-Model-Selection-Using-Data-Science-Techniques.git
cd SaaS-Pricing-Model-Selection-Using-Data-Science-Techniques
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. **Data Cleaning**:
   - Open `Notebooks/01_cleaning.ipynb`
   - Run all cells to process raw data

2. **Model Training & Optimization**:
   - Open `Notebooks/02_modeling_CORRECTED.ipynb` (leakage-free rebuild)
   - Execute cells to train models and generate recommendations

3. **View Results**:
   - Check `Optimization Results/` for CSV outputs
   - Review `Visualizations/` for charts and graphs
   - Read `business_narrative_CORRECTED.txt` for the executive summary
   - Read `WHAT_I_FIXED.md` for the model audit and corrections

## 📊 Customer Segments

| Segment | Size | Characteristics | Recommended Pricing |
|---------|------|----------------|---------------------|
| Premium | 693 | Highest LTV/CAC (~146), low churn (~2%) | Supports premium pricing |
| Standard | 2,039 | Strong LTV/CAC (~34), low churn (~2%) | Base pricing |
| Growth | 1,277 | Moderate LTV/CAC (~28), higher churn (~7%) | Competitive pricing |
| At-Risk | 3,048 | Lowest LTV/CAC (~9), highest churn (~8%) | Retention pricing |

## 📉 Key Metrics

- **Mean monthly churn (observed)**: 5.31%
- **LTV/CAC guardrail (optimization constraint)**: ≥3.0
- **Model Performance (honest, hold-out / cross-validated)**:
  - Churn Prediction ROC-AUC: ~0.89
  - Revenue (ARPU) R²: ~0.97 (5-fold CV 0.968 ± 0.001)

## 💡 Recommendations

### Tiered Pricing Strategy
Segment-based pricing, differentiated by each segment's LTV/CAC and churn profile:
- **Premium**: highest willingness-to-pay; supports above-baseline pricing
- **Standard**: healthy economics; anchor at the baseline price
- **Growth**: price-sensitive; competitive pricing to expand the base
- **At-Risk**: retention-focused pricing to protect lifetime value

Exact tier prices depend on the elasticity assumptions in the optimization and should be validated by A/B testing before rollout.

### Implementation Roadmap
1. **Months 1-2**: A/B test with 20% of new customers
2. **Months 3-4**: Monitor churn, LTV, and conversion metrics
3. **Months 5-6**: Full rollout based on results
4. **Ongoing**: Continuous monitoring and quarterly reviews

## ⚠️ Limitations

- Analysis assumes price as primary driver; other factors (features, support, UX) also influence outcomes
- Real-world A/B testing required for validation
- External market factors not fully captured
- Customer survey validation recommended
- The price→churn link in the optimization is a bounded modelling assumption, not a fitted causal estimate; elasticity is cross-sectional, not experimental

## 🔮 Future Enhancements

- [ ] Incorporate competitive pricing data
- [ ] Add feature-based value pricing analysis
- [ ] Implement real-time churn prediction API
- [ ] Develop automated pricing recommendation dashboard
- [ ] Include seasonality and market trend analysis

## 📄 License

This project is available for educational and research purposes.

## 👤 Author

**Sanskriti Dabhade**
- GitHub: [@sanskritidabhade](https://github.com/sanskritidabhade)

## 🙏 Acknowledgments

- Data sources: Multiple SaaS business datasets
- Inspired by modern SaaS pricing optimization practices
- Built using open-source data science tools

---

**Note**: This analysis provides data-driven recommendations but should be validated through real-world testing before full implementation. Always consider market conditions, competitive landscape, and customer feedback when making pricing decisions.