# SaaS Pricing Model Selection Using Data Science Techniques

A comprehensive data science project that analyzes SaaS business metrics to determine optimal pricing strategies that maximize customer lifetime value (LTV) while maintaining sustainable customer acquisition costs (CAC) and churn rates.

## 📊 Project Overview

This project leverages machine learning and statistical analysis to optimize SaaS pricing strategies by:
- Predicting customer churn using ensemble methods
- Forecasting revenue based on pricing scenarios
- Segmenting customers by behavior and value
- Calculating price elasticity of demand
- Recommending tiered pricing structures

**Key Achievement**: Identified optimal pricing strategy that achieves a 4.76 LTV/CAC ratio while maintaining acceptable churn rates.

## 🎯 Business Problem

SaaS companies face the challenge of setting prices that:
- Maximize revenue and profitability
- Minimize customer churn
- Optimize the LTV-to-CAC ratio
- Remain competitive in the market

This project provides a data-driven framework to solve this multi-objective optimization problem.

## 🔍 Key Findings

- **Optimal Price Point**: $1,753.88 (34.7% decrease from baseline)
- **Expected LTV/CAC Ratio**: 4.76 (exceeds healthy threshold of 3.0)
- **Price Elasticity**: -2.191 (elastic demand)
- **Customer Segments**: 4 distinct segments identified
- **Projected MRR**: $8,958,995

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
- Cleaned and standardized 4,222 customer records
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
   - Open `Notebooks/02_modeling.ipynb`
   - Execute cells to train models and generate recommendations

3. **View Results**:
   - Check `Optimization Results/` for CSV outputs
   - Review `Visualizations/` for charts and graphs
   - Read `business_narrative.txt` for executive summary

## 📊 Customer Segments

| Segment | Size | Characteristics | Recommended Pricing |
|---------|------|----------------|---------------------|
| Premium | 219 | High LTV/CAC, low churn | $2,630.83 |
| Standard | 760 | Balanced metrics | $1,753.88 |
| Growth | 1,538 | High potential | $1,403.11 |
| At-Risk | 1,705 | High churn risk | $1,052.33 |

## 📉 Key Metrics

- **Churn Rate Target**: ≤20%
- **LTV/CAC Ratio Target**: ≥3.0
- **Model Performance**:
  - Churn Prediction AUC: 0.940
  - Revenue Forecast R²: 1.000

## 💡 Recommendations

### Tiered Pricing Strategy
Implement a four-tier pricing structure tailored to customer segments:
- **Premium Tier**: $2,630.83 for high-value customers
- **Standard Tier**: $1,753.88 as primary offering
- **Growth Tier**: $1,403.11 for price-sensitive market
- **Retention Tier**: $1,052.33 for at-risk customers

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