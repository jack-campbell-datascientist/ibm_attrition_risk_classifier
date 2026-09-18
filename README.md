# Workforce Attrition Risk & Retention Simulator

An end-to-end workforce analytics project that predicts employee attrition risk and turns those predictions into an interactive decision-support tool for HR and people leaders.

Streamlit app link: https://ibmattritionriskclassifier-pw8xir5s6euwrd3nkrw9k6.streamlit.app/

## Why this project

Attrition is one of the most expensive and hardest-to-manage problems in workforce planning. This project treats it as a business question rather than a pure modeling exercise: it doesn't just predict who might leave, it surfaces why, and gives a way to test how changes to overtime, compensation, or workload might shift that risk before those changes happen.

## Dataset

IBM HR Analytics Employee Attrition dataset (Kaggle), 1,470 employee records with demographic, compensation, satisfaction, and role-based features, plus a historical attrition label.
Link: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data
## Approach

1. **Data preparation** - cleaned and encoded categorical features, removed non-predictive identifier columns
2. **Modeling** - trained and compared Random Forest and XGBoost classifiers, evaluating on accuracy, precision, recall, F1, and ROC AUC
3. **Model selection** - selected Random Forest based on stronger recall, since in a retention context, missing an at-risk employee is more costly than flagging someone who wasn't actually at risk
4. **Explainability** - used SHAP values to show which factors are driving individual risk predictions, not just aggregate feature importance
5. **Dashboard** - built an interactive Streamlit app for exploring risk at the company, department, and individual level

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | 0.830 | 0.465 | 0.426 | 0.444 | 0.787 |
| XGBoost | 0.833 | 0.474 | 0.383 | 0.424 | 0.778 |

## Dashboard features

- **Executive summary** - company-wide attrition rate, predicted at-risk headcount, and department-level risk comparison
- **Department drilldown** - filterable views by department and role, with a ranked list of highest-risk employees
- **Retention simulator** - an interactive scenario tool where adjusting factors like overtime, income, or work-life balance updates predicted risk in real time, with SHAP-based explanations and suggested retention actions

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly

## Running it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Next steps

Planned addition: a causal analysis estimating the effect of overtime on attrition probability while controlling for income, tenure, and role, to move beyond correlation-based feature importance toward an estimate that could inform an actual policy decision.