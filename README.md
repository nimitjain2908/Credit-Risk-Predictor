# 🏦 AI Credit Risk Advisor
### *End-to-End Predictive Modeling & Deployment*

## 📊 Business Problem
Financial institutions face the challenge of expanding credit access while minimizing default rates. This project develops a high-performance machine learning pipeline to predict the probability of loan default, providing automated decision support for underwriters.

## 🚀 Key Results
* **Model Performance:** Achieved a **0.76 AUC-ROC** using LightGBM, outperforming baseline models by 5%.
* **Explainability:** Integrated **SHAP** values to identify key risk drivers (External Credit Ratings, Age, and Education).
* **Deployment:** Developed a live **Streamlit** dashboard for real-time risk assessment and "what-if" analysis.

## 🛠️ Tech Stack
* **Languages:** Python (Pandas, NumPy, Scikit-learn)
* **Algorithms:** Random Forest, LightGBM (Gradient Boosting)
* **Explainable AI:** SHAP
* **Deployment:** Streamlit, Joblib

## 📈 Key Insights
1. **External Ratings:** Features `EXT_SOURCE_2` and `EXT_SOURCE_3` are the strongest predictors of repayment behavior.
2. **Age & Stability:** Older applicants showed a statistically significant lower risk profile.
3. **Education:** Higher education levels correlate with higher creditworthiness in this dataset.

## 💻 How to Run Locally
1. Clone the repo: `git clone https://github.com/nimitjain2908/Credit-Risk-Predictor.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
