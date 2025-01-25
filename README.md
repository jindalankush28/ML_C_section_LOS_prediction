# Predicting Post-Cesarean Hospital Stay

## Project Overview
This project aims to develop machine learning models to predict the length of stay (LOS) for post-cesarean hospitalizations using the Healthcare Cost and Utilization Project (HCUP) dataset. The study compares multiple machine learning approaches and evaluates the impact of different features, including demographic, clinical, and social determinants of health, on prediction accuracy.

---

## Objectives
- Establish predictive models for post-cesarean LOS.
- Identify the most relevant features for LOS prediction using mRMR.
- Compare performance across machine learning algorithms.
- Evaluate the generalizability of models using large-scale population data.

---

## Dataset
- **Source:** Healthcare Cost and Utilization Project (HCUP) State Inpatient Database (SID).
- **Study Population:** Cesarean section cases from Maryland (2016–2019).
- **Features:** Demographics, ICD-10 and PCS codes, and social determinants of health.
- **Sample Size:** 86,889 cases after inclusion and exclusion criteria.

---

## Methodology
1. **Preprocessing:**
   - Numerical features: Z-score normalization.
   - Categorical features: One-hot encoding.
   - Derived features: Gestational age and Elixhauser comorbidity scores.

2. **Feature Selection:**
   - Minimum Redundancy Maximum Relevance (mRMR) method to rank feature importance.

3. **Models:**
   - Linear Regression
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - Extreme Gradient Boosting (XGBoost)
   - Multilayer Perceptron (MLP)

4. **Evaluation Metrics:**
   - Accuracy (ACC)
   - Area Under the Curve (AUC)
   - F1 Score

---

## Key Results
### Table 3: Classification Performance on Combined Features

| Model               | ACC            | AUC            | F1             |
|---------------------|----------------|----------------|----------------|
| Linear Model        | 0.7323 (0.0200)| 0.5417 (0.0155)| 0.1898 (0.0478)|
| Logistic Regression | 0.7317 (0.0085)| 0.5465 (0.0138)| 0.2131 (0.0385)|
| Random Forest       | 0.7295 (0.0028)| 0.5189 (0.0055)| 0.0882 (0.0214)|
| SVM                 | 0.7546 (0.0068)| 0.5739 (0.0086)| 0.3045 (0.0486)|
| XGBoost             | 0.7326 (0.0059)| 0.5390 (0.0086)| 0.1790 (0.0256)|
| MLP                 | 0.7670 (0.0085)| 0.7088 (0.0220)| 0.4775 (0.0152)|

### Insights:
- MLP exhibited the best performance, with over 10% improvement in AUC and F1 score compared to other models.
- ICD-based features provided better predictive power than social determinants or combined features.

---

## Key Findings
- **Feature Importance:** Gestational age was the most critical predictor, followed by clinical interventions (e.g., blood transfusion, complications).
- **Model Performance:** MLP demonstrated superior results but required higher computational resources.
- **Social Determinants:** Limited utility in improving LOS prediction.
- **Future Directions:** Integration of advanced deep learning techniques and comparisons across states for enhanced generalizability.

---

## References
1. World Health Organization. *Caesarean section rates continue to rise, amid growing inequalities in access*, 2021.
2. Alsinglawi et al., *An explainable machine learning framework for lung cancer hospital length of stay prediction*, Scientific Reports, 2022.
3. Lequertier et al., *Hospital length of stay prediction methods: a systematic review*, Medical Care, 2021.
4. Zhao et al., *Maximum relevance and minimum redundancy feature selection methods for a marketing machine learning platform*, 2019.

---

For further details, refer to the project documentation or contact the research team.
