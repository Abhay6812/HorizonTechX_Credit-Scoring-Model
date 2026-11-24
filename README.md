# CreditScope ML — Credit Scoring & Creditworthiness Prediction Model (TASK 1)

## 1. Project Overview & Objective

**CreditScope ML** is an end-to-end machine learning project designed to predict customer creditworthiness based on historical financial data, debt obligations, payment behavior, credit utilization, and transaction patterns.

Banks and financial institutions rely on credit scoring models to assess loan and credit card eligibility while managing default exposure. This project classifies each applicant into one of three risk tiers:

- **Low Risk** — Strong creditworthiness, low debt burden, consistent on-time payments, and healthy liquid reserves.
- **Medium Risk** — Moderate leverage or occasional late payments; typically eligible with standard or adjusted underwriting terms.
- **High Risk** — Elevated credit utilization, high debt-to-income ratio, past defaults, or irregular payment behavior indicating significant default risk.

The project delivers **both**:
1. A standalone, reproducible **Jupyter Notebook** (`public/Credit_Scoring_Model.ipynb`) written in Python 3 (`nbformat 4.5`) using `scikit-learn`, `xgboost`, `shap`, `matplotlib`, and `seaborn`.
2. An **Interactive Web Workbench** built with React, TypeScript, and Tailwind CSS that lets you inspect and copy notebook cells, explore Matplotlib/Seaborn diagnostic visualizations, audit benchmark tables, export the `.ipynb` notebook and 5,000-row `.csv` dataset, and run live credit risk simulations with **SHAP feature importance explanations**.

---

## 2. Repository & File Structure

```text
├── public/
│   └── Credit_Scoring_Model.ipynb        # Standalone runnable Jupyter Notebook (nbformat 4.5)
├── src/
│   ├── App.tsx                           # Main workspace layout (TopBar, Sidebar, 4 active views)
│   ├── index.css                         # Tailwind CSS v4 config + typography & tabular numerals
│   ├── main.tsx                          # React DOM entry point
│   ├── components/
│   │   ├── TopBar.tsx                    # Single-row header navigation + .ipynb export button
│   │   ├── Sidebar.tsx                   # Notebook cell outline, dataset metadata & export actions
│   │   ├── NotebookView.tsx              # Interactive Jupyter cell viewer (Run Cell / Run All / Copy)
│   │   ├── VisualizationsView.tsx        # Interactive gallery of Matplotlib & Seaborn diagnostic plots
│   │   ├── ChartsGallery.tsx             # SVG diagnostic charts (EDA, ROC-AUC, Confusion Matrix, Heatmap)
│   │   ├── MetricsTableView.tsx          # Model benchmark table, per-class report & dataset explorer
│   │   ├── RiskPredictorView.tsx         # Live applicant credit risk simulator & model consensus
│   │   └── FeatureImportancePanel.tsx    # Local SHAP waterfall chart & Global algorithm weights
│   ├── data/
│   │   └── notebookCells.ts              # Structured Markdown + Python code cells & cell outputs
│   └── utils/
│       ├── creditScoringEngine.ts        # Deterministic dataset generator, metrics & SHAP inference
│       └── notebookExporter.ts           # Browser exporters for .ipynb (JSON) and .csv (5,000 rows)
├── index.html                            # Document metadata & Google Fonts (Plus Jakarta Sans, JetBrains Mono)
├── package.json                          # Project scripts and dependencies
└── README.md                             # Project documentation (this file)
```

---

## 3. Dataset Schema & Feature Dictionary

The benchmark dataset (`n = 5,000` customer records, fixed random seed `42`) models realistic financial, lending, and transaction distributions.

### A. Raw Input Features (9 Features)

| Feature | Type | Range / Distribution | Description |
| :--- | :--- | :--- | :--- |
| `annual_income` | `float` | `$20,000` – `$260,000` | Gross annual customer income in USD (log-normal distribution). |
| `existing_debt` | `float` | `$500` – `$195,000` | Total active revolving and installment debt balances. |
| `loan_history_count` | `int` | `0` – `14` | Number of historical and active loan accounts on file. |
| `past_defaults` | `int` | `0` – `6` | Count of historical loan defaults or charge-offs. |
| `on_time_payment_rate` | `float` | `0.45` – `1.00` | Share of scheduled billing cycles paid on or before due date. |
| `credit_utilization` | `float` | `0.02` – `0.99` | Ratio of used revolving credit to total available credit limit. |
| `account_balance` | `float` | `$150` – `$150,000` | Liquid checking and savings account reserves. |
| `credit_history_months` | `int` | `6` – `264` | Total age of credit history in months (`0.5` – `22` years). |
| `monthly_tx_volume` | `float` | Derived from income | Average monthly banking and card transaction volume in USD. |

### B. Target Variable (`risk_category`)

| Risk Tier | Class Index | Sample Count | Share | Typical Profile |
| :--- | :---: | :---: | :---: | :--- |
| **Low Risk** | `0` | `2,100` | `42.0%` | Avg income `$84.2k`, utilization `19%`, on-time rate `96%`, reserves `$29.8k` |
| **Medium Risk** | `1` | `1,700` | `34.0%` | Avg income `$61.9k`, utilization `44%`, on-time rate `87%`, reserves `$9.4k` |
| **High Risk** | `2` | `1,200` | `24.0%` | Avg income `$43.3k`, utilization `73%`, on-time rate `72%`, reserves `$2.7k` |

---

## 4. Data Preprocessing & Feature Engineering

To capture non-linear financial stress signals beyond raw dollar amounts, **Cell 4** (`engineer_features`) constructs **5 domain-specific features** (bringing total features from `9` to `14`):

1. **`debt_to_income_ratio`** = $\frac{\text{existing\_debt}}{\text{annual\_income} + 1}$
   Measures overall leverage relative to earning capacity.
2. **`balance_to_debt_ratio`** = $\frac{\text{account\_balance}}{\text{existing\_debt} + 1}$
   Measures liquid coverage buffer against outstanding debt obligations.
3. **`utilization_debt_interaction`** = $\text{credit\_utilization} \times \text{debt\_to\_income\_ratio}$
   Captures compounding risk when high revolving utilization coincides with heavy total leverage.
4. **`payment_reliability_index`** = $\frac{\text{on\_time\_payment\_rate}}{1 + \text{past\_defaults}}$
   Penalizes historical defaults alongside recent payment punctuality.
5. **`tx_to_income_ratio`** = $\frac{\text{monthly\_tx\_volume} \times 12}{\text{annual\_income} + 1}$
   Measures annualized transaction velocity relative to gross income.

### Splitting & Scaling
- **Missing Values Check**: Verified `0` missing entries across all `5,000 × 14` engineered cells.
- **Train / Test Split**: Stratified `80% / 20%` split (`4,000` training / `1,000` holdout test samples) preserving exact class proportions (`42% / 34% / 24%`).
- **Feature Scaling**: `StandardScaler` is applied inside a scikit-learn `Pipeline` for **Logistic Regression** to prevent scale leakage during cross-validation, while tree-based models operate directly on unscaled features.

---

## 5. Machine Learning Algorithms & Hyperparameter Tuning

All four classifiers are tuned using **5-Fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) and `RandomizedSearchCV` optimizing **weighted One-vs-Rest ROC-AUC** (`roc_auc_ovr_weighted`).

| Algorithm | Tuned Hyperparameters | Inductive Strength |
| :--- | :--- | :--- |
| **XGBoost** (`XGBClassifier`) | `n_estimators=250`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.9` | Gradient-boosted trees capturing subtle feature interactions (`utilization × DTI`) with regularized leaf weights. |
| **Random Forest** (`RandomForestClassifier`) | `n_estimators=250`, `max_depth=12`, `min_samples_split=4` | Bagged ensemble reducing variance and providing stable out-of-sample probability estimates. |
| **Logistic Regression** (`LogisticRegression`) | `StandardScaler` + multinomial, `C=5.0`, `max_iter=1000` | Fast, interpretable linear baseline with well-calibrated baseline probabilities. |
| **Decision Tree** (`DecisionTreeClassifier`) | `max_depth=7`, `min_samples_split=10`, `min_samples_leaf=5` | Transparent rule-based decision paths; useful as an interpretable single-tree benchmark. |

---

## 6. Evaluation Metrics & Benchmark Results

Models are evaluated on the **1,000-sample holdout test set** across all five required classification metrics:

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC (OvR) | 5-Fold CV ROC-AUC | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **93.40%** | **93.42%** | **93.40%** | **93.39%** | **0.9886** | **0.9871** | `1.84s` |
| **Random Forest** | `91.80%` | `91.84%` | `91.80%` | `91.78%` | `0.9834` | `0.9819` | `2.31s` |
| **Logistic Regression** | `89.60%` | `89.63%` | `89.60%` | `89.58%` | `0.9742` | `0.9735` | `0.28s` |
| **Decision Tree** | `86.40%` | `86.51%` | `86.40%` | `86.42%` | `0.9285` | `0.9241` | `0.14s` |

### XGBoost Per-Class Classification Report (`n = 1,000`)

| Risk Tier | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low Risk** | `0.9549` | `0.9571` | `0.9560` | `420` |
| **Medium Risk** | `0.9032` | `0.9059` | `0.9045` | `340` |
| **High Risk** | `0.9412` | `0.9333` | `0.9372` | `240` |

---

## 7. Visualizations (Matplotlib & Seaborn)

The notebook and the **Visualizations** tab include six diagnostic visual suites:

1. **EDA Risk Tier Distributions**: Side-by-side Seaborn `countplot` of class balance and `boxplot` comparisons of `credit_utilization` and `on_time_payment_rate` across risk tiers.
2. **Pearson Feature Correlation Matrix**: `sns.heatmap` (`cmap="coolwarm"`, `center=0`) highlighting positive co-movement (`annual_income` vs. `account_balance`: `+0.68`; `loan_history_count` vs. `credit_history_months`: `+0.72`) and inverse risk relationships (`on_time_payment_rate` vs. `past_defaults`: `-0.64`).
3. **Model Performance Comparison Bar Chart**: Grouped `sns.barplot` comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC across all four algorithms.
4. **Confusion Matrix Heatmaps (2×2 Grid)**: Actual vs. Predicted 3×3 matrices with raw counts and row-normalized recall percentages per classifier.
5. **Multi-Class ROC-AUC Curves (One-vs-Rest)**: True Positive Rate vs. False Positive Rate curves and per-class AUC scores for Low Risk, Medium Risk, and High Risk.
6. **Global Feature Importance Ranking**: Horizontal comparison of top credit risk predictors (`credit_utilization`, `debt_to_income_ratio`, `payment_reliability_index`, `on_time_payment_rate`, `past_defaults`).

---

## 8. Model Interpretability & SHAP Feature Importance

Both **Cell 9** of the Jupyter Notebook (`explain_prediction_shap`) and the interactive **Feature Importance & SHAP Prediction Explanation** panel in the **Risk Predictor** view explain *why* a specific customer received a given prediction:

- **Algorithm-Specific Explainers**:
  - **TreeSHAP** (`XGBoost`, `Random Forest`, `Decision Tree`): Explains non-linear tree splits and feature interactions (`utilization × DTI`, default + delinquency compounding).
  - **LinearSHAP** (`Logistic Regression`): Computes exact standardized linear attributions relative to the training distribution.
- **Additive Efficiency Invariant**:
  $$\text{Population Baseline } E[f(x)] + \sum_{i=1}^{9} \phi_i = f(x) \text{ (Final Risk Score)}$$
  With `E[f(x)] = 1.620`, positive SHAP values (`+φ`, shown in crimson) push the prediction toward **High Risk**, while negative SHAP values (`-φ`, shown in emerald) push the prediction toward **Low Risk**.
- **Local vs. Global View**: Toggle between **Local SHAP Explanation** (diverging attribution bars + plain-English rationale for the current applicant) and **Global Model Weights** (algorithm-level Gain / Gini / $|\beta|$ importance vs. local $|\phi_i|$ share).

---

## 9. Running in Jupyter Notebook (Local & Google Colab)

> **Important**: The core ML pipeline is self-contained inside `Credit_Scoring_Model.ipynb` and runs directly in Jupyter Notebook, JupyterLab, or Google Colab.

### Option A: Local Jupyter Notebook
1. Download `Credit_Scoring_Model.ipynb` using the **Download .ipynb** button in the app (or use `public/Credit_Scoring_Model.ipynb` directly).
2. Create a Python virtual environment (`Python 3.9+` recommended) and install required libraries:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install numpy pandas scikit-learn xgboost matplotlib seaborn shap jupyter
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook public/Credit_Scoring_Model.ipynb
   ```
4. Click **Kernel → Restart & Run All** to generate the dataset, train all 4 classifiers, print evaluation reports, render all Matplotlib/Seaborn plots, and run SHAP local explanations.

### Option B: Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook** and upload `Credit_Scoring_Model.ipynb`.
3. Uncomment `!pip install -q xgboost ...` in Cell 1 (and `!pip install -q shap` in Cell 9) if running in a fresh runtime, then select **Runtime → Run all**.

---

## 10. Running the Interactive Web Workbench

The web interface mirrors the notebook pipeline and provides interactive exploration, export, and live inference:

```bash
# 1. Install Node.js dependencies
npm install

# 2. Start the development server on port 3000
npm run dev

# 3. Type-check and build production bundle
npm run lint
npm run build
```

### Workspace Views
- **Notebook**: Read, copy, and execute all 10 Markdown and Python cells step-by-step with inline outputs and plots.
- **Visualizations**: Switch between algorithms to inspect ROC-AUC curves, Confusion Matrices, Feature Importance, EDA plots, and Correlation Heatmaps.
- **Model Metrics**: Compare all 4 algorithms, inspect per-class Precision/Recall/F1, and search/filter the synthetic dataset (or export all 5,000 rows as CSV).
- **Risk Predictor**: Adjust all 9 financial inputs (or load 4 applicant presets), compare predictions across all 4 classifiers simultaneously, and inspect the **SHAP Feature Importance** panel explaining each prediction.
