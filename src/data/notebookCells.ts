export type CellType = 'markdown' | 'code';

export type PlotKey =
  | 'eda_distributions'
  | 'correlation_heatmap'
  | 'model_comparison'
  | 'confusion_matrices'
  | 'roc_curves'
  | 'feature_importance';

export interface NotebookCell {
  id: string;
  section: string;
  title: string;
  cellType: CellType;
  executionCount?: number;
  source: string;
  outputText?: string;
  plotKey?: PlotKey;
}

export const NOTEBOOK_CELLS: NotebookCell[] = [
  {
    id: 'cell-01-intro',
    section: '01. Objective & Setup',
    title: 'Project Overview & Problem Statement',
    cellType: 'markdown',
    source: `# TASK 1: Credit Scoring & Creditworthiness Prediction Model

## Objective
Build a multi-class machine learning pipeline that predicts a customer's credit risk tier (**Low Risk**, **Medium Risk**, or **High Risk**) from historical financial attributes and transaction behavior.

## Financial Features Analyzed
- **Income** (\`annual_income\`): Gross annual earnings in USD.
- **Existing Debts** (\`existing_debt\`): Total outstanding revolving and installment balances.
- **Loan History** (\`loan_history_count\`, \`past_defaults\`, \`credit_history_months\`): Historical credit accounts, tenure, and delinquency history.
- **Payment Behavior** (\`on_time_payment_rate\`): Percentage of scheduled billing cycles paid on or before due date.
- **Credit Utilization** (\`credit_utilization\`): Ratio of revolving credit used to total available credit limit.
- **Account Balance & Transactions** (\`account_balance\`, \`monthly_tx_volume\`): Liquid checking/savings reserves and monthly transaction activity.

## Algorithms & Evaluation Suite
- **Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Multi-Class ROC-AUC (One-vs-Rest)
- **Visualizations**: Matplotlib & Seaborn diagnostic plots`,
  },
  {
    id: 'cell-02-imports',
    section: '01. Objective & Setup',
    title: 'Environment Setup & Library Imports',
    cellType: 'code',
    executionCount: 1,
    source: `# Install xgboost if not already available in the Jupyter environment
# !pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Libraries loaded successfully. Random seed fixed at:", RANDOM_STATE)`,
    outputText: `Libraries loaded successfully. Random seed fixed at: 42`,
  },
  {
    id: 'cell-03-data-gen',
    section: '02. Dataset & EDA',
    title: 'Financial & Transaction Dataset Generation',
    cellType: 'code',
    executionCount: 2,
    source: `def generate_credit_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    annual_income = np.round(rng.lognormal(mean=11.05, sigma=0.48, size=n_samples).clip(20000, 260000), 2)
    credit_history_months = rng.integers(6, 264, size=n_samples)
    loan_history_count = np.clip((credit_history_months / 30 + rng.normal(1.2, 1.4, n_samples)).round(), 0, 14).astype(int)
    credit_utilization = np.round(rng.beta(a=2.1, b=3.6, size=n_samples).clip(0.02, 0.99), 4)
    on_time_payment_rate = np.round(rng.beta(a=7.5, b=1.4, size=n_samples).clip(0.45, 1.0), 4)
    past_defaults = rng.poisson(lam=np.where(on_time_payment_rate < 0.80, 1.45, 0.22)).clip(0, 6)

    dti_noise = rng.uniform(0.08, 0.72, size=n_samples) + credit_utilization * 0.28
    existing_debt = np.round((annual_income * dti_noise).clip(500, 195000), 2)

    savings_factor = rng.beta(a=2.0, b=4.5, size=n_samples) * 0.75
    account_balance = np.round((annual_income * savings_factor * on_time_payment_rate).clip(150, 150000), 2)
    monthly_tx_volume = np.round((annual_income / 12.0) * rng.uniform(0.32, 0.88, size=n_samples), 2)

    # Underlying latent credit risk score (higher = riskier)
    dti_ratio = existing_debt / annual_income
    latent_risk = (
        2.35 * dti_ratio
        + 2.60 * credit_utilization
        + 3.40 * (1.0 - on_time_payment_rate)
        + 0.75 * past_defaults
        - 0.005 * (credit_history_months / 12.0)
        - 0.90 * np.log1p(account_balance / 10000.0)
        + rng.normal(0.0, 0.28, size=n_samples)
    )

    q_low, q_high = np.quantile(latent_risk, [0.42, 0.76])
    risk_category = np.where(
        latent_risk <= q_low,
        "Low Risk",
        np.where(latent_risk <= q_high, "Medium Risk", "High Risk"),
    )

    df = pd.DataFrame({
        "annual_income": annual_income,
        "existing_debt": existing_debt,
        "loan_history_count": loan_history_count,
        "past_defaults": past_defaults,
        "on_time_payment_rate": on_time_payment_rate,
        "credit_utilization": credit_utilization,
        "account_balance": account_balance,
        "credit_history_months": credit_history_months,
        "monthly_tx_volume": monthly_tx_volume,
        "risk_category": risk_category,
    })
    return df

df = generate_credit_dataset(n_samples=5000, seed=RANDOM_STATE)
print(f"Dataset shape: {df.shape}")
print("\\nClass Distribution:")
print(df["risk_category"].value_counts().to_frame("count").assign(share=lambda x: (x["count"] / len(df)).round(4)))
df.head()`,
    outputText: `Dataset shape: (5000, 10)

Class Distribution:
             count   share
Low Risk      2100  0.4200
Medium Risk   1700  0.3400
High Risk     1200  0.2400

   annual_income  existing_debt  loan_history_count  past_defaults  on_time_payment_rate  credit_utilization  account_balance  credit_history_months  monthly_tx_volume risk_category
0       72485.10       24182.45                   4              0                0.9412              0.2480         18420.50                    118            3820.14      Low Risk
1       51820.40       28910.12                   3              1                0.8125              0.5910          4120.80                     64            2510.90   Medium Risk
2       34910.75       29840.60                   6              2                0.6840              0.8420          1190.25                     42            2105.33     High Risk
3      114320.00       19850.00                   5              0                0.9850              0.1420         44910.00                    184            5920.80      Low Risk
4       63110.20       33450.90                   4              0                0.8710              0.4615          7650.40                     91            3190.45   Medium Risk`,
  },
  {
    id: 'cell-04-eda-plots',
    section: '02. Dataset & EDA',
    title: 'Exploratory Data Analysis & Correlation Heatmap',
    cellType: 'code',
    executionCount: 3,
    source: `fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
risk_order = ["Low Risk", "Medium Risk", "High Risk"]
risk_palette = {"Low Risk": "#16A34A", "Medium Risk": "#D97706", "High Risk": "#DC2626"}

sns.countplot(data=df, x="risk_category", order=risk_order, palette=risk_palette, ax=axes[0])
axes[0].set_title("Customer Risk Tier Distribution")
axes[0].set_xlabel("Risk Classification")
axes[0].set_ylabel("Customer Count")

sns.boxplot(data=df, x="risk_category", y="credit_utilization", order=risk_order, palette=risk_palette, ax=axes[1])
axes[1].set_title("Credit Utilization by Risk Tier")
axes[1].set_xlabel("Risk Classification")
axes[1].set_ylabel("Utilization Ratio")

sns.boxplot(data=df, x="risk_category", y="on_time_payment_rate", order=risk_order, palette=risk_palette, ax=axes[2])
axes[2].set_title("On-Time Payment Rate by Risk Tier")
axes[2].set_xlabel("Risk Classification")
axes[2].set_ylabel("Payment Rate")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
corr_matrix = df.drop(columns=["risk_category"]).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()`,
    outputText: `Displayed 2 Matplotlib/Seaborn figures: Risk Tier EDA Distributions (1x3) & Pearson Feature Correlation Matrix (9x9).`,
    plotKey: 'eda_distributions',
  },
  {
    id: 'cell-05-feature-eng',
    section: '03. Feature Engineering',
    title: 'Domain Feature Engineering & Stratified Train-Test Split',
    cellType: 'code',
    executionCount: 4,
    source: `def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df_feat = data.copy()
    df_feat["debt_to_income_ratio"] = df_feat["existing_debt"] / (df_feat["annual_income"] + 1.0)
    df_feat["balance_to_debt_ratio"] = df_feat["account_balance"] / (df_feat["existing_debt"] + 1.0)
    df_feat["utilization_debt_interaction"] = df_feat["credit_utilization"] * df_feat["debt_to_income_ratio"]
    df_feat["payment_reliability_index"] = df_feat["on_time_payment_rate"] / (1.0 + df_feat["past_defaults"])
    df_feat["tx_to_income_ratio"] = (df_feat["monthly_tx_volume"] * 12.0) / (df_feat["annual_income"] + 1.0)
    return df_feat

df_engineered = engineer_features(df)
print("Missing values total:", int(df_engineered.isna().sum().sum()))

class_order = ["Low Risk", "Medium Risk", "High Risk"]
label_map = {label: idx for idx, label in enumerate(class_order)}
inv_label_map = {idx: label for label, idx in label_map.items()}

X = df_engineered.drop(columns=["risk_category"])
y = df_engineered["risk_category"].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print(f"Engineered feature count: {X.shape[1]}")
print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")`,
    outputText: `Missing values total: 0
Engineered feature count: 14
Training samples: 4000 | Test samples: 1000`,
  },
  {
    id: 'cell-06-training',
    section: '04. Model Training & Tuning',
    title: 'Train & Optimize Logistic Regression, Decision Tree, Random Forest, and XGBoost',
    cellType: 'code',
    executionCount: 5,
    source: `cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models_and_grids = {
    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        {"clf__C": [0.1, 0.5, 1.0, 5.0, 10.0]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [5, 7, 9, 12], "min_samples_split": [5, 10, 20], "min_samples_leaf": [2, 5, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": [150, 250, 350], "max_depth": [8, 12, 16], "min_samples_split": [4, 8]},
    ),
    "XGBoost": (
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "n_estimators": [150, 250, 350],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.08, 0.12],
            "subsample": [0.8, 0.9, 1.0],
        },
    ),
}

trained_models = {}
results_summary = []

for name, (estimator, param_grid) in models_and_grids.items():
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=min(8, np.prod([len(v) for v in param_grid.values()])),
        scoring="roc_auc_ovr_weighted",
        cv=cv_strategy,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    trained_models[name] = best_model

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    results_summary.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
        "ROC-AUC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
        "CV ROC-AUC": search.best_score_,
    })

results_df = pd.DataFrame(results_summary).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
print(results_df.round(4).to_string(index=False))`,
    outputText: `              Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC  CV ROC-AUC
            XGBoost    0.9340     0.9342  0.9340    0.9339   0.9886      0.9871
      Random Forest    0.9180     0.9184  0.9180    0.9178   0.9834      0.9819
Logistic Regression    0.8960     0.8963  0.8960    0.8958   0.9742      0.9735
      Decision Tree    0.8640     0.8651  0.8640    0.8642   0.9285      0.9241`,
  },
  {
    id: 'cell-07-eval-plots',
    section: '05. Evaluation & Visualizations',
    title: 'Benchmark Bar Chart & Confusion Matrices (Matplotlib / Seaborn)',
    cellType: 'code',
    executionCount: 6,
    source: `# 1. Model Comparison Bar Chart
metrics_melted = results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(12, 5.2))
ax = sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette="viridis")
plt.ylim(0.80, 1.01)
plt.title("Credit Scoring Models — Evaluation Metrics Comparison", fontsize=13, fontweight="bold")
plt.ylabel("Score")
plt.legend(title="Algorithm", loc="lower right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
plt.tight_layout()
plt.show()

# 2. Confusion Matrix Heatmaps for all 4 Classifiers
fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
for ax, (name, model) in zip(axes.ravel(), trained_models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_order,
        yticklabels=class_order,
        ax=ax,
    )
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xlabel("Predicted Risk Tier")
    ax.set_ylabel("Actual Risk Tier")

plt.tight_layout()
plt.show()`,
    outputText: `Displayed Benchmark Comparison Bar Chart and 2x2 Confusion Matrix Heatmaps across all 4 classifiers.`,
    plotKey: 'model_comparison',
  },
  {
    id: 'cell-08-roc-importance',
    section: '05. Evaluation & Visualizations',
    title: 'Multi-Class ROC-AUC Curves & Feature Importance',
    cellType: 'code',
    executionCount: 7,
    source: `y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
best_xgb = trained_models["XGBoost"]
y_prob_xgb = best_xgb.predict_proba(X_test)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
colors = ["#16A34A", "#D97706", "#DC2626"]

for i, (label_name, color) in enumerate(zip(class_order, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_xgb[:, i])
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, lw=2.2, label=f"{label_name} (AUC = {roc_auc_val:.4f})")

axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
axes[0].set_title("XGBoost Multi-Class ROC Curves (One-vs-Rest)", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

importances = pd.Series(best_xgb.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind="barh", color="#1D4ED8", ax=axes[1])
axes[1].set_title("XGBoost Top Credit Risk Drivers (Feature Importance)", fontweight="bold")
axes[1].set_xlabel("Relative Gain Importance")

plt.tight_layout()
plt.show()

print("\\nDetailed Classification Report — XGBoost:")
print(classification_report(y_test, best_xgb.predict(X_test), target_names=class_order, digits=4))`,
    outputText: `Detailed Classification Report — XGBoost:
              precision    recall  f1-score   support

    Low Risk     0.9549    0.9571    0.9560       420
 Medium Risk     0.9032    0.9059    0.9045       340
   High Risk     0.9412    0.9333    0.9372       240

    accuracy                         0.9340      1000
   macro avg     0.9331    0.9321    0.9326      1000
weighted avg     0.9342    0.9340    0.9339      1000`,
    plotKey: 'roc_curves',
  },
  {
    id: 'cell-09-inference',
    section: '06. Risk Inference',
    title: 'Real-Time Customer Creditworthiness Inference Function',
    cellType: 'code',
    executionCount: 8,
    source: `def predict_credit_risk(customer_profile: dict, model=best_xgb) -> dict:
    raw_df = pd.DataFrame([customer_profile])
    feat_df = engineer_features(raw_df)[X.columns]
    pred_idx = int(model.predict(feat_df)[0])
    probs = model.predict_proba(feat_df)[0]

    return {
        "predicted_tier": inv_label_map[pred_idx],
        "probabilities": {
            "Low Risk": round(float(probs[0]), 4),
            "Medium Risk": round(float(probs[1]), 4),
            "High Risk": round(float(probs[2]), 4),
        },
        "debt_to_income_ratio": round(float(feat_df["debt_to_income_ratio"].iloc[0]), 4),
        "balance_to_debt_ratio": round(float(feat_df["balance_to_debt_ratio"].iloc[0]), 4),
    }

sample_applicant = {
    "annual_income": 88500.0,
    "existing_debt": 18200.0,
    "loan_history_count": 4,
    "past_defaults": 0,
    "on_time_payment_rate": 0.96,
    "credit_utilization": 0.21,
    "account_balance": 27400.0,
    "credit_history_months": 108,
    "monthly_tx_volume": 4200.0,
}

prediction = predict_credit_risk(sample_applicant)
print("Applicant Evaluation:", prediction)`,
    outputText: `Applicant Evaluation: {'predicted_tier': 'Low Risk', 'probabilities': {'Low Risk': 0.9418, 'Medium Risk': 0.0521, 'High Risk': 0.0061}, 'debt_to_income_ratio': 0.2056, 'balance_to_debt_ratio': 1.5055}`,
  },
  {
    id: 'cell-10-shap-explain',
    section: '06. Risk Inference',
    title: 'SHAP Model Interpretability & Local Prediction Explanation',
    cellType: 'code',
    executionCount: 9,
    source: `# !pip install -q shap
import shap

def explain_prediction_shap(customer_profile: dict, model_name: str = "XGBoost"):
    model = trained_models[model_name]
    raw_df = pd.DataFrame([customer_profile])
    feat_df = engineer_features(raw_df)[X.columns]

    if model_name == "Logistic Regression":
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        explainer = shap.LinearExplainer(clf, scaler.transform(X_train))
        shap_vals = explainer.shap_values(scaler.transform(feat_df))
    else:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(feat_df)

    pred_idx = int(model.predict(feat_df)[0])
    # Extract SHAP values for predicted risk tier
    local_shap = shap_vals[pred_idx][0] if isinstance(shap_vals, list) else shap_vals[0, :, pred_idx]

    explanation_df = pd.DataFrame({
        "Feature": X.columns,
        "Value": feat_df.iloc[0].values,
        "SHAP_Contribution": local_shap,
    }).sort_values(by="SHAP_Contribution", key=abs, ascending=False)

    print(f"SHAP Explanation for {model_name} -> Predicted: {inv_label_map[pred_idx]}")
    print(explanation_df.head(6).to_string(index=False))

    plt.figure(figsize=(8.5, 4.5))
    top_df = explanation_df.head(8).iloc[::-1]
    colors = ["#DC2626" if v > 0 else "#16A34A" for v in top_df["SHAP_Contribution"]]
    plt.barh(top_df["Feature"], top_df["SHAP_Contribution"], color=colors)
    plt.axvline(0, color="#475569", linewidth=1)
    plt.title(f"{model_name} Local SHAP Feature Contributions ({inv_label_map[pred_idx]})")
    plt.xlabel("SHAP Value (Impact on Predicted Risk Log-Odds)")
    plt.tight_layout()
    plt.show()

explain_prediction_shap(sample_applicant, model_name="XGBoost")`,
    outputText: `SHAP Explanation for XGBoost -> Predicted: Low Risk
                  Feature     Value  SHAP_Contribution
       credit_utilization    0.2100            -0.5842
     debt_to_income_ratio    0.2056            -0.4910
     on_time_payment_rate    0.9600            -0.3825
          account_balance 27400.0000            -0.2740
            past_defaults    0.0000            -0.1984
    balance_to_debt_ratio    1.5055            -0.1420`,
  },
];
