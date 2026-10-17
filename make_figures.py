import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, accuracy_score, ConfusionMatrixDisplay)

os.makedirs('figures', exist_ok=True)

# --- same pipeline as experiment.ipynb ---
df = pd.read_csv('diabetes.csv')
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train_fe, X_test_fe = X_train.copy(), X_test.copy()
for c in ['Insulin', 'SkinThickness']:
    X_train_fe[c + '_missing'] = X_train[c].isna().astype(int)
    X_test_fe[c + '_missing'] = X_test[c].isna().astype(int)

orig_cols = list(X_train.columns)
imputer = SimpleImputer(strategy='median')
X_train_fe[orig_cols] = imputer.fit_transform(X_train[orig_cols])
X_test_fe[orig_cols] = imputer.transform(X_test[orig_cols])

for d in (X_train_fe, X_test_fe):
    d['Glucose_BMI'] = d['Glucose'] * d['BMI']
    d['Age_group'] = pd.cut(d['Age'], bins=[20, 30, 40, 50, 120],
                            labels=[0, 1, 2, 3]).astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_test_scaled = scaler.transform(X_test_fe)

log_reg = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train_fe, y_train)

lr_pred = log_reg.predict(X_test_scaled)
lr_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
rf_pred = rf.predict(X_test_fe)
rf_prob = rf.predict_proba(X_test_fe)[:, 1]

# --- Table 1 ---
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'ROC-AUC': [roc_auc_score(y_test, lr_prob), roc_auc_score(y_test, rf_prob)],
    'Recall': [recall_score(y_test, lr_pred), recall_score(y_test, rf_pred)],
    'Precision': [precision_score(y_test, lr_pred), precision_score(y_test, rf_pred)],
    'F1': [f1_score(y_test, lr_pred), f1_score(y_test, rf_pred)],
    'Accuracy': [accuracy_score(y_test, lr_pred), accuracy_score(y_test, rf_pred)],
}).round(3)
results.to_csv('figures/table1_results.csv', index=False)
print(results.to_string(index=False))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                        X_train_fe, y_train, cv=cv, scoring='roc_auc')
rf_cv = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42),
                        X_train_fe, y_train, cv=cv, scoring='roc_auc')
print(f'LR CV ROC-AUC: {lr_cv.mean():.3f} (+/- {lr_cv.std():.3f})')
print(f'RF CV ROC-AUC: {rf_cv.mean():.3f} (+/- {rf_cv.std():.3f})')

# --- Figure 1: metric comparison ---
metrics = ['ROC-AUC', 'Recall', 'Precision', 'F1', 'Accuracy']
x = np.arange(len(metrics))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w/2, results.loc[0, metrics], w, label='Logistic Regression')
b2 = ax.bar(x + w/2, results.loc[1, metrics], w, label='Random Forest')
ax.bar_label(b1, fmt='%.2f', fontsize=8)
ax.bar_label(b2, fmt='%.2f', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Figure 1. Test-set performance of the two models')
ax.legend()
plt.tight_layout()
plt.savefig('figures/fig1_metrics.png', dpi=200)
plt.close()

# --- Figure 2: feature importance ---
imp = pd.Series(rf.feature_importances_, index=X_train_fe.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
imp.plot(kind='barh', ax=ax)
ax.set_xlabel('Importance')
ax.set_title('Figure 2. Random forest feature importance')
plt.tight_layout()
plt.savefig('figures/fig2_importance.png', dpi=200)
plt.close()

# --- Figure 3: confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
ConfusionMatrixDisplay.from_predictions(y_test, lr_pred, ax=axes[0], colorbar=False)
axes[0].set_title('Logistic Regression')
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, ax=axes[1], colorbar=False)
axes[1].set_title('Random Forest')
fig.suptitle('Figure 3. Confusion matrices on the test set')
plt.tight_layout()
plt.savefig('figures/fig3_confusion.png', dpi=200)
plt.close()

print('Saved figures in the figures folder')