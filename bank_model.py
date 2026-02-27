import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & peek ──────────────────────────────────────────────────────────
df = pd.read_csv("/mnt/user-data/uploads/bank.csv")
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['deposit'].value_counts()}\n")

# ── 2. Feature engineering ──────────────────────────────────────────────────
data = df.copy()

# Encode target
data["deposit"] = (data["deposit"] == "yes").astype(int)

# Bin age into life-stage buckets (numeric codes)
data["age_bucket"] = pd.cut(data["age"],
                            bins=[0, 30, 40, 55, 100],
                            labels=[0, 1, 2, 3]).astype(int)

# Call efficiency: duration per campaign contact
data["call_efficiency"] = data["duration"] / (data["campaign"] + 1)

# Was this client contacted before?
data["was_contacted_before"] = (data["pdays"] != -1).astype(int)

# Month to numeric (seasonal signal)
month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
             "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
data["month_num"] = data["month"].map(month_map)

# Drop raw columns we've already encoded or don't need
data.drop(columns=["month"], inplace=True)

# Label-encode all remaining object columns
le = LabelEncoder()
cat_cols = data.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    data[col] = le.fit_transform(data[col].astype(str))

# ── 3. Split ────────────────────────────────────────────────────────────────
X = data.drop(columns=["deposit"])
y = data["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train ────────────────────────────────────────────────────────────────
clf = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.07,
    max_depth=4,
    subsample=0.85,
    min_samples_leaf=20,
    random_state=7
)
clf.fit(X_train_sc, y_train)

# ── 5. Evaluate ─────────────────────────────────────────────────────────────
y_pred  = clf.predict(X_test_sc)
y_proba = clf.predict_proba(X_test_sc)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Deposit","Subscribed"]))

cv_scores = cross_val_score(clf, X_train_sc, y_train, cv=5, scoring="roc_auc")
print(f"5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Plots ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Bank Term Deposit Subscription – Model Results", fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# (a) Confusion matrix
ax0 = fig.add_subplot(gs[0, 0])
cm  = confusion_matrix(y_test, y_pred)
im  = ax0.imshow(cm, cmap="Blues")
ax0.set_xticks([0,1]); ax0.set_yticks([0,1])
ax0.set_xticklabels(["No","Yes"]); ax0.set_yticklabels(["No","Yes"])
ax0.set_xlabel("Predicted"); ax0.set_ylabel("Actual")
ax0.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax0.text(j, i, cm[i,j], ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=13)

# (b) ROC curve
ax1 = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.3f}")
ax1.plot([0,1],[0,1],"--", color="gray", lw=1)
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve"); ax1.legend(loc="lower right")

# (c) Feature importances (top 10)
ax2 = fig.add_subplot(gs[0, 2])
feat_imp = pd.Series(clf.feature_importances_, index=X.columns).nlargest(10)
feat_imp[::-1].plot(kind="barh", ax=ax2, color="#10B981")
ax2.set_title("Top 10 Feature Importances"); ax2.set_xlabel("Importance")

# (d) Predicted probability distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(y_proba[y_test==0], bins=30, alpha=0.6, label="No Deposit", color="#EF4444")
ax3.hist(y_proba[y_test==1], bins=30, alpha=0.6, label="Subscribed", color="#3B82F6")
ax3.set_xlabel("Predicted Probability"); ax3.set_ylabel("Count")
ax3.set_title("Probability Distribution by Class"); ax3.legend()

# (e) CV scores
ax4 = fig.add_subplot(gs[1, 1])
folds = [f"Fold {i+1}" for i in range(5)]
bars  = ax4.bar(folds, cv_scores, color="#8B5CF6", alpha=0.8)
ax4.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean={cv_scores.mean():.3f}")
ax4.set_ylim(0.7, 1.0); ax4.set_title("5-Fold CV AUC Scores"); ax4.legend()
ax4.set_ylabel("AUC")

# (f) Metrics summary text box
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
summary = (
    f"  ── Model Summary ──\n\n"
    f"  Algorithm  : Gradient Boosting\n"
    f"  Accuracy   : {acc:.2%}\n"
    f"  ROC-AUC    : {auc:.4f}\n"
    f"  CV AUC     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n"
    f"  Train size : {len(X_train)}\n"
    f"  Test size  : {len(X_test)}\n"
    f"  Features   : {X.shape[1]}"
)
ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
         fontsize=11, verticalalignment="top",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#F0FDF4", edgecolor="#10B981"))

plt.savefig("/mnt/user-data/outputs/bank_model_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved.")

# ── 7. Quick prediction demo ─────────────────────────────────────────────────
print("\n── Sample Predictions (first 5 test records) ──")
sample = X_test.iloc[:5].copy()
sample_sc = scaler.transform(sample)
preds  = clf.predict(sample_sc)
probas = clf.predict_proba(sample_sc)[:,1]
for i, (p, pr) in enumerate(zip(preds, probas)):
    label = "Subscribed ✓" if p == 1 else "No Deposit ✗"
    print(f"  Record {i+1}: {label}  (confidence: {pr:.2%})")
