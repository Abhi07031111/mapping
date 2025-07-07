# scripts/supervised_predict.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

# === Step 1: Load training data
df = pd.read_pickle("./data/supervised_training_data.pkl")

X = np.vstack(df["Embedding"])

# === Step 2: Prepare labels
mlb_nfri = MultiLabelBinarizer()
mlb_kpci = MultiLabelBinarizer()
mlb_policy = MultiLabelBinarizer()

y_nfri = mlb_nfri.fit_transform(df["Mapped NFRI"])
y_kpci = mlb_kpci.fit_transform(df["KPCI ID"])
y_policy = mlb_policy.fit_transform(df["Policy ID"])

# === Step 3: Train/Test split
X_train, X_test, y_nfri_train, y_nfri_test = train_test_split(X, y_nfri, test_size=0.2, random_state=42)
_, _, y_kpci_train, y_kpci_test = train_test_split(X, y_kpci, test_size=0.2, random_state=42)
_, _, y_policy_train, y_policy_test = train_test_split(X, y_policy, test_size=0.2, random_state=42)

# === Step 4: Train Models
clf_nfri = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf_kpci = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf_policy = MultiOutputClassifier(LogisticRegression(max_iter=1000))

print("🔄 Training models...")
clf_nfri.fit(X_train, y_nfri_train)
clf_kpci.fit(X_train, y_kpci_train)
clf_policy.fit(X_train, y_policy_train)
print("✅ Training complete.")

# === Step 5: Predict
y_nfri_pred = clf_nfri.predict(X_test)
y_kpci_pred = clf_kpci.predict(X_test)
y_policy_pred = clf_policy.predict(X_test)

# === Step 6: Evaluate
def evaluate_model(y_true, y_pred, label):
    print(f"\n📊 Evaluation for {label}")
    print(classification_report(y_true, y_pred, zero_division=0))

evaluate_model(y_nfri_test, y_nfri_pred, "Mapped NFRI")
evaluate_model(y_kpci_test, y_kpci_pred, "KPCI ID")
evaluate_model(y_policy_test, y_policy_pred, "Policy ID")

# === Step 7: Optional - Save Models
import joblib
joblib.dump(clf_nfri, "./models/clf_nfri.pkl")
joblib.dump(clf_kpci, "./models/clf_kpci.pkl")
joblib.dump(clf_policy, "./models/clf_policy.pkl")

print("\n💾 Models saved to ./models/")
