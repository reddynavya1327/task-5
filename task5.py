

# Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Loaded Successfully")
print("Shape:", X.shape)

# -----------------------------
# Step 2: Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 3: Decision Tree
# -----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# -----------------------------
# Step 4: Control Overfitting (limit depth)
# -----------------------------
dt_depth = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_depth.fit(X_train, y_train)

y_pred_dt_depth = dt_depth.predict(X_test)
print("Decision Tree (Depth=3) Accuracy:", accuracy_score(y_test, y_pred_dt_depth))

# -----------------------------
# Step 5: Visualize Tree
# -----------------------------
plt.figure(figsize=(15,8))
plot_tree(dt_depth, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# -----------------------------
# Step 6: Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# -----------------------------
# Step 7: Feature Importance
# -----------------------------
importances = rf.feature_importances_
features = pd.Series(importances, index=data.feature_names).sort_values(ascending=False)

print("\nTop Features:\n", features.head())

# Plot feature importance
features.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

# -----------------------------
# Step 8: Cross Validation
# -----------------------------
cv_scores = cross_val_score(rf, X, y, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))
