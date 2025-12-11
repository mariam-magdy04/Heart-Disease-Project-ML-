import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier   

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv(r"C:\Users\HP\Desktop\archive\heart_statlog_cleveland_hungary_final.csv")

# EDA 
print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe(), "\n")
print("Missing values:\n", df.isna().sum(), "\n")

df.hist(figsize=(12, 10)); plt.tight_layout(); plt.show()
corr = df.corr(numeric_only=True)
plt.matshow(corr); plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(); plt.show()
plt.scatter(df["age"], df["max heart rate"], c=df["target"])
plt.xlabel("Age"); plt.ylabel("Max Heart Rate"); plt.show()


continuous = ["age","resting bp s","cholesterol","max heart rate","oldpeak"]
def remove_outliers(df, cols):
    for c in cols:
        Q1, Q3 = df[c].quantile([0.25,0.75])
        IQR = Q3 - Q1
        df = df[(df[c] >= Q1-1.5*IQR) & (df[c] <= Q3+1.5*IQR)]
    return df

df = remove_outliers(df, continuous)

# Preprocessing
X = df.drop("target", axis=1)
y = df["target"]

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.columns.difference(categorical).tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modeling 
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    results.append([name, acc, prec, rec, f1])
 
results_df = pd.DataFrame(results, columns=["Model","Acc","Prec","Rec","F1"])
print("\n=== Comparison ===")
print(results_df.sort_values("F1", ascending=False))
