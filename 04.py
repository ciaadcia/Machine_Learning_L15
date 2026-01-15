import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_column = train.columns[-1]

y = train[target_column]
train = train.drop(target_column, axis=1)

combined = pd.concat([train, test], axis=0)

encoder = LabelEncoder()
for col in combined.columns:
    if combined[col].dtype == 'object':
        combined[col] = encoder.fit_transform(combined[col].astype(str))

X = combined.iloc[:len(train)]
X_test = combined.iloc[len(train):]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_val)
print("Accuracy Logistic Regression (No Scaling):",
      accuracy_score(y_val, pred))

test["Prediction"] = model.predict_proba(X_test)[:, 1]
test.to_csv("prediction_logistic_noscale.csv", index=False)
