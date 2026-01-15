import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_column = train.columns[-1]
y = train[target_column]
X_train_data = train.drop(target_column, axis=1)

combined = pd.concat([X_train_data, test], axis=0)

encoder = LabelEncoder()
for col in combined.columns:
    if combined[col].dtype == 'object':
        combined[col] = encoder.fit_transform(combined[col].astype(str))

scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

kmeans = KMeans(n_clusters=3, random_state=42)
combined["User_Cluster"] = kmeans.fit_predict(combined_scaled)

X = combined.iloc[:len(X_train_data)]
X_test = combined.iloc[len(X_train_data):]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred_val = model.predict(X_val)
print("Hipotesis 3 - Clustering Based Model Accuracy:",
      accuracy_score(y_val, pred_val))

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    test.columns[0]: test[test.columns[0]],
    target_column: test_predictions
})

submission.to_csv("submission_cluster_based.csv", index=False)
