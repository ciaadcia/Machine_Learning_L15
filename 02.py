import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Ambil kolom target (diasumsikan kolom terakhir)
target_column = train.columns[-1]

y = train[target_column]
X_train_data = train.drop(target_column, axis=1)

# Gabungkan train + test untuk encoding (ANTI UNSEEN LABEL)
combined = pd.concat([X_train_data, test], axis=0)

# Encode kolom kategorikal
encoder = LabelEncoder()
for col in combined.columns:
    if combined[col].dtype == 'object':
        combined[col] = encoder.fit_transform(combined[col])

# Pisahkan kembali
X = combined.iloc[:len(X_train_data)]
X_test = combined.iloc[len(X_train_data):]

# Split untuk evaluasi
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluasi
pred_val = model.predict(X_val)
print("Hipotesis 2 - Random Forest Accuracy:", accuracy_score(y_val, pred_val))

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    test.columns[0]: test[test.columns[0]],  # kolom ID
    target_column: test_predictions          # nama target HARUS SAMA
})

submission.to_csv("submission_random_forest.csv", index=False)

