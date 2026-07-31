import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_column = train.columns[-1]

y = train[target_column]
train_features = train.drop(target_column, axis=1)

combined = pd.concat([train_features, test], axis=0)

encoder = LabelEncoder()
for col in combined.columns:
    if combined[col].dtype == 'object':
        combined[col] = encoder.fit_transform(combined[col])

X = combined.iloc[:len(train_features)]
X_test = combined.iloc[len(train_features):]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_val)

print("Accuracy Logistic Regression:", accuracy_score(y_val, pred))
result = pd.DataFrame({'ID': test['id'],'result': model.predict(X_test)})
result.to_csv('file_name.csv', index=False)
