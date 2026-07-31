import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target_col = 'result'

behavior_features = [
    'sex',
    'has_photo',
    'has_mobile',
    'followers_count',
    'relation',
    'city',
    'last_seen'
]

X_train = train_df[behavior_features].fillna("Unknown")
y_train = train_df[target_col]

X_test = test_df[behavior_features].fillna("Unknown")

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

model = RandomForestClassifier(n_estimators=300,max_depth=12,random_state=42,n_jobs=-1)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission = pd.DataFrame({"id": test_df["id"], "result": predictions})
submission.to_csv("submission_model3.csv", index=False)
