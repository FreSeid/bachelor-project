import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV

# Feature Engineering
df = pd.read_csv("./data/data_labeled_final.csv", delimiter=";", dtype=str, index_col=0)

X = df.drop(labels=["Name","Anomalie"], axis=1) # leave out Name
y = df["Anomalie"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=4712834)

# Create Pipeline
pipeline = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2, encoded_missing_value=-1,dtype=int)),
    ('isolation_forest', IsolationForest(contamination=0.01))
])

# Hyperparameter tuning for contamination parameter
param_grid = {
    'isolation_forest__contamination': [0.001, 0.01, 0.1, 0.2, 0.3]
}

# Custom scorer for maximizing precision of the outliers (class 1)
def outlier_precision(y_true, y_pred):
    y_pred = (y_pred == -1).astype(int)  # Convert IsolationForest predictions to binary labels: -1 -> 1 (outlier), 1 -> 0 (normal)
    return precision_score(y_true, y_pred, pos_label=1)

scorer = make_scorer(outlier_precision)

grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Evaluation
y_pred = best_estimator.predict(X_test)

y_pred[y_pred == 1] = 0  # Normal
y_pred[y_pred == -1] = 1  # Outlier

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(best_params)