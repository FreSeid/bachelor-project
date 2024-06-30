import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
from Levenshtein import distance
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV

# Feature Engineering
df = pd.read_csv("./data/data_labeled_final.csv", delimiter=";", dtype=str, index_col=0)

X = df.drop(labels=["Anomalie"], axis=1)
X["Name"] = X["Name"].fillna("")
y = df["Anomalie"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=4712834)

# Define custom Transformers
class HammingDistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.encoder.fit(X[self.categorical_features])
        return self

    def transform(self, X):
        X_encoded = self.encoder.transform(X[self.categorical_features])
        hamming_matrix = pairwise_distances(X_encoded, metric='hamming')
        hamming_matrix_scaled = self.scaler.fit_transform(hamming_matrix)
        return hamming_matrix_scaled

    def transform_test(self, X_train, X_test):
        X_train_encoded = self.encoder.transform(X_train[self.categorical_features])
        X_test_encoded = self.encoder.transform(X_test[self.categorical_features])
        hamming_matrix = pairwise_distances(X_test_encoded, X_train_encoded, metric='hamming')
        hamming_matrix_scaled = self.scaler.transform(hamming_matrix)
        return hamming_matrix_scaled

class LevenshteinDistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_feature):
        self.text_feature = text_feature
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text_data = X[self.text_feature].values
        dist_matrix = np.array([[distance(text_data[i], text_data[j]) for j in range(len(text_data))] for i in range(len(text_data))])
        dist_matrix_scaled = self.scaler.fit_transform(dist_matrix)
        return dist_matrix_scaled

    def transform_test(self, X_train, X_test):
        train_data = X_train[self.text_feature].values
        test_data = X_test[self.text_feature].values
        dist_matrix = np.array([[distance(test_data[i], train_data[j]) for j in range(len(train_data))] for i in range(len(test_data))])
        dist_matrix_scaled = self.scaler.transform(dist_matrix)
        return dist_matrix_scaled

# Create Transformers
hamming_transformer = HammingDistanceTransformer(categorical_features=["Kategorie", "Typ", "Klasse", "Segment", "Gruppe", "Division", "Teil"])
levenshtein_transformer = LevenshteinDistanceTransformer(text_feature="Name")

hamming_matrix_train = hamming_transformer.fit_transform(X_train)
levenshtein_matrix_train = levenshtein_transformer.fit_transform(X_train)

# Hyperparameter tuning
lof = LocalOutlierFactor(metric='precomputed', novelty=True)
svm = OneClassSVM(kernel="precomputed")

# Custom scorer for maximizing precision of the outliers 
def outlier_precision(y_true, y_pred):
    y_pred = (y_pred == -1).astype(int)  # Convert predictions to binary labels: -1 -> 1 (outlier), 1 -> 0 (normal)
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0.0)

scorer = make_scorer(outlier_precision)

lof_grid_search = GridSearchCV(lof, {"contamination": [0.001, 0.01, 0.1]}, cv=2, scoring=scorer, error_score="raise")
svm_grid_search = GridSearchCV(svm, {"nu": [ 0.1, 0.3, 0.5, 0.8]}, cv=2, scoring=scorer, error_score="raise")

# Manual tuning for levenshtein scale so distance matrix is not completely recalculated every loop
def tune_levenshtein_distance(grid_search):
    lscale_values = [0, 0.3, 0.6]
    best_score = 0
    best_estimator = None
    best_lscale = 0
    best_params = None

    for levenshtein_scale in lscale_values:
        combined_matrix_train = hamming_matrix_train + levenshtein_matrix_train * levenshtein_scale
        grid_search.fit(combined_matrix_train, y_train)
        score = grid_search.best_score_
        if score > best_score:
            best_score = score
            best_estimator = grid_search.best_estimator_
            best_lscale = levenshtein_scale
            best_params = grid_search.best_params_

    return best_estimator, best_lscale, best_params

best_lof, lof_lscale, best_params_lof = tune_levenshtein_distance(lof_grid_search)

best_svm, svm_lscale, best_params_svm = tune_levenshtein_distance(svm_grid_search)

# Evaluation
lof_test_distance_matrix = hamming_transformer.transform_test(X_train, X_test) + levenshtein_transformer.transform_test(X_train, X_test) * lof_lscale

y_pred_lof = best_lof.predict(lof_test_distance_matrix)

y_pred_lof[y_pred_lof == 1] = 0  # Normal
y_pred_lof[y_pred_lof == -1] = 1  # Outlier

print("Best lscale LOF:")
print(lof_lscale)
print("\nBest contamination LOF:")
print(best_params_lof)
print("\nConfusion Matrix LOF:")
print(confusion_matrix(y_test, y_pred_lof))
print("\nClassification Report LOF:")
print(classification_report(y_test, y_pred_lof))

svm_test_distance_matrix = hamming_transformer.transform_test(X_train, X_test) + levenshtein_transformer.transform_test(X_train, X_test) * svm_lscale

y_pred_svm = best_svm.predict(svm_test_distance_matrix)

y_pred_svm[y_pred_svm == 1] = 0  # Normal
y_pred_svm[y_pred_svm == -1] = 1  # Outlier

print("Best lscale SVM:")
print(svm_lscale)
print("\nBest contamination SVM:")
print(best_params_svm)
print("Confusion Matrix SVM:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report SVM:")
print(classification_report(y_test, y_pred_svm))