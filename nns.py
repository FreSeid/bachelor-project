import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import matplotlib.pyplot as plt

# Data Preperation
df = pd.read_csv("./data/data_labeled_final.csv", delimiter=";", dtype=str, index_col=0)
df = df[df["Anomalie"] != 1]
df = df.drop(labels=["Name","Anomalie"], axis=1)
data = df.drop_duplicates()

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
data_encoded = encoder.fit_transform(data)

anomalies = pd.read_csv("./data/anomalies.csv", delimiter=";", dtype=str)
anomalies = anomalies.drop(labels=["Name","Anomalie"], axis=1)

# Create Ball Tree
nn = NearestNeighbors(n_neighbors=5, algorithm="ball_tree", metric="hamming")
nn.fit(data_encoded)

# Function to find wrong values in an anomaly
def find_differences(anomaly, other):
    differences = {}
    for col in anomaly.index:
        if anomaly[col] != other[col]:
            differences[col] = other[col]
    return differences

# Dictionary for gathering results
percentage_dict = defaultdict(int)

# Iterate Anomalies
for i in range(0, len(anomalies), 2):
    anomaly = anomalies.iloc[i]
    correct_version = anomalies.iloc[i+1]
    
    corrected_columns = find_differences(anomaly, correct_version)
    
    # Find Neighbors
    anomaly_encoded = encoder.transform(anomalies.iloc[[i]])
    distances, indices = nn.kneighbors(anomaly_encoded)

    best_match_percentage = 0
    
    # Check if any neighbors match
    for neighbor_index in indices[0]:
        neighbor = data.iloc[neighbor_index]
        neighbor_differences = find_differences(anomaly, neighbor)
        
        # Variable used to check if all columns are corrected in neighbor
        all_col_corrected = True
        for col in corrected_columns:
            column_different_in_neighbor = col in neighbor_differences
            neighbor_wrong = neighbor_differences[col] != corrected_columns[col]
            if not column_different_in_neighbor or neighbor_wrong:
                all_col_corrected = False
                break
                
        if all_col_corrected:
            # Calculate percentage of matching columns
            match_count = 0
            for col in anomaly.index:
                if anomaly[col] == neighbor[col]:
                    match_count += 1
            match_percentage = (match_count / len(anomaly.index)) * 100

            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage

    percentage_dict[best_match_percentage] += 1

# Results
for percent, count in percentage_dict.items():
    print(f"{percent:.2f}%: {count} times")