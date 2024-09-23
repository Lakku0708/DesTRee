import pandas as pd
import numpy as np
import math

# Load the dataset
dataset_path = 'Enjoy sports.csv'
dataset = pd.read_csv(dataset_path)

# Drop unnecessary columns
dataset = dataset.drop(columns=['Day'])


# Function to calculate uncertainty (entropy)
def calculate_uncertainty(target_column):
    unique_values, value_counts = np.unique(target_column, return_counts=True)
    uncertainty = 0
    for count in value_counts:
        probability = count / sum(value_counts)
        uncertainty += -probability * np.log2(probability)
    return uncertainty


# Function to compute gain (information gain)
def compute_gain(dataframe, feature, label):
    total_uncertainty = calculate_uncertainty(dataframe[label])

    # Calculate the weighted uncertainty for each unique value of the feature
    feature_values, value_counts = np.unique(dataframe[feature], return_counts=True)
    weighted_uncertainty = 0
    for i in range(len(feature_values)):
        subset = dataframe[dataframe[feature] == feature_values[i]]
        prob = value_counts[i] / np.sum(value_counts)
        weighted_uncertainty += prob * calculate_uncertainty(subset[label])

    # Information gain is the difference between total uncertainty and weighted uncertainty
    return total_uncertainty - weighted_uncertainty


# Recursive function to generate the decision structure
def create_decision_structure(current_data, full_data, remaining_features, label, default_class=None):
    # Case 1: If all labels are the same, return the label
    if len(np.unique(current_data[label])) <= 1:
        return np.unique(current_data[label])[0]

    # Case 2: If no data is left, return the most frequent class in the original dataset
    elif len(current_data) == 0:
        return np.unique(full_data[label])[np.argmax(np.unique(full_data[label], return_counts=True)[1])]

    # Case 3: If no features are left, return the default class
    elif len(remaining_features) == 0:
        return default_class

    # Case 4: Build the decision tree
    else:
        # Default class is the most common label in the current data
        default_class = np.unique(current_data[label])[np.argmax(np.unique(current_data[label], return_counts=True)[1])]

        # Find the best feature with the highest information gain
        feature_gains = [compute_gain(current_data, feature, label) for feature in remaining_features]
        best_feature_index = np.argmax(feature_gains)
        best_feature = remaining_features[best_feature_index]

        # Create the root node of the decision tree
        decision_tree = {best_feature: {}}

        # Remove the best feature from the list of remaining features
        remaining_features = [i for i in remaining_features if i != best_feature]

        # Build the branches for each unique value of the best feature
        for value in np.unique(current_data[best_feature]):
            sub_data = current_data[current_data[best_feature] == value]
            subtree = create_decision_structure(sub_data, full_data, remaining_features, label, default_class)
            decision_tree[best_feature][value] = subtree

        return decision_tree


# Define the list of features
attributes = list(dataset.columns[:-1])

# Build the decision structure
decision_structure = create_decision_structure(dataset, dataset, attributes, label='Decision')

# Pretty print the decision tree
import pprint

pprint.pprint(decision_structure)
