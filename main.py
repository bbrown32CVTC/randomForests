# This is a Python script for Random Forest Classification.

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create the path to the dataset and set dataframe to the data
path_to_file = "../CSC 419 HW 3 Spambase 1000.csv"
df = pd.read_csv(path_to_file)

# Test 1
# Create X and y dataframes
y = df['spam']
X = df[['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']]

# Create train_test_split of 70% training and 30% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# Create the Random Forest Regressor model with all default parameters and train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Classify the data with a prediction using the model
y_pred = rf_model.predict(X_test)

# # Test 2
# # Create X and y dataframes
# y = df['spam']
# X = df[['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']]
#
# # Create train_test_split of 70% training and 30% testing data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
#
# # Create the Random Forest Regressor model with all default parameters and train the model
# rf_model = RandomForestClassifier(max_depth=2)
# rf_model.fit(X_train, y_train)
#
# # Classify the data with a prediction using the model
# y_pred = rf_model.predict(X_test)


# # Test 3
# # Create X and y dataframes
# y = df['spam']
# X = df[['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'char_freq_!']]
#
# # Create train_test_split of 70% training and 30% testing data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
#
# # Create the Random Forest Regressor model with all default parameters and train the model
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
#
# # Classify the data with a prediction using the model
# y_pred = rf_model.predict(X_test)


# Evaluate the accuracy of the model using accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Accuracy Percent:", round(accuracy * 100, 2), "%")

# Report the relative importance of each feature to the model
importances = rf_model.feature_importances_
columns = X.columns
i = 0

while i < len(columns):
    print(f" The importance of feature '{columns[i]}' is {round(importances[i] * 100, 2)}%")
    i += 1
