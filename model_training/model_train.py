"""
Training model module
"""

import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import pandas as pd

from data_preparation.data_prep import (change_column_names,
                                        encode_categorical_features,
                                        replace_column_values,
                                        remove_duplicates,
                                        split_data)

def train_logistic_regression(X_train, y_train, max_iter=50000):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=10, random_state=101):
    """
    Train a Random Forest model.
    """
    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    forest.fit(X_train, y_train)
    print(f"Las: {forest.score(X_train, y_train)}")
    return forest

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree model.
    """
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    print(f"Drzewa decyzyjne: {tree.score(X_train, y_train)}")
    return tree

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using test data and print a classification report.
    """
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

def export_model(model, file_path):
    """
    Export the trained model to a file.
    """
    dump(model, open(file_path, "wb"))

def main():
    """
    Main function to load data, preprocess it, train a model, evaluate it, and export the model.
    """
    df = pd.read_csv("data/ObesityDataSet.csv")
    df = remove_duplicates(df)
    new_column_names = {
                        'Gender': 'gender',
                        'Age': 'age',
                        'Height': 'height',
                        'Weight': 'weight',
                        'SMOKE': 'smoke',
                        'FAVC': 'highcal_intake',
                        'FCVC': 'veg_intake',
                        'NCP': 'meals_daily',
                        'CAEC': 'snacking',
                        'CH2O': 'water_intake_daily',
                        'SCC': 'track_cal_intake',
                        'FAF': 'physical_weekly',
                        'TUE': 'tech_usage_daily',
                        'CALC': 'alcohol_intake',
                        'MTRANS': 'transport_mode'
                       }
    df = change_column_names(df, new_column_names)
    categorical_features = df.select_dtypes('object').columns.drop('NObeyesdad')
    df = encode_categorical_features(df, categorical_features)
    value_map = {'Insufficient_Weight':0,
                 'Normal_Weight':1,
                 'Overweight_Level_I':2,
                 'Overweight_Level_II':3,
                 'Obesity_Type_I':4,
                 'Obesity_Type_II':5,
                 'Obesity_Type_III':6}
    df = replace_column_values(df, 'NObeyesdad', value_map)
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    file_name = "random_forest_model.pkl"
    export_model(model, f"ml_models/{file_name}")

if __name__ == "__main__":
    main()
