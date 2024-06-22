import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder

def remove_duplicates(df):
    return df.drop_duplicates()

def change_column_names(df, new_column_names):
    return df.rename(columns=new_column_names)


def encode_categorical_features(df, categorical_features):
    encoder = OrdinalEncoder()
    df[categorical_features] = encoder.fit_transform(df[categorical_features])
    return df


def replace_column_values(df, column, value_map):
    df[column] = df[column].replace(value_map)
    return df


def split_data(X, y, test_size=0.20, random_state=101):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_logistic_regression(X_train, y_train, max_iter=50000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))