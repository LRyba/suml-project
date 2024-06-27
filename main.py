import pandas as pd

from data_preparation.data_prep import change_column_names, encode_categorical_features, replace_column_values, remove_duplicates, split_data
from model_training.model_train import train_logistic_regression, train_decision_tree, train_random_forest, evaluate_model, export_model

def main():
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

    #model = train_logistic_regression(X_train, y_train)
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    file_name = "random_forest_model.pkl"
    export_model(model, f"ml_models/{file_name}")

if __name__ == "__main__":
    main()
