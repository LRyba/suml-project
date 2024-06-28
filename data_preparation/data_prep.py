from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

def remove_duplicates(df):
    """
    Return new DataFrame with removed duplicate rows.
    """
    return df.drop_duplicates()

def change_column_names(df, new_column_names):
    """
    Return changed DataFrame wih new column names.
    """
    return df.rename(columns=new_column_names)

def encode_categorical_features(df, categorical_features):
    """
    Return encoded categorical features using OrdinalEncoder.
    """
    encoder = OrdinalEncoder()
    df[categorical_features] = encoder.fit_transform(df[categorical_features])
    return df

def replace_column_values(df, column, value_map):
    """
    Return new DataFrame with replaced values in a specific column using a mapping dictionary.
    """
    df[column] = df[column].replace(value_map)
    return df

def split_data(X, y, test_size=0.20, random_state=101):
    """
    Return splitted data into training and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
