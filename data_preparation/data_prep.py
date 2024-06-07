import os
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    """
    A class used to preprocess and manage a dataset.
    """

    def __init__(self, filename, train=0.8, test=0.15, validation=0.05, seed=50):
        """
        Initializes the Dataset class with a dataset file and split proportions.

        filename : The name of the dataset file.
        train : The proportion of data to be used for training (default is 0.8).
        test : The proportion of data to be used for testing (default is 0.15).
        validation : The proportion of data to be used for validation (default is 0.05).
        seed : The random seed for data splitting (default is 50).
        """
        self.full_dataset_path = os.path.join(os.path.curdir, 'datasets', filename)
        self.full_dataset = pd.read_csv(self.full_dataset_path)
        self.train_set = pd.DataFrame()
        self.validate_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        self.train_proportion = train
        self.test_proportion = test
        self.validate_proportion = validation
        self.seed = seed

    def clean_missing_vals(self):
        """
        Removes rows with missing values from the dataset.
        """
        self.full_dataset.dropna(axis=0, inplace=True)

    def fill_missing_vals(self):
        """
        Fills missing values in the dataset with the mean of each column.
        """
        for column in self.full_dataset.columns:
            self.full_dataset[column] = self.full_dataset[column].fillna(self.full_dataset[column].mean())

    def clean_outliers(self):
        """
        Removes outliers from the dataset using the IQR method.
        """
        q1 = self.full_dataset.quantile(0.25)
        q3 = self.full_dataset.quantile(0.75)
        iqr = q3 - q1
        outliers = (self.full_dataset < (q1 - 1.5 * iqr)) | (self.full_dataset > (q3 + 1.5 * iqr))
        self.full_dataset = self.full_dataset[~outliers.any(axis=1)]

    def remove_columns(self, cols_to_remove):
        """
        Removes specified columns from the dataset.

        cols_to_remove : List of column names to be removed.
        """
        self.full_dataset.drop(cols_to_remove, axis=1, inplace=True)

    def normalize(self, cols_to_normalize):
        """
        Normalizes specified columns in the dataset.

        cols_to_normalize : List of column names to be normalized.
        """
        for column in self.full_dataset.columns:
            if column in cols_to_normalize:
                self.full_dataset[column] = self.full_dataset[column] / self.full_dataset[column].abs().max()

    def split_data(self):
        """
        Splits the dataset into training, validation, and test sets based on the specified proportions.
        """
        self.train_set, self.test_set = train_test_split(
            self.full_dataset,
            test_size=1 - self.train_proportion,
            random_state=self.seed
        )
        validate_size = self.validate_proportion / (self.validate_proportion + self.test_proportion)
        self.test_set, self.validate_set = train_test_split(
            self.test_set,
            test_size=validate_size,
            random_state=self.seed
        )

    def transform_text_values(self, trans_dict):
        """
        Transforms text values in the dataset using a provided dictionary.

        trans_dict : Dictionary where keys are column names and values are dictionaries mapping original values to new values.
        """
        for column, mapping in trans_dict.items():
            if column in self.full_dataset.columns:
                self.full_dataset[column].replace(mapping, inplace=True)
    
    def one_hot_encode(self, column):
        """
        Performs one-hot encoding on a specified column.

        column : The name of the column to be one-hot encoded.
        """
        dummies = pd.get_dummies(self.full_dataset[column], prefix=column)
        self.full_dataset.drop(column, axis=1, inplace=True)
        self.full_dataset = pd.concat([self.full_dataset, dummies], axis=1)